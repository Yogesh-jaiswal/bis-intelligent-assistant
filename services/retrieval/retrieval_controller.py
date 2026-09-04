"""
services/retrieval/retrieval_controller.py
============================================

Evidence-based Retrieval Controller for the BIS Intelligent Assistant.

Orchestrates multi-hop retrieval and evidence-based clarification dynamically.
After initial retrieval, the controller inspects the accumulated database records
and documentary passages to decide whether:
  1. COMPLETE: Current evidence is sufficient to synthesize an authoritative answer.
  2. RETRIEVE_MORE: Additional structured or vector retrieval is required for related
     entities (e.g., standard → certification scheme → testing laboratories).
  3. NEED_CLARIFICATION: Evidence contains multiple distinct alternatives or ambiguities
     that require user disambiguation (e.g., voltage ratings differentiating IS 694 vs IS 7098).

Enforces safety bounds:
  - Maximum retrieval hops (from BaseAppSettings.MAX_RETRIEVAL_HOPS)
  - Maximum evidence count (from BaseAppSettings.MAX_EVIDENCE_COUNT)
  - Operation signature deduplication (prevents infinite retrieval loops)
  - No arbitrary SQL generation — all operations are strictly typed DatabaseOperations.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

from configs import get_settings
from services.ai.engine import AIEngine
from services.query_analyser.analyser_schema import DatabaseOperation, QueryPlan
from services.query_executor.executor_schema import QueryExecutionResult
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import ClarificationQuestion

logger = logging.getLogger(__name__)


class ControllerDecision(str, Enum):
    """High-level decision emitted by the Retrieval Controller."""
    COMPLETE = "COMPLETE"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"


class StructuredControllerOutput(BaseModel):
    """Structured LLM schema for retrieval controller decisions."""
    decision: ControllerDecision = Field(
        ...,
        description="COMPLETE if evidence is sufficient, RETRIEVE_MORE if another step is needed, NEED_CLARIFICATION if user input is required."
    )
    reason: str = Field(
        ...,
        description="Reasoning explaining the decision based on the current evidence."
    )
    next_operations: list[DatabaseOperation] = Field(
        default_factory=list,
        description="Next DatabaseOperations to execute if decision is RETRIEVE_MORE."
    )
    next_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the next database operations (e.g. standard_number, product_name, state)."
    )
    next_rag_query: str | None = Field(
        default=None,
        description="Optional query for document vector search if RAG retrieval is required."
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask the user if decision is NEED_CLARIFICATION."
    )
    clarification_options: list[str] | None = Field(
        default=None,
        description="List of options if clarification is multiple-choice."
    )
    clarification_input_type: Literal["text", "number", "select", "multi_select", "boolean"] = Field(
        default="text",
        description="Input type for the clarification question."
    )


class RetrievalController:
    """
    Evaluates accumulated retrieval evidence and determines next retrieval steps or clarification.
    """

    def __init__(self, ai_engine: AIEngine | None = None):
        self.ai_engine = ai_engine or AIEngine(get_settings().AI_MODEL)

    def evaluate(
        self,
        user_query: str,
        plan: QueryPlan,
        accumulated_db_results: dict[str, list[dict[str, Any]]],
        accumulated_rag_chunks: list[RetrievedChunk],
        hop_count: int,
        executed_signatures: set[str],
    ) -> StructuredControllerOutput:
        """
        Evaluate current evidence state and decide next action.

        :param user_query: Original user query.
        :param plan: Initial QueryPlan.
        :param accumulated_db_results: Grouped DB records collected so far.
        :param accumulated_rag_chunks: RAG passages collected so far.
        :param hop_count: Current retrieval hop index (1-based).
        :param executed_signatures: Set of operation signatures already executed to prevent loops.
        :return: StructuredControllerOutput with decision and next actions.
        """
        settings = get_settings()

        # Hard bounds: Max hops or max evidence reached -> force COMPLETE
        total_records = sum(len(recs) for recs in accumulated_db_results.values())
        total_evidence = total_records + len(accumulated_rag_chunks)

        if hop_count >= settings.MAX_RETRIEVAL_HOPS:
            logger.info("[CONTROLLER] Max retrieval hops limit (%d) reached -> forcing COMPLETE", settings.MAX_RETRIEVAL_HOPS)
            return StructuredControllerOutput(
                decision=ControllerDecision.COMPLETE,
                reason=f"Reached maximum permitted retrieval hops ({settings.MAX_RETRIEVAL_HOPS}).",
            )

        if total_evidence >= settings.MAX_EVIDENCE_COUNT:
            logger.info("[CONTROLLER] Max evidence count limit (%d) reached -> forcing COMPLETE", settings.MAX_EVIDENCE_COUNT)
            return StructuredControllerOutput(
                decision=ControllerDecision.COMPLETE,
                reason="Sufficient evidence accumulated across retrieval hops.",
            )

        if self.ai_engine.provider == "FAKE":
            logger.info("[CONTROLLER] FAKE provider active -> using heuristic evidence evaluation")
            return self._heuristic_fallback(
                user_query=user_query,
                plan=plan,
                accumulated_db_results=accumulated_db_results,
                executed_signatures=executed_signatures,
            )

        # Build prompt summarizing current evidence and query goals
        prompt = self._build_controller_prompt(
            user_query=user_query,
            plan=plan,
            accumulated_db_results=accumulated_db_results,
            accumulated_rag_chunks=accumulated_rag_chunks,
            hop_count=hop_count,
            executed_signatures=executed_signatures,
        )

        start_t = time.perf_counter()
        try:
            logger.info("[CONTROLLER: LLM] Evaluating evidence via AI model '%s'...", self.ai_engine.provider)
            raw = self.ai_engine.complete(prompt, StructuredControllerOutput)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info("[CONTROLLER: LLM] Evaluation completed in %.2f ms", elapsed_ms)

            if isinstance(raw, StructuredControllerOutput):
                decision_obj = raw
            elif isinstance(raw, dict):
                decision_obj = StructuredControllerOutput(**raw)
            else:
                decision_obj = StructuredControllerOutput(
                    decision=ControllerDecision.COMPLETE,
                    reason="Could not parse controller decision; completing with current evidence.",
                )

            # Sanity check: prevent re-executing identical signatures in loop
            if decision_obj.decision == ControllerDecision.RETRIEVE_MORE:
                next_ops = [
                    op for op in decision_obj.next_operations
                    if f"{op}:{sorted(decision_obj.next_parameters.items())}" not in executed_signatures
                ]
                if not next_ops and not decision_obj.next_rag_query:
                    logger.info("[CONTROLLER] All proposed operations already executed -> marking COMPLETE to prevent loop")
                    return StructuredControllerOutput(
                        decision=ControllerDecision.COMPLETE,
                        reason="All planned operations completed without new parameters.",
                    )
                decision_obj.next_operations = next_ops

            return decision_obj

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning("[CONTROLLER: FALLBACK] AI model evaluation failed after %.2f ms (%s) -> using heuristic fallback", elapsed_ms, exc)
            return self._heuristic_fallback(
                user_query=user_query,
                plan=plan,
                accumulated_db_results=accumulated_db_results,
                executed_signatures=executed_signatures,
            )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_controller_prompt(
        self,
        user_query: str,
        plan: QueryPlan,
        accumulated_db_results: dict[str, list[dict[str, Any]]],
        accumulated_rag_chunks: list[RetrievedChunk],
        hop_count: int,
        executed_signatures: set[str],
    ) -> str:
        # Format database records summary
        db_summary = []
        for op, recs in accumulated_db_results.items():
            db_summary.append(f"[{op}] ({len(recs)} records):")
            for r in recs[:3]:
                key_fields = {k: v for k, v in r.items() if k in ("is_number", "title", "product_name", "scheme_code", "name", "mandatory", "scope")}
                db_summary.append(f"  - {key_fields}")

        rag_summary = [f"- {c.filename}: {c.chunk.text[:120]}..." for c in accumulated_rag_chunks[:3]]

        return f"""You are the Multi-Hop Retrieval Controller for the Bureau of Indian Standards (BIS) Assistant.

Analyze the user query and the evidence retrieved so far to decide if we need more data, need clarification from the user, or have enough to formulate a complete answer.

==================================================
USER QUERY & CONTEXT
==================================================
User Query: "{user_query}"
Normalized Query: "{plan.normalized_query}"
Classified Intent: {plan.intent}
Current Hop Count: {hop_count}

==================================================
CURRENT RETRIEVED EVIDENCE
==================================================
Database Records:
{chr(10).join(db_summary) if db_summary else "None"}

Document Passages:
{chr(10).join(rag_summary) if rag_summary else "None"}

Executed Operations Signatures:
{list(executed_signatures)}

==================================================
DECISION RULES
==================================================
1. "COMPLETE":
   - Use if the accumulated evidence answers all components of the user query (e.g. standard details found, or service found, or certification + labs found).
   - Use if no further database operations can help.

2. "RETRIEVE_MORE":
   - Use when the user query asks for related downstream information that is NOT yet in the evidence, but we now have the required entity from the previous step.
   - Example 1: User asked "Do I need BIS certification and testing for my cables?" -> We found applicable standard IS 694, but have NOT yet fetched GET_CERTIFICATION_REQUIREMENT or FIND_LABORATORIES for IS 694. -> RETRIEVE_MORE with next_operations=["GET_CERTIFICATION_REQUIREMENT", "FIND_LABORATORIES"], next_parameters={{"standard_number": "IS 694:2010"}}.
   - Example 2: User asked a technical clause question and needs_rag is true, but no RAG chunks retrieved yet -> RETRIEVE_MORE with next_rag_query.

3. "NEED_CLARIFICATION":
   - Use ONLY when the retrieved evidence shows multiple mutually exclusive standards or categories and the system cannot determine which one applies without user input.
   - Example: For generic "cables", evidence shows IS 694 (PVC up to 1100V) and IS 7098 (XLPE for higher voltages). -> NEED_CLARIFICATION with clarification_question="What is the voltage rating or insulation type of your cable (e.g., up to 1100V for PVC, or higher voltage)?", options=["Up to 1100V (PVC)", "Above 1100V / High Voltage (XLPE)", "Other"].

Choose your decision now and output the structured JSON:"""

    def _heuristic_fallback(
        self,
        user_query: str,
        plan: QueryPlan,
        accumulated_db_results: dict[str, list[dict[str, Any]]],
        executed_signatures: set[str],
    ) -> StructuredControllerOutput:
        """Deterministic heuristic fallback when LLM controller call is unavailable."""
        q_lower = user_query.lower()

        # Check if user asked about certification / testing and we have standards but no cert/lab data
        has_standards = bool(accumulated_db_results.get("FIND_APPLICABLE_STANDARDS") or accumulated_db_results.get("FIND_STANDARD"))
        has_cert = bool(accumulated_db_results.get("GET_CERTIFICATION_REQUIREMENT") or accumulated_db_results.get("GET_CERTIFICATION_SCHEME"))
        has_labs = bool(accumulated_db_results.get("FIND_LABORATORIES"))

        if has_standards and ("certif" in q_lower or "isi" in q_lower or "mandatory" in q_lower or "scheme" in q_lower) and not has_cert:
            std_records = accumulated_db_results.get("FIND_APPLICABLE_STANDARDS") or accumulated_db_results.get("FIND_STANDARD") or []
            std_num = std_records[0].get("is_number") if std_records else None
            sig = f"GET_CERTIFICATION_REQUIREMENT:[('standard_number', '{std_num}')]"
            if sig not in executed_signatures:
                logger.info("[CONTROLLER: HEURISTIC] Standards found -> triggering hop for certification requirements")
                return StructuredControllerOutput(
                    decision=ControllerDecision.RETRIEVE_MORE,
                    reason="Standards discovered; retrieving certification requirements.",
                    next_operations=[DatabaseOperation.GET_CERTIFICATION_REQUIREMENT],
                    next_parameters={"standard_number": std_num} if std_num else {},
                )

        if has_standards and ("lab" in q_lower or "test" in q_lower) and not has_labs:
            std_records = accumulated_db_results.get("FIND_APPLICABLE_STANDARDS") or accumulated_db_results.get("FIND_STANDARD") or []
            std_num = std_records[0].get("is_number") if std_records else None
            sig = f"FIND_LABORATORIES:[('standard_number', '{std_num}')]"
            if sig not in executed_signatures:
                logger.info("[CONTROLLER: HEURISTIC] Standards found -> triggering hop for recognized testing laboratories")
                return StructuredControllerOutput(
                    decision=ControllerDecision.RETRIEVE_MORE,
                    reason="Standards discovered; retrieving testing laboratories.",
                    next_operations=[DatabaseOperation.FIND_LABORATORIES],
                    next_parameters={"standard_number": std_num} if std_num else {},
                )

        logger.info("[CONTROLLER: HEURISTIC] Evaluation complete -> sufficient evidence gathered")
        return StructuredControllerOutput(
            decision=ControllerDecision.COMPLETE,
            reason="Heuristic evaluation complete; sufficient evidence retrieved.",
        )
