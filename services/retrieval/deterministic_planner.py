"""
services/retrieval/deterministic_planner.py
===========================================

Deterministic Retrieval Planner for the BIS Intelligent Assistant.

Replaces the per-hop LLM Retrieval Controller with high-performance, predictable,
and deterministic retrieval planning and RAG routing.

Responsibilities:
  1. Decides whether DB and/or RAG retrieval is required.
  2. Enforces deterministic RAG-routing rules (R1 to R7).
  3. Derives downstream parameters across multi-hop relational dependencies
     (e.g., FIND_PRODUCT -> FIND_APPLICABLE_STANDARDS -> GET_CERTIFICATION_REQUIREMENT / FIND_LABORATORIES).
  4. Generates standard/document-filtered RAG queries without arbitrary keywords.
  5. Prevents redundant retrieval loops via executed signature deduplication.
  6. Enforces hard bounds on MAX_RETRIEVAL_HOPS and MAX_EVIDENCE_COUNT.
  7. Detects genuine ambiguity and triggers structured clarification deterministically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from configs import get_settings
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.retrieval.retrieval_dataclasses import RetrievedChunk

logger = logging.getLogger(__name__)

# Explicit documentary/technical keyword patterns for Rule R1
_DOCUMENT_KEYWORDS = [
    r"\brequirement(s)?\b",
    r"\bspecification(s)?\b",
    r"\bscope\b",
    r"\bclause(s)?\b",
    r"\btesting\b",
    r"\btest\s+method(s)?\b",
    r"\bsafety\s+requirement(s)?\b",
    r"\btechnical\s+requirement(s)?\b",
    r"\bconformity\s+requirement(s)?\b",
    r"\bmarking\s+requirement(s)?\b",
    r"\blabeling\s+requirement(s)?\b",
    r"\binstallation\s+requirement(s)?\b",
    r"\bapplicable\s+provision(s)?\b",
    r"\bwhat\s+does\s+the\s+standard\s+say\b",
    r"\baccording\s+to\s+the\s+standard\b",
    r"\bstandard\s+document\b",
    r"\bexplain\s+the\s+standard\b",
    r"\bas\s+per\s+is\b",
    r"\bdimension(s)?\b",
    r"\bsampling\b",
    r"\blimit(s)?\b",
    r"\bcondition(s)?\b",
    r"\bvoltage\s+rating(s)?\b",
    r"\bconductor\b",
]

_DOC_REGEX = re.compile("|".join(_DOCUMENT_KEYWORDS), re.IGNORECASE)


@dataclass
class DeterministicRetrievalPlan:
    """Plan specifying the exact actions to take for a retrieval hop."""
    execute_db: bool = False
    db_operations: list[DatabaseOperation] = field(default_factory=list)
    db_parameters: dict[str, Any] = field(default_factory=dict)

    execute_rag: bool = False
    rag_query: str | None = None
    rag_standard_filter: str | None = None
    rag_filename_filter: str | None = None

    continue_retrieval: bool = True
    is_clarification: bool = False
    clarification_question: str | None = None
    clarification_options: list[str] | None = None
    clarification_input_type: Literal["text", "number", "select", "multi_select", "boolean"] = "text"

    reason: str = ""


class DeterministicPlanner:
    """
    Deterministic retrieval planner coordinating structured DB operations,
    parameter propagation, document-first RAG routing, and termination conditions.
    """

    @classmethod
    def plan_initial_hop(cls, initial_plan: QueryPlan) -> DeterministicRetrievalPlan:
        """
        Produce the initial retrieval plan for Hop 1 based on Query Analyzer output
        and deterministic RAG routing rules.
        """
        intent_val = getattr(initial_plan.intent, "value", str(initial_plan.intent))
        norm_q = initial_plan.normalized_query
        q_lower = norm_q.lower()

        # Rule R1: Check for explicit documentary concepts in query
        has_doc_keywords = bool(_DOC_REGEX.search(q_lower))

        # Check if a specific standard is already identified in initial parameters
        std_num = (
            initial_plan.parameters.get("standard_number")
            or initial_plan.parameters.get("is_number")
        )

        db_ops = list(initial_plan.db_operations)
        db_params = dict(initial_plan.parameters)
        execute_db = initial_plan.needs_db and bool(db_ops)
        execute_rag = initial_plan.needs_rag

        reconciliation_notes: list[str] = []

        # ------------------------------------------------------------------
        # Intent-to-Retrieval Compatibility Enforcement & Contradiction Resolution
        # ------------------------------------------------------------------
        if intent_val == QueryIntent.BIS_SERVICE_LOOKUP.value:
            if DatabaseOperation.GET_BIS_SERVICE not in db_ops:
                db_ops.append(DatabaseOperation.GET_BIS_SERVICE)
                execute_db = True
                reconciliation_notes.append("Injected GET_BIS_SERVICE for BIS_SERVICE_LOOKUP")
            # Vector store contains standard technical PDFs, not service metadata
            if execute_rag and not has_doc_keywords:
                execute_rag = False
                reconciliation_notes.append("Suppressed RAG for structured BIS_SERVICE_LOOKUP")

        elif intent_val == QueryIntent.LABORATORY_LOOKUP.value:
            if DatabaseOperation.FIND_LABORATORIES not in db_ops:
                db_ops.append(DatabaseOperation.FIND_LABORATORIES)
                execute_db = True
                reconciliation_notes.append("Injected FIND_LABORATORIES for LABORATORY_LOOKUP")
            if execute_rag and not has_doc_keywords:
                execute_rag = False
                reconciliation_notes.append("Suppressed RAG for structured LABORATORY_LOOKUP")

        elif intent_val == QueryIntent.CERTIFICATION_PROCESS.value:
            if DatabaseOperation.GET_CERTIFICATION_SCHEME not in db_ops:
                db_ops.append(DatabaseOperation.GET_CERTIFICATION_SCHEME)
                execute_db = True
                reconciliation_notes.append("Injected GET_CERTIFICATION_SCHEME for CERTIFICATION_PROCESS")
            if execute_rag and not has_doc_keywords:
                execute_rag = False
                reconciliation_notes.append("Suppressed RAG for procedural CERTIFICATION_PROCESS")

        elif intent_val == QueryIntent.PRODUCT_STANDARD_RECOMMENDATION.value:
            if DatabaseOperation.FIND_APPLICABLE_STANDARDS not in db_ops:
                db_ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
                execute_db = True
                reconciliation_notes.append("Injected FIND_APPLICABLE_STANDARDS for PRODUCT_STANDARD_RECOMMENDATION")

        elif intent_val == QueryIntent.CERTIFICATION_REQUIREMENT.value:
            if std_num:
                if DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in db_ops:
                    db_ops.append(DatabaseOperation.GET_CERTIFICATION_REQUIREMENT)
                    execute_db = True
                    reconciliation_notes.append("Injected GET_CERTIFICATION_REQUIREMENT for standard necessity")
            else:
                if DatabaseOperation.FIND_APPLICABLE_STANDARDS not in db_ops:
                    db_ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
                    execute_db = True
                if DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in db_ops:
                    db_ops.append(DatabaseOperation.GET_CERTIFICATION_REQUIREMENT)
                    execute_db = True
                reconciliation_notes.append("Injected standard mapping & certification requirement")

        elif intent_val == QueryIntent.STANDARD_LOOKUP.value:
            if DatabaseOperation.FIND_STANDARD not in db_ops:
                db_ops.append(DatabaseOperation.FIND_STANDARD)
                execute_db = True
                reconciliation_notes.append("Injected FIND_STANDARD for STANDARD_LOOKUP")

        elif intent_val == QueryIntent.GENERAL_BIS_QUERY.value:
            # General overview of BIS requires documentary RAG passages
            execute_rag = True
            reconciliation_notes.append("Enabled RAG for GENERAL_BIS_QUERY overview")

        elif intent_val == QueryIntent.OUT_OF_SCOPE.value:
            execute_db = False
            db_ops = []
            execute_rag = False

        # Support secondary intents
        for sec in getattr(initial_plan, "secondary_intents", []):
            sec_val = getattr(sec, "value", str(sec))
            if sec_val == QueryIntent.CERTIFICATION_REQUIREMENT.value and DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in db_ops:
                db_ops.append(DatabaseOperation.GET_CERTIFICATION_REQUIREMENT)
                execute_db = True
            elif sec_val == QueryIntent.PRODUCT_STANDARD_RECOMMENDATION.value and DatabaseOperation.FIND_APPLICABLE_STANDARDS not in db_ops:
                db_ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
                execute_db = True
            elif sec_val == QueryIntent.LABORATORY_LOOKUP.value and DatabaseOperation.FIND_LABORATORIES not in db_ops:
                db_ops.append(DatabaseOperation.FIND_LABORATORIES)
                execute_db = True
            elif sec_val in (QueryIntent.TESTING_REQUIREMENT.value, QueryIntent.TECHNICAL_QUESTION.value):
                execute_rag = True

        if bool(db_ops):
            execute_db = True

        # Rule R1: Documentary keyword match triggers RAG for technical/standards queries
        non_rag_intents = {
            QueryIntent.BIS_SERVICE_LOOKUP.value,
            QueryIntent.LABORATORY_LOOKUP.value,
            QueryIntent.OUT_OF_SCOPE.value,
        }
        if has_doc_keywords and intent_val not in non_rag_intents:
            execute_rag = True

        rag_query = None
        rag_filter = None
        if execute_rag:
            rag_query = norm_q
            if std_num:
                rag_filter = str(std_num)

        reason = "Initial hop planned from QueryPlan"
        if reconciliation_notes:
            reason += " | Reconciled: " + "; ".join(reconciliation_notes)
        elif has_doc_keywords and not initial_plan.needs_rag:
            reason += " (RAG enabled by Rule R1 documentary keyword matching)"

        logger.info(
            "[PLANNER] Initial Hop Planned -> DB: %s (Ops: %s), RAG: %s (Query: '%s', Filter: '%s') | Reason: %s",
            execute_db,
            [getattr(op, "value", str(op)) for op in db_ops],
            execute_rag,
            rag_query,
            rag_filter,
            reason,
        )

        return DeterministicRetrievalPlan(
            execute_db=execute_db,
            db_operations=db_ops,
            db_parameters=db_params,
            execute_rag=execute_rag,
            rag_query=rag_query,
            rag_standard_filter=rag_filter,
            continue_retrieval=True,
            reason=reason,
        )

    @classmethod
    def plan_next_hop(
        cls,
        user_query: str,
        initial_plan: QueryPlan,
        accumulated_db_results: dict[str, list[dict[str, Any]]],
        accumulated_rag_chunks: list[RetrievedChunk],
        hop_count: int,
        executed_signatures: set[str],
    ) -> DeterministicRetrievalPlan:
        """
        Evaluate current retrieval state and determine next retrieval actions or termination.
        Zero LLM calls.
        """
        settings = get_settings()
        q_lower = user_query.lower() + " " + initial_plan.normalized_query.lower()

        total_db_records = sum(len(recs) for recs in accumulated_db_results.values())
        total_evidence = total_db_records + len(accumulated_rag_chunks)

        # ------------------------------------------------------------------
        # Rule R7: Hard Limit on Maximum Hops
        # ------------------------------------------------------------------
        if hop_count >= settings.MAX_RETRIEVAL_HOPS:
            logger.info("[PLANNER] Maximum hops (%d) reached -> stopping retrieval", settings.MAX_RETRIEVAL_HOPS)
            return DeterministicRetrievalPlan(
                continue_retrieval=False,
                reason=f"Reached maximum permitted retrieval hops ({settings.MAX_RETRIEVAL_HOPS}).",
            )

        # ------------------------------------------------------------------
        # Rule R6: Hard Limit on Maximum Evidence Count
        # ------------------------------------------------------------------
        if total_evidence >= settings.MAX_EVIDENCE_COUNT:
            logger.info("[PLANNER] Maximum evidence count (%d) reached -> stopping retrieval", settings.MAX_EVIDENCE_COUNT)
            return DeterministicRetrievalPlan(
                continue_retrieval=False,
                reason=f"Sufficient evidence accumulated ({total_evidence} items).",
            )

        # ------------------------------------------------------------------
        # Extract Discovered Entities from Accumulated Evidence
        # ------------------------------------------------------------------
        discovered_standards: list[str] = []
        for op in ("FIND_STANDARD", "FIND_APPLICABLE_STANDARDS"):
            for rec in accumulated_db_results.get(op, []):
                num = rec.get("is_number") or rec.get("standard_number")
                if num and num not in discovered_standards:
                    discovered_standards.append(num)

        discovered_products: list[dict[str, Any]] = accumulated_db_results.get("FIND_PRODUCT", [])

        # ------------------------------------------------------------------
        # Deterministic Clarification Detection
        # ------------------------------------------------------------------
        # If product/standards search found multiple distinct standards and user query is generic
        if (
            len(discovered_standards) >= settings.CLARIFICATION_THRESHOLD
            and not any(k in initial_plan.parameters for k in ("maximum_voltage", "minimum_voltage", "voltage", "conductor", "product_type"))
            and ("cable" in q_lower or "wire" in q_lower or "conductor" in q_lower)
            and not any(w in q_lower for w in ("all", "list", "which", "compare", "what are"))
        ):
            # Check if there is ambiguity between low voltage (IS 694) and high voltage (IS 7098)
            options = [
                "Low Voltage PVC Cables (up to 1100 V - IS 694)",
                "High Voltage XLPE Cables (above 1100 V / 3.3 kV - 33 kV - IS 7098)",
                "Winding Wires (Enamelled Copper - IS 13730)",
                "Other Specifications",
            ]
            question = "Multiple Indian Standards apply depending on the cable voltage rating and insulation type. Could you please specify your application?"
            logger.info("[PLANNER] Ambiguity detected across %d standards -> requesting clarification", len(discovered_standards))
            return DeterministicRetrievalPlan(
                continue_retrieval=False,
                is_clarification=True,
                clarification_question=question,
                clarification_options=options,
                clarification_input_type="select",
                reason="Multiple standards match generic product term; voltage disambiguation required.",
            )

        # ------------------------------------------------------------------
        # Multi-Hop Relational Operation Chaining & Parameter Derivation
        # ------------------------------------------------------------------
        next_db_ops: list[DatabaseOperation] = []
        next_db_params: dict[str, Any] = dict(initial_plan.parameters)

        primary_std = discovered_standards[0] if discovered_standards else None
        if primary_std:
            next_db_params["standard_number"] = primary_std

        # Dependency Rule 1: Standards Discovered -> Certification Requirements
        has_cert = bool(accumulated_db_results.get("GET_CERTIFICATION_REQUIREMENT") or accumulated_db_results.get("GET_CERTIFICATION_SCHEME"))
        asked_cert_in_plan = (
            DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in initial_plan.db_operations
            or DatabaseOperation.GET_CERTIFICATION_SCHEME in initial_plan.db_operations
        )
        asked_cert_in_query = bool(re.search(r"\b(certif\w*|isi\s*mark|scheme|qco|licen[cs]e|mandatory|compulsory|necessary|need to take|required|obligation|zaroori|anivarya)\b", q_lower))
        wants_cert = asked_cert_in_plan or asked_cert_in_query
        if discovered_standards and wants_cert and not has_cert:
            sig = f"{DatabaseOperation.GET_CERTIFICATION_REQUIREMENT.value}:[('standard_number', '{primary_std}')]"
            if sig not in executed_signatures:
                next_db_ops.append(DatabaseOperation.GET_CERTIFICATION_REQUIREMENT)
                logger.info("[PLANNER] Standards found (%s) & certification asked -> queueing GET_CERTIFICATION_REQUIREMENT", primary_std)

        # Dependency Rule 2: Standards Discovered -> Testing Laboratories
        has_labs = bool(accumulated_db_results.get("FIND_LABORATORIES"))
        asked_lab_in_plan = DatabaseOperation.FIND_LABORATORIES in initial_plan.db_operations
        asked_lab_in_query = bool(re.search(r"\b(lab\w*|test\w*|where can i test|testing facilit\w*)\b", q_lower))
        wants_labs = asked_lab_in_plan or asked_lab_in_query
        if discovered_standards and wants_labs and not has_labs:
            sig = f"{DatabaseOperation.FIND_LABORATORIES.value}:[('standard_number', '{primary_std}')]"
            if sig not in executed_signatures:
                next_db_ops.append(DatabaseOperation.FIND_LABORATORIES)
                logger.info("[PLANNER] Standards found (%s) & testing asked -> queueing FIND_LABORATORIES", primary_std)

        # Dependency Rule 3: Product Discovered -> Applicable Standards
        has_mappings = bool(accumulated_db_results.get("FIND_APPLICABLE_STANDARDS"))
        if discovered_products and not has_mappings:
            first_prod = discovered_products[0]
            prod_name = first_prod.get("name")
            sig = f"{DatabaseOperation.FIND_APPLICABLE_STANDARDS.value}:[('product', '{prod_name}')]"
            if sig not in executed_signatures:
                next_db_ops.append(DatabaseOperation.FIND_APPLICABLE_STANDARDS)
                next_db_params["product"] = prod_name

        # Dependency Rule 4: BIS Services Requested -> Services Execution
        has_services = bool(accumulated_db_results.get("GET_BIS_SERVICE"))
        wants_services = (
            getattr(initial_plan.intent, "value", str(initial_plan.intent)) == QueryIntent.BIS_SERVICE_LOOKUP.value
            or DatabaseOperation.GET_BIS_SERVICE in initial_plan.db_operations
        )
        if wants_services and not has_services:
            sig = f"{DatabaseOperation.GET_BIS_SERVICE.value}:{sorted(next_db_params.items())}"
            if sig not in executed_signatures:
                next_db_ops.append(DatabaseOperation.GET_BIS_SERVICE)
                logger.info("[PLANNER] BIS Services requested & not yet retrieved -> queueing GET_BIS_SERVICE")

        # Filter out already executed operation signatures
        valid_next_ops = []
        for op in next_db_ops:
            sig = f"{op.value}:{sorted(next_db_params.items())}"
            if sig not in executed_signatures:
                valid_next_ops.append(op)

        # ------------------------------------------------------------------
        # Document-Grounded RAG Routing (Rules R1, R2, R3)
        # ------------------------------------------------------------------
        execute_rag = False
        rag_query = None
        rag_std_filter = None

        has_rag = len(accumulated_rag_chunks) > 0
        has_doc_keywords = bool(_DOC_REGEX.search(q_lower))

        # Pure metadata query check (Rule R3): pure factual attributes don't need RAG unless document keywords present
        is_pure_metadata = any(
            w in q_lower for w in (
                "publication year", "publication date", "status", "revision number",
                "department", "validity", "lab code", "email", "phone", "address",
                "list service", "what service", "services offered"
            )
        ) and not has_doc_keywords

        # Rule R2: Standard identified & documentary grounding requested/useful
        if discovered_standards and not has_rag and not is_pure_metadata:
            if initial_plan.needs_rag or has_doc_keywords or any(w in q_lower for w in ("applicable", "detail", "about", "specification", "requirement", "provision", "scope", "clause", "testing", "cable")):
                execute_rag = True
                rag_std_filter = primary_std
                rag_query = f"{initial_plan.normalized_query} {primary_std}" if primary_std else initial_plan.normalized_query
                logger.info("[PLANNER] Standard identified (%s) -> triggering document-first RAG for grounding", primary_std)

        # Check if there is any new work for this hop
        if not valid_next_ops and not execute_rag:
            db_count = sum(len(r) for r in accumulated_db_results.values())
            rag_count = len(accumulated_rag_chunks)
            if db_count > 0 or rag_count > 0:
                reason = f"Sufficient evidence gathered & all scheduled operations completed ({db_count} DB records, {rag_count} RAG chunks gathered)."
            else:
                reason = "All scheduled retrieval operations completed (no matching records found in database)."
            logger.info("[PLANNER] %s -> stopping retrieval", reason)
            return DeterministicRetrievalPlan(
                continue_retrieval=False,
                reason=reason,
            )

        return DeterministicRetrievalPlan(
            execute_db=bool(valid_next_ops),
            db_operations=valid_next_ops,
            db_parameters=next_db_params,
            execute_rag=execute_rag,
            rag_query=rag_query,
            rag_standard_filter=rag_std_filter,
            continue_retrieval=True,
            reason="Advancing multi-hop retrieval with derived parameters.",
        )
