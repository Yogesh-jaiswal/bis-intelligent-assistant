"""
services/query/query_service.py
===============================

Main orchestration service coordinating:
  1. Initial query analysis
  2. Multi-hop structured and semantic database retrieval
  3. Retrieval controller decisions (COMPLETE, RETRIEVE_MORE, NEED_CLARIFICATION)
  4. Evidence-based clarification generation
  5. Context assembly and citation construction
  6. Final response synthesis with technical terminology preservation
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from configs import get_settings
from services.ai.engine import AIEngine
from services.query_analyser import QueryAnalyzer, QueryIntent, QueryPlan
from services.query_executor import QueryExecutor, QueryExecutionResult
from services.retrieval.citation_builder import CitationBuilder
from services.retrieval.deterministic_planner import (
    DeterministicPlanner,
    DeterministicRetrievalPlan,
)
from services.retrieval.retrieval_controller import (
    ControllerDecision,
    RetrievalController,
    StructuredControllerOutput,
)
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from services.retrieval.similarity_search_service import SimilaritySearchService
from validators.chat_responses import (
    CertificationCard,
    ChatRequest,
    ChatResponse,
    Citation,
    ClarificationQuestion,
    DataBlock,
    LaboratoryCard,
    ServiceCard,
    StandardCard,
)

from services.query.synthesis_prompt import (
    SYNTHESIS_FALLBACK_TEMPLATES,
    SynthesisResponse,
    build_synthesis_prompt,
)

logger = logging.getLogger(__name__)


OUT_OF_SCOPE_MESSAGES = {
    "en": (
        "I am the Bureau of Indian Standards (BIS) Intelligent Assistant. "
        "I can help you with Indian Standards, conformity assessment, "
        "BIS certifications, testing laboratories, and BIS services. "
        "Your query appears to be outside of this scope."
    ),

    "hi": (
        "मैं भारतीय मानक ब्यूरो (BIS) का इंटेलिजेंट असिस्टेंट हूँ। "
        "मैं भारतीय मानकों, अनुरूपता मूल्यांकन, BIS प्रमाणन, "
        "परीक्षण प्रयोगशालाओं और BIS सेवाओं से संबंधित जानकारी में आपकी सहायता कर सकता हूँ। "
        "आपका प्रश्न इस दायरे से बाहर प्रतीत होता है।"
    ),
}



class QueryService:
    """
    Main orchestration service coordinating query analysis, multi-hop retrieval,
    evidence-based clarification, and final localized response generation.
    """

    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        executor: QueryExecutor | None = None,
        retriever: SimilaritySearchService | None = None,
        controller: RetrievalController | None = None,
        ai_engine: AIEngine | None = None,
    ):
        settings = get_settings()
        self.ai_engine = ai_engine or AIEngine(settings.AI_MODEL)
        self.analyzer = analyzer or QueryAnalyzer(ai_engine=self.ai_engine)
        self.executor = executor or QueryExecutor()
        self.retriever = retriever or SimilaritySearchService()
        self.controller = controller or RetrievalController(ai_engine=self.ai_engine)

    def process(self, payload: ChatRequest) -> dict[str, Any]:
        """
        Execute the end-to-end conversational pipeline for a user query.

        :param payload: Incoming ChatRequest payload.
        :return: Serialized ChatResponse dictionary.
        """
        settings = get_settings()
        user_message = payload.message.content.strip()
        conv_id = payload.conversation_id or f"conv_{uuid.uuid4().hex[:10]}"

        logger.info(
            "[PIPELINE: START] Processing query for conversation_id='%s' (user_query='%s')",
            conv_id,
            user_message,
        )

        # Step 0: Safely retrieve previous conversation summary (if any)
        try:
            prior_summary = self._load_conversation_summary(conv_id)
        except Exception as exc:
            logger.warning("[CONVERSATION: LOAD FAILED] Non-fatal load failure for '%s': %s", conv_id, exc)
            prior_summary = None

        # ------------------------------------------------------------------
        # Stage 1: Initial Query Analysis
        # ------------------------------------------------------------------
        logger.info("[PIPELINE: STAGE 1 - QUERY ANALYSIS] Analyzing user query intent and extracting entities...")
        plan: QueryPlan = self.analyzer.analyze(user_message, conversation_summary=prior_summary)

        if payload.message.language:
            plan.response_language = payload.message.language

        logger.info(
            "[PIPELINE: STAGE 1 RESULT] Intent=%s | Lang=%s | Relevant=%s | NeedsDB=%s (Ops: %s) | NeedsRAG=%s | Params=%s | Normalised query=%s",
            plan.intent,
            plan.response_language,
            plan.relevant,
            plan.needs_db,
            [getattr(op, "value", str(op)) for op in plan.db_operations] if plan.db_operations else [],
            plan.needs_rag,
            plan.parameters,
            plan.normalized_query
        )

        # ------------------------------------------------------------------
        # Stage 2: Out-of-Scope Handling
        # ------------------------------------------------------------------
        if not plan.relevant:
            logger.warning(
                "[PIPELINE: OUT-OF-SCOPE] Query marked non-relevant to BIS domain (query='%s')",
                user_message,
            )
            language = plan.response_language or "en"

            message = OUT_OF_SCOPE_MESSAGES.get(
                language,
                OUT_OF_SCOPE_MESSAGES["en"],
            )
            return ChatResponse(
                message_type="answer",
                conversation_id=conv_id,
                message=message,
                citations=[],
                data=[],
            ).model_dump(mode="json")

        # ------------------------------------------------------------------
        # Stage 3: Deterministic Multi-Hop Retrieval Loop
        # ------------------------------------------------------------------
        accumulated_db_results: dict[str, list[dict[str, Any]]] = {}
        accumulated_rag_chunks: list[RetrievedChunk] = []
        executed_signatures: set[str] = set()
        seen_chunk_keys: set[tuple[str, str]] = set()

        hop_count = 1
        current_hop_plan = DeterministicPlanner.plan_initial_hop(plan)

        logger.info(
            "[PIPELINE: STAGE 2 - DETERMINISTIC RETRIEVAL] Starting retrieval loop (max_hops=%d)...",
            settings.MAX_RETRIEVAL_HOPS,
        )

        while hop_count <= settings.MAX_RETRIEVAL_HOPS:
            logger.info(
                "[RETRIEVAL: HOP %d/%d] Executing retrieval iteration for normalized_query='%s'...",
                hop_count,
                settings.MAX_RETRIEVAL_HOPS,
                plan.normalized_query,
            )

            # 3A: Execute Structured Database Operations for this hop
            if current_hop_plan.execute_db and current_hop_plan.db_operations:
                logger.info(
                    "[RETRIEVAL: DB EXECUTION] Running %d DB operations: %s with parameters: %s",
                    len(current_hop_plan.db_operations),
                    [getattr(op, "value", str(op)) for op in current_hop_plan.db_operations],
                    current_hop_plan.db_parameters,
                )
                db_sub_plan = QueryPlan(
                    normalized_query=plan.normalized_query,
                    relevant=True,
                    intent=plan.intent,
                    response_language=plan.response_language,
                    needs_db=True,
                    needs_rag=False,
                    db_operations=current_hop_plan.db_operations,
                    parameters=current_hop_plan.db_parameters,
                )
                db_result: QueryExecutionResult = self.executor.execute(db_sub_plan)
                
                for op_name, recs in db_result.results.items():
                    if op_name not in accumulated_db_results:
                        accumulated_db_results[op_name] = []
                    new_added = 0
                    for rec in recs:
                        if rec not in accumulated_db_results[op_name]:
                            accumulated_db_results[op_name].append(rec)
                            new_added += 1
                    logger.info(
                        "[RETRIEVAL: DB RESULT] Operation '%s' returned %d records (%d new accumulated, total: %d)",
                        op_name,
                        len(recs),
                        new_added,
                        len(accumulated_db_results[op_name]),
                    )

                for op in current_hop_plan.db_operations:
                    sig = f"{getattr(op, 'value', str(op))}:{sorted(current_hop_plan.db_parameters.items())}"
                    executed_signatures.add(sig)

            # 3B: Execute Vector RAG Retrieval if required
            if current_hop_plan.execute_rag and current_hop_plan.rag_query:
                rag_query = current_hop_plan.rag_query
                rag_std_filter = current_hop_plan.rag_standard_filter
                logger.info(
                    "[RETRIEVAL: VECTOR RAG] Searching vector store for documentary evidence (query='%s', filter='%s', top_k=%d)...",
                    rag_query,
                    rag_std_filter,
                    settings.DEFAULT_RAG_TOP_K,
                )
                try:
                    new_chunks = self.retriever.search(
                        query=rag_query,
                        k=settings.DEFAULT_RAG_TOP_K,
                        standard_number=rag_std_filter,
                    )
                    added_chunks = 0
                    for c in new_chunks:
                        key = (c.filename or "", c.chunk.text[:80])
                        if key not in seen_chunk_keys:
                            seen_chunk_keys.add(key)
                            accumulated_rag_chunks.append(c)
                            added_chunks += 1
                    logger.info(
                        "[RETRIEVAL: VECTOR RAG RESULT] Retrieved %d chunks (%d new, total accumulated: %d)",
                        len(new_chunks),
                        added_chunks,
                        len(accumulated_rag_chunks),
                    )
                except Exception as e:
                    logger.warning("[RETRIEVAL: VECTOR RAG ERROR] RAG search encountered an issue: %s", e)

            # 3C: Evaluate Next Hop with Deterministic Planner (0 LLM calls)
            total_db_records = sum(len(r) for r in accumulated_db_results.values())
            logger.info(
                "[RETRIEVAL: DETERMINISTIC PLANNER] Evaluating evidence (%d DB records, %d RAG chunks)...",
                total_db_records,
                len(accumulated_rag_chunks),
            )
            next_hop_plan = DeterministicPlanner.plan_next_hop(
                user_query=user_message,
                initial_plan=plan,
                accumulated_db_results=accumulated_db_results,
                accumulated_rag_chunks=accumulated_rag_chunks,
                hop_count=hop_count,
                executed_signatures=executed_signatures,
            )

            logger.info(
                "[RETRIEVAL: PLANNER DECISION] Hop %d decision: continue=%s, is_clarification=%s | Reason: %s",
                hop_count,
                next_hop_plan.continue_retrieval,
                next_hop_plan.is_clarification,
                next_hop_plan.reason,
            )

            # Handle Evidence-Based Clarification Decision
            if next_hop_plan.is_clarification:
                logger.info("[RETRIEVAL: CLARIFICATION NEEDED] Building clarification response for user...")
                data_cards, citations = self._build_cards_and_citations(
                    accumulated_db_results, accumulated_rag_chunks, plan=plan
                )
                clarification_id = f"q_clarification_{uuid.uuid4().hex[:6]}"
                lang = "hi" if (plan.response_language or "").lower().startswith("hi") else "en"
                tpl = SYNTHESIS_FALLBACK_TEMPLATES.get(lang, SYNTHESIS_FALLBACK_TEMPLATES["en"])
                question_text = next_hop_plan.clarification_question or (
                    tpl.get("clarification_default", "Could you please specify more details to help identify the exact standard?")
                )
                questions = [
                    ClarificationQuestion(
                        id=clarification_id,
                        question=question_text,
                        input_type=next_hop_plan.clarification_input_type or "text",
                        options=next_hop_plan.clarification_options,
                        required=True,
                    )
                ]
                try:
                    self._save_conversation_summary(
                        conv_id,
                        f"{prior_summary or ''}\n[Clarification requested: {question_text}]".strip(),
                    )
                except Exception as exc:
                    logger.warning("[CONVERSATION: SAVE FAILED] Non-fatal save failure during clarification: %s", exc)
                return ChatResponse(
                    message_type="clarification",
                    conversation_id=conv_id,
                    message=question_text,
                    citations=citations,
                    data=data_cards,
                    questions=questions,
                ).model_dump(mode="json")

            # Handle Next Retrieval Hop
            if next_hop_plan.continue_retrieval and hop_count < settings.MAX_RETRIEVAL_HOPS:
                logger.info(
                    "[RETRIEVAL: MULTI-HOP ADVANCE] Advancing to hop %d with next ops: %s, next RAG: '%s' (filter: '%s')",
                    hop_count + 1,
                    [getattr(op, "value", str(op)) for op in next_hop_plan.db_operations] if next_hop_plan.db_operations else [],
                    next_hop_plan.rag_query,
                    next_hop_plan.rag_standard_filter,
                )
                current_hop_plan = next_hop_plan
                hop_count += 1
                continue

            # COMPLETE or max hops reached
            break

        # ------------------------------------------------------------------
        # Stage 4: Assemble Structured Data Cards and Citations
        # ------------------------------------------------------------------
        logger.info("[PIPELINE: STAGE 3 - CARD & CITATION ASSEMBLY] Assembling data cards and verified citations...")
        data_cards, citations = self._build_cards_and_citations(
            accumulated_db_results, accumulated_rag_chunks, plan=plan
        )
        logger.info(
            "[PIPELINE: STAGE 3 RESULT] Assembled %d DataCards and %d Citations",
            len(data_cards),
            len(citations),
        )

        # ------------------------------------------------------------------
        # Stage 5: Synthesize Final Response & Update Conversation Summary
        # ------------------------------------------------------------------
        logger.info(
            "[PIPELINE: STAGE 4 - RESPONSE SYNTHESIS] Synthesizing final answer (language='%s')...",
            plan.response_language,
        )
        final_message, updated_summary = self._synthesize_response(
            user_query=user_message,
            plan=plan,
            accumulated_db_results=accumulated_db_results,
            rag_chunks=accumulated_rag_chunks,
            citations=citations,
            conversation_summary=prior_summary,
        )

        # Step 6: Persist updated conversation summary
        try:
            self._save_conversation_summary(conv_id, updated_summary)
        except Exception as exc:
            logger.warning("[CONVERSATION: SAVE FAILED] Non-fatal save failure for '%s': %s", conv_id, exc)

        logger.info(
            "[PIPELINE: COMPLETE] Query successfully processed (type='answer', message_len=%d, cards=%d, citations=%d)",
            len(final_message),
            len(data_cards),
            len(citations),
        )

        # Step 7: Construct Final ChatResponse
        response = ChatResponse(
            message_type="answer",
            conversation_id=conv_id,
            message=final_message,
            citations=citations,
            data=data_cards,
        )

        return response.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_cards_and_citations(
        self,
        db_results: dict[str, list[dict[str, Any]]],
        rag_chunks: list[RetrievedChunk],
        plan: QueryPlan | None = None,
    ) -> tuple[list[DataBlock], list[Citation]]:
        """Construct validated frontend DataBlock cards and Citation objects."""
        cards: list[DataBlock] = []
        citations: list[Citation] = []
        seen_card_keys: set[tuple[str, str]] = set()

        fallback_url = "https://www.bis.gov.in/"

        # 1. Process Database Results into Cards
        for op_name, records in db_results.items():
            for rec in records:
                if op_name in ("FIND_STANDARD", "FIND_APPLICABLE_STANDARDS"):
                    is_num = rec.get("is_number") or rec.get("standard_number") or "IS Standard"
                    key = ("standard", is_num)
                    raw_rel = str(rec.get("relevance") or "Primary").lower()
                    if "support" in raw_rel:
                        card_rel = "Supporting"
                    elif "relat" in raw_rel:
                        card_rel = "Related"
                    else:
                        card_rel = "Primary"

                    if key not in seen_card_keys:
                        seen_card_keys.add(key)
                        cards.append(
                            StandardCard(
                                data_type="standard",
                                id=f"std_{rec.get('standard_id', rec.get('id', uuid.uuid4().hex[:6]))}",
                                is_number=is_num,
                                title=rec.get("title") or rec.get("standard_title") or is_num,
                                revision_number=rec.get("revision_number"),
                                publication_year=rec.get("publication_year"),
                                status=rec.get("status") or "Active",
                                technical_department=rec.get("technical_department"),
                                relevance=card_rel,
                                source_url=rec.get("source_url") or "https://standards.bis.gov.in/",
                                document_url=rec.get("document_url"),
                            )
                        )

                elif op_name in ("GET_CERTIFICATION_SCHEME", "GET_CERTIFICATION_REQUIREMENT"):
                    scheme_name = rec.get("scheme_name") or rec.get("name") or "BIS Certification Scheme"
                    key = ("certification", scheme_name)
                    if key not in seen_card_keys:
                        seen_card_keys.add(key)
                        cards.append(
                            CertificationCard(
                                data_type="certification",
                                id=f"cert_{rec.get('scheme_id', rec.get('id', uuid.uuid4().hex[:6]))}",
                                name=scheme_name,
                                scheme_code=rec.get("scheme_code") or "Scheme-I",
                                certification_type=rec.get("certification_type") or "Product Certification",
                                mandatory=rec.get("mandatory") or "Yes",
                                authority=rec.get("authority") or "Bureau of Indian Standards",
                                requirements=[rec["conditions"]] if rec.get("conditions") else None,
                                source_url=rec.get("source_url") or "https://www.bis.gov.in/product-certification/",
                            )
                        )

                elif op_name == "FIND_LABORATORIES":
                    lab_name = rec.get("name") or "BIS Recognized Laboratory"
                    key = ("laboratory", lab_name)
                    if key not in seen_card_keys:
                        seen_card_keys.add(key)
                        cards.append(
                            LaboratoryCard(
                                data_type="laboratory",
                                id=f"lab_{rec.get('id', uuid.uuid4().hex[:6])}",
                                lab_code=rec.get("lab_code"),
                                name=lab_name,
                                address=rec.get("address"),
                                state=rec.get("state"),
                                district=rec.get("district"),
                                phone=rec.get("phone"),
                                email=rec.get("email"),
                                validity_date=rec.get("validity_date"),
                                scope=rec.get("scope"),
                                source_url=rec.get("source_url") or "https://lims.bis.gov.in/",
                            )
                        )

                elif op_name == "GET_BIS_SERVICE":
                    srv_name = rec.get("name") or "BIS Service"
                    key = ("service", srv_name)
                    if key not in seen_card_keys:
                        seen_card_keys.add(key)
                        cards.append(
                            ServiceCard(
                                data_type="service",
                                id=f"srv_{rec.get('id', uuid.uuid4().hex[:6])}",
                                name=srv_name,
                                service_type=rec.get("service_type") or "General Service",
                                description=rec.get("description"),
                                source_url=rec.get("source_url") or fallback_url,
                            )
                        )

        # 2. Process RAG Chunks into Citations via CitationBuilder
        # Citation relevance filtering: suppress standard PDF citations for structured-only intents
        # unless technical/testing secondary intents are present
        filtered_rag_chunks = rag_chunks
        if plan:
            intent_val = getattr(plan.intent, "value", str(plan.intent))
            non_doc_intents = {
                QueryIntent.BIS_SERVICE_LOOKUP.value,
                QueryIntent.LABORATORY_LOOKUP.value,
                QueryIntent.OUT_OF_SCOPE.value,
            }
            has_tech_secondary = any(
                getattr(sec, "value", str(sec)) in {QueryIntent.TESTING_REQUIREMENT.value, QueryIntent.TECHNICAL_QUESTION.value}
                for sec in getattr(plan, "secondary_intents", [])
            )
            if intent_val in non_doc_intents and not has_tech_secondary:
                logger.info(
                    "[CITATIONS: FILTER] Excluding %d RAG chunks from citation assembly for structured intent '%s'",
                    len(rag_chunks),
                    intent_val,
                )
                filtered_rag_chunks = []

        rag_citations = CitationBuilder.build_api_citations(filtered_rag_chunks)
        for i, cit in enumerate(rag_citations, start=1):
            citations.append(
                Citation(
                    id=f"cit_{i}",
                    source_type=cit.source_type,
                    title=cit.title,
                    reference=cit.reference,
                    source_url=cit.source_url,
                )
            )

        return cards, citations

    def _load_conversation_summary(self, conversation_id: str) -> str | None:
        """Safely load existing conversation summary from database."""
        if not conversation_id:
            return None
        try:
            from models.conversation import Conversation
            conv = Conversation.query.filter_by(conversation_id=conversation_id).first()
            if conv and conv.summary:
                logger.info(
                    "[CONVERSATION: LOAD] Loaded summary for conversation_id='%s' (len=%d)",
                    conversation_id,
                    len(conv.summary),
                )
                return conv.summary
            return None
        except Exception as exc:
            logger.warning(
                "[CONVERSATION: LOAD FAILED] Non-fatal: failed to load summary for '%s': %s",
                conversation_id,
                exc,
            )
            return None

    def _save_conversation_summary(self, conversation_id: str, summary: str | None) -> None:
        """Safely persist or update conversation summary in database."""
        if not conversation_id or not summary or not summary.strip():
            return
        try:
            from app.extensions import db
            from models.conversation import Conversation
            conv = Conversation.query.filter_by(conversation_id=conversation_id).first()
            if conv:
                conv.summary = summary.strip()
            else:
                conv = Conversation(conversation_id=conversation_id, summary=summary.strip())
                db.session.add(conv)
            db.session.commit()
            logger.info("[CONVERSATION: SAVED] Persisted summary for conversation_id='%s'", conversation_id)
        except Exception as exc:
            try:
                from app.extensions import db
                db.session.rollback()
            except Exception:
                pass
            logger.warning(
                "[CONVERSATION: SAVE FAILED] Non-fatal: failed to save summary for '%s': %s",
                conversation_id,
                exc,
            )

    def _synthesize_response(
        self,
        user_query: str,
        plan: QueryPlan,
        accumulated_db_results: dict[str, list[dict[str, Any]]],
        rag_chunks: list[RetrievedChunk],
        citations: list[Citation],
        conversation_summary: str | None = None,
    ) -> tuple[str, str | None]:
        """
        Prompt the LLM to synthesize an evidence-based answer and updated conversation summary.
        Treats retrieved evidence as source material, avoiding raw RAG dumping and verbatim copying.
        Preserves technical terminology, Indian Standard numbers, units, and valid citation tags.
        """
        prompt = build_synthesis_prompt(
            user_query=user_query,
            plan=plan,
            accumulated_db_results=accumulated_db_results,
            rag_chunks=rag_chunks,
            citations=citations,
            conversation_summary=conversation_summary,
        )

        try:
            if self.ai_engine.provider == "FAKE":
                raise ValueError("Using deterministic evidence summary under FAKE provider")
            raw = self.ai_engine.complete(
                prompt,
                SynthesisResponse,
                think=False,
                options={"temperature": 0.2, "num_predict": 700},
            )
            if isinstance(raw, SynthesisResponse):
                return raw.answer, raw.conversation_summary
            if isinstance(raw, dict):
                return raw.get("answer") or str(raw), raw.get("conversation_summary")
            return str(raw), None

        except Exception as e:
            logger.info("Synthesis LLM completion unavailable (%s); generating structured evidence summary.", e)
            lang = "hi" if (plan.response_language or "").lower().startswith("hi") else "en"
            tpl = SYNTHESIS_FALLBACK_TEMPLATES.get(lang, SYNTHESIS_FALLBACK_TEMPLATES["en"])
            summary_parts = []

            # Summarize standards
            for op, recs in accumulated_db_results.items():
                if op in ("FIND_STANDARD", "FIND_APPLICABLE_STANDARDS") and recs:
                    stds = [f"{r.get('is_number', '')} ({r.get('title', '')})" for r in recs[:3] if r.get('is_number')]
                    if stds:
                        summary_parts.append(tpl["applicable_standards"].format(standards=", ".join(stds)))
                elif op in ("GET_CERTIFICATION_SCHEME", "GET_CERTIFICATION_REQUIREMENT") and recs:
                    schemes = [
                        f"{r.get('scheme_code', 'Scheme-I')} ({tpl['mandatory'] if r.get('mandatory') == 'Yes' else tpl['voluntary']})"
                        for r in recs[:2]
                    ]
                    if schemes:
                        summary_parts.append(tpl["certification_schemes"].format(schemes=", ".join(schemes)))
                elif op == "FIND_LABORATORIES" and recs:
                    labs = [f"{r.get('name', 'Laboratory')} ({r.get('state', '')})" for r in recs[:2]]
                    if labs:
                        summary_parts.append(tpl["laboratories"].format(labs=", ".join(labs)))
                elif op == "GET_BIS_SERVICE" and recs:
                    srvs = [r.get('name', '') for r in recs[:10] if r.get('name')]
                    if srvs:
                        summary_parts.append(tpl["services"].format(services=", ".join(srvs)))

            # Summarize RAG passages (only for documentary/technical queries or if relevant to intent)
            if rag_chunks and plan.intent not in (QueryIntent.BIS_SERVICE_LOOKUP, QueryIntent.LABORATORY_LOOKUP, QueryIntent.OUT_OF_SCOPE):
                top_text = rag_chunks[0].chunk.text.strip().replace("\n", " ")
                cit_tag = "<cit_1>" if citations else ""
                summary_parts.append(f"{top_text} {cit_tag}".strip())

            display_query = user_query if lang == "hi" else plan.normalized_query
            answer = (
                " ".join(summary_parts)
                if summary_parts
                else tpl["no_records_found"].format(query=display_query)
            )

            # Build deterministic compact conversation summary using language templates
            summary_items = []
            if plan.parameters.get("product"):
                summary_items.append(f"{tpl['product_label']}: {plan.parameters['product']}")
            if plan.parameters.get("standard_number"):
                summary_items.append(f"{tpl['standard_label']}: {plan.parameters['standard_number']}")
            for op, recs in accumulated_db_results.items():
                if recs and op in ("FIND_STANDARD", "FIND_APPLICABLE_STANDARDS"):
                    std = recs[0].get("is_number")
                    if std and not any(std in s for s in summary_items):
                        summary_items.append(f"{tpl['standard_label']}: {std}")
            summary_items.append(f"{tpl['language_label']}: {plan.response_language}")
            fallback_summary = tpl["summary_topic"].format(topic=plan.normalized_query, facts=", ".join(summary_items))

            return answer, fallback_summary



# Singleton instance and module-level function
_query_service = QueryService()


def process_query(payload: ChatRequest) -> dict[str, Any]:
    """
    Module-level entry point called by the Flask /v1/query route.
    """
    return _query_service.process(payload)