import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from configs import get_settings
from services.ai.engine import AIEngine
from services.query_analyser import QueryAnalyzer, QueryPlan
from services.query_executor import QueryExecutor, QueryExecutionResult
from services.retrieval.similarity_search_service import SimilaritySearchService
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import (
    ChatRequest,
    ChatResponse,
    Citation,
    CertificationCard,
    DataBlock,
    LaboratoryCard,
    ServiceCard,
    StandardCard,
)

logger = logging.getLogger(__name__)


class SynthesisResponse(BaseModel):
    """Structured response schema for the final synthesis LLM."""
    answer: str = Field(
        ...,
        description="Authoritative, evidence-based response synthesized from the database and document evidence.",
    )


class QueryService:
    """
    Main orchestration service coordinating query analysis, deterministic database execution,
    document RAG retrieval, context assembly, and final response generation.
    """

    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
        executor: QueryExecutor | None = None,
        retriever: SimilaritySearchService | None = None,
        ai_engine: AIEngine | None = None,
    ):
        self.ai_engine = ai_engine or AIEngine(get_settings().AI_MODEL)
        self.analyzer = analyzer or QueryAnalyzer(ai_engine=self.ai_engine)
        self.executor = executor or QueryExecutor()
        self.retriever = retriever or SimilaritySearchService()

    def process(self, payload: ChatRequest) -> dict[str, Any]:
        """
        Execute the end-to-end conversational pipeline for a user query.

        :param payload: Incoming ChatRequest payload.
        :return: Serialized ChatResponse dictionary.
        """
        user_message = payload.message.content.strip()
        conv_id = payload.conversation_id or f"conv_{uuid.uuid4().hex[:10]}"

        # Step 1: Query Analysis
        plan: QueryPlan = self.analyzer.analyze(user_message)

        # Step 2: Handle Out-of-Scope Queries
        if not plan.relevant:
            return ChatResponse(
                message_type="answer",
                conversation_id=conv_id,
                message=(
                    "I am the Bureau of Indian Standards (BIS) Intelligent Assistant. "
                    "I can help you with Indian Standards, conformity assessment, BIS certifications, "
                    "testing laboratories, and BIS services. Your query appears to be outside of this scope."
                ),
                citations=[],
                data=[],
            ).model_dump()

        # Step 3: Deterministic Database Execution
        db_result: QueryExecutionResult = self.executor.execute(plan)

        # Step 4: Vector RAG Retrieval (when required by plan)
        rag_chunks: list[RetrievedChunk] = []
        if plan.needs_rag:
            try:
                rag_chunks = self.retriever.search(
                    query=plan.normalized_query,
                    k=5,
                )
            except Exception as e:
                logger.warning(f"RAG retrieval encountered an issue, proceeding with DB data: {e}")

        # Step 5: Assemble Structured Data Cards and Citations
        data_cards, citations = self._build_cards_and_citations(db_result, rag_chunks)

        # Step 6: Build Synthesis Context & Final LLM Generation
        final_message = self._synthesize_response(
            user_query=user_message,
            plan=plan,
            db_result=db_result,
            rag_chunks=rag_chunks,
        )

        # Step 7: Construct ChatResponse
        response = ChatResponse(
            message_type="answer",
            conversation_id=conv_id,
            message=final_message,
            citations=citations,
            data=data_cards,
        )

        return response.model_dump()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_cards_and_citations(
        self,
        db_result: QueryExecutionResult,
        rag_chunks: list[RetrievedChunk],
    ) -> tuple[list[DataBlock], list[Citation]]:
        """Construct validated frontend DataBlock cards and Citation objects."""
        cards: list[DataBlock] = []
        citations: list[Citation] = []
        seen_card_keys: set[tuple[str, str]] = set()
        cit_index = 1

        fallback_url = "https://www.bis.gov.in/"

        # 1. Process Database Results into Cards
        for op_name, records in db_result.results.items():
            for rec in records:
                if op_name in ("FIND_STANDARD", "FIND_APPLICABLE_STANDARDS"):
                    is_num = rec.get("is_number") or rec.get("standard_number") or "IS Standard"
                    key = ("standard", is_num)
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
                                relevance=rec.get("relevance") or "Primary",
                                applicable_when=f"Applicable for {rec.get('product_name')}" if rec.get("product_name") else None,
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

        # 2. Process RAG Chunks into Citations
        for chunk in rag_chunks:
            citations.append(
                Citation(
                    id=f"cit_{cit_index}",
                    source_type="document",
                    title=chunk.filename or "BIS Technical Document",
                    reference=chunk.author or chunk.filename,
                    source_url="https://standards.bis.gov.in/",
                )
            )
            cit_index += 1

        return cards, citations

    def _synthesize_response(
        self,
        user_query: str,
        plan: QueryPlan,
        db_result: QueryExecutionResult,
        rag_chunks: list[RetrievedChunk],
    ) -> str:
        """Prompt the LLM to synthesize the retrieved evidence into a natural-language answer."""
        prompt = f"""You are the Bureau of Indian Standards (BIS) Conversational Assistant.
Synthesize a clear, authoritative, and direct response to the user's question using ONLY the provided evidence.

==================================================
EVIDENCE SUMMARY
==================================================
User Query: "{user_query}"
Normalized English Query: "{plan.normalized_query}"
Target Response Language: "{plan.response_language}"

--- STRUCTURED DATABASE RECORDS ---
{db_result.results if db_result.executed else "No database records retrieved."}

--- DOCUMENTARY RAG PASSAGES ---
{chr(10).join([f"[Passage {i+1}] (Source: {c.filename}): {c.chunk.text}" for i, c in enumerate(rag_chunks)]) if rag_chunks else "No document passages retrieved."}

--- MISSING INFORMATION / GAPS ---
{plan.missing_information if plan.missing_information else "None"}

==================================================
INSTRUCTIONS
==================================================
1. Base your answer strictly on the provided database and document evidence.
2. If citing document passages, embed inline tags like <cit_1>, <cit_2>.
3. Mention applicable standard numbers (e.g., IS 694), certification requirements, or testing laboratories where provided.
4. If certain parameters were missing or records were not found, state that clearly without guessing.
5. Respond in the requested response language (e.g., respond in Hindi if language is 'hi', or English if 'en').
6. Provide ONLY the final answer message.
"""

        try:
            raw = self.ai_engine.complete(prompt, SynthesisResponse)
            if isinstance(raw, SynthesisResponse):
                return raw.answer
            return raw.get("answer") or str(raw)
        except Exception as e:
            logger.exception(f"Synthesis LLM generation failed: {e}")
            # Fallback natural language construction if LLM call fails
            if db_result.executed and db_result.record_count > 0:
                return f"Based on BIS records, here is the information matching your query: {plan.normalized_query}."
            return "We could not find specific BIS records matching your query. Please provide additional product details."


# Singleton instance and module-level function
_query_service = QueryService()


def process_query(payload: ChatRequest) -> dict[str, Any]:
    """
    Module-level entry point called by the Flask /v1/query route.
    """
    return _query_service.process(payload)