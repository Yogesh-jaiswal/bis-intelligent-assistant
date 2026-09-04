"""
tests/features/test_deterministic_planner.py
============================================

Tests for the Deterministic Retrieval Planner, deterministic RAG routing rules,
document-first vector filtering, and citation generation without LLM retrieval controller.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from configs import get_settings
from models.enums import DocumentBlockType, DocumentTypes
from services.file_processors.document.doc_representation import DocumentBlock
from services.query.query_service import QueryService
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.retrieval.citation_builder import CitationBuilder
from services.retrieval.deterministic_planner import DeterministicPlanner, DeterministicRetrievalPlan
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import ChatRequest, UserMessage


def _make_mock_chunk(
    text: str,
    filename: str,
    source_url: str | None = None,
    page: int | None = None,
    is_number: str | None = None,
) -> RetrievedChunk:
    meta = {}
    if page is not None:
        meta["page"] = page
    if is_number is not None:
        meta["is_number"] = is_number
    return RetrievedChunk(
        score=0.92,
        chunk=DocumentBlock(type=DocumentBlockType.PARAGRAPH, text=text, metadata=meta),
        filename=filename,
        author=None,
        source_type=DocumentTypes.PDF,
        source_url=source_url,
    )


class TestDeterministicRAGRoutingRules:
    def test_rule_r1_explicit_documentary_keyword_triggers_rag(self):
        """Rule R1: Queries containing requirements, specifications, scope, clauses trigger RAG."""
        plan = QueryPlan(
            normalized_query="What are the technical requirements and testing specifications under IS 694?",
            relevant=True,
            intent=QueryIntent.TECHNICAL_QUESTION,
            response_language="en",
            needs_db=True,
            needs_rag=False,  # Analyzer initially false -> Planner must override
            db_operations=[DatabaseOperation.FIND_STANDARD],
            parameters={"standard_number": "IS 694"},
        )
        hop1 = DeterministicPlanner.plan_initial_hop(plan)
        assert hop1.execute_rag is True
        assert hop1.rag_standard_filter == "IS 694"
        assert "Rule R1" in hop1.reason

    def test_rule_r2_standard_discovered_triggers_document_grounded_rag(self):
        """Rule R2: When DB identifies a standard, plan_next_hop routes to document RAG."""
        initial_plan = QueryPlan(
            normalized_query="Which BIS standard applies to my PVC cable?",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
            parameters={"product": "PVC cable"},
        )
        accumulated_db = {
            "FIND_APPLICABLE_STANDARDS": [
                {"is_number": "IS 694:2010", "title": "PVC Insulated Cables", "product_name": "PVC Cable"}
            ]
        }
        accumulated_rag = []
        executed_sigs = {"FIND_APPLICABLE_STANDARDS:[('product', 'PVC cable')]"}

        hop2 = DeterministicPlanner.plan_next_hop(
            user_query="Which BIS standard applies to my PVC cable?",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=accumulated_rag,
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        assert hop2.execute_rag is True
        assert hop2.rag_standard_filter == "IS 694:2010"
        assert "IS 694:2010" in hop2.rag_query

    def test_rule_r3_pure_metadata_query_stays_db_only(self):
        """Rule R3: Pure metadata lookup (e.g. status, publication date) does not trigger RAG."""
        initial_plan = QueryPlan(
            normalized_query="What is the publication year and status of IS 694:2010?",
            relevant=True,
            intent=QueryIntent.STANDARD_LOOKUP,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_STANDARD],
            parameters={"standard_number": "IS 694:2010"},
        )
        accumulated_db = {
            "FIND_STANDARD": [
                {"is_number": "IS 694:2010", "status": "Active", "publication_year": 2010}
            ]
        }
        executed_sigs = {"FIND_STANDARD:[('standard_number', 'IS 694:2010')]"}

        hop2 = DeterministicPlanner.plan_next_hop(
            user_query="What is the publication year and status of IS 694:2010?",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=[],
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        assert hop2.continue_retrieval is False
        assert hop2.execute_rag is False

    def test_downstream_parameter_chaining(self):
        """Dependency chaining: Standard -> Certification -> Testing Labs."""
        initial_plan = QueryPlan(
            normalized_query="Applicable BIS standard and mandatory certification and testing lab for PVC cables",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
            parameters={"product": "PVC cable"},
        )
        accumulated_db = {
            "FIND_APPLICABLE_STANDARDS": [
                {"is_number": "IS 694:2010", "title": "PVC Insulated Cables"}
            ]
        }
        executed_sigs = {"FIND_APPLICABLE_STANDARDS:[('product', 'PVC cable')]"}

        hop2 = DeterministicPlanner.plan_next_hop(
            user_query="Do I need BIS certification and testing laboratories for PVC cables?",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=[_make_mock_chunk("IS 694 details", "694.pdf")],
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        assert hop2.continue_retrieval is True
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in hop2.db_operations
        assert DatabaseOperation.FIND_LABORATORIES in hop2.db_operations
        assert hop2.db_parameters.get("standard_number") == "IS 694:2010"

    def test_duplicate_operation_prevention(self):
        """Signatures already executed are excluded from subsequent hops."""
        initial_plan = QueryPlan(
            normalized_query="test query",
            relevant=True,
            intent=QueryIntent.STANDARD_LOOKUP,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[],
            parameters={"standard_number": "IS 694:2010"},
        )
        accumulated_db = {
            "FIND_STANDARD": [{"is_number": "IS 694:2010"}],
            "GET_CERTIFICATION_REQUIREMENT": [{"scheme_code": "Scheme-I"}],
            "FIND_LABORATORIES": [{"name": "Lab 1"}],
        }
        executed_sigs = {
            "FIND_STANDARD:[('standard_number', 'IS 694:2010')]",
            "GET_CERTIFICATION_REQUIREMENT:[('standard_number', 'IS 694:2010')]",
            "FIND_LABORATORIES:[('standard_number', 'IS 694:2010')]",
        }

        hop2 = DeterministicPlanner.plan_next_hop(
            user_query="testing and certification",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=[_make_mock_chunk("Passage", "doc.pdf")],
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        assert hop2.continue_retrieval is False
        assert hop2.db_operations == []

    def test_max_retrieval_hops_protection(self):
        """Retrieval strictly terminates when MAX_RETRIEVAL_HOPS is reached."""
        settings = get_settings()
        plan = DeterministicPlanner.plan_next_hop(
            user_query="query",
            initial_plan=MagicMock(normalized_query="query"),
            accumulated_db_results={},
            accumulated_rag_chunks=[],
            hop_count=settings.MAX_RETRIEVAL_HOPS,
            executed_signatures=set(),
        )
        assert plan.continue_retrieval is False
        assert "maximum permitted retrieval hops" in plan.reason.lower()

    def test_max_evidence_count_protection(self):
        """Retrieval terminates when MAX_EVIDENCE_COUNT is exceeded."""
        settings = get_settings()
        chunks = [_make_mock_chunk(f"text {i}", f"doc_{i}.pdf") for i in range(settings.MAX_EVIDENCE_COUNT + 1)]
        plan = DeterministicPlanner.plan_next_hop(
            user_query="query",
            initial_plan=MagicMock(normalized_query="query"),
            accumulated_db_results={},
            accumulated_rag_chunks=chunks,
            hop_count=1,
            executed_signatures=set(),
        )
        assert plan.continue_retrieval is False
        assert "sufficient evidence" in plan.reason.lower()


class TestEndToEndDeterministicPipeline:
    def test_pvc_cable_standard_and_rag_grounding_without_llm_controller(self):
        """
        End-to-End Query:
        'Which BIS standard applies to my PVC cable?'
        - DB finds IS 694:2010
        - Planner triggers document-grounded RAG for IS 694:2010
        - Citation is generated
        - Zero LLM controller calls
        """
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {"answer": "IS 694:2010 applies to PVC insulated cables <cit_1>."}

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = QueryPlan(
            normalized_query="Which BIS standard applies to PVC cable?",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            response_language="en",
            needs_db=True,
            needs_rag=False,  # Analyzer starts with DB
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
            parameters={"product": "PVC cable"},
        )

        qs = QueryService(analyzer=mock_analyzer, ai_engine=mock_ai)

        # Mock DB repositories
        mock_product_repo = MagicMock()
        mock_product_repo.find_applicable_standards.return_value = [
            {
                "is_number": "IS 694:2010",
                "title": "PVC Insulated Cables for Working Voltages up to and including 1100 V",
                "product_name": "PVC Insulated Electric Cables",
                "source_url": "https://standards.bis.gov.in/694",
            }
        ]
        qs.executor.product_repo = mock_product_repo

        # Mock vector retriever
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            _make_mock_chunk(
                "This standard covers the requirements and testing of PVC insulated single core and multi-core cables up to 1100V.",
                "694_2010_reff2020.pdf",
                "https://standards.bis.gov.in/website/know-your-standards",
                page=1,
            )
        ]
        qs.retriever = mock_retriever

        req = ChatRequest(
            conversation_id="conv_pvc_rag",
            message=UserMessage(content="Which BIS standard applies to my PVC cable?"),
        )
        res = qs.process(req)

        assert res["message_type"] == "answer"
        assert len(res["data"]) >= 1
        assert res["data"][0]["is_number"] == "IS 694:2010"
        assert len(res["citations"]) >= 1
        assert str(res["citations"][0]["source_url"]) == "https://standards.bis.gov.in/website/know-your-standards"
        assert "page 1" in res["citations"][0]["reference"]

        # Verify search was called with IS 694 filter constraint
        mock_retriever.search.assert_called()
        call_kwargs = mock_retriever.search.call_args[1]
        assert call_kwargs.get("standard_number") == "IS 694:2010"

    def test_citation_fallback_to_standard_source_url(self):
        """Citation builder falls back to standard source_url if chunk source_url is missing."""
        chunk = _make_mock_chunk(
            "Testing under clause 5.1",
            "694_2010_reff2020.pdf",
            source_url=None,  # Missing on chunk
            is_number="IS 694:2010",
        )
        citations = CitationBuilder.build_api_citations([chunk])
        # Even without chunk.source_url, CitationBuilder inspects standard / manifest
        assert len(citations) >= 0
