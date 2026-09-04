"""
tests/features/test_service_routing_and_rag_prevention.py
=========================================================

Regression and integration tests verifying:
1. Deterministic reconciliation of contradictory QueryPlans for structured lookups.
2. Prevention of arbitrary vector RAG searches over Indian Standard PDFs for services, labs, and schemes.
3. Preservation of RAG for GENERAL_BIS_QUERY and technical/testing queries.
4. Citation relevance filtering ensuring standard document PDFs are not cited for BIS service queries.
5. End-to-end pipeline execution for English and Hindi BIS service queries.
"""

from unittest.mock import MagicMock
import pytest

from services.query_analyser.analyser_schema import (
    DatabaseOperation,
    QueryIntent,
    QueryPlan,
)
from services.retrieval.deterministic_planner import DeterministicPlanner
from services.file_processors.document.doc_representation import DocumentBlock
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from services.query.query_service import QueryService
from services.query_analyser.analyser import QueryAnalyzer
from validators.chat_responses import ChatRequest, UserMessage


def _make_mock_chunk(text: str, filename: str, source_url: str = "https://standards.bis.gov.in/") -> RetrievedChunk:
    return RetrievedChunk(
        score=0.45,
        chunk=DocumentBlock(type="paragraph", text=text, metadata={"page": 1}),
        filename=filename,
        author="BIS",
        source_type="standard",
        source_url=source_url,
    )


class TestIntentToRetrievalCompatibility:
    """Tests for DeterministicPlanner intent-to-retrieval validation and reconciliation."""

    def test_bis_service_lookup_reconciles_contradiction_and_disables_rag(self):
        """
        When the analyzer produces the exact bug:
        Intent=BIS_SERVICE_LOOKUP, NeedsDB=False, Ops=[], NeedsRAG=True
        The planner must reconcile it to execute GET_BIS_SERVICE and suppress RAG.
        """
        buggy_plan = QueryPlan(
            normalized_query="What services does BIS provide?",
            relevant=True,
            intent=QueryIntent.BIS_SERVICE_LOOKUP,
            response_language="en",
            needs_db=False,
            needs_rag=True,
            db_operations=[],
            parameters={},
        )

        hop = DeterministicPlanner.plan_initial_hop(buggy_plan)

        assert hop.execute_db is True
        assert DatabaseOperation.GET_BIS_SERVICE in hop.db_operations
        assert hop.execute_rag is False
        assert "Injected GET_BIS_SERVICE" in hop.reason
        assert "Suppressed RAG" in hop.reason

    def test_laboratory_lookup_reconciles_and_disables_rag(self):
        """
        LABORATORY_LOOKUP without DB operations must be reconciled to FIND_LABORATORIES
        and RAG must be suppressed.
        """
        plan = QueryPlan(
            normalized_query="Where can I test my cables in Gujarat?",
            relevant=True,
            intent=QueryIntent.LABORATORY_LOOKUP,
            response_language="en",
            needs_db=False,
            needs_rag=True,
            db_operations=[],
            parameters={"product": "cables", "state": "Gujarat"},
        )

        hop = DeterministicPlanner.plan_initial_hop(plan)

        assert hop.execute_db is True
        assert DatabaseOperation.FIND_LABORATORIES in hop.db_operations
        assert hop.execute_rag is False

    def test_certification_process_reconciles_and_disables_rag(self):
        """
        CERTIFICATION_PROCESS without DB operations must be reconciled to GET_CERTIFICATION_SCHEME
        and RAG suppressed.
        """
        plan = QueryPlan(
            normalized_query="What is the process to apply for FMCS certification?",
            relevant=True,
            intent=QueryIntent.CERTIFICATION_PROCESS,
            response_language="en",
            needs_db=False,
            needs_rag=True,
            db_operations=[],
            parameters={"scheme_name": "FMCS"},
        )

        hop = DeterministicPlanner.plan_initial_hop(plan)

        assert hop.execute_db is True
        assert DatabaseOperation.GET_CERTIFICATION_SCHEME in hop.db_operations
        assert hop.execute_rag is False

    def test_general_bis_query_preserves_rag(self):
        """
        GENERAL_BIS_QUERY (e.g. 'What is BIS?') legitimately requires documentary RAG passages
        describing BIS itself.
        """
        plan = QueryPlan(
            normalized_query="What is the Bureau of Indian Standards and what is its mandate?",
            relevant=True,
            intent=QueryIntent.GENERAL_BIS_QUERY,
            response_language="en",
            needs_db=False,
            needs_rag=True,
            db_operations=[],
            parameters={},
        )

        hop = DeterministicPlanner.plan_initial_hop(plan)

        assert hop.execute_rag is True
        assert hop.rag_standard_filter is None

    def test_technical_question_preserves_rag_and_standard_filter(self):
        """
        TECHNICAL_QUESTION with identified standard must keep RAG enabled and filtered to standard.
        """
        plan = QueryPlan(
            normalized_query="What is the maximum permissible conductor temperature under IS 694?",
            relevant=True,
            intent=QueryIntent.TECHNICAL_QUESTION,
            response_language="en",
            needs_db=True,
            needs_rag=True,
            db_operations=[DatabaseOperation.FIND_STANDARD],
            parameters={"standard_number": "IS 694:2010"},
        )

        hop = DeterministicPlanner.plan_initial_hop(plan)

        assert hop.execute_rag is True
        assert hop.rag_standard_filter == "IS 694:2010"


class TestCitationRelevanceAndFiltering:
    """Tests that standard PDF chunks are not converted to citations for structured lookups."""

    def test_citation_assembly_excludes_standard_chunks_for_service_intent(self):
        """
        For BIS_SERVICE_LOOKUP, stray standard PDF chunks must not be converted to citations.
        """
        qs = QueryService(analyzer=MagicMock(), controller=MagicMock(), ai_engine=MagicMock())
        db_results = {
            "GET_BIS_SERVICE": [
                {"id": "1", "name": "ISI Mark Licence", "service_type": "Product Certification", "source_url": "https://www.bis.gov.in/product-certification/"}
            ]
        }
        stray_standard_chunks = [
            _make_mock_chunk("Pressure cookers shall comply with safety valves clause 4.2", "IS_2347_cookers.pdf", "https://standards.bis.gov.in/is2347.pdf"),
            _make_mock_chunk("High strength deformed steel bars specification", "IS_1786_steel.pdf", "https://standards.bis.gov.in/is1786.pdf"),
        ]
        plan = QueryPlan(
            normalized_query="What services does BIS provide?",
            relevant=True,
            intent=QueryIntent.BIS_SERVICE_LOOKUP,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.GET_BIS_SERVICE],
            parameters={},
        )

        cards, citations = qs._build_cards_and_citations(db_results, stray_standard_chunks, plan=plan)

        assert len(cards) == 1
        assert cards[0].data_type == "service"
        assert len(citations) == 0, "Stray standard PDF chunks must NOT be cited for BIS_SERVICE_LOOKUP"

    def test_citation_assembly_preserves_citations_for_technical_intent(self):
        """
        For TECHNICAL_QUESTION, standard PDF chunks must be preserved as citations.
        """
        qs = QueryService(analyzer=MagicMock(), controller=MagicMock(), ai_engine=MagicMock())
        chunks = [
            _make_mock_chunk("Maximum operating temperature is 70 degrees C.", "IS_694.pdf", "https://standards.bis.gov.in/is694.pdf")
        ]
        plan = QueryPlan(
            normalized_query="What is the temperature limit in IS 694?",
            relevant=True,
            intent=QueryIntent.TECHNICAL_QUESTION,
            response_language="en",
            needs_db=True,
            needs_rag=True,
            db_operations=[DatabaseOperation.FIND_STANDARD],
            parameters={"standard_number": "IS 694:2010"},
        )

        cards, citations = qs._build_cards_and_citations({}, chunks, plan=plan)

        assert len(citations) == 1
        assert citations[0].source_type in ("document", "standard")


class TestServiceQueryEndToEnd:
    """End-to-end tests for BIS service queries under English and Hindi."""

    @pytest.fixture
    def mock_services(self):
        return [
            {"id": "1", "name": "Grant of BIS Product Certification Licence (ISI Mark)", "service_type": "Product Certification", "source_url": "https://www.bis.gov.in/product-certification/"},
            {"id": "2", "name": "Renewal of BIS Product Certification Licence", "service_type": "Product Certification", "source_url": "https://www.bis.gov.in/product-certification/"},
            {"id": "3", "name": "Grant of BIS Licence under Foreign Manufacturers Certification Scheme (FMCS)", "service_type": "Product Certification", "source_url": "https://www.bis.gov.in/fmcs/"},
            {"id": "4", "name": "Compulsory Registration of Electronics and IT Goods (CRS)", "service_type": "Product Certification", "source_url": "https://www.bis.gov.in/crs/"},
            {"id": "5", "name": "BIS Hallmarking Registration for Jewellers", "service_type": "Hallmarking", "source_url": "https://www.bis.gov.in/hallmarking/"},
            {"id": "6", "name": "BIS Laboratory Recognition (Third-Party Testing Labs)", "service_type": "Laboratory Service", "source_url": "https://www.bis.gov.in/labs/"},
            {"id": "7", "name": "Training Programmes on Indian Standards (NITS)", "service_type": "Training", "source_url": "https://www.bis.gov.in/nits/"},
        ]

    def test_english_service_query_reconciles_buggy_analyzer_output(self, mock_services):
        """
        End-to-end test for 'What are services provided BIS?':
        Even if the analyzer emitted the buggy QueryPlan (NeedsDB=False, Ops=[], NeedsRAG=True),
        the pipeline must execute GET_BIS_SERVICE, suppress standard PDF RAG, return ServiceCards,
        and attach 0 unrelated standard citations.
        """
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {
            "answer": "The Bureau of Indian Standards (BIS) provides several key services including ISI Mark Certification, FMCS, CRS, Hallmarking, Laboratory Recognition, and NITS Training.",
            "conversation_summary": "Topic: BIS Services overview."
        }

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = QueryPlan(
            normalized_query="What services does BIS provide?",
            relevant=True,
            intent=QueryIntent.BIS_SERVICE_LOOKUP,
            response_language="en",
            needs_db=False,  # BUGGY analyzer output
            needs_rag=True,  # BUGGY analyzer output
            db_operations=[],  # BUGGY analyzer output
            parameters={},
        )

        qs = QueryService(analyzer=mock_analyzer, controller=MagicMock(), ai_engine=mock_ai)
        mock_service_repo = MagicMock()
        mock_service_repo.get_bis_service.return_value = mock_services
        qs.executor.service_repo = mock_service_repo

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            _make_mock_chunk("Pressure cookers safety valve clause", "IS_2347.pdf")
        ]
        qs.retriever = mock_retriever

        req = ChatRequest(
            conversation_id="conv_services_test",
            message=UserMessage(content="What are services provided BIS?"),
        )
        res = qs.process(req)

        assert res["message_type"] == "answer"
        assert len(res["data"]) == len(mock_services)
        assert all(d["data_type"] == "service" for d in res["data"])
        assert len(res["citations"]) == 0, "No standard PDF citations must be attached for service queries"
        assert "ISI Mark" in res["message"]

    def test_hindi_service_query_e2e(self, mock_services):
        """
        End-to-end test for Hindi service query:
        'BIS की कौन-कौन सी सेवाएँ हैं?'
        """
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {
            "answer": "भारतीय मानक ब्यूरो (BIS) कई महत्वपूर्ण सेवाएँ प्रदान करता है जैसे ISI मार्क प्रमाणन, हॉलमार्किंग, और प्रयोगशाला परीक्षण सेवाएँ।",
            "conversation_summary": "विषय: BIS सेवाएँ।"
        }

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = QueryPlan(
            normalized_query="What services does BIS provide?",
            relevant=True,
            intent=QueryIntent.BIS_SERVICE_LOOKUP,
            response_language="hi",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.GET_BIS_SERVICE],
            parameters={},
        )

        qs = QueryService(analyzer=mock_analyzer, controller=MagicMock(), ai_engine=mock_ai)
        mock_service_repo = MagicMock()
        mock_service_repo.get_bis_service.return_value = mock_services
        qs.executor.service_repo = mock_service_repo

        req = ChatRequest(
            conversation_id="conv_hindi_services",
            message=UserMessage(content="BIS की कौन-कौन सी सेवाएँ हैं?", language="hi"),
        )
        res = qs.process(req)

        assert res["message_type"] == "answer"
        assert len(res["data"]) == len(mock_services)
        assert len(res["citations"]) == 0


class TestHeuristicFallbackPreservation:
    """Tests for QueryAnalyzer._heuristic_fallback for service and general queries."""

    def test_heuristic_service_query_preserves_question_and_routes_to_db(self):
        analyzer = QueryAnalyzer(ai_engine=MagicMock())
        plan = analyzer._heuristic_fallback("What are services provided BIS?")

        assert plan.intent == QueryIntent.BIS_SERVICE_LOOKUP
        assert plan.normalized_query == "What services does BIS provide?"
        assert plan.needs_db is True
        assert plan.needs_rag is False
        assert plan.db_operations == [DatabaseOperation.GET_BIS_SERVICE]

    def test_heuristic_hindi_service_query(self):
        analyzer = QueryAnalyzer(ai_engine=MagicMock())
        plan = analyzer._heuristic_fallback("BIS की कौन-कौन सी सेवाएँ हैं?")

        assert plan.intent == QueryIntent.BIS_SERVICE_LOOKUP
        assert plan.response_language == "hi"
        assert plan.needs_db is True
        assert plan.needs_rag is False
        assert plan.db_operations == [DatabaseOperation.GET_BIS_SERVICE]
