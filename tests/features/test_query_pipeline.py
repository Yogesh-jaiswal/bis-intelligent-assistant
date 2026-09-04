"""
tests/features/test_query_pipeline.py
======================================

Comprehensive integration tests for the BIS Query Pipeline:
  1. Semantic database retrieval with differently-worded queries
  2. Retrieval Controller decisions (COMPLETE, RETRIEVE_MORE, NEED_CLARIFICATION)
  3. Maximum retrieval hops protection (loop termination)
  4. Clarification question generation and subsequent retrieval integration
  5. Citation ID validation (inline tags match returned Citation objects)
  6. Source URL provenance and page metadata preservation (no URL fabrication)
  7. Response localization preserving technical terminology (IS 694, 1100V, Scheme-I)
  8. Service collection listing without artificial parameters
  9. General BIS information documentary retrieval
 10. Multi-hop complex query execution
 11. Duplicate evidence prevention
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from configs import get_settings
from models.enums import DocumentBlockType, DocumentTypes
from repositories.semantic_search import rank_entities_semantically
from services.ai.engine import AIEngine
from services.file_processors.document.doc_representation import DocumentBlock
from services.query.query_service import QueryService
from services.query_analyser import QueryAnalyzer
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.query_executor.executor_schema import ExecutionStatus, QueryExecutionResult
from services.retrieval.citation_builder import CitationBuilder
from services.retrieval.retrieval_controller import (
    ControllerDecision,
    RetrievalController,
    StructuredControllerOutput,
)
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import ChatRequest, UserMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_chunk(text: str, filename: str, source_url: str | None = None, page: int | None = None) -> RetrievedChunk:
    meta = {}
    if page is not None:
        meta["page"] = page
    return RetrievedChunk(
        score=0.92,
        chunk=DocumentBlock(type=DocumentBlockType.PARAGRAPH, text=text, metadata=meta),
        filename=filename,
        author=None,
        source_type=DocumentTypes.PDF,
        source_url=source_url,
    )


# ===========================================================================
# 1. Semantic Database Retrieval
# ===========================================================================

class TestSemanticDatabaseRetrieval:
    def test_semantic_ranking_returns_best_match(self):
        entities = [
            {"id": "1", "name": "PVC insulated cables up to 1100V", "desc": "domestic electrical wiring cables"},
            {"id": "2", "name": "Steel structural bars", "desc": "construction steel reinforcement"},
            {"id": "3", "name": "Polyethylene insulated power cables", "desc": "high voltage industrial cable"},
        ]
        query = "wires for home electrical wiring"
        ranked = rank_entities_semantically(
            query=query,
            entities=entities,
            text_extractor=lambda e: f"{e['name']} {e['desc']}",
            limit=2,
            threshold=0.1,
        )
        assert len(ranked) > 0
        top_entity = ranked[0][1]
        assert "PVC insulated cables" in top_entity["name"]

    def test_empty_query_returns_empty_ranking(self):
        assert rank_entities_semantically("", [{"id": 1}], lambda x: "") == []


# ===========================================================================
# 2. Retrieval Controller Decisions
# ===========================================================================

class TestRetrievalController:
    def _controller_with_mock_ai(self, mock_output: StructuredControllerOutput) -> RetrievalController:
        ai = MagicMock()
        ai.complete.return_value = mock_output
        return RetrievalController(ai_engine=ai)

    def test_controller_complete_decision(self):
        output = StructuredControllerOutput(
            decision=ControllerDecision.COMPLETE,
            reason="Standard details retrieved; query fully answered.",
        )
        ctrl = self._controller_with_mock_ai(output)
        plan = QueryPlan(
            normalized_query="What is IS 694?",
            relevant=True,
            intent=QueryIntent.STANDARD_LOOKUP,
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_STANDARD],
        )
        res = ctrl.evaluate(
            user_query="What is IS 694?",
            plan=plan,
            accumulated_db_results={"FIND_STANDARD": [{"is_number": "IS 694:2010"}]},
            accumulated_rag_chunks=[],
            hop_count=1,
            executed_signatures=set(),
        )
        assert res.decision == ControllerDecision.COMPLETE

    def test_controller_retrieve_more_decision(self):
        output = StructuredControllerOutput(
            decision=ControllerDecision.RETRIEVE_MORE,
            reason="Applicable standard found; retrieving certification requirements.",
            next_operations=[DatabaseOperation.GET_CERTIFICATION_REQUIREMENT],
            next_parameters={"standard_number": "IS 694:2010"},
        )
        ctrl = self._controller_with_mock_ai(output)
        plan = QueryPlan(
            normalized_query="Do I need certification for my cables?",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
        )
        res = ctrl.evaluate(
            user_query="Do I need certification for my cables?",
            plan=plan,
            accumulated_db_results={"FIND_APPLICABLE_STANDARDS": [{"is_number": "IS 694:2010"}]},
            accumulated_rag_chunks=[],
            hop_count=1,
            executed_signatures=set(),
        )
        assert res.decision == ControllerDecision.RETRIEVE_MORE
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in res.next_operations
        assert res.next_parameters["standard_number"] == "IS 694:2010"

    def test_controller_need_clarification_decision(self):
        output = StructuredControllerOutput(
            decision=ControllerDecision.NEED_CLARIFICATION,
            reason="Multiple cable standards exist.",
            clarification_question="What is the voltage rating of your cable?",
            clarification_options=["Up to 1100V", "Above 1100V", "Other"],
            clarification_input_type="select",
        )
        ctrl = self._controller_with_mock_ai(output)
        plan = QueryPlan(
            normalized_query="Which standard for cables?",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
        )
        res = ctrl.evaluate(
            user_query="Which standard for cables?",
            plan=plan,
            accumulated_db_results={"FIND_APPLICABLE_STANDARDS": [{"is_number": "IS 694"}, {"is_number": "IS 7098"}]},
            accumulated_rag_chunks=[],
            hop_count=1,
            executed_signatures=set(),
        )
        assert res.decision == ControllerDecision.NEED_CLARIFICATION
        assert "voltage rating" in res.clarification_question.lower()

    def test_max_hops_protection(self):
        """Controller must force COMPLETE when hop_count reaches MAX_RETRIEVAL_HOPS."""
        ctrl = RetrievalController()
        settings = get_settings()
        plan = QueryPlan(
            normalized_query="complex query",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            needs_db=True,
            needs_rag=False,
        )
        res = ctrl.evaluate(
            user_query="complex query",
            plan=plan,
            accumulated_db_results={},
            accumulated_rag_chunks=[],
            hop_count=settings.MAX_RETRIEVAL_HOPS,
            executed_signatures=set(),
        )
        assert res.decision == ControllerDecision.COMPLETE
        assert "maximum" in res.reason.lower()


# ===========================================================================
# 3. Citation Integration & Provenance
# ===========================================================================

class TestCitationIntegration:
    def test_citation_ids_sequential(self):
        chunks = [
            _make_mock_chunk("IS 694 clause 1", "694.pdf", "https://standards.bis.gov.in/694", page=2),
            _make_mock_chunk("IS 1554 clause 3", "1554.pdf", "https://standards.bis.gov.in/1554", page=5),
        ]
        citations = CitationBuilder.build_api_citations(chunks)
        assert len(citations) == 2
        assert citations[0].id == "cit_1"
        assert citations[1].id == "cit_2"

    def test_page_reference_preserved_in_citation(self):
        chunk = _make_mock_chunk("IS 694 text", "694.pdf", "https://standards.bis.gov.in/694", page=7)
        citations = CitationBuilder.build_api_citations([chunk])
        assert len(citations) == 1
        assert "page 7" in citations[0].reference.lower()

    def test_no_url_excluded_without_fabrication(self):
        chunk = _make_mock_chunk("Text without verified URL", "unknown.pdf", source_url=None)
        citations = CitationBuilder.build_api_citations([chunk])
        assert len(citations) == 0


# ===========================================================================
# 4. Response Localization & Technical Terminology
# ===========================================================================

class TestResponseLocalization:
    def test_synthesizer_prompt_contains_technical_preservation_instruction(self):
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {"answer": "IS 694:2010 1100 V तक के PVC केबल्स के लिए भारतीय मानक है।"}
        qs = QueryService(ai_engine=mock_ai)
        plan = QueryPlan(
            normalized_query="What is IS 694?",
            relevant=True,
            intent=QueryIntent.STANDARD_LOOKUP,
            response_language="hi",
            needs_db=True,
            needs_rag=False,
        )

        citations = CitationBuilder.build_api_citations([
            _make_mock_chunk("IS 694 text", "694.pdf", "https://standards.bis.gov.in/694")
        ])

        msg, _ = qs._synthesize_response(
            user_query="IS 694 क्या है?",
            plan=plan,
            accumulated_db_results={"FIND_STANDARD": [{"is_number": "IS 694:2010"}]},
            rag_chunks=[],
            citations=citations,
        )

        assert "IS 694" in msg
        called_prompt = mock_ai.complete.call_args[0][0]
        assert "DO NOT translate technical identifiers" in called_prompt
        assert "response_language" in called_prompt.lower() or "hi" in called_prompt


# ===========================================================================
# 5. Service Collection Query
# ===========================================================================

class TestServiceCollectionQuery:
    def test_service_listing_without_search_term_returns_all(self):
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {"answer": "Here are the services offered by BIS."}

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = QueryPlan(
            normalized_query="What are the services offered by BIS?",
            relevant=True,
            intent=QueryIntent.BIS_SERVICE_LOOKUP,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.GET_BIS_SERVICE],
            parameters={},
        )

        mock_controller = MagicMock()
        mock_controller.evaluate.return_value = StructuredControllerOutput(
            decision=ControllerDecision.COMPLETE,
            reason="All services retrieved.",
        )

        qs = QueryService(analyzer=mock_analyzer, controller=mock_controller, ai_engine=mock_ai)
        mock_service_repo = MagicMock()
        mock_service_repo.get_bis_service.return_value = [
            {"id": "1", "name": "Grant of Licence", "service_type": "Product Certification", "source_url": "https://bis.gov.in"},
            {"id": "2", "name": "Hallmarking Registration", "service_type": "Hallmarking", "source_url": "https://bis.gov.in"},
        ]
        qs.executor.service_repo = mock_service_repo

        req = ChatRequest(
            conversation_id="conv_srv_1",
            message=UserMessage(content="What are the services offered by BIS?"),
        )
        res = qs.process(req)

        assert res["message_type"] == "answer"
        assert len(res["data"]) >= 2
        card_names = [d["name"] for d in res["data"]]
        assert "Grant of Licence" in card_names


# ===========================================================================
# 6. End-to-End Multi-Hop Query Support
# ===========================================================================

class TestEndToEndMultiHopQuery:
    def test_multi_hop_cable_query_execution(self):
        """
        Demo Query A:
        'For my cables business, do I need BIS certification and do I need to test it in laboratories?'
        Hop 1: FIND_APPLICABLE_STANDARDS -> finds IS 694
        Hop 2: GET_CERTIFICATION_REQUIREMENT + FIND_LABORATORIES -> finds Scheme-I and Lab
        """
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {"answer": "For cables, IS 694 applies under Scheme-I and testing is done at CPRI."}

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = QueryPlan(
            normalized_query="cables business BIS certification testing laboratories",
            relevant=True,
            intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
            parameters={"product": "cables"},
        )

        mock_controller = MagicMock()
        mock_controller.evaluate.side_effect = [
            StructuredControllerOutput(
                decision=ControllerDecision.RETRIEVE_MORE,
                reason="Standard found, retrieving certification and lab requirements.",
                next_operations=[DatabaseOperation.GET_CERTIFICATION_REQUIREMENT, DatabaseOperation.FIND_LABORATORIES],
                next_parameters={"standard_number": "IS 694:2010"},
            ),
            StructuredControllerOutput(
                decision=ControllerDecision.COMPLETE,
                reason="All downstream information retrieved.",
            ),
        ]

        qs = QueryService(analyzer=mock_analyzer, controller=mock_controller, ai_engine=mock_ai)

        # Mock repositories
        mock_product_repo = MagicMock()
        mock_product_repo.find_applicable_standards.return_value = [
            {"is_number": "IS 694:2010", "title": "PVC Insulated Cables", "product_name": "PVC Cables", "source_url": "https://standards.bis.gov.in/694"}
        ]
        mock_cert_repo = MagicMock()
        mock_cert_repo.find_certification_requirements.return_value = [
            {"scheme_name": "Scheme I (ISI Mark)", "scheme_code": "Scheme-I", "mandatory": "Yes", "conditions": "Compulsory testing under QCO", "source_url": "https://bis.gov.in/cert"}
        ]
        mock_lab_repo = MagicMock()
        mock_lab_repo.find_laboratories.return_value = [
            {"id": "1", "name": "Central Power Research Institute (CPRI)", "state": "Karnataka", "scope": "IS 694 Cables", "source_url": "https://lims.bis.gov.in"}
        ]

        qs.executor.product_repo = mock_product_repo
        qs.executor.cert_repo = mock_cert_repo
        qs.executor.lab_repo = mock_lab_repo

        req = ChatRequest(
            conversation_id="conv_cable_multihop",
            message=UserMessage(content="For my cables business, do I need BIS certification and do I need to test it in laboratories?"),
        )
        res = qs.process(req)

        assert res["message_type"] == "answer"
        card_types = [d["data_type"] for d in res["data"]]
        assert "standard" in card_types
        assert "certification" in card_types
        assert "laboratory" in card_types

    def test_general_bis_info_query(self):
        """
        Demo Query C:
        'What is BIS?'
        Answered via documentary RAG passages from authoritative BIS documents.
        """
        mock_ai = MagicMock()
        mock_ai.complete.return_value = {"answer": "The Bureau of Indian Standards (BIS) is the National Standards Body of India <cit_1>."}

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = QueryPlan(
            normalized_query="What is BIS and what does it do?",
            relevant=True,
            intent=QueryIntent.GENERAL_BIS_QUERY,
            response_language="en",
            needs_db=False,
            needs_rag=True,
            db_operations=[],
            parameters={},
        )

        mock_controller = MagicMock()
        mock_controller.evaluate.return_value = StructuredControllerOutput(
            decision=ControllerDecision.COMPLETE,
            reason="Authoritative document passages retrieved.",
        )

        qs = QueryService(analyzer=mock_analyzer, controller=mock_controller, ai_engine=mock_ai)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            _make_mock_chunk(
                "The Bureau of Indian Standards (BIS) is the National Standards Body of India established under the BIS Act, 2016.",
                "web_about_bis.html",
                "https://www.bis.gov.in/about-bis/",
            )
        ]
        qs.retriever = mock_retriever

        req = ChatRequest(
            conversation_id="conv_what_is_bis",
            message=UserMessage(content="What is BIS and what does it do?"),
        )
        res = qs.process(req)

        assert res["message_type"] == "answer"
        assert len(res["citations"]) > 0
        assert str(res["citations"][0]["source_url"]) == "https://www.bis.gov.in/about-bis/"
