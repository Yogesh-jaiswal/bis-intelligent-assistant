"""
tests/features/test_semantic_preservation.py
============================================

Comprehensive regression tests for:
1. Query Analyzer Semantic Preservation:
   - Applicability queries ("Which BIS standard applies to my product?")
   - Mandatory certification queries ("Is BIS certification mandatory for my product?")
   - Existing standard necessity queries ("Is IS 1234 required for my product?")
   - Follow-up with reference resolution ("Is this standard mandatory?")
   - Conditional queries ("What if my product operates below the rated limit?")
   - Multilingual Hindi queries ("क्या इस उत्पाद के लिए BIS प्रमाणन अनिवार्य है?")
   - Hinglish queries ("Mere product ke liye BIS certification mandatory hai kya?")
   - Multi-intent queries ("Which standard applies to my product and is BIS certification mandatory?")
   - Negation preservation ("Is BIS certification not required for this product?")
   - Generic follow-up ("Do I need it?")
2. Dictionary-Driven Hindi Synthesis Fallback:
   - Verifies fallback produces Hindi answers and summaries when synthesis fails for Hindi queries.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from flask import Flask

from app.extensions import db
from models.enums import DocumentBlockType, DocumentTypes
from services.file_processors.document.doc_representation import DocumentBlock
from services.query.query_service import QueryService
from services.query_analyser.analyser import QueryAnalyzer
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.retrieval.citation_builder import CitationBuilder
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import ChatRequest, UserMessage


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


def _make_chunk(text: str, filename: str, url: str) -> RetrievedChunk:
    return RetrievedChunk(
        score=0.92,
        chunk=DocumentBlock(type=DocumentBlockType.PARAGRAPH, text=text, metadata={"page": 1}),
        filename=filename,
        author=None,
        source_type=DocumentTypes.PDF,
        source_url=url,
    )


class TestSemanticPreservation:
    """Tests that the Query Analyzer strictly preserves question semantics across queries."""

    def test_applicability_intent(self):
        """'Which BIS standard applies to my product?' -> PRODUCT_STANDARD_RECOMMENDATION + FIND_APPLICABLE_STANDARDS."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Which BIS standard applies to my product?")

        assert plan.intent == QueryIntent.PRODUCT_STANDARD_RECOMMENDATION
        assert DatabaseOperation.FIND_APPLICABLE_STANDARDS in plan.db_operations
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in plan.db_operations
        assert "which" in plan.normalized_query.lower()

    def test_mandatory_certification_intent(self):
        """'Is BIS certification mandatory for my product?' -> CERTIFICATION_REQUIREMENT + GET_CERTIFICATION_REQUIREMENT."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Is BIS certification mandatory for my product?")

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert "mandatory" in plan.normalized_query.lower() or "required" in plan.normalized_query.lower()

    def test_existing_standard_necessity_does_not_become_recommendation(self):
        """'Is IS 1234 required for my product?' -> MUST NOT become PRODUCT_STANDARD_RECOMMENDATION."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Is IS 1234 required for my product?")

        # Must be classified as certification requirement or necessity, NOT product standard recommendation
        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert plan.intent != QueryIntent.PRODUCT_STANDARD_RECOMMENDATION
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert DatabaseOperation.FIND_APPLICABLE_STANDARDS not in plan.db_operations
        assert plan.parameters.get("standard_number") == "IS 1234"
        assert "is 1234" in plan.normalized_query.lower()

    def test_followup_reference_resolution_preserves_question(self):
        """'Is this standard mandatory?' with context -> resolves 'this standard' while preserving mandatory question."""
        analyzer = QueryAnalyzer()
        summary = "Topic: Solar Inverter. Standard: IS 16221. Language: en."
        plan = analyzer._heuristic_fallback("Is this standard mandatory?", conversation_summary=summary)

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert plan.parameters.get("standard_number") == "IS 16221"
        # The question must remain asking if it's mandatory, NOT 'which standard applies'
        assert "mandatory" in plan.normalized_query.lower() or "necessary" in plan.normalized_query.lower()
        assert "which bis standard applies" not in plan.normalized_query.lower()

    def test_conditional_query_preserves_condition(self):
        """'Then what if my product operates below 1110V? Is it necessary to take this standard?' -> preserves condition."""
        analyzer = QueryAnalyzer()
        summary = "Topic: Electrical Equipment. Standard: IS 694. Language: en."
        query = "Then what if my product operates below 1110V? Is it necessary to take this standard?"
        plan = analyzer._heuristic_fallback(query, conversation_summary=summary)

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert plan.parameters.get("standard_number") == "IS 694"
        assert "voltage" in plan.parameters or "1110" in plan.normalized_query
        assert "necessary" in plan.normalized_query.lower() or "is 694" in plan.normalized_query.lower()
        # Must NOT be converted into a generic applicability search
        assert DatabaseOperation.FIND_APPLICABLE_STANDARDS not in plan.db_operations

    def test_hindi_certification_intent(self):
        """'क्या इस उत्पाद के लिए BIS प्रमाणन अनिवार्य है?' -> normalizes to English question preserving mandatory intent."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("क्या इस उत्पाद के लिए BIS प्रमाणन अनिवार्य है?")

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert plan.response_language == "hi"
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert "mandatory" in plan.normalized_query.lower() or "certification" in plan.normalized_query.lower()

    def test_hinglish_certification_intent(self):
        """'Mere product ke liye BIS certification mandatory hai kya?' -> preserves certification question."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Mere product ke liye BIS certification mandatory hai kya?")

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert plan.response_language == "hi"
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert "mandatory" in plan.normalized_query.lower()

    def test_multi_intent_query_preserves_both_requests(self):
        """'Which standard applies to my product and is BIS certification mandatory?' -> preserves both requests."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Which standard applies to my product and is BIS certification mandatory?")

        # Primary intent: applicability
        assert plan.intent == QueryIntent.PRODUCT_STANDARD_RECOMMENDATION
        # Secondary intent: certification
        assert QueryIntent.CERTIFICATION_REQUIREMENT in plan.secondary_intents
        # Both operations queued
        assert DatabaseOperation.FIND_APPLICABLE_STANDARDS in plan.db_operations
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations

    def test_negation_preservation(self):
        """'Is BIS certification not required for this product?' -> negation preserved in normalized_query."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Is BIS certification not required for this product?")

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert "not" in plan.normalized_query.lower()

    def test_generic_followup_preserves_action(self):
        """'Do I need it?' with context -> resolves reference without changing actual question."""
        analyzer = QueryAnalyzer()
        summary = "Topic: Safety Helmet. Standard: IS 2925. Language: en."
        plan = analyzer._heuristic_fallback("Do I need it?", conversation_summary=summary)

        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert plan.parameters.get("standard_number") == "IS 2925"
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert "is 2925" in plan.normalized_query.lower() or "need" in plan.normalized_query.lower() or "mandatory" in plan.normalized_query.lower()

    def test_llm_query_analyzer_prompt_and_schema_with_secondary_intents(self):
        """Verify prompt contains the 6-step hierarchy, and QueryPlan parses secondary_intents from model."""
        from services.query_analyser.analyser_prompt import build_analyser_prompt
        prompt = build_analyser_prompt(
            "Then what if my wires are less than 1110V? Is it necessary to take this standard?",
            conversation_summary="Topic: Cables. Standard: IS 694:2010. Language: en."
        )
        assert "6-STEP MANDATORY ANALYSIS HIERARCHY" in prompt
        assert "Step 1 — Understand what the user is asking" in prompt
        assert "CONTRACT FOR `normalized_query`" in prompt
        assert "Do NOT turn this into FIND_APPLICABLE_STANDARDS" in prompt

        # Test model output validation with secondary_intents
        mock_model_output = {
            "normalized_query": "Then what if my wires are less than 1110 V? Is it necessary to take IS 694:2010?",
            "relevant": True,
            "intent": "CERTIFICATION_REQUIREMENT",
            "secondary_intents": ["TECHNICAL_QUESTION"],
            "response_language": "en",
            "needs_db": True,
            "needs_rag": True,
            "db_operations": ["GET_CERTIFICATION_REQUIREMENT"],
            "parameters": {"standard_number": "IS 694:2010", "voltage": "less than 1110V"},
            "missing_information": [],
        }
        plan = QueryPlan.model_validate(mock_model_output)
        assert plan.intent == QueryIntent.CERTIFICATION_REQUIREMENT
        assert QueryIntent.TECHNICAL_QUESTION in plan.secondary_intents
        assert plan.db_operations == [DatabaseOperation.GET_CERTIFICATION_REQUIREMENT]
        assert plan.parameters["voltage"] == "less than 1110V"


class TestHindiSynthesisFallback:
    """Tests that synthesis fallback correctly returns localized Hindi responses when synthesis LLM is unavailable."""

    def test_hindi_fallback_when_synthesis_fails(self, app):
        """When synthesis LLM fails on a Hindi query, fallback returns Hindi responses using dictionary templates."""
        with app.app_context():
            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"

            # Analyzer succeeds, Synthesis fails
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="Which BIS standard applies to PVC cables",
                    relevant=True,
                    intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
                    response_language="hi",
                    needs_db=True,
                    needs_rag=False,
                    db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
                    parameters={"product": "PVC cable"},
                ),
                Exception("Synthesis LLM Timeout"),
            ]

            qs = QueryService(ai_engine=mock_ai)
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            qs.retriever = mock_retriever

            # Mock executor to return standard
            mock_product_repo = MagicMock()
            mock_product_repo.find_applicable_standards.return_value = [
                {"is_number": "IS 694:2010", "title": "PVC Insulated Cables"}
            ]
            qs.executor.product_repo = mock_product_repo

            req = ChatRequest(
                conversation_id="conv_hindi_fallback_test",
                message=UserMessage(content="मेरी पीवीसी केबल के लिए कौन-सा मानक लागू होता है?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            # Must contain Hindi fallback template wording
            assert "लागू भारतीय मानक" in resp["message"]
            assert "IS 694:2010" in resp["message"]

    def test_hindi_fallback_when_no_records_found(self, app):
        """When no records found on Hindi query, fallback returns localized no_records_found message."""
        with app.app_context():
            mock_ai = MagicMock()
            mock_ai.provider = "OLLAMA"
            mock_ai.complete.side_effect = [
                QueryPlan(
                    normalized_query="Unknown exotic product",
                    relevant=True,
                    intent=QueryIntent.PRODUCT_STANDARD_RECOMMENDATION,
                    response_language="hi",
                    needs_db=True,
                    needs_rag=False,
                    db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS],
                    parameters={"product": "Unknown product"},
                ),
                Exception("Synthesis LLM unavailable"),
            ]

            qs = QueryService(ai_engine=mock_ai)
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            qs.retriever = mock_retriever
            mock_product_repo = MagicMock()
            mock_product_repo.find_applicable_standards.return_value = []
            qs.executor.product_repo = mock_product_repo

            req = ChatRequest(
                conversation_id="conv_hindi_no_records",
                message=UserMessage(content="अज्ञात वस्तु के लिए मानक क्या है?"),
            )
            resp = qs.process(req)

            assert resp["message_type"] == "answer"
            assert "BIS अभिलेखों के अनुसार" in resp["message"]
