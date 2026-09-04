"""
tests/features/test_backend_stabilization.py
============================================

Tests for the final backend stabilization requirements:
  A. Ollama empty response -> AIResponseError
  B. Ollama malformed JSON -> AIResponseError
  C. Ollama schema validation failure -> AISchemaValidationError
  D. Ollama connection failure -> AIConnectionError
  E. Query analyzer successful structured response -> correct QueryPlan
  F. Query analyzer fallback for standard-only query -> only FIND_APPLICABLE_STANDARDS
  G. Query analyzer fallback for combined query -> all corresponding operations
  H. Deterministic planner stops after standard retrieval for standard-only query
  I. Deterministic planner schedules certification when certification is requested
  J. Deterministic planner schedules testing when testing is requested
  K. Embedding model loaded once and reused
  L. Embedding device selects CUDA when available, else CPU
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, Field

from exceptions import (
    AIConnectionError,
    AIProviderError,
    AIResponseError,
    AISchemaValidationError,
)
from services.ai.engine import AIEngine
from services.ai.models.ollama_provider import OllamaProvider
from services.file_processors.embeddings.embeddings_generator import (
    EmbeddingGenerator,
    get_embedding_device,
    init_embedding_model,
)
from services.query_analyser.analyser import QueryAnalyzer
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from services.retrieval.deterministic_planner import DeterministicPlanner


class DummySchema(BaseModel):
    name: str
    count: int = Field(default=1)


class TestOllamaErrorHandling:
    def test_a_ollama_empty_response_raises_ai_response_error(self):
        """Ollama HTTP 200 with empty/blank response content must raise AIResponseError, NOT connection error."""
        provider = OllamaProvider()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "   "}}
        provider.client = mock_client

        with pytest.raises(AIResponseError) as exc_info:
            provider.generate("test prompt", DummySchema)

        assert "empty response content" in str(exc_info.value).lower()
        assert exc_info.value.provider == "OLLAMA"
        # Must NOT be an AIConnectionError
        assert not isinstance(exc_info.value, AIConnectionError)

    def test_b_ollama_malformed_json_raises_ai_response_error(self):
        """Ollama returning invalid/truncated JSON must raise AIResponseError with JSON decode context."""
        provider = OllamaProvider()
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Invalid JSON: EOF while parsing a value"}}
        provider.client = mock_client

        with pytest.raises(AIResponseError) as exc_info:
            provider.generate("test prompt", DummySchema)

        assert "malformed json" in str(exc_info.value).lower()
        assert not isinstance(exc_info.value, AIConnectionError)

    def test_c_ollama_schema_validation_failure_raises_ai_schema_validation_error(self):
        """Ollama returning valid JSON that fails Pydantic schema validation raises AISchemaValidationError."""
        provider = OllamaProvider()
        mock_client = MagicMock()
        # Missing required field 'name'
        mock_client.chat.return_value = {"message": {"content": json.dumps({"wrong_field": 123})}}
        provider.client = mock_client

        with pytest.raises(AISchemaValidationError) as exc_info:
            provider.generate("test prompt", DummySchema)

        assert "schema" in str(exc_info.value).lower()
        assert isinstance(exc_info.value, AIProviderError)
        assert not isinstance(exc_info.value, AIConnectionError)

    def test_d_ollama_connection_failure_raises_ai_connection_error(self):
        """Ollama transport/network/timeout failure raises AIConnectionError."""
        from ollama import RequestError

        provider = OllamaProvider()
        mock_client = MagicMock()
        mock_client.chat.side_effect = RequestError("Connection refused to http://localhost:11434")
        provider.client = mock_client

        with pytest.raises(AIConnectionError) as exc_info:
            provider.generate("test prompt", DummySchema)

        assert "connection" in str(exc_info.value).lower()
        assert exc_info.value.provider == "OLLAMA"


class TestQueryAnalyzerFallback:
    def test_e_query_analyzer_successful_llm_response(self):
        """QueryAnalyzer returns validated QueryPlan on successful LLM completion."""
        mock_engine = MagicMock(spec=AIEngine)
        mock_engine.provider = "OLLAMA"
        mock_engine.complete.return_value = {
            "normalized_query": "Which BIS standard applies to PVC cable?",
            "relevant": True,
            "intent": "product_standard_recommendation",
            "response_language": "en",
            "needs_db": True,
            "needs_rag": False,
            "db_operations": ["FIND_APPLICABLE_STANDARDS"],
            "parameters": {"product": "PVC cable"},
        }

        analyzer = QueryAnalyzer(ai_engine=mock_engine)
        plan = analyzer.analyze("Which BIS standard applies to my PVC cable?")

        assert plan.relevant is True
        assert plan.intent == QueryIntent.PRODUCT_STANDARD_RECOMMENDATION
        assert plan.db_operations == [DatabaseOperation.FIND_APPLICABLE_STANDARDS]
        assert plan.parameters.get("product") == "PVC cable"

    def test_f_query_analyzer_fallback_for_standard_only_query(self):
        """Fallback for 'Which BIS standard applies to my PVC cable?' produces only FIND_APPLICABLE_STANDARDS."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Which BIS standard applies to my PVC cable?")

        assert plan.relevant is True
        assert plan.intent == QueryIntent.PRODUCT_STANDARD_RECOMMENDATION
        # MUST ONLY contain FIND_APPLICABLE_STANDARDS, NOT certification or laboratories
        assert plan.db_operations == [DatabaseOperation.FIND_APPLICABLE_STANDARDS]
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in plan.db_operations
        assert DatabaseOperation.FIND_LABORATORIES not in plan.db_operations
        assert "pvc cable" in plan.parameters.get("product", "").lower()
        # Normalized query must NOT contain artificial 'certification, and testing'
        assert "certification" not in plan.normalized_query.lower()
        assert "testing" not in plan.normalized_query.lower()

    def test_query_analyzer_fallback_for_certification_only_query(self):
        """Fallback for 'What certification is required for PVC cable?' includes certification requirement."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("What certification is required for PVC cable?")

        assert plan.relevant is True
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert DatabaseOperation.FIND_LABORATORIES not in plan.db_operations

    def test_query_analyzer_fallback_for_laboratory_only_query(self):
        """Fallback for 'Where can I get this cable tested?' includes laboratory lookup."""
        analyzer = QueryAnalyzer()
        plan = analyzer._heuristic_fallback("Where can I get this cable tested?")

        assert plan.relevant is True
        assert plan.intent == QueryIntent.LABORATORY_LOOKUP
        assert DatabaseOperation.FIND_LABORATORIES in plan.db_operations
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in plan.db_operations

    def test_g_query_analyzer_fallback_for_combined_query(self):
        """Fallback for explicit combined query includes standard, certification, and lab operations."""
        analyzer = QueryAnalyzer()
        query = "For my cables business, do I need BIS certification and do I need to test it in laboratories?"
        plan = analyzer._heuristic_fallback(query)

        assert plan.relevant is True
        assert DatabaseOperation.FIND_APPLICABLE_STANDARDS in plan.db_operations
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in plan.db_operations
        assert DatabaseOperation.FIND_LABORATORIES in plan.db_operations


class TestDeterministicPlannerTargetedDownstream:
    def test_h_planner_stops_after_standard_retrieval_when_only_standard_requested(self):
        """
        When user asks 'Which BIS standard applies to my PVC cable?',
        after finding IS 694:2010 and retrieving document RAG chunks,
        the planner MUST stop (continue_retrieval=False) without queuing cert or labs.
        """
        initial_plan = QueryPlan(
            normalized_query="Which BIS standard applies to PVC cable",
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
        # Simulate having executed document RAG for IS 694
        from models.enums import DocumentBlockType, DocumentTypes
        from services.file_processors.document.doc_representation import DocumentBlock
        from services.retrieval.retrieval_dataclasses import RetrievedChunk

        rag_chunk = RetrievedChunk(
            score=0.95,
            chunk=DocumentBlock(type=DocumentBlockType.PARAGRAPH, text="IS 694 scope and specs", metadata={"page": 1}),
            filename="694_2010_reff2020.pdf",
            author=None,
            source_type=DocumentTypes.PDF,
            source_url="https://standards.bis.gov.in/694",
        )
        executed_sigs = {"FIND_APPLICABLE_STANDARDS:[('product', 'PVC cable')]"}

        next_plan = DeterministicPlanner.plan_next_hop(
            user_query="Which BIS standard applies to my PVC cable?",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=[rag_chunk],
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        # Must stop cleanly!
        assert next_plan.continue_retrieval is False
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT not in next_plan.db_operations
        assert DatabaseOperation.FIND_LABORATORIES not in next_plan.db_operations
        assert "sufficient" in next_plan.reason.lower()

    def test_i_planner_schedules_certification_when_certification_requested(self):
        """When certification is requested, planner derives standard and queues GET_CERTIFICATION_REQUIREMENT."""
        initial_plan = QueryPlan(
            normalized_query="What certification is required for PVC cable?",
            relevant=True,
            intent=QueryIntent.CERTIFICATION_REQUIREMENT,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS, DatabaseOperation.GET_CERTIFICATION_REQUIREMENT],
            parameters={"product": "PVC cable"},
        )
        accumulated_db = {
            "FIND_APPLICABLE_STANDARDS": [{"is_number": "IS 694:2010"}]
        }
        executed_sigs = {"FIND_APPLICABLE_STANDARDS:[('product', 'PVC cable')]"}

        next_plan = DeterministicPlanner.plan_next_hop(
            user_query="What certification is required for PVC cable?",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=[],
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        assert next_plan.continue_retrieval is True
        assert DatabaseOperation.GET_CERTIFICATION_REQUIREMENT in next_plan.db_operations
        assert next_plan.db_parameters.get("standard_number") == "IS 694:2010"

    def test_j_planner_schedules_testing_when_testing_requested(self):
        """When testing/laboratory is requested, planner derives standard and queues FIND_LABORATORIES."""
        initial_plan = QueryPlan(
            normalized_query="Where can I test PVC cable?",
            relevant=True,
            intent=QueryIntent.LABORATORY_LOOKUP,
            response_language="en",
            needs_db=True,
            needs_rag=False,
            db_operations=[DatabaseOperation.FIND_APPLICABLE_STANDARDS, DatabaseOperation.FIND_LABORATORIES],
            parameters={"product": "PVC cable"},
        )
        accumulated_db = {
            "FIND_APPLICABLE_STANDARDS": [{"is_number": "IS 694:2010"}]
        }
        executed_sigs = {"FIND_APPLICABLE_STANDARDS:[('product', 'PVC cable')]"}

        next_plan = DeterministicPlanner.plan_next_hop(
            user_query="Where can I test my cable?",
            initial_plan=initial_plan,
            accumulated_db_results=accumulated_db,
            accumulated_rag_chunks=[],
            hop_count=1,
            executed_signatures=executed_sigs,
        )

        assert next_plan.continue_retrieval is True
        assert DatabaseOperation.FIND_LABORATORIES in next_plan.db_operations
        assert next_plan.db_parameters.get("standard_number") == "IS 694:2010"


class TestEmbeddingModelLifecycleAndDevice:
    def test_k_embedding_model_singleton_reused(self):
        import services.file_processors.embeddings.embeddings_generator as eg_mod
        orig_model = eg_mod._MODEL
        try:
            with patch("services.file_processors.embeddings.embeddings_generator.SentenceTransformer") as mock_st_cls:
                mock_instance = MagicMock()
                mock_st_cls.return_value = mock_instance

                # First load
                m1 = init_embedding_model(force_reload=True)
                # Second call should reuse singleton without calling SentenceTransformer constructor again
                m2 = init_embedding_model(force_reload=False)

                assert m1 is m2
                assert mock_st_cls.call_count == 1
        finally:
            eg_mod._MODEL = orig_model

    def test_l_embedding_device_selection(self):
        """Embedding device selection selects CUDA if available, CPU otherwise."""
        mock_settings = MagicMock()
        mock_settings.EMBEDDINGS_DEVICE = None
        with patch("services.file_processors.embeddings.embeddings_generator.get_settings", return_value=mock_settings):
            with patch("torch.cuda.is_available", return_value=True):
                device = get_embedding_device()
                assert device == "cuda"

            with patch("torch.cuda.is_available", return_value=False):
                device = get_embedding_device()
                assert device == "cpu"
