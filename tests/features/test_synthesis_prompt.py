"""
tests/features/test_synthesis_prompt.py
=======================================

Unit tests for build_synthesis_prompt and SynthesisResponse schema in services/query/synthesis_prompt.py.
"""

from services.query_analyser.analyser_schema import QueryIntent, QueryPlan
from services.query.synthesis_prompt import (
    SYNTHESIS_FALLBACK_TEMPLATES,
    SynthesisResponse,
    build_synthesis_prompt,
)
from services.retrieval.retrieval_dataclasses import RetrievedChunk
from validators.chat_responses import Citation


def test_build_synthesis_prompt_structure():
    """Verify build_synthesis_prompt properly formats context, evidence, and rules."""
    plan = QueryPlan(
        normalized_query="What services does BIS provide?",
        relevant=True,
        intent=QueryIntent.BIS_SERVICE_LOOKUP,
        response_language="en",
        needs_db=True,
        needs_rag=False,
    )

    db_results = {
        "GET_BIS_SERVICE": [{"name": "Product Certification", "service_type": "Licensing"}]
    }

    citations = [
        Citation(
            id="cit_1",
            source_type="service",
            title="BIS Product Certification",
            source_url="https://bis.gov.in/service1",
        )
    ]

    prompt = build_synthesis_prompt(
        user_query="What are services provided BIS?",
        plan=plan,
        accumulated_db_results=db_results,
        rag_chunks=[],
        citations=citations,
        conversation_summary="User previously asked about BIS general overview.",
    )

    assert "User Query: \"What are services provided BIS?\"" in prompt
    assert "Normalized English Query: \"What services does BIS provide?\"" in prompt
    assert "Classified User Intent: \"BIS_SERVICE_LOOKUP\"" in prompt
    assert "PREVIOUS CONVERSATION CONTEXT" in prompt
    assert "User previously asked about BIS general overview." in prompt
    assert "Product Certification" in prompt
    assert "<cit_1>" in prompt
    assert "INTENT-COMPATIBLE EVIDENCE GROUNDING" in prompt


def test_synthesis_response_schema():
    """Verify SynthesisResponse validation."""
    resp = SynthesisResponse(
        answer="BIS provides product certification services.",
        conversation_summary="Discussed BIS certification services.",
    )
    assert resp.answer.startswith("BIS provides")
    assert resp.conversation_summary is not None
