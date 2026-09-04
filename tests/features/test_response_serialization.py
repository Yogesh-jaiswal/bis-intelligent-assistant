"""
tests/features/test_response_serialization.py
=============================================

Regression tests for response serialization boundary:
  - Verifies that Pydantic models containing HttpUrl (like Citation) serialize cleanly
    into JSON-compatible dictionaries (mode="json") without triggering Flask jsonify/json.dumps 500 errors.
  - Verifies that POST /v1/query returns HTTP 200 with valid JSON envelope when fallback responses are produced.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import HttpUrl

from app.factory import create_app
from services.query_analyser.analyser_schema import DatabaseOperation, QueryIntent, QueryPlan
from validators.chat_responses import ChatResponse, Citation, StandardCard


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestResponseSerialization:
    def test_chat_response_model_dump_json_mode_converts_httpurl_to_str(self):
        """Direct test that model_dump(mode='json') serializes HttpUrl to string."""
        cit = Citation(
            id="cit_1",
            source_type="document",
            title="IS 694:2010 PVC Cables",
            reference="IS 694:2010, page 1",
            source_url=HttpUrl("https://standards.bis.gov.in/website/know-your-standards"),
        )
        card = StandardCard(
            data_type="standard",
            id="std_1",
            is_number="IS 694:2010",
            title="PVC Insulated Cables",
            status="Active",
            source_url="https://standards.bis.gov.in/694",
        )
        resp = ChatResponse(
            message_type="answer",
            conversation_id="conv_test_serialization",
            message="Applicable standard is IS 694:2010 <cit_1>.",
            citations=[cit],
            data=[card],
        )

        serialized = resp.model_dump(mode="json")

        # Must be native string, not HttpUrl object
        assert isinstance(serialized["citations"][0]["source_url"], str)
        assert serialized["citations"][0]["source_url"] == "https://standards.bis.gov.in/website/know-your-standards"

        # Must serialize cleanly with json.dumps
        json_str = json.dumps(serialized)
        assert "https://standards.bis.gov.in/website/know-your-standards" in json_str

    def test_query_endpoint_returns_200_with_citations_and_cards(self, client):
        """Integration test for POST /v1/query returning 200 with citations and cards."""
        mock_result = {
            "message_type": "answer",
            "conversation_id": "conv_regression_1",
            "message": "Applicable standard is IS 694:2010 <cit_1>.",
            "citations": [
                {
                    "id": "cit_1",
                    "source_type": "document",
                    "title": "IS 694:2010 PVC Cables",
                    "reference=" : "IS 694:2010, page 1",
                    "source_url": "https://standards.bis.gov.in/website/know-your-standards",
                }
            ],
            "data": [
                {
                    "data_type": "standard",
                    "id": "std_1",
                    "is_number": "IS 694:2010",
                    "title": "PVC Insulated Cables",
                }
            ],
        }

        with patch("routes.v1.query.process_query", return_value=mock_result):
            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": "conv_regression_1",
                    "message": {"content": "Which standard applies to PVC cables?"},
                },
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert data["error"] is None
            assert data["data"]["message_type"] == "answer"
            assert len(data["data"]["citations"]) == 1
            assert data["data"]["citations"][0]["source_url"] == "https://standards.bis.gov.in/website/know-your-standards"

    def test_query_endpoint_with_ollama_fallback_returns_200(self, client):
        """Pipeline fallback (e.g. when LLM synthesis is unavailable) returns HTTP 200, not 500."""
        # Query for standard lookup
        response = client.post(
            "/v1/query",
            json={
                "conversation_id": "conv_fallback_test",
                "message": {"content": "What is IS 694?"},
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["error"] is None
        assert data["data"]["message_type"] in ("answer", "clarification")
        assert len(data["data"]["message"]) > 0
