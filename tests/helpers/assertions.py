"""
tests/helpers/assertions.py
===========================

Reusable assertion utilities for chat response envelopes and data cards.
"""

from typing import Any


def assert_success_envelope(response_data: dict[str, Any]) -> dict[str, Any]:
    """Assert standard API success envelope format and return inner data."""
    assert isinstance(response_data, dict), f"Response is not a dict: {type(response_data)}"
    assert response_data.get("success") is True, f"Expected success=True, got {response_data.get('success')}"
    assert response_data.get("error") is None, f"Expected error=None, got {response_data.get('error')}"
    inner = response_data.get("data")
    assert isinstance(inner, dict), f"Expected inner data dict, got {type(inner)}"
    return inner


def assert_valid_chat_response(data: dict[str, Any]) -> None:
    """Assert valid ChatResponse payload shape."""
    assert "conversation_id" in data, "Missing conversation_id in ChatResponse"
    assert "message_type" in data, "Missing message_type in ChatResponse"
    assert data["message_type"] in ("answer", "clarification"), f"Invalid message_type: {data['message_type']}"
    assert "message" in data and isinstance(data["message"], str) and data["message"].strip(), "Empty message in ChatResponse"
    assert "citations" in data and isinstance(data["citations"], list), "Missing or invalid citations list"
    assert "data" in data and isinstance(data["data"], list), "Missing or invalid data cards list"
