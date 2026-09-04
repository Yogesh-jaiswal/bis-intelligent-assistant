"""
tests/fixtures/chat_fixtures.py
===============================

Reusable fixtures, sample queries, and factory helpers for chat and query services.
"""

from __future__ import annotations

import pytest
from validators.chat_responses import ChatRequest, UserMessage


def make_chat_request(
    content: str,
    conversation_id: str | None = None,
    language: str | None = None,
) -> ChatRequest:
    """Helper to construct a valid ChatRequest model."""
    return ChatRequest(
        conversation_id=conversation_id,
        message=UserMessage(content=content, language=language),
    )


SAMPLE_CABLE_QUERY = "Which BIS standard applies to my PVC cable?"
SAMPLE_SERVICE_QUERY = "What services does BIS provide?"
SAMPLE_WATER_QUERY = "What is the Indian Standard for packaged drinking water?"
SAMPLE_HINDI_QUERY = "आईएसआई मार्क प्राप्त करने की क्या प्रक्रिया है?"


@pytest.fixture
def sample_cable_chat_request() -> ChatRequest:
    """Sample chat request for PVC cable standard query."""
    return make_chat_request(SAMPLE_CABLE_QUERY)


@pytest.fixture
def sample_service_chat_request() -> ChatRequest:
    """Sample chat request for BIS services query."""
    return make_chat_request(SAMPLE_SERVICE_QUERY)
