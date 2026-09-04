"""
tests/fakes/fake_services.py
============================

Shared mock and fake service implementations for testing without external services.
"""

from unittest.mock import MagicMock
from services.ai.engine import AIEngine
from services.ai.models.fake_provider import FakeProvider


def create_fake_ai_engine() -> AIEngine:
    """Create an AIEngine using the FakeAIProvider."""
    engine = AIEngine("FAKE")
    return engine


def create_mock_repositories() -> dict[str, MagicMock]:
    """Create standard dictionary of mock repositories."""
    return {
        "standard_repo": MagicMock(),
        "product_repo": MagicMock(),
        "cert_repo": MagicMock(),
        "lab_repo": MagicMock(),
        "service_repo": MagicMock(),
        "upload_repo": MagicMock(),
    }
