"""
tests/conftest.py
=================

Shared root pytest configuration and fixtures for BIS Assistant tests.
"""

from unittest.mock import MagicMock
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.factory import create_app
from tests.fakes.fake_services import create_mock_repositories


@pytest.fixture(scope="session")
def app() -> Flask:
    """Session-scoped application configured for testing."""
    test_app = create_app(testing=True)
    return test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Test client for issuing requests against the application."""
    return app.test_client()


@pytest.fixture
def shared_mock_repos() -> dict[str, MagicMock]:
    """Shared mocked repository dictionary."""
    return create_mock_repositories()
