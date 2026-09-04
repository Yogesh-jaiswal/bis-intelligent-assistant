"""
tests/fixtures/sample_payloads.py
=================================

Composite / legacy entry point re-exporting service-specific fixtures.
Prefer importing directly from specific service fixture modules:
- tests.fixtures.chat_fixtures
- tests.fixtures.standard_fixtures
- tests.fixtures.service_fixtures
"""

from tests.fixtures.chat_fixtures import (
    make_chat_request,
    SAMPLE_CABLE_QUERY,
    SAMPLE_SERVICE_QUERY,
    SAMPLE_WATER_QUERY,
    SAMPLE_HINDI_QUERY,
    sample_cable_chat_request,
    sample_service_chat_request,
)
from tests.fixtures.standard_fixtures import (
    SAMPLE_STANDARD_RECORDS,
    sample_standard_records,
)
from tests.fixtures.service_fixtures import (
    SAMPLE_SERVICE_RECORDS,
    sample_service_records,
)

__all__ = [
    "make_chat_request",
    "SAMPLE_CABLE_QUERY",
    "SAMPLE_SERVICE_QUERY",
    "SAMPLE_WATER_QUERY",
    "SAMPLE_HINDI_QUERY",
    "sample_cable_chat_request",
    "sample_service_chat_request",
    "SAMPLE_STANDARD_RECORDS",
    "sample_standard_records",
    "SAMPLE_SERVICE_RECORDS",
    "sample_service_records",
]
