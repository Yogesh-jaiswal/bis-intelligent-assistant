"""
tests/fixtures
==============

Service-organized test fixtures, sample queries, and mock payloads.

Modules:
- chat_fixtures: Query schemas, chat requests, and sample user questions
- standard_fixtures: Indian standard records and catalog mocks
- service_fixtures: BIS scheme and conformity assessment service records
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
