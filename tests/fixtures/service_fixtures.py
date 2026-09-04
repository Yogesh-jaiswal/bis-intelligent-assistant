"""
tests/fixtures/service_fixtures.py
==================================

Reusable fixtures and mock records for BIS Conformity & Certification services.
"""

from __future__ import annotations

from typing import Any
import pytest


SAMPLE_SERVICE_RECORDS: list[dict[str, Any]] = [
    {
        "id": "srv_001",
        "service_name": "Grant of BIS Product Certification Licence (ISI Mark)",
        "service_type": "Product Certification",
        "description": "Licence granting use of standard mark for manufactured products.",
        "source_url": "https://bis.gov.in/service/isi-mark",
    },
    {
        "id": "srv_002",
        "service_name": "Compulsory Registration Scheme (CRS)",
        "service_type": "Registration",
        "description": "Self-declaration of conformity for electronics and IT goods.",
        "source_url": "https://bis.gov.in/service/crs",
    },
]


@pytest.fixture
def sample_service_records() -> list[dict[str, Any]]:
    """Return a fresh copy of sample BIS service records."""
    return [dict(rec) for rec in SAMPLE_SERVICE_RECORDS]
