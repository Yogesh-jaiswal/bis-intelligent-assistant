"""
tests/fixtures/standard_fixtures.py
===================================

Reusable fixtures and mock records for Indian Standards services and repositories.
"""

from __future__ import annotations

from typing import Any
import pytest


SAMPLE_STANDARD_RECORDS: list[dict[str, Any]] = [
    {
        "id": "std_001",
        "is_number": "IS 694:2010",
        "title": "Polyvinyl Chloride Insulated Unsheathed and Sheathed Cables/Cords",
        "status": "Active",
        "source_url": "https://standards.bis.gov.in/is694",
    },
    {
        "id": "std_002",
        "is_number": "IS 14543:2024",
        "title": "Packaged Drinking Water (Other than Packaged Natural Mineral Water) — Specification",
        "status": "Active",
        "source_url": "https://standards.bis.gov.in/is14543",
    },
]


@pytest.fixture
def sample_standard_records() -> list[dict[str, Any]]:
    """Return a fresh copy of sample standard records."""
    return [dict(rec) for rec in SAMPLE_STANDARD_RECORDS]
