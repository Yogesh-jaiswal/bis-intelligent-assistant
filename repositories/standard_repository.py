import logging
from typing import Any
from sqlalchemy import select, or_

from app.extensions import db
from models.standard import Standard
from models.standard_version import StandardVersion
from models.standard_amendment import StandardAmendment
from exceptions import DatabaseError

logger = logging.getLogger(__name__)


def _standard_to_dict(standard: Standard) -> dict[str, Any]:
    """Helper to serialize a Standard ORM model into a standard JSON dictionary."""
    return {
        "id": str(standard.id),
        "is_number": standard.is_number,
        "title": standard.title,
        "revision_number": standard.revision_no,
        "publication_year": standard.publication_year,
        "status": standard.status,
        "technical_department": standard.technical_department,
        "source_url": standard.source_url,
        "document_url": standard.document_url,
        "last_verified_at": standard.last_verified_at.isoformat() if standard.last_verified_at else None,
    }


class StandardRepository:
    """Repository for querying Indian Standards and their associated metadata."""

    @staticmethod
    def find_standard(
        standard_number: str | None = None,
        title: str | None = None,
        status: str | None = None,
        technical_department: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for Indian Standards matching the given criteria.

        :param standard_number: IS number or substring (e.g., 'IS 694', '694', 'IS 694:2010').
        :param title: Keyword or phrase in standard title.
        :param status: Standard status (e.g., 'Active', 'Withdrawn').
        :param technical_department: Department name or code (e.g., 'ETD').
        :param limit: Maximum number of records to return (capped at 50).
        :return: List of serialized standard dictionaries.
        """
        try:
            stmt = select(Standard)

            conditions = []
            if standard_number and standard_number.strip():
                clean_num = standard_number.strip()
                conditions.append(Standard.is_number.ilike(f"%{clean_num}%"))

            if title and title.strip():
                conditions.append(Standard.title.ilike(f"%{title.strip()}%"))

            if status and status.strip():
                conditions.append(Standard.status.ilike(f"%{status.strip()}%"))

            if technical_department and technical_department.strip():
                conditions.append(Standard.technical_department.ilike(f"%{technical_department.strip()}%"))

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(Standard.publication_year.desc().nullslast(), Standard.id.asc()).limit(safe_limit)

            standards = db.session.scalars(stmt).all()
            return [_standard_to_dict(s) for s in standards]

        except Exception as e:
            logger.exception("Failed to query standards from database")
            raise DatabaseError("Failed to query standards from database") from e

    @staticmethod
    def get_standard_by_is_number(is_number: str) -> dict[str, Any] | None:
        """
        Find an exact standard by its IS number.

        :param is_number: Formal IS number.
        :return: Serialized standard dictionary or None.
        """
        try:
            stmt = select(Standard).where(Standard.is_number.ilike(is_number.strip()))
            standard = db.session.scalar(stmt)
            return _standard_to_dict(standard) if standard else None
        except Exception as e:
            logger.exception(f"Failed to fetch standard by is_number '{is_number}'")
            raise DatabaseError(f"Failed to fetch standard: {is_number}") from e


# Functional aliases for direct import
find_standard = StandardRepository.find_standard
get_standard_by_is_number = StandardRepository.get_standard_by_is_number
