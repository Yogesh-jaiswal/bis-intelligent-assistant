import logging
from typing import Any
from sqlalchemy import select, or_

from app.extensions import db
from models.laboratory import Laboratory
from exceptions import DatabaseError

logger = logging.getLogger(__name__)


def _lab_to_dict(lab: Laboratory) -> dict[str, Any]:
    """Helper to serialize a Laboratory ORM model."""
    return {
        "id": str(lab.id),
        "lab_code": lab.lab_code,
        "name": lab.name,
        "address": lab.address,
        "state": lab.state,
        "district": lab.district,
        "contact_person": lab.contact_person,
        "phone": lab.phone,
        "email": lab.email,
        "validity_date": lab.validity_date.isoformat() if lab.validity_date else None,
        "scope": lab.scope,
        "source_url": lab.source_url,
    }


class LaboratoryRepository:
    """Repository for searching and filtering BIS recognized laboratories."""

    @staticmethod
    def find_laboratories(
        state: str | None = None,
        district: str | None = None,
        scope_keyword: str | None = None,
        standard_number: str | None = None,
        product: str | None = None,
        lab_code: str | None = None,
        name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search for testing laboratories matching location, testing scope, or standard criteria.

        :param state: State name (e.g., 'Tamil Nadu', 'Haryana', 'Delhi').
        :param district: District name (e.g., 'Thiruvallur', 'Sonipat').
        :param scope_keyword: Keyword to match inside laboratory scope description.
        :param standard_number: Standard code to search inside lab scope.
        :param product: Product name/keyword to search inside lab scope.
        :param lab_code: BIS lab identification code.
        :param name: Lab organization name.
        :param limit: Maximum results to return (capped at 50 to prevent huge payloads).
        :return: List of serialized laboratory dictionaries.
        """
        try:
            stmt = select(Laboratory)
            conditions = []

            if state and state.strip():
                conditions.append(Laboratory.state.ilike(f"%{state.strip()}%"))

            if district and district.strip():
                conditions.append(Laboratory.district.ilike(f"%{district.strip()}%"))

            if lab_code and lab_code.strip():
                conditions.append(Laboratory.lab_code.ilike(f"%{lab_code.strip()}%"))

            if name and name.strip():
                conditions.append(Laboratory.name.ilike(f"%{name.strip()}%"))

            # Scope filtering (searches standard, product, or scope keywords inside testing scope)
            scope_terms = []
            if standard_number and standard_number.strip():
                scope_terms.append(f"%{standard_number.strip()}%")
            if product and product.strip():
                scope_terms.append(f"%{product.strip()}%")
            if scope_keyword and scope_keyword.strip():
                scope_terms.append(f"%{scope_keyword.strip()}%")

            if scope_terms:
                scope_conditions = [Laboratory.scope.ilike(term) for term in scope_terms]
                conditions.append(or_(*scope_conditions))

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(Laboratory.id.asc()).limit(safe_limit)

            labs = db.session.scalars(stmt).all()
            return [_lab_to_dict(l) for l in labs]

        except Exception as e:
            logger.exception("Failed to query laboratories from database")
            raise DatabaseError("Failed to query laboratories from database") from e


# Functional aliases for direct import
find_laboratories = LaboratoryRepository.find_laboratories
