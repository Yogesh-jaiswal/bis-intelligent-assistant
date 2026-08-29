import logging
from typing import Any
from sqlalchemy import select, or_

from app.extensions import db
from models.certification_scheme import CertificationScheme
from models.standard_certification import StandardCertification
from models.standard import Standard
from exceptions import DatabaseError

logger = logging.getLogger(__name__)


def _scheme_to_dict(scheme: CertificationScheme) -> dict[str, Any]:
    """Helper to serialize a CertificationScheme ORM model."""
    return {
        "id": str(scheme.id),
        "name": scheme.name,
        "scheme_code": scheme.scheme_code,
        "description": scheme.description,
        "certification_type": scheme.certification_type,
        "mandatory": scheme.mandatory,
        "authority": scheme.authority,
        "source_url": scheme.source_url,
    }


class CertificationRepository:
    """Repository for querying Certification Schemes and Standard Certification Requirements."""

    @staticmethod
    def find_certification_scheme(
        scheme_code: str | None = None,
        name: str | None = None,
        certification_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for BIS certification schemes.

        :param scheme_code: Scheme identifier (e.g., 'Scheme-I', 'FMCS', 'CRS').
        :param name: Scheme title or keyword.
        :param certification_type: Type of certification.
        :param limit: Maximum records to return.
        :return: List of serialized scheme dictionaries.
        """
        try:
            stmt = select(CertificationScheme)
            conditions = []

            if scheme_code and scheme_code.strip():
                conditions.append(CertificationScheme.scheme_code.ilike(f"%{scheme_code.strip()}%"))

            if name and name.strip():
                conditions.append(CertificationScheme.name.ilike(f"%{name.strip()}%"))

            if certification_type and certification_type.strip():
                conditions.append(CertificationScheme.certification_type.ilike(f"%{certification_type.strip()}%"))

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(CertificationScheme.id.asc()).limit(safe_limit)

            schemes = db.session.scalars(stmt).all()
            return [_scheme_to_dict(s) for s in schemes]

        except Exception as e:
            logger.exception("Failed to query certification schemes from database")
            raise DatabaseError("Failed to query certification schemes from database") from e

    @staticmethod
    def find_certification_requirements(
        standard_number: str | None = None,
        scheme_code: str | None = None,
        mandatory: str | None = None,
        requirement_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve certification requirements and QCO conditions for a standard and/or scheme.

        :param standard_number: IS number to check (e.g. 'IS 694').
        :param scheme_code: Filter by specific scheme code.
        :param mandatory: Mandatory filter ('Yes', 'No').
        :param requirement_type: Requirement type ('Compulsory under QCO', 'Factory Inspection', etc.).
        :param limit: Maximum records to return.
        :return: List of requirement dictionaries.
        """
        try:
            stmt = (
                select(StandardCertification, Standard, CertificationScheme)
                .join(Standard, StandardCertification.standard_id == Standard.id)
                .join(CertificationScheme, StandardCertification.certification_scheme_id == CertificationScheme.id)
            )

            conditions = []

            if standard_number and standard_number.strip():
                conditions.append(Standard.is_number.ilike(f"%{standard_number.strip()}%"))

            if scheme_code and scheme_code.strip():
                conditions.append(CertificationScheme.scheme_code.ilike(f"%{scheme_code.strip()}%"))

            if mandatory and mandatory.strip():
                conditions.append(StandardCertification.mandatory.ilike(f"%{mandatory.strip()}%"))

            if requirement_type and requirement_type.strip():
                conditions.append(StandardCertification.requirement_type.ilike(f"%{requirement_type.strip()}%"))

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(StandardCertification.id.asc()).limit(safe_limit)

            rows = db.session.execute(stmt).all()

            results = []
            for cert_req, standard, scheme in rows:
                results.append({
                    "id": str(cert_req.id),
                    "standard_id": str(standard.id),
                    "is_number": standard.is_number,
                    "standard_title": standard.title,
                    "scheme_id": str(scheme.id),
                    "scheme_name": scheme.name,
                    "scheme_code": scheme.scheme_code,
                    "requirement_type": cert_req.requirement_type,
                    "mandatory": cert_req.mandatory,
                    "conditions": cert_req.conditions,
                    "authority": scheme.authority,
                    "source_url": cert_req.source_url or scheme.source_url,
                })

            return results

        except Exception as e:
            logger.exception("Failed to query certification requirements from database")
            raise DatabaseError("Failed to query certification requirements from database") from e


# Functional aliases for direct import
find_certification_scheme = CertificationRepository.find_certification_scheme
find_certification_requirements = CertificationRepository.find_certification_requirements
