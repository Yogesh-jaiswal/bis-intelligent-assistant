import logging
from typing import Any
from sqlalchemy import select, or_

from app.extensions import db
from models.service import Service
from exceptions import DatabaseError
from repositories.semantic_search import rank_entities_semantically

logger = logging.getLogger(__name__)


def _service_to_dict(service: Service) -> dict[str, Any]:
    """Helper to serialize a Service ORM model."""
    return {
        "id": str(service.id),
        "name": service.name,
        "service_type": service.service_type,
        "description": service.description,
        "eligibility": service.eligibility,
        "documents_required": service.documents_required,
        "source_url": service.source_url,
    }


class ServiceRepository:
    """Repository for querying BIS Services, Portal Operations, and Licensing Services."""

    @staticmethod
    def get_bis_service(
        name: str | None = None,
        service_type: str | None = None,
        keyword: str | None = None,
        limit: int = 10,
        enable_semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search for BIS services by name, type, or keyword with semantic fallback.

        :param name: Service title or keyword.
        :param service_type: Type of service (e.g., 'Product Certification', 'Foreign Certification', 'Licence Amendment').
        :param keyword: General search term across name and description.
        :param limit: Maximum records to return.
        :param enable_semantic: Enable semantic ranking when keyword search is performed.
        :return: List of serialized service dictionaries.
        """
        try:
            stmt = select(Service)
            conditions = []

            if name and name.strip():
                conditions.append(Service.name.ilike(f"%{name.strip()}%"))

            if service_type and service_type.strip():
                conditions.append(Service.service_type.ilike(f"%{service_type.strip()}%"))

            if keyword and keyword.strip():
                term = f"%{keyword.strip()}%"
                conditions.append(
                    or_(
                        Service.name.ilike(term),
                        Service.description.ilike(term),
                        Service.eligibility.ilike(term),
                    )
                )

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(Service.id.asc()).limit(safe_limit)

            services = list(db.session.scalars(stmt).all())
            seen_ids = {s.id for s in services}

            search_query = keyword or name
            if enable_semantic and search_query and len(services) < safe_limit:
                all_services = list(db.session.scalars(select(Service)).all())
                ranked = rank_entities_semantically(
                    query=search_query,
                    entities=all_services,
                    text_extractor=lambda s: f"{s.name} {s.service_type} {s.description or ''} {s.eligibility or ''}",
                    limit=safe_limit - len(services),
                )
                for _score, srv in ranked:
                    if srv.id not in seen_ids:
                        services.append(srv)
                        seen_ids.add(srv.id)

            return [_service_to_dict(s) for s in services]

        except Exception as e:
            logger.exception("Failed to query BIS services from database")
            raise DatabaseError("Failed to query BIS services from database") from e



# Functional aliases for direct import
get_bis_service = ServiceRepository.get_bis_service
