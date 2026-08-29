import logging
from typing import Any
from sqlalchemy import select, or_

from app.extensions import db
from models.product import Product
from models.standard import Standard
from models.product_standard_mapping import ProductStandardMapping
from exceptions import DatabaseError

logger = logging.getLogger(__name__)


def _product_to_dict(product: Product) -> dict[str, Any]:
    """Helper to serialize a Product ORM model into a dictionary."""
    return {
        "id": str(product.id),
        "name": product.name,
        "category": product.category,
        "description": product.description,
        "keywords": product.keywords,
        "created_at": product.created_at.isoformat() if product.created_at else None,
    }


class ProductRepository:
    """Repository for querying Products and Product-to-Standard mappings."""

    @staticmethod
    def find_product(
        name: str | None = None,
        category: str | None = None,
        keyword: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for products by name, category, or keyword.

        :param name: Product name or partial match.
        :param category: Product category.
        :param keyword: Search term across name, keywords, and description.
        :param limit: Maximum records to return.
        :return: List of serialized product dictionaries.
        """
        try:
            stmt = select(Product)
            conditions = []

            if name and name.strip():
                conditions.append(Product.name.ilike(f"%{name.strip()}%"))

            if category and category.strip():
                conditions.append(Product.category.ilike(f"%{category.strip()}%"))

            if keyword and keyword.strip():
                term = f"%{keyword.strip()}%"
                conditions.append(
                    or_(
                        Product.name.ilike(term),
                        Product.keywords.ilike(term),
                        Product.description.ilike(term),
                    )
                )

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(Product.id.asc()).limit(safe_limit)

            products = db.session.scalars(stmt).all()
            return [_product_to_dict(p) for p in products]

        except Exception as e:
            logger.exception("Failed to query products from database")
            raise DatabaseError("Failed to query products from database") from e

    @staticmethod
    def find_applicable_standards(
        product_name: str | None = None,
        category: str | None = None,
        standard_number: str | None = None,
        relevance: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve applicable standards mapped to a specific product or product category.

        :param product_name: Name or keyword of the product (e.g., 'PVC cable', 'cables').
        :param category: Category filter.
        :param standard_number: Filter by specific IS number.
        :param relevance: 'Primary', 'Supporting', or 'Related'.
        :param limit: Maximum records to return.
        :return: List of dictionaries combining standard data, product data, and relevance.
        """
        try:
            stmt = (
                select(ProductStandardMapping, Product, Standard)
                .join(Product, ProductStandardMapping.product_id == Product.id)
                .join(Standard, ProductStandardMapping.standard_id == Standard.id)
            )

            conditions = []

            if product_name and product_name.strip():
                term = f"%{product_name.strip()}%"
                conditions.append(
                    or_(
                        Product.name.ilike(term),
                        Product.keywords.ilike(term),
                        Product.description.ilike(term),
                        Product.category.ilike(term),
                    )
                )

            if category and category.strip():
                conditions.append(Product.category.ilike(f"%{category.strip()}%"))

            if standard_number and standard_number.strip():
                conditions.append(Standard.is_number.ilike(f"%{standard_number.strip()}%"))

            if relevance and relevance.strip():
                conditions.append(ProductStandardMapping.relevance.ilike(f"%{relevance.strip()}%"))

            if conditions:
                stmt = stmt.where(*conditions)

            safe_limit = min(max(1, limit), 50)
            stmt = stmt.order_by(ProductStandardMapping.id.asc()).limit(safe_limit)

            rows = db.session.execute(stmt).all()

            results = []
            for mapping, product, standard in rows:
                results.append({
                    "mapping_id": str(mapping.id),
                    "product_id": str(product.id),
                    "product_name": product.name,
                    "product_category": product.category,
                    "standard_id": str(standard.id),
                    "is_number": standard.is_number,
                    "title": standard.title,
                    "revision_number": standard.revision_no,
                    "publication_year": standard.publication_year,
                    "status": standard.status,
                    "technical_department": standard.technical_department,
                    "relevance": mapping.relevance or "Primary",
                    "source_url": mapping.source_url or standard.source_url,
                    "document_url": standard.document_url,
                })

            return results

        except Exception as e:
            logger.exception("Failed to query applicable standards from database")
            raise DatabaseError("Failed to query applicable standards from database") from e


# Functional aliases for direct import
find_product = ProductRepository.find_product
find_applicable_standards = ProductRepository.find_applicable_standards
