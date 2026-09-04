import logging
import time
from typing import Any
from sqlalchemy import select, or_

from app.extensions import db
from models.product import Product
from models.standard import Standard
from models.product_standard_mapping import ProductStandardMapping
from exceptions import DatabaseError
from repositories.semantic_search import rank_entities_semantically

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
        enable_semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search for products by name, category, or keyword with semantic fallback.
        """
        start_t = time.perf_counter()
        logger.info(
            "[REPO: PRODUCT] Querying products (name='%s', category='%s', keyword='%s', limit=%d)",
            name,
            category,
            keyword,
            limit,
        )
        try:
            stmt = select(Product)
            conditions = []

            if name and name.strip():
                clean_n = name.strip()
                n_terms = [clean_n]
                if clean_n.lower().endswith("s") and len(clean_n) > 3:
                    n_terms.append(clean_n[:-1])
                n_or = [Product.name.ilike(f"%{t}%") for t in n_terms]
                conditions.append(or_(*n_or))

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

            products = list(db.session.scalars(stmt).all())
            seen_ids = {p.id for p in products}

            # Semantic fallback if keyword provided and no exact results found
            search_query = name or keyword
            if enable_semantic and search_query and len(products) == 0:
                logger.info("[REPO: PRODUCT] SQL returned %d records (< %d) -> triggering semantic embedding search for '%s'", len(products), safe_limit, search_query)
                all_products = list(db.session.scalars(select(Product)).all())
                ranked = rank_entities_semantically(
                    query=search_query,
                    entities=all_products,
                    text_extractor=lambda p: f"{p.name} {p.category or ''} {p.keywords or ''} {p.description or ''}",
                    limit=safe_limit - len(products),
                )
                for _score, prod in ranked:
                    if prod.id not in seen_ids:
                        products.append(prod)
                        seen_ids.add(prod.id)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info("[REPO: PRODUCT] Query completed in %.2f ms -> %d products found", elapsed_ms, len(products))
            return [_product_to_dict(p) for p in products]

        except Exception as e:
            logger.error("[REPO: PRODUCT ERROR] Failed to query products from database: %s", e, exc_info=True)
            raise DatabaseError("Failed to query products from database") from e

    @staticmethod
    def find_applicable_standards(
        product_name: str | None = None,
        category: str | None = None,
        standard_number: str | None = None,
        relevance: str | None = None,
        limit: int = 10,
        enable_semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve applicable standards mapped to a specific product or product category.
        Uses semantic product discovery if natural-language wording does not match exact names.
        """
        start_t = time.perf_counter()
        logger.info(
            "[REPO: APPLICABLE STANDARDS] Querying product-standard mappings (product='%s', category='%s', std='%s', rel='%s', limit=%d)",
            product_name,
            category,
            standard_number,
            relevance,
            limit,
        )
        try:
            stmt = (
                select(ProductStandardMapping, Product, Standard)
                .join(Product, ProductStandardMapping.product_id == Product.id)
                .join(Standard, ProductStandardMapping.standard_id == Standard.id)
            )

            conditions = []

            if product_name and product_name.strip():
                clean_p = product_name.strip()
                p_terms = [clean_p]
                if clean_p.lower().endswith("s") and len(clean_p) > 3:
                    p_terms.append(clean_p[:-1])
                p_or = []
                for t in p_terms:
                    term = f"%{t}%"
                    p_or.extend([
                        Product.name.ilike(term),
                        Product.keywords.ilike(term),
                        Product.description.ilike(term),
                        Product.category.ilike(term),
                    ])
                conditions.append(or_(*p_or))

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

            rows = list(db.session.execute(stmt).all())
            seen_mapping_ids = {mapping.id for mapping, _, _ in rows}

            # Semantic fallback: if no direct product matches found, find products semantically
            if enable_semantic and product_name and len(rows) == 0:
                logger.info("[REPO: APPLICABLE STANDARDS] Exact match yielded %d records -> searching products semantically for '%s'", len(rows), product_name)
                all_products = list(db.session.scalars(select(Product)).all())
                ranked_products = rank_entities_semantically(
                    query=product_name,
                    entities=all_products,
                    text_extractor=lambda p: f"{p.name} {p.category or ''} {p.keywords or ''} {p.description or ''}",
                    limit=5,
                )

                if ranked_products:
                    matched_prod_ids = [p.id for _score, p in ranked_products]
                    fallback_stmt = (
                        select(ProductStandardMapping, Product, Standard)
                        .join(Product, ProductStandardMapping.product_id == Product.id)
                        .join(Standard, ProductStandardMapping.standard_id == Standard.id)
                        .where(ProductStandardMapping.product_id.in_(matched_prod_ids))
                    )
                    if standard_number and standard_number.strip():
                        fallback_stmt = fallback_stmt.where(Standard.is_number.ilike(f"%{standard_number.strip()}%"))

                    fallback_stmt = fallback_stmt.order_by(ProductStandardMapping.id.asc()).limit(safe_limit - len(rows))
                    fallback_rows = db.session.execute(fallback_stmt).all()

                    for mapping, prod, std in fallback_rows:
                        if mapping.id not in seen_mapping_ids:
                            rows.append((mapping, prod, std))
                            seen_mapping_ids.add(mapping.id)

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

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info("[REPO: APPLICABLE STANDARDS] Query completed in %.2f ms -> %d mappings found", elapsed_ms, len(results))
            return results

        except Exception as e:
            logger.error("[REPO: APPLICABLE STANDARDS ERROR] Failed to query applicable standards: %s", e, exc_info=True)
            raise DatabaseError("Failed to query applicable standards from database") from e


# Functional aliases for direct import
find_product = ProductRepository.find_product
find_applicable_standards = ProductRepository.find_applicable_standards
