"""
Semantic search helper for database entities.

Allows natural-language discovery of database entities (Products, Standards, Services)
when exact keyword/ILIKE searches return zero or insufficient results.

Preserves parameterized relational queries:
1. Computes cosine similarity between query embedding and entity textual fields.
2. Returns ranked entities above the configured SEMANTIC_DB_SIMILARITY_THRESHOLD.
3. Downstream services use normal SQLAlchemy relationships to retrieve related records.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from configs import get_settings
from services.file_processors.embeddings.embeddings_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)

_embedder: EmbeddingGenerator | None = None


def _get_embedder() -> EmbeddingGenerator:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        if settings.AI_PROVIDER == "FAKE":
            from services.file_processors.embeddings.fake_embeddings import FakeEmbeddingGenerator
            _embedder = FakeEmbeddingGenerator()
        else:
            _embedder = EmbeddingGenerator()
    return _embedder


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def rank_entities_semantically(
    query: str,
    entities: list[Any],
    text_extractor: Callable[[Any], str],
    limit: int = 10,
    threshold: float | None = None,
) -> list[tuple[float, Any]]:
    """
    Rank in-memory or database entities against a natural language query using embeddings.

    :param query: Natural language search string.
    :param entities: List of model instances or dictionaries.
    :param text_extractor: Function mapping an entity to its text representation.
    :param limit: Maximum entities to return.
    :param threshold: Cosine similarity cutoff (defaults to SEMANTIC_DB_SIMILARITY_THRESHOLD).
    :return: List of (similarity_score, entity) tuples sorted descending by score.
    """
    if not query or not query.strip() or not entities:
        return []

    settings = get_settings()
    cutoff = threshold if threshold is not None else settings.SEMANTIC_DB_SIMILARITY_THRESHOLD

    try:
        embedder = _get_embedder()
        query_vec = embedder.embed(query.strip())

        texts = [text_extractor(e).strip() for e in entities]
        entity_vecs = embedder.embed_many(texts)

        scored: list[tuple[float, Any]] = []
        for entity, vec in zip(entities, entity_vecs):
            score = cosine_similarity(query_vec, vec)
            if score >= cutoff:
                scored.append((score, entity))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    except Exception as exc:
        logger.warning("Semantic entity ranking encountered an error: %s", exc)
        return []
