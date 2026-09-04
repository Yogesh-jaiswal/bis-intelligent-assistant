import logging
import time
from repositories.embedding_repository import retrieve_similar_chunks
from services.file_processors.embeddings.embeddings_generator import EmbeddingGenerator
from services.file_processors.document.doc_representation import DocumentBlock

from .retrieval_dataclasses import RetrievedChunk

logger = logging.getLogger(__name__)


class SimilaritySearchService:
    """Service for performing similarity searches on document chunks based on embeddings."""
    def __init__(self, embedder=None):
        self._embedding_cache: dict[str, list[float]] = {}
        if embedder is not None:
            self.embedder = embedder
        else:
            from configs import get_settings
            if get_settings().AI_PROVIDER == "FAKE":
                from services.file_processors.embeddings.fake_embeddings import FakeEmbeddingGenerator
                self.embedder = FakeEmbeddingGenerator()
            else:
                self.embedder = EmbeddingGenerator()

    def search(
        self,
        query: str,
        k: int = 5,
        upload_id: str | None = None,
        filename: str | None = None,
        standard_number: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search query in stored embeddings and return top k results with optional document filtering.
        """
        start_t = time.perf_counter()
        normalized_q = query.strip()
        if normalized_q in self._embedding_cache:
            query_embedding = self._embedding_cache[normalized_q]
            embed_ms = 0.0
            logger.info("[VECTOR SEARCH: CACHE] Using cached query embedding for: '%s'...", query[:80])
        else:
            logger.info("[VECTOR SEARCH: START] Generating embedding vector for query: '%s'...", query[:100])
            query_embedding = self.embedder.embed(query)
            self._embedding_cache[normalized_q] = query_embedding
            embed_ms = (time.perf_counter() - start_t) * 1000.0

        db_start = time.perf_counter()
        result = retrieve_similar_chunks(
            query_embedding=query_embedding,
            k=k,
            upload_id=upload_id,
            filename=filename,
            standard_number=standard_number,
        )
        db_ms = (time.perf_counter() - db_start) * 1000.0

        chunks = [
            RetrievedChunk(
                score=score,
                chunk=DocumentBlock(
                    type=chunk.block_type,
                    text=chunk.content,
                    metadata=chunk.chunk_metadata,
                ),
                filename=upload.filename,
                author=upload.author,
                source_type=upload.source_type,
                source_url=upload.source_url,
            ) for (score, chunk, upload) in result
        ]

        logger.info(
            "[VECTOR SEARCH: COMPLETE] Embedded query in %.2f ms | DB cosine search in %.2f ms -> Retrieved %d chunks (Top score: %.4f, Doc: '%s')",
            embed_ms,
            db_ms,
            len(chunks),
            chunks[0].score if chunks else 0.0,
            chunks[0].filename if chunks else "None",
        )
        return chunks