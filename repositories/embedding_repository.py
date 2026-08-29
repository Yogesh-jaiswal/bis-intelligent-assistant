import logging
from configs import get_settings
from app.extensions import db
from models.chunk_embeddings import ChunkEmbedding
from models.document_chunk import DocumentChunk
from models.upload import Upload
from exceptions import DatabaseError

logger = logging.getLogger(__name__)


def retrieve_similar_chunks(
    query_embedding: list[float],
    k: int = 5,
    upload_id: str | None = None,
) -> list[tuple[float, DocumentChunk, Upload]]:
    """
    Retrieve the top-k most similar document chunks from the vector store together with their upload metadata.

    :param query_embedding: Embedding vector of the query (e.g. 384 dimensions).
    :param k: Number of top results to return.
    :param upload_id: Optional filter for a specific upload document ID.
    :return: List of tuples (similarity_score, DocumentChunk, Upload).
    """
    settings = get_settings()

    try:
        # Configure HNSW search parameter for session
        db.session.execute(
            db.text(f"SET LOCAL hnsw.ef_search = {settings.HNSW_EF_SEARCH};")
        )

        distance_expr = ChunkEmbedding.vector.cosine_distance(query_embedding)

        stmt = (
            db.select(
                DocumentChunk,
                Upload,
                distance_expr.label("distance"),
            )
            .join(ChunkEmbedding.chunk)
            .join(Upload, DocumentChunk.upload_id == Upload.id)
        )

        if upload_id:
            stmt = stmt.where(Upload.id == upload_id)

        stmt = stmt.order_by(distance_expr).limit(k)

        rows = db.session.execute(stmt)

        results: list[tuple[float, DocumentChunk, Upload]] = []

        for chunk, upload, distance in rows:
            score = max(0.0, 1.0 - float(distance))

            if score < settings.MIN_SIMILARITY:
                continue

            results.append(
                (
                    round(score, 4),
                    chunk,
                    upload,
                )
            )

        return results

    except Exception as e:
        logger.exception("Failed to retrieve similar chunks from vector store")
        raise DatabaseError("Failed to retrieve similar chunks from vector store") from e