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
    filename: str | None = None,
    standard_number: str | None = None,
) -> list[tuple[float, DocumentChunk, Upload]]:
    """
    Retrieve the top-k most similar document chunks from the vector store together with their upload metadata.

    :param query_embedding: Embedding vector of the query (e.g. 384 dimensions).
    :param k: Number of top results to return.
    :param upload_id: Optional filter for a specific upload document ID.
    :param filename: Optional exact filename filter (e.g. '694_2010_reff2020.pdf').
    :param standard_number: Optional IS standard filter (e.g. 'IS 694' or 'IS 694:2010').
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
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == DocumentChunk.id)
            .join(Upload, DocumentChunk.upload_id == Upload.id)
        )

        conditions = []
        if upload_id:
            conditions.append(Upload.id == upload_id)
        if filename:
            conditions.append(Upload.filename.ilike(f"%{filename.strip()}%"))
        if standard_number and standard_number.strip():
            # Extract standard digits/pattern, e.g. "IS 694:2010" -> "694"
            clean_std = standard_number.strip()
            # Extract main number part after "IS"
            std_parts = clean_std.replace("IS", "").replace(":", " ").replace("-", " ").split()
            if std_parts:
                num_part = std_parts[0]
                conditions.append(
                    db.or_(
                        Upload.filename.ilike(f"%{num_part}%"),
                        DocumentChunk.content.ilike(f"%IS {num_part}%"),
                        DocumentChunk.content.ilike(f"%IS:{num_part}%"),
                    )
                )

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(distance_expr).limit(k)

        rows = db.session.execute(stmt).all()

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

        # Fallback to unfiltered search if filtered search returned 0 results
        if not results and (standard_number or filename or upload_id):
            logger.info("[VECTOR RETRIEVAL] Filtered vector search yielded 0 results; falling back to unfiltered search")
            fallback_stmt = (
                db.select(
                    DocumentChunk,
                    Upload,
                    distance_expr.label("distance"),
                )
                .join(ChunkEmbedding, ChunkEmbedding.chunk_id == DocumentChunk.id)
                .join(Upload, DocumentChunk.upload_id == Upload.id)
                .order_by(distance_expr)
                .limit(k)
            )
            fallback_rows = db.session.execute(fallback_stmt).all()
            for chunk, upload, distance in fallback_rows:
                score = max(0.0, 1.0 - float(distance))
                if score >= settings.MIN_SIMILARITY:
                    results.append((round(score, 4), chunk, upload))

        return results

    except Exception as e:
        logger.exception("Failed to retrieve similar chunks from vector store")
        raise DatabaseError("Failed to retrieve similar chunks from vector store") from e