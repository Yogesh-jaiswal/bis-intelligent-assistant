import uuid
from datetime import datetime
from sqlalchemy import (
    func,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.extensions import db
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document_chunk import DocumentChunk


class ChunkEmbedding(db.Model):
    __tablename__ = "chunk_embeddings"

    id: Mapped[str] = mapped_column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    chunk_id: Mapped[str] = mapped_column(
        db.ForeignKey("document_chunks.id"),
        nullable=False
    )

    vector: Mapped[list[float]] = mapped_column(
        Vector(384),
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    chunk: Mapped["DocumentChunk"] = db.relationship("DocumentChunk", back_populates="embedding", lazy="raise_on_sql")

    __table_args__ = (
        Index(
            "hnsw_chunk_vector_idx",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )