import uuid
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db
from typing import TYPE_CHECKING

from .enums import DocumentBlockType

if TYPE_CHECKING:
    from .upload import Upload
    from .chunk_embeddings import ChunkEmbedding

class DocumentChunk(db.Model):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    upload_id: Mapped[str] = mapped_column(
        db.ForeignKey("uploads.id"),
        nullable=False
    )

    chunk_index: Mapped[int] = mapped_column(
        db.Integer,
        nullable=False
    )

    block_type: Mapped[DocumentBlockType] = mapped_column(
        db.Enum(DocumentBlockType),
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        db.Text,
        nullable=False
    )

    chunk_metadata: Mapped[dict] = mapped_column(
        db.JSON,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    upload: Mapped["Upload"] = db.relationship("Upload", back_populates="chunks", lazy="raise_on_sql")
    embedding: Mapped["ChunkEmbedding"] = db.relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )