import uuid
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document_chunk import DocumentChunk


class Upload(db.Model):
    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    filename: Mapped[str] = mapped_column(
        db.String(255),
        nullable=False
    )

    file_type: Mapped[str] = mapped_column(
        db.String(50),
        nullable=False
    )

    file_path: Mapped[str] = mapped_column(
        db.Text,
        nullable=False
    )

    author: Mapped[str | None] = mapped_column(
        db.String(255),
        nullable=True
    )

    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )
    """Authoritative external URL for this document (e.g. verified BIS page).
    Null when no reliable URL is known. Never use a generic homepage as fallback.
    """

    created_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    @property
    def source_type(self) -> str:
        return self.file_type

    chunks: Mapped[list["DocumentChunk"]] = db.relationship(
        "DocumentChunk",
        back_populates="upload",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )
