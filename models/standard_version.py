from datetime import date
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .standard import Standard


class StandardVersion(db.Model):
    __tablename__ = "standard_versions"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    standard_id: Mapped[int] = mapped_column(
        db.ForeignKey("standards.id"),
        nullable=False
    )

    version: Mapped[str] = mapped_column(
        db.String(100),
        nullable=False
    )

    version_type: Mapped[str | None] = mapped_column(
        db.String(50),
        nullable=True
    )

    publication_date: Mapped[date | None] = mapped_column(
        db.Date,
        nullable=True
    )

    effective_date: Mapped[date | None] = mapped_column(
        db.Date,
        nullable=True
    )

    status: Mapped[str | None] = mapped_column(
        db.String(50),
        nullable=True,
        default="Active"
    )

    document_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    supersedes_version_id: Mapped[int | None] = mapped_column(
        db.ForeignKey("standard_versions.id"),
        nullable=True
    )

    standard: Mapped["Standard"] = db.relationship(
        "Standard",
        back_populates="versions",
        lazy="raise_on_sql"
    )

    supersedes_version: Mapped["StandardVersion | None"] = db.relationship(
        "StandardVersion",
        remote_side=[id],
        lazy="raise_on_sql"
    )
