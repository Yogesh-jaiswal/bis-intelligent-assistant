from datetime import date
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .standard import Standard


class StandardAmendment(db.Model):
    __tablename__ = "standard_amendments"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    standard_id: Mapped[int] = mapped_column(
        db.ForeignKey("standards.id"),
        nullable=False
    )

    amendment_number: Mapped[str] = mapped_column(
        db.String(50),
        nullable=False
    )

    title: Mapped[str | None] = mapped_column(
        db.Text,
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

    document_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    standard: Mapped["Standard"] = db.relationship(
        "Standard",
        back_populates="amendments",
        lazy="raise_on_sql"
    )
