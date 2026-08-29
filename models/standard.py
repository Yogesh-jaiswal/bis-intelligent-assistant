from datetime import date
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .standard_version import StandardVersion
    from .standard_amendment import StandardAmendment
    from .standard_certification import StandardCertification
    from .product_standard_mapping import ProductStandardMapping


class Standard(db.Model):
    __tablename__ = "standards"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    is_number: Mapped[str] = mapped_column(
        db.String(100),
        unique=True,
        nullable=False
    )

    title: Mapped[str] = mapped_column(
        db.Text,
        nullable=False
    )

    revision_no: Mapped[int | None] = mapped_column(
        db.Integer,
        nullable=True
    )

    publication_year: Mapped[int | None] = mapped_column(
        db.Integer,
        nullable=True
    )

    status: Mapped[str | None] = mapped_column(
        db.String(50),
        nullable=True,
        default="Active"
    )

    technical_department: Mapped[str | None] = mapped_column(
        db.String(150),
        nullable=True
    )

    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    document_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    last_verified_at: Mapped[date | None] = mapped_column(
        db.Date,
        nullable=True
    )

    versions: Mapped[list["StandardVersion"]] = db.relationship(
        "StandardVersion",
        back_populates="standard",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )

    amendments: Mapped[list["StandardAmendment"]] = db.relationship(
        "StandardAmendment",
        back_populates="standard",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )

    certifications: Mapped[list["StandardCertification"]] = db.relationship(
        "StandardCertification",
        back_populates="standard",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )

    product_mappings: Mapped[list["ProductStandardMapping"]] = db.relationship(
        "ProductStandardMapping",
        back_populates="standard",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )
