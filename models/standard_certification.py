from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .standard import Standard
    from .certification_scheme import CertificationScheme


class StandardCertification(db.Model):
    __tablename__ = "standard_certification"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    standard_id: Mapped[int] = mapped_column(
        db.ForeignKey("standards.id"),
        nullable=False
    )

    certification_scheme_id: Mapped[int] = mapped_column(
        db.ForeignKey("certification_schemes.id"),
        nullable=False
    )

    requirement_type: Mapped[str | None] = mapped_column(
        db.String(100),
        nullable=True
    )

    mandatory: Mapped[str | None] = mapped_column(
        db.String(50),
        nullable=True
    )

    conditions: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    standard: Mapped["Standard"] = db.relationship(
        "Standard",
        back_populates="certifications",
        lazy="raise_on_sql"
    )

    certification_scheme: Mapped["CertificationScheme"] = db.relationship(
        "CertificationScheme",
        back_populates="standard_certifications",
        lazy="raise_on_sql"
    )
