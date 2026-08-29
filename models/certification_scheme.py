from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .standard_certification import StandardCertification


class CertificationScheme(db.Model):
    __tablename__ = "certification_schemes"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    name: Mapped[str] = mapped_column(
        db.String(255),
        nullable=False
    )

    scheme_code: Mapped[str] = mapped_column(
        db.String(50),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    certification_type: Mapped[str | None] = mapped_column(
        db.String(100),
        nullable=True
    )

    mandatory: Mapped[str | None] = mapped_column(
        db.String(255),
        nullable=True
    )

    authority: Mapped[str | None] = mapped_column(
        db.String(150),
        nullable=True,
        default="Bureau of Indian Standards (BIS)"
    )

    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    standard_certifications: Mapped[list["StandardCertification"]] = db.relationship(
        "StandardCertification",
        back_populates="certification_scheme",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )
