from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .product_standard_mapping import ProductStandardMapping


class Product(db.Model):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    name: Mapped[str] = mapped_column(
        db.String(255),
        nullable=False
    )

    category: Mapped[str | None] = mapped_column(
        db.String(255),
        nullable=True
    )

    description: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    keywords: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=True
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True
    )

    standard_mappings: Mapped[list["ProductStandardMapping"]] = db.relationship(
        "ProductStandardMapping",
        back_populates="product",
        cascade="all, delete-orphan",
        lazy="raise_on_sql"
    )
