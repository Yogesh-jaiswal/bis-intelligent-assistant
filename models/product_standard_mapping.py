from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db

if TYPE_CHECKING:
    from .product import Product
    from .standard import Standard


class ProductStandardMapping(db.Model):
    __tablename__ = "product_standard_mapping"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    product_id: Mapped[int] = mapped_column(
        db.ForeignKey("products.id"),
        nullable=False
    )

    standard_id: Mapped[int] = mapped_column(
        db.ForeignKey("standards.id"),
        nullable=False
    )

    relevance: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True,
        default="Primary"
    )


    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    product: Mapped["Product"] = db.relationship(
        "Product",
        back_populates="standard_mappings",
        lazy="raise_on_sql"
    )

    standard: Mapped["Standard"] = db.relationship(
        "Standard",
        back_populates="product_mappings",
        lazy="raise_on_sql"
    )
