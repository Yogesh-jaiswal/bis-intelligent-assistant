from datetime import date
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db


class Laboratory(db.Model):
    __tablename__ = "laboratories"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    lab_code: Mapped[str | None] = mapped_column(
        db.String(50),
        nullable=True
    )

    name: Mapped[str] = mapped_column(
        db.String(255),
        nullable=False
    )

    address: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    state: Mapped[str | None] = mapped_column(
        db.String(100),
        nullable=True
    )

    district: Mapped[str | None] = mapped_column(
        db.String(100),
        nullable=True
    )

    contact_person: Mapped[str | None] = mapped_column(
        db.String(150),
        nullable=True
    )

    phone: Mapped[str | None] = mapped_column(
        db.String(100),
        nullable=True
    )

    email: Mapped[str | None] = mapped_column(
        db.String(150),
        nullable=True
    )

    validity_date: Mapped[date | None] = mapped_column(
        db.Date,
        nullable=True
    )

    scope: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )
