from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db


class Service(db.Model):
    __tablename__ = "services"

    id: Mapped[int] = mapped_column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    name: Mapped[str] = mapped_column(
        db.String(255),
        nullable=False
    )

    service_type: Mapped[str] = mapped_column(
        db.String(100),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    eligibility: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    documents_required: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )

    source_url: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True
    )
