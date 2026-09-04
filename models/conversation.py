from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db


class Conversation(db.Model):
    """
    Minimal conversation persistence model for MVP multi-turn context.
    Identified solely by conversation_id without user accounts or auth tokens.
    """
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(
        db.String(100),
        primary_key=True,
    )

    summary: Mapped[str | None] = mapped_column(
        db.Text,
        nullable=True,
    )

    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=True,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )
