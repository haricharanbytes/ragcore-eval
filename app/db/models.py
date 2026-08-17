import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class IngestionStatus(str, PyEnum):
    """Tracks where a document is in the ingestion pipeline, so the
    frontend can show 'processing...' instead of assuming upload = ready."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, index=True, default="default_user")

    filename: Mapped[str] = mapped_column(String)
    file_extension: Mapped[str] = mapped_column(String)
    file_size_bytes: Mapped[int] = mapped_column(Integer)

    status: Mapped[IngestionStatus] = mapped_column(
        Enum(IngestionStatus), default=IngestionStatus.PENDING
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    # Matches the Chroma collection name this document's chunks were stored in,
    chroma_collection: Mapped[str] = mapped_column(String)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )