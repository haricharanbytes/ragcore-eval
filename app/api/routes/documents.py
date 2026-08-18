import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.vectorstore import delete_document as delete_document_vectors
from app.db.models import Document
from app.db.session import get_db
from app.models.schemas import DocumentListResponse, DocumentResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)) -> DocumentListResponse:
    user_id = settings.default_user_id  # single-user for now; see db/models.py note

    stmt = select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    documents = db.execute(stmt).scalars().all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=len(documents),
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    user_id = settings.default_user_id

    document = db.get(Document, document_id)
    if document is None or document.user_id != user_id:
        # Same 404 whether the doc doesn't exist or belongs to another
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        delete_document_vectors(user_id=user_id, document_id=document_id)
    except Exception:
        logger.exception("Failed to delete vectors for document_id=%s", document_id)

    file_path = Path(settings.upload_dir) / f"{document.id}{document.file_extension}"
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            logger.exception("Failed to delete file on disk for document_id=%s", document_id)

    db.delete(document)
    db.commit()
    logger.info("Deleted document_id=%s (user=%s)", document_id, user_id)