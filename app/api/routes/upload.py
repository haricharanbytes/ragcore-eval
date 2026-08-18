import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.core.ingestion import ingest_document
from app.core.vectorstore import _collection_name
from app.db.models import Document, IngestionStatus
from app.db.session import get_db
from app.models.schemas import DocumentResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    extension = Path(file.filename).suffix.lower()
    if extension not in settings.allowed_file_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {', '.join(settings.allowed_file_extensions_list)}",
        )

    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size_bytes > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max allowed size of {settings.max_upload_size_mb}MB.",
        )

    user_id = settings.default_user_id  # single-user for now; see db/models.py note

    # Create the DB row up front (status=PENDING) so a record exists even
    # if something fails while saving the file to disk.
    document = Document(
        user_id=user_id,
        filename=file.filename,
        file_extension=extension,
        file_size_bytes=size_bytes,
        status=IngestionStatus.PENDING,
        chroma_collection=_collection_name(user_id),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Save with a UUID-based name (document.id) to avoid filename collisions
    # and directory traversal issues from user-supplied filenames.
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{document.id}{extension}"

    try:
        saved_path.write_bytes(content)
    except OSError:
        logger.exception("Failed to write uploaded file to disk: %s", file.filename)
        document.status = IngestionStatus.FAILED
        document.error_message = "Failed to save uploaded file to disk."
        db.commit()
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    document = ingest_document(db=db, document=document, file_path=saved_path)

    return DocumentResponse.model_validate(document)