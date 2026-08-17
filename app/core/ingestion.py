import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.document_loader import DocumentLoadError, load_document
from app.core.text_splitter import split_documents
from app.core.vectorstore import add_chunks
from app.db.models import Document, IngestionStatus

logger = logging.getLogger(__name__)


def ingest_document(
    db: Session,
    document: Document,
    file_path: Path,
) -> Document:
    
    document.status = IngestionStatus.PROCESSING
    db.commit()

    try:
        loaded_docs = load_document(file_path)

        for doc in loaded_docs:
            doc.metadata["source"] = document.filename

        chunks = split_documents(loaded_docs)

        if not chunks:
            raise DocumentLoadError("Document produced no usable chunks after splitting.")

        stored_count = add_chunks(user_id=document.user_id, document_id=document.id, chunks=chunks)

        document.status = IngestionStatus.COMPLETED
        document.chunk_count = stored_count
        document.error_message = None
        logger.info("Ingestion completed for document_id=%s (%d chunks)", document.id, stored_count)

    except DocumentLoadError as exc:
        # Expected, user-facing failures (bad file, empty content, unsupported type).
        document.status = IngestionStatus.FAILED
        document.error_message = str(exc)
        logger.warning("Ingestion failed for document_id=%s: %s", document.id, exc)

    except Exception as exc:
        # Unexpected failures (e.g. embedding model crash, disk full).
        document.status = IngestionStatus.FAILED
        document.error_message = "An unexpected error occurred during processing."
        logger.exception("Unexpected ingestion error for document_id=%s", document.id)

    finally:
        db.commit()
        db.refresh(document)

    return document