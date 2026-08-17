import logging
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from app.config import settings
from app.core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def _collection_name(user_id: str) -> str:
    return f"user_{user_id}_docs"


@lru_cache
def _get_chroma_client(user_id: str) -> Chroma:
    """
    Cached per user_id, so repeated calls for the same user reuse the
    same Chroma client instead of reopening the persisted DB each time.
    """
    return Chroma(
        collection_name=_collection_name(user_id),
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_chunks(user_id: str, document_id: str, chunks: list[LCDocument]) -> int:
    """
    Embeds and stores chunks for one document & Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v is not None}

    store = _get_chroma_client(user_id)
    ids = [f"{document_id}_{chunk.metadata.get('chunk_index', i)}" for i, chunk in enumerate(chunks)]

    store.add_documents(documents=chunks, ids=ids)
    logger.info("Stored %d chunk(s) for document_id=%s (user=%s)", len(chunks), document_id, user_id)
    return len(chunks)


def similarity_search(
    user_id: str,
    query: str,
    k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[tuple[LCDocument, float]]:
    
    store = _get_chroma_client(user_id)
    top_k = k or settings.retrieval_candidate_k

    where_filter = {"document_id": {"$in": document_ids}} if document_ids else None

    results = store.similarity_search_with_relevance_scores(
        query, k=top_k, filter=where_filter
    )
    logger.info("Retrieved %d chunk(s) for query (user=%s)", len(results), user_id)
    return results


def get_all_chunks(user_id: str, document_ids: list[str] | None = None) -> list[LCDocument]:
    store = _get_chroma_client(user_id)
    where_filter = {"document_id": {"$in": document_ids}} if document_ids else None

    raw = store.get(where=where_filter, include=["documents", "metadatas"])
    chunks = [
        LCDocument(page_content=doc, metadata=metadata or {})
        for doc, metadata in zip(raw["documents"], raw["metadatas"])
    ]
    return chunks


def delete_document(user_id: str, document_id: str) -> None:
    """Removes all vectors belonging to a document — when a user deletes an uploaded file."""
    store = _get_chroma_client(user_id)
    store.delete(where={"document_id": document_id})
    logger.info("Deleted vectors for document_id=%s (user=%s)", document_id, user_id)