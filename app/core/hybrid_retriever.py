"""
Hybrid retrieval.

Combines two retrieval strategies to catch what either one alone would miss:

- Vector search (Chroma): strong at SEMANTIC similarity — "refund process"
  matches a chunk about "how to get your money back" even without shared words.
- BM25 (keyword): strong at EXACT matches — a specific product code, name,
  or number that vector search might blur past in favor of "similar-sounding"
  text.

This stage does NOT try to rank the merged results — it just gathers a
generous, deduplicated candidate pool (~20-30 chunks). Precise ranking is
the reranker's job (see reranker.py), which scores everything against the
original question using a model built for that purpose.

Design note: the BM25 index is built fresh on every query from whatever
chunks currently exist in Chroma (see vectorstore.get_all_chunks), rather
than maintained as a separate persistent index kept in sync with uploads/
deletes. At portfolio scale (dozens-hundreds of chunks per user) rebuilding
per-query is fast and keeps the system simple; a persisted BM25 index is
the natural next step if this needed to scale to much larger corpora.
"""

import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

from app.config import settings
from app.core.vectorstore import get_all_chunks, similarity_search

logger = logging.getLogger(__name__)


def _chunk_key(doc: LCDocument) -> str:
    """Unique identity for a chunk, used for deduplication when a chunk
    is retrieved by both vector search and BM25."""
    return f"{doc.metadata.get('document_id', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"


def hybrid_retrieve(
    user_id: str,
    query: str,
    document_ids: list[str] | None = None,
) -> list[LCDocument]:
    """
    Returns a deduplicated pool of candidate chunks from both vector and
    BM25 retrieval, ready to be handed to the reranker. Order is not
    meaningful here — both retrievers' results are merged, not ranked
    against each other.
    """
    candidate_k = settings.retrieval_candidate_k

    # --- Vector candidates ---
    vector_results = similarity_search(
        user_id=user_id, query=query, k=candidate_k, document_ids=document_ids
    )
    vector_docs = [doc for doc, _score in vector_results]

    # --- BM25 candidates ---
    all_chunks = get_all_chunks(user_id=user_id, document_ids=document_ids)
    bm25_docs: list[LCDocument] = []
    if all_chunks:
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = candidate_k
        bm25_docs = bm25_retriever.invoke(query)

    # --- Merge + dedupe ---
    merged: dict[str, LCDocument] = {}
    for doc in vector_docs + bm25_docs:
        merged[_chunk_key(doc)] = doc

    candidates = list(merged.values())
    logger.info(
        "Hybrid retrieval: %d vector + %d BM25 -> %d unique candidate(s) (user=%s)",
        len(vector_docs), len(bm25_docs), len(candidates), user_id,
    )
    return candidates