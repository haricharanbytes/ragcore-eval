"""
Reranking.

Vector and BM25 retrieval are both fast but approximate — they're built to
quickly narrow millions of documents down to a candidate pool, not to
precisely judge relevance. A cross-encoder reranker does the opposite: it's
slower (scores each candidate individually against the query, rather than
comparing precomputed embeddings) but much more accurate, because it looks
at the query and passage TOGETHER rather than comparing separate vectors.

This is the stage that actually decides what the LLM sees. Feeding it a
generous, imprecise candidate pool (from hybrid_retriever.py) and trusting
the reranker to find the real top 3-5 is a well-established pattern for
improving RAG answer quality over naive top-k vector search alone.

Runs locally via sentence-transformers' CrossEncoder (already a dependency
for embeddings) — no extra API or key needed, consistent with the rest of
this app's "local-first" model choices.
"""

import logging
from functools import lru_cache

from langchain_core.documents import Document as LCDocument
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_reranker_model() -> CrossEncoder:
    """
    Cached so the model (a few hundred MB) loads into memory once per
    process, same reasoning as get_embedding_model() in embeddings.py.
    """
    logger.info("Loading reranker model: %s (first call downloads/caches weights)", settings.reranker_model)
    model = CrossEncoder(settings.reranker_model)
    logger.info("Reranker model ready.")
    return model


def rerank(
    query: str,
    candidates: list[LCDocument],
    top_n: int | None = None,
) -> list[tuple[LCDocument, float]]:
    """
    Scores each candidate chunk against the ORIGINAL question (not the
    rewritten query — reranking should reflect what the user actually
    asked) and returns the top_n highest-scoring chunks, sorted best-first.

    Returns (chunk, score) pairs. Scores are raw cross-encoder relevance
    scores — useful for sorting and relative comparison, not a 0-1
    probability like the vector similarity scores elsewhere in the app.
    """
    if not candidates:
        return []

    n = top_n or settings.rerank_top_n
    model = _get_reranker_model()

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_results = scored[:n]
    pre_filter_top_score = top_results[0][1] if top_results else 0.0

    # Drop weakly-relevant chunks even if it means returning fewer than
    # n (or zero) — a chunk that only barely outscored the rest of a weak
    # candidate pool shouldn't get forced into the LLM's context just to
    # fill a quota. See settings.rerank_min_score for the rationale.
    if settings.rerank_min_score is not None:
        before = len(top_results)
        top_results = [(doc, score) for doc, score in top_results if score >= settings.rerank_min_score]
        dropped = before - len(top_results)
        if dropped:
            logger.info("Dropped %d chunk(s) below rerank_min_score=%.2f", dropped, settings.rerank_min_score)

    logger.info(
        "Reranked %d candidate(s) -> top %d survived filtering (best raw score before filtering=%.3f)",
        len(candidates), len(top_results), pre_filter_top_score,
    )
    return top_results