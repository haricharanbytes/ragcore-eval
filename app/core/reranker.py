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

    # Deliberately NOT filtering by an absolute score threshold here.
    # Cross-encoder scores are raw, uncalibrated logits from a binary
    # classifier trained on a different dataset (MS MARCO) — there's no
    # principled reason a fixed cutoff like 0.0 means "irrelevant" for
    # OUR documents. A "negative" score can still be the most useful
    # chunk available for a given question. We trust the LLM's own
    # prompt instructions (see rag_chain.py: "say so if the context
    # doesn't contain enough information") to handle genuinely weak
    # context, rather than the pipeline silently discarding it first.
    logger.info(
        "Reranked %d candidate(s) -> top %d (best score=%.3f, worst kept score=%.3f)",
        len(candidates), len(top_results),
        top_results[0][1] if top_results else 0.0,
        top_results[-1][1] if top_results else 0.0,
    )
    return top_results