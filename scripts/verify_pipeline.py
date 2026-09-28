"""
Standalone smoke test for the hybrid retrieval + reranking pipeline.

Run this BEFORE starting uvicorn, to catch issues stage-by-stage instead
of guessing which part of a 4-stage pipeline broke from one FastAPI error.

Usage (from the repo root, with your venv active):
    python3 scripts/verify_pipeline.py

Requires: at least one document already uploaded and successfully
ingested (status=completed) for the default user — this script reads
from your existing Chroma data, it doesn't create new documents.
"""

import sys
import traceback


def check(label: str, fn):
    print(f"\n--- {label} ---")
    try:
        result = fn()
        print(f"PASS: {label}")
        return result
    except Exception as exc:
        print(f"FAIL: {label}")
        print(f"  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


def main():
    from app.config import settings

    print(f"Using GROQ_MODEL={settings.groq_model}")
    print(f"Using QUERY_REWRITE_MODEL={settings.query_rewrite_model}")
    print(f"Using RERANKER_MODEL={settings.reranker_model}")
    print(f"RETRIEVAL_CANDIDATE_K={settings.retrieval_candidate_k}  RERANK_TOP_N={settings.rerank_top_n}")

    # --- Stage 0: is there any data to test against? ---
    from app.core.vectorstore import get_all_chunks

    chunks = check(
        "Stage 0: Chroma has existing data",
        lambda: get_all_chunks(user_id=settings.default_user_id),
    )
    if not chunks:
        print("\nNo chunks found for the default user. Upload a document first "
              "(via the running app or the API) before running this script.")
        sys.exit(1)
    print(f"  Found {len(chunks)} existing chunk(s) to test against.")

    sample_question = "What is this document about?"

    # --- Stage 1: query rewrite ---
    from app.core.query_rewrite import rewrite_query

    rewritten = check(
        "Stage 1: Query rewrite (Groq)",
        lambda: rewrite_query(sample_question),
    )
    if rewritten:
        print(f"  Original:  {sample_question}")
        print(f"  Rewritten: {rewritten}")

    # --- Stage 2: hybrid retrieval ---
    from app.core.hybrid_retriever import hybrid_retrieve

    candidates = check(
        "Stage 2: Hybrid retrieval (vector + BM25)",
        lambda: hybrid_retrieve(user_id=settings.default_user_id, query=rewritten or sample_question),
    )
    if candidates:
        print(f"  Got {len(candidates)} candidate chunk(s).")

    # --- Stage 3: reranking ---
    from app.core.reranker import rerank

    reranked = check(
        "Stage 3: Reranking (local cross-encoder — will download model on first run)",
        lambda: rerank(query=sample_question, candidates=candidates or []),
    )
    if reranked:
        print(f"  Top {len(reranked)} after reranking, scores:")
        for doc, score in reranked:
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"    {score:.3f}  {preview}...")

    # --- Stage 4: full pipeline end-to-end ---
    from app.core.rag_chain import generate_answer

    result = check(
        "Stage 4: Full pipeline (generate_answer)",
        lambda: generate_answer(user_id=settings.default_user_id, question=sample_question),
    )
    if result:
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Sources: {len(result['sources'])}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()