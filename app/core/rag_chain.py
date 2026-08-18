"""
RAG chain.

This is where the full pipeline comes together:
1. Rewrite the question into a clearer retrieval query (query_rewrite.py)
2. Hybrid retrieval: vector + BM25 search, merged into a candidate pool
   (hybrid_retriever.py)
3. Rerank candidates against the ORIGINAL question with a local
   cross-encoder, keep the top few (reranker.py)
4. Stuff those chunks into a prompt that instructs the LLM to answer
   ONLY from the provided context (reduces hallucination)
5. Call the Groq-hosted LLM to generate the answer
6. Return the answer alongside the source chunks that backed it, so the
   UI can render "Source: report.pdf, page 3" citations

Each stage lives in its own module and is swappable/testable on its own —
this function's job is purely orchestration, not logic.
"""

import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings
from app.core.hybrid_retriever import hybrid_retrieve
from app.core.query_rewrite import rewrite_query
from app.core.reranker import rerank
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context from the user's uploaded documents.

Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so clearly \
instead of guessing.
- Be concise and direct.
- Do not fabricate document names, page numbers, or facts not present in the context.

Context:
{context}
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


@lru_cache
def _get_llm() -> ChatGroq:
    """Cached so we don't reconstruct the Groq client on every request."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
    )


def _format_context(results: list[tuple]) -> str:
    """Turns reranked (chunk, score) pairs into a numbered context block
    the LLM can reference. Numbering helps the model stay grounded in
    specific passages rather than blending everything together."""
    parts = []
    for i, (doc, _score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}" + (f", page {page}" if page else "")
        parts.append(f"[{i}] (Source: {location})\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_answer(
    user_id: str,
    question: str,
    document_ids: list[str] | None = None,
) -> dict:
    """
    Runs the full rewrite -> hybrid retrieve -> rerank -> generate
    pipeline for one question.

    Returns:
        {"answer": str, "sources": list[SourceChunk]}
    """
    retrieval_query = rewrite_query(question)

    candidates = hybrid_retrieve(
        user_id=user_id, query=retrieval_query, document_ids=document_ids
    )

    if not candidates:
        logger.info("No candidates found for question (user=%s)", user_id)
        return {
            "answer": (
                "I couldn't find anything relevant to that question in your "
                "uploaded documents. Try rephrasing, or upload a document that "
                "covers this topic."
            ),
            "sources": [],
        }

    # Rerank against the ORIGINAL question, not the rewrite — the rewrite
    # is a retrieval aid, but final relevance should reflect what the user
    # actually asked.
    results = rerank(query=question, candidates=candidates)

    if not results:
        # Every candidate scored below rerank_min_score — the pipeline
        # correctly refuses to answer from weakly-related chunks rather
        # than padding the context just to have something to show.
        logger.info("All candidates filtered by reranker (user=%s)", user_id)
        return {
            "answer": (
                "I couldn't find anything relevant to that question in your "
                "uploaded documents. Try rephrasing, or upload a document that "
                "covers this topic."
            ),
            "sources": [],
        }

    context = _format_context(results)
    llm = _get_llm()

    chain = _prompt | llm
    response = chain.invoke({"context": context, "question": question})

    sources = [
        SourceChunk(
            document_id=doc.metadata.get("document_id", "unknown"),
            filename=doc.metadata.get("source", "unknown"),
            chunk_text=doc.page_content,
            chunk_index=doc.metadata.get("chunk_index", 0),
            page_number=doc.metadata.get("page"),
            # Raw cross-encoder relevance score (not a 0-1 probability like
            # the old vector-only score) — still meaningful for sorting
            # and relative comparison between sources.
            relevance_score=round(float(score), 4),
        )
        for doc, score in results
    ]

    logger.info("Generated answer for question (user=%s), %d source(s)", user_id, len(sources))
    return {"answer": response.content, "sources": sources}