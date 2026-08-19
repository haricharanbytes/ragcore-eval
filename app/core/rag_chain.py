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
5. Call the Groq-hosted LLM to generate a STRUCTURED response — both the
   answer text AND an explicit "did the context actually support this?"
   flag, so we know whether to show sources at all (see AnswerWithGrounding)
6. Return the answer alongside source chunks — but ONLY if the LLM
   itself confirmed it actually used the context. If retrieval returned
   weak/irrelevant chunks and the LLM correctly said "I don't know,"
   showing source citations for chunks that didn't actually inform the
   answer would be misleading, even though those chunks technically
   exist and got reranked.

Each stage lives in its own module and is swappable/testable on its own —
this function's job is purely orchestration, not logic.
"""

import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import settings
from app.core.hybrid_retriever import hybrid_retrieve
from app.core.query_rewrite import rewrite_query
from app.core.reranker import rerank
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)


class AnswerWithGrounding(BaseModel):
    """
    Structured output shape for the LLM's response. Asking for
    found_in_context as an explicit field — rather than trying to infer
    it later from the answer's wording (e.g. matching phrases like
    "I don't know") — gets a direct, reliable signal from the model
    itself instead of a brittle keyword guess that breaks if the
    prompt or model ever changes.
    """
    answer: str = Field(description="The answer to the user's question, written naturally.")
    found_in_context: bool = Field(
        description=(
            "True if the provided context actually contained enough information "
            "to answer the question. False if the context was irrelevant or "
            "insufficient, and the answer is essentially 'I don't know' / "
            "'not found in the documents'."
        )
    )


_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context from the user's uploaded documents.

Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so clearly \
in the answer, AND set found_in_context to false.
- Only set found_in_context to true if the context genuinely supports your answer.
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


@lru_cache
def _get_structured_llm():
    """
    Same cached Groq client, wrapped to return AnswerWithGrounding
    directly instead of raw text — LangChain handles the underlying
    function-calling/JSON-mode plumbing.
    """
    return _get_llm().with_structured_output(AnswerWithGrounding)


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

    sources is EMPTY whenever the LLM indicates the context didn't
    actually support the answer — even though retrieval/reranking still
    ran and found *some* chunks (there's always a "closest match," even
    when nothing is truly relevant). Citing chunks that didn't actually
    inform the answer would be misleading.
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
    # actually asked. No score threshold applied here — see reranker.py
    # for why a fixed cutoff on raw cross-encoder scores isn't meaningful.
    results = rerank(query=question, candidates=candidates)

    context = _format_context(results)
    structured_llm = _get_structured_llm()

    chain = _prompt | structured_llm
    response: AnswerWithGrounding = chain.invoke({"context": context, "question": question})

    sources = []
    if response.found_in_context:
        sources = [
            SourceChunk(
                document_id=doc.metadata.get("document_id", "unknown"),
                filename=doc.metadata.get("source", "unknown"),
                chunk_text=doc.page_content,
                chunk_index=doc.metadata.get("chunk_index", 0),
                page_number=doc.metadata.get("page"),
                relevance_score=round(float(score), 4),
            )
            for doc, score in results
        ]
    else:
        logger.info(
            "LLM indicated context did not support an answer (user=%s) — omitting sources.",
            user_id,
        )

    logger.info("Generated answer for question (user=%s), %d source(s)", user_id, len(sources))
    return {"answer": response.answer, "sources": sources}