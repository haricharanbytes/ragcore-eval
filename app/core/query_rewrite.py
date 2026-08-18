"""
Query rewriting.

Raw user questions are often conversational or under-specified for
retrieval purposes — e.g. "what about the pricing thing?" embeds poorly
compared to "What is the pricing structure described in the document?".
This step asks a small, fast LLM to rewrite the question into a clearer,
keyword-rich standalone query before it ever reaches retrieval.

Deliberately uses a SEPARATE, smaller Groq model (settings.query_rewrite_model,
e.g. llama-3.1-8b-instant) from the one used for answer generation
(settings.groq_model, e.g. llama-3.3-70b-versatile) — rewriting a question
doesn't need a large model's reasoning power, and using a cheaper/faster
model here keeps the added latency of this extra pipeline stage small.

If this step fails for any reason (Groq error, timeout), we fall back to
the original question rather than failing the whole request — query
rewriting is an enhancement, not a hard dependency.
"""

import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = """You rewrite user questions into clear, standalone \
search queries optimized for document retrieval.

Rules:
- Preserve the original meaning and intent exactly. Do not answer the question.
- Make implicit references explicit (e.g. resolve "it", "that", "the thing").
- Prefer concrete keywords and terms over vague phrasing.
- Output ONLY the rewritten query text — no explanation, no quotes, no preamble.
- If the original question is already clear and well-formed, return it unchanged.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _REWRITE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


@lru_cache
def _get_rewrite_llm() -> ChatGroq:
    """Cached so we don't reconstruct the Groq client on every request.
    Low temperature — this is a mechanical rewriting task, not a
    creative one, so we want consistent, literal output."""
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.query_rewrite_model,
        temperature=0.0,
    )


def rewrite_query(question: str) -> str:
    """
    Returns a retrieval-optimized rewrite of the question, or the
    original question unchanged if rewriting fails or the model
    returns something clearly unusable (empty output).
    """
    try:
        llm = _get_rewrite_llm()
        chain = _prompt | llm
        response = chain.invoke({"question": question})
        rewritten = response.content.strip()

        if not rewritten:
            logger.warning("Query rewrite returned empty output, using original question.")
            return question

        logger.info("Query rewritten: %r -> %r", question, rewritten)
        return rewritten

    except Exception:
        logger.exception("Query rewrite failed, falling back to original question.")
        return question