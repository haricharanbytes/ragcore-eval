import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.rag_chain import generate_answer
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    user_id = settings.default_user_id  # single-user for now; see db/models.py note

    try:
        result = generate_answer(
            user_id=user_id,
            question=request.question,
            document_ids=request.document_ids,
        )
    except Exception:
        logger.exception("Failed to generate answer for question: %s", request.question)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer right now. Please try again in a moment.",
        )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        question=request.question,
    )