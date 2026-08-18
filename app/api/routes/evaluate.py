"""
Evaluation API route.

POST /evaluate scores an already-generated answer using RAGAS Faithfulness
and Answer Relevancy. Called on-demand from the frontend when a user
clicks "Check this answer" under a chat message — NOT run automatically
on every query, since each call costs a couple of extra Groq requests
(the judge calls) and takes a few seconds.

Kept thin like the other routes: request validation and error mapping
live here, the actual scoring logic lives in core/answer_evaluator.py.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.core.answer_evaluator import evaluate_answer
from app.models.schemas import EvaluateAnswerRequest, EvaluateAnswerResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluateAnswerResponse)
async def evaluate(request: EvaluateAnswerRequest) -> EvaluateAnswerResponse:
    try:
        result = await evaluate_answer(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
        )
    except Exception:
        # Covers Groq errors, RAGAS internal failures, etc. Evaluation is
        # a bonus feature, not core functionality — fail with a clear
        # message rather than a raw 500, so the frontend can show
        # something sensible instead of a generic crash.
        logger.exception("Failed to evaluate answer for question: %s", request.question)
        raise HTTPException(
            status_code=502,
            detail="Couldn't evaluate this answer right now. Please try again in a moment.",
        )

    return EvaluateAnswerResponse(**result)