from datetime import datetime

from pydantic import BaseModel, Field

from app.db.models import IngestionStatus

#comments
class DocumentResponse(BaseModel):
    """What the API returns to describe a single uploaded document."""
    id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    status: IngestionStatus
    error_message: str | None = None
    chunk_count: int
    created_at: datetime
    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int

#Query / Chat
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Optional: restrict retrieval to specific documents rather than
    # the whole knowledge base. None/empty = search everything.
    document_ids: list[str] | None = None


class SourceChunk(BaseModel):
    """A single retrieved chunk that backed the answer — this is what
    lets the UI show 'Source: report.pdf, page 3' style citations."""
    document_id: str
    filename: str
    chunk_text: str
    chunk_index: int
    page_number: int | None = None
    relevance_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str

#Answer evaluation
class EvaluateAnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(
        ..., min_length=1, description="The source chunk texts the answer was generated from."
    )


class EvaluateAnswerResponse(BaseModel):
    """Scores are 0-1 (RAGAS convention), higher is better."""
    faithfulness: float
    answer_relevancy: float

class ErrorResponse(BaseModel):
    """Consistent error shape across every endpoint, so the frontend only needs one error-handling code path."""
    detail: str
    error_code: str | None = None