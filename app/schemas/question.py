from datetime import datetime

from pydantic import BaseModel, Field


class QuestionCreate(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[int] | None = None


class CitationOut(BaseModel):
    document_id: int
    filename: str
    page_number: int
    chunk_id: int
    source_id: str
    snippet: str
    text: str


class QuestionResponse(BaseModel):
    question_id: int
    answer: str
    citations: list[CitationOut]
    latency_ms: int


class QuestionDetail(QuestionResponse):
    question_text: str
    generation_model: str | None
    created_at: datetime
