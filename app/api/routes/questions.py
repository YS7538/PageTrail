from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.feedback import Feedback
from app.models.question import Question
from app.models.retrieved_chunk import RetrievedChunk
from app.schemas.feedback import FeedbackCreate, FeedbackOut
from app.schemas.question import QuestionCreate, QuestionDetail, QuestionResponse
from app.services.rag_pipeline import AnswerGenerationError, answer_question

router = APIRouter()


@router.post("/questions", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED)
def create_question(
    request: QuestionCreate, db: Session = Depends(get_db)
) -> QuestionResponse:
    if request.document_ids is not None:
        existing_ids = set(
            db.scalars(select(Document.id).where(Document.id.in_(request.document_ids))).all()
        )
        if existing_ids != set(request.document_ids):
            raise HTTPException(status_code=404, detail="One or more documents not found")
    try:
        question, citations = answer_question(db, request.question, request.document_ids)
    except AnswerGenerationError:
        raise HTTPException(status_code=502, detail="Answer generation failed") from None
    return QuestionResponse(
        question_id=question.id,
        answer=question.answer_text or "",
        citations=citations,
        latency_ms=question.latency_ms or 0,
    )


@router.get("/questions/{question_id}", response_model=QuestionDetail)
def get_question(question_id: int, db: Session = Depends(get_db)) -> QuestionDetail:
    question = db.get(Question, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    statement = (
        select(
            RetrievedChunk.chunk_id,
            RetrievedChunk.source_id,
            Chunk.document_id,
            Document.filename,
            Chunk.page_number,
            Chunk.content,
        )
        .join(Chunk, Chunk.id == RetrievedChunk.chunk_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(RetrievedChunk.question_id == question_id, RetrievedChunk.cited.is_(True))
        .order_by(RetrievedChunk.rank)
    )
    citations = [
        {
            "document_id": row.document_id,
            "filename": row.filename,
            "page_number": row.page_number,
            "chunk_id": row.chunk_id,
            "source_id": row.source_id,
            "snippet": row.content[:300],
            "text": row.content,
        }
        for row in db.execute(statement)
    ]
    return QuestionDetail(
        question_id=question.id,
        answer=question.answer_text or "",
        citations=citations,
        latency_ms=question.latency_ms or 0,
        question_text=question.question_text,
        generation_model=question.generation_model,
        created_at=question.created_at,
    )


@router.post(
    "/questions/{question_id}/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback(
    question_id: int,
    request: FeedbackCreate,
    db: Session = Depends(get_db),
) -> Feedback:
    if db.get(Question, question_id) is None:
        raise HTTPException(status_code=404, detail="Question not found")
    feedback = Feedback(
        question_id=question_id,
        rating=request.rating,
        comment=request.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback
