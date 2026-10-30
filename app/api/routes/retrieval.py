from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.schemas.retrieval import RetrievalRequest
from app.services.retriever import retrieve

router = APIRouter()


@router.post("/retrieve")
def retrieve_chunks(
    request: RetrievalRequest, db: Session = Depends(get_db)
) -> list[dict[str, int | str | float]]:
    # temporary, replaced by /questions in Phase 5
    if request.document_ids is not None:
        existing_ids = set(
            db.scalars(select(Document.id).where(Document.id.in_(request.document_ids))).all()
        )
        if existing_ids != set(request.document_ids):
            raise HTTPException(status_code=404, detail="One or more documents not found")
    return retrieve(db, request.question, request.document_ids, request.top_k)
