from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedder import embed_query


def retrieve(
    db: Session,
    question: str,
    document_ids: list[int] | None,
    top_k: int,
) -> list[dict[str, int | str | float]]:
    query_vector = embed_query(question)
    distance = Chunk.embedding.cosine_distance(query_vector)
    statement = (
        select(
            Chunk.id,
            Chunk.document_id,
            Document.filename,
            Chunk.page_number,
            Chunk.content,
            distance.label("distance"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(top_k)
    )
    if document_ids is not None:
        statement = statement.where(Chunk.document_id.in_(document_ids))

    return [
        {
            "chunk_id": row.id,
            "document_id": row.document_id,
            "filename": row.filename,
            "page_number": row.page_number,
            "content": row.content,
            "score": round(1 - row.distance, 4),
        }
        for row in db.execute(statement)
    ]
