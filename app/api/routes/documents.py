from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import require_api_key
from app.db.session import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.document import ChunkOut, DocumentOut
from app.services.chunker import chunk_pages, clean_pages
from app.services.embedder import embed_texts
from app.services.pdf_parser import extract_pages

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> Document:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must have a .pdf filename")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the maximum upload size")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not have PDF magic bytes")

    destination = settings.upload_dir / f"{uuid4().hex}_{filename}"
    destination.write_bytes(content)
    try:
        try:
            pages = extract_pages(destination)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if sum(len(text) for _, text in pages) < 200:
            raise HTTPException(
                status_code=422,
                detail="No extractable text. Scanned PDFs are not supported yet.",
            )

        chunks = chunk_pages(clean_pages(pages))
        embeddings = embed_texts([chunk_text for _, chunk_text in chunks])
        document = Document(
            filename=filename,
            storage_path=str(destination),
            status="processed",
            page_count=len(pages),
        )
        db.add(document)
        db.flush()
        db.add_all(
            [
                Chunk(
                    document_id=document.id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    content=chunk_text,
                    token_count=len(chunk_text.split()),  # Word count is a rough token-count proxy.
                    embedding=embedding,
                    embedding_model=settings.embedding_model,
                )
                for chunk_index, ((page_number, chunk_text), embedding) in enumerate(
                    zip(chunks, embeddings, strict=True)
                )
            ]
        )
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise
    return document


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.id)).all())


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkOut])
def list_document_chunks(document_id: int, db: Session = Depends(get_db)) -> list[Chunk]:
    # temporary, for inspection
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    statement = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    return list(db.scalars(statement).all())


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    db.commit()
    Path(document.storage_path).unlink(missing_ok=True)
