from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document


def test_upload_pdf_creates_chunks_and_embeddings(client: TestClient, pdf_bytes: bytes) -> None:
    response = client.post("/documents", files={"file": ("paper.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 201
    result = response.json()
    assert result["page_count"] == 2
    with SessionLocal() as db:
        chunks = list(db.scalars(select(Chunk).where(Chunk.document_id == result["id"])))
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(len(chunk.embedding) == 384 for chunk in chunks)
    assert all(chunk.embedding_model == settings.embedding_model for chunk in chunks)


def test_txt_extension_is_rejected(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("paper.txt", b"text", "text/plain")})
    assert response.status_code == 400


def test_non_pdf_content_is_rejected(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("paper.pdf", b"not a PDF", "application/pdf")})
    assert response.status_code == 400


def test_oversized_upload_is_rejected(client: TestClient, pdf_bytes: bytes, monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    response = client.post("/documents", files={"file": ("paper.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 413


def test_pdf_without_text_leaves_no_file_or_row(client: TestClient) -> None:
    document = pymupdf.open()
    document.new_page()            # a page with no text; zero pages can't be saved
    blank_pdf = document.tobytes()
    document.close()
    response = client.post("/documents", files={"file": ("blank.pdf", blank_pdf, "application/pdf")})
    assert response.status_code == 422
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Document)) == 0
    assert not list(Path(settings.upload_dir).iterdir())


def test_documents_list_get_missing_and_delete(
    client: TestClient, upload_document
) -> None:
    created = upload_document()
    assert [item["id"] for item in client.get("/documents").json()] == [created["id"]]
    assert client.get(f"/documents/{created['id']}").json()["filename"] == "paper.pdf"
    assert client.get("/documents/999").status_code == 404
    path = next(Path(settings.upload_dir).glob(f"*_{created['filename']}"))
    assert client.delete(f"/documents/{created['id']}").status_code == 204
    assert not path.exists()
    with SessionLocal() as db:
        assert db.get(Document, created["id"]) is None
        assert db.scalar(select(func.count()).select_from(Chunk)) == 0
    assert client.delete(f"/documents/{created['id']}").status_code == 404
