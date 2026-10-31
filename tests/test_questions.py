from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.document import Document
from app.models.question import Question
from app.models.retrieved_chunk import RetrievedChunk


def ask(client: TestClient, question: str, document_ids: list[int] | None = None):
    return client.post("/questions", json={"question": question, "document_ids": document_ids})


def test_question_retrieves_only_selected_documents(client: TestClient, upload_document) -> None:
    first = upload_document("first.pdf")
    upload_document("second.pdf")
    response = ask(client, "climate research findings", [first["id"]])
    assert response.status_code == 201
    assert response.json()["citations"]
    assert {citation["document_id"] for citation in response.json()["citations"]} == {first["id"]}


def test_question_unknown_document_returns_404(client: TestClient) -> None:
    assert ask(client, "question", [999]).status_code == 404


def test_invalid_citation_is_removed_and_valid_citations_map_to_chunks(
    client: TestClient, upload_document, fake_models
) -> None:
    document = upload_document()
    fake_models["answer"] = "Supported [S1] and invalid [S99]."
    response = ask(client, "climate research findings", [document["id"]])
    result = response.json()
    assert response.status_code == 201
    assert "[S99]" not in result["answer"]
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    with SessionLocal() as db:
        from app.models.chunk import Chunk

        chunk = db.get(Chunk, citation["chunk_id"])
        assert chunk is not None
        assert chunk.document_id == citation["document_id"]
        assert chunk.page_number == citation["page_number"]


def test_insufficient_evidence_skips_generation(
    client: TestClient, upload_document, fake_models, monkeypatch
) -> None:
    from app.core.config import settings

    upload_document()
    monkeypatch.setattr(settings, "min_score", 1.1)
    monkeypatch.setitem(fake_models, "answer", AssertionError("generator must not run"))
    response = ask(client, "climate research findings")
    assert response.status_code == 201
    assert response.json()["answer"] == (
        "The selected documents do not contain enough relevant evidence to answer this question."
    )
    assert response.json()["citations"] == []
    assert fake_models["calls"] == 0


def test_generation_failure_returns_502_without_saving_question(
    client: TestClient, upload_document, fake_models
) -> None:
    upload_document()
    fake_models["answer"] = RuntimeError("private failure details")
    response = ask(client, "climate research findings")
    assert response.status_code == 502
    assert "private failure details" not in response.text
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Question)) == 0


def test_question_get_persists_answer_citations_and_document_cascade(
    client: TestClient, upload_document
) -> None:
    document = upload_document()
    response = ask(client, "climate research findings", [document["id"]])
    question_id = response.json()["question_id"]
    detail = client.get(f"/questions/{question_id}").json()
    assert detail["answer"] == "Evidence supports this. [S1]"
    assert detail["citations"] == response.json()["citations"]
    assert client.delete(f"/documents/{document['id']}").status_code == 204
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(RetrievedChunk)) == 0
