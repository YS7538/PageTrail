import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import fitz
import pytest
from dotenv import dotenv_values


_env_file = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
test_database_url = os.environ.get("TEST_DATABASE_URL") or _env_file.get("TEST_DATABASE_URL")
database_url = os.environ.get("DATABASE_URL") or _env_file.get("DATABASE_URL")
if not test_database_url:
    pytest.exit("TEST_DATABASE_URL is required; configure a dedicated test database.", returncode=2)
if test_database_url == database_url:
    pytest.exit("TEST_DATABASE_URL must differ from DATABASE_URL; refusing to run tests.", returncode=2)

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
_upload_dir = Path(tempfile.mkdtemp(prefix=".researchpaper-tests-", dir=project_root))
os.environ["DATABASE_URL"] = test_database_url
os.environ["UPLOAD_DIR"] = str(_upload_dir)

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.session import Base, SessionLocal, engine
from app.main import app
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.feedback import Feedback
from app.models.question import Question
from app.models.retrieved_chunk import RetrievedChunk


@pytest.fixture(scope="session", autouse=True)
def database() -> Iterator[None]:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()
    shutil.rmtree(_upload_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_database(database: None) -> Iterator[None]:
    with engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE TABLE feedback, retrieved_chunks, questions, chunks, documents "
                "RESTART IDENTITY CASCADE"
            )
        )
    for path in _upload_dir.iterdir():
        if path.is_file():
            path.unlink()
    yield


@pytest.fixture
def client() -> Iterator[TestClient]:
    test_client = TestClient(app)
    yield test_client
    test_client.close()


def _vector(text_value: str) -> list[float]:
    vector = [0.0] * 384
    for word in re.findall(r"\w+", text_value.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % len(vector)
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude:
        vector = [value / magnitude for value in vector]
    return vector


@pytest.fixture(autouse=True)
def fake_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    import app.api.routes.documents as documents_route
    from app.core.config import settings
    import app.services.rag_pipeline as pipeline
    import app.services.retriever as retriever

    monkeypatch.setattr(settings, "min_score", 0.0)
    monkeypatch.setattr(
        documents_route,
        "embed_texts",
        lambda texts: [_vector(value) for value in texts],
    )

    def embed_query(question: str) -> list[float]:
        prefix = "Represent this sentence for searching relevant passages: "
        return _vector(question.removeprefix(prefix))

    monkeypatch.setattr(retriever, "embed_query", embed_query)
    monkeypatch.setattr(
        pipeline,
        "rerank",
        lambda question, candidates: sorted(
            candidates,
            key=lambda item: len(set(question.lower().split()) & set(item["content"].lower().split())),
            reverse=True,
        ),
    )
    state: dict[str, object] = {"answer": "Evidence supports this. [S1]", "calls": 0}

    def generate_answer(question: str, passages: list[dict]) -> str:
        state["calls"] = int(state["calls"]) + 1
        answer = state["answer"]
        if isinstance(answer, BaseException):
            raise answer
        return str(answer)

    monkeypatch.setattr(pipeline, "generate_answer", generate_answer)
    return state


@pytest.fixture
def pdf_bytes() -> bytes:
    document = fitz.open()
    for page_number in range(2):
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            (f"Page {page_number + 1} evidence about climate research and findings. " * 18),
        )
    content = document.tobytes()
    document.close()
    return content


@pytest.fixture
def upload_document(client: TestClient, pdf_bytes: bytes):
    def upload(filename: str = "paper.pdf", content: bytes | None = None) -> dict:
        response = client.post(
            "/documents",
            files={"file": (filename, content if content is not None else pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201, response.text
        return response.json()

    return upload
