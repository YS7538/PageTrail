import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import Document

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval_questions.json"


def load_eval_questions(db: Session) -> list[dict]:
    with EVAL_PATH.open(encoding="utf-8") as input_file:
        questions = json.load(input_file)

    filenames = {item["document"] for item in questions}
    document_ids = dict(
        db.execute(
            select(Document.filename, func.max(Document.id))
            .where(Document.filename.in_(filenames))
            .group_by(Document.filename)
        ).all()
    )
    resolved = []
    for item in questions:
        document_id = document_ids.get(item["document"])
        if document_id is None:
            raise ValueError(f"Document not found: {item['document']}")
        resolved.append({**item, "document_id": document_id})
    return resolved


def first_hit_rank(results: list[dict], expected_pages: set[int]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result["page_number"] in expected_pages:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def mrr(rank: int | None) -> float:
    return 1 / rank if rank is not None else 0.0
