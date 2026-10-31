import re
from time import perf_counter

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.question import Question
from app.models.retrieved_chunk import RetrievedChunk
from app.services.generator import generate_answer
from app.services.reranker import rerank
from app.services.retriever import retrieve

_CITATION = re.compile(r"\[S(\d+)\]")
_NO_EVIDENCE = "The selected documents do not contain enough relevant evidence to answer this question."


class AnswerGenerationError(Exception):
    pass


def answer_question(
    db: Session, question: str, document_ids: list[int] | None
) -> tuple[Question, list[dict[str, int | str]]]:
    started_at = perf_counter()
    candidates = retrieve(db, question, document_ids, top_k=settings.candidates)
    enough_evidence = bool(candidates and candidates[0]["score"] >= settings.min_score)
    if enough_evidence:
        if settings.use_reranker:
            chunks = rerank(question, candidates)[: settings.top_k]
        else:
            chunks = candidates[: settings.top_k]
    else:
        chunks = []

    passages = [
        {**chunk, "source_id": f"S{rank}"}
        for rank, chunk in enumerate(chunks, start=1)
    ]
    used_model: str | None = None

    if not enough_evidence:
        answer = _NO_EVIDENCE
        cited_source_ids: set[str] = set()
    else:
        used_model = settings.generation_model
        try:
            answer = generate_answer(question, passages)
        except Exception as exc:
            raise AnswerGenerationError from exc

        valid_source_ids = {passage["source_id"] for passage in passages}
        cited_source_ids = set()

        def validate_citation(match: re.Match[str]) -> str:
            source_id = f"S{int(match.group(1))}"
            if source_id in valid_source_ids:
                cited_source_ids.add(source_id)
                return f"[{source_id}]"
            return ""

        answer = _CITATION.sub(validate_citation, answer)

    citations = [
        {
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "source_id": passage["source_id"],
            "snippet": chunk["content"][:300],
            "text": chunk["content"],
        }
        for passage, chunk in zip(passages, chunks, strict=True)
        if passage["source_id"] in cited_source_ids
    ]
    latency_ms = int((perf_counter() - started_at) * 1000)
    question_row = Question(
        question_text=question,
        answer_text=answer,
        latency_ms=latency_ms,
        generation_model=used_model,
    )

    try:
        db.add(question_row)
        db.flush()
        db.add_all(
            [
                RetrievedChunk(
                    question_id=question_row.id,
                    chunk_id=chunk["chunk_id"],
                    source_id=passage["source_id"],
                    rank=rank,
                    score=chunk["score"],
                    rerank_score=chunk.get("rerank_score"),
                    cited=passage["source_id"] in cited_source_ids,
                )
                for rank, (passage, chunk) in enumerate(zip(passages, chunks, strict=True), start=1)
            ]
        )
        db.commit()
        db.refresh(question_row)
    except Exception:
        db.rollback()
        raise

    return question_row, citations
