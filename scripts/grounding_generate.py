import json
import re
import time
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.retrieved_chunk import RetrievedChunk
from app.services.rag_pipeline import AnswerGenerationError, answer_question
from scripts.eval_utils import ROOT, load_eval_questions

RESULTS_DIR = ROOT / "data" / "results"
SUFFIX = "rerank" if settings.use_reranker else "vector"
OUTPUT_PATH = RESULTS_DIR / f"grounding_answers_{SUFFIX}.jsonl"
RETRY_DELAYS = (20, 60, 120)


def print_call_error(question_index: int, attempt: int, error: Exception) -> None:
    cause = error.__cause__ or error
    message = str(cause)
    for secret in (settings.gemini_api_key, settings.app_api_key):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?is)\bheaders\b.*", "headers=[REDACTED]", message)
    message = re.sub(
        r"(?i)(authorization|x-goog-api-key|x-api-key)(?:['\"]?\s*[:=]\s*['\"]?)(?:bearer\s+)?[^'\"\s,}]+",
        r"\1=[REDACTED]",
        message,
    )
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", message)
    print(
        f"Generation attempt {attempt}/4 failed for question_index={question_index}: "
        f"{type(cause).__name__}: {message[:200]}"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_records: dict[int, dict[str, Any]] = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as input_file:
            for line in input_file:
                if line.strip():
                    record = json.loads(line)
                    if record.get("generation_model") != settings.generation_model:
                        print(
                            "Warning: existing grounding answers use a different or unknown "
                            "generation_model; stopping without appending."
                        )
                        return
                    completed_records[record["question_index"]] = record
    completed = set(completed_records)

    with SessionLocal() as db:
        questions = load_eval_questions(db)

    question_ids: list[int] = [
        record["question_id"] for record in completed_records.values()
    ]
    with OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
        for question_index, item in enumerate(questions, start=1):
            if question_index in completed:
                continue

            for attempt in range(1, len(RETRY_DELAYS) + 2):
                try:
                    with SessionLocal() as db:
                        question, _ = answer_question(
                            db, item["question"], [item["document_id"]]
                        )
                        statement = (
                            select(
                                RetrievedChunk.source_id,
                                RetrievedChunk.rank,
                                Chunk.page_number,
                                RetrievedChunk.cited,
                                Chunk.content,
                            )
                            .join(Chunk, Chunk.id == RetrievedChunk.chunk_id)
                            .where(RetrievedChunk.question_id == question.id)
                            .order_by(RetrievedChunk.rank)
                        )
                        passages = [
                            {
                                "source_id": row.source_id,
                                "rank": row.rank,
                                "page_number": row.page_number,
                                "cited": row.cited,
                                "text": row.content,
                            }
                            for row in db.execute(statement)
                        ]
                        record = {
                            "question_index": question_index,
                            "question": item["question"],
                            "answerable": item["answerable"],
                            "question_id": question.id,
                            "answer": question.answer_text,
                            "generation_model": settings.generation_model,
                            "passages": passages,
                            "gated": not passages,
                            "latency_ms": question.latency_ms,
                        }
                        if "on_topic" in item:
                            record["on_topic"] = item["on_topic"]
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_file.flush()
                        question_ids.append(question.id)
                    break
                except AnswerGenerationError as exc:
                    print_call_error(question_index, attempt, exc)
                    if attempt > len(RETRY_DELAYS):
                        print(
                            f"Generation failed for question_index={question_index} "
                            "after 4 attempts; skipping."
                        )
                        break
                    time.sleep(RETRY_DELAYS[attempt - 1])

            if question_index < len(questions):
                time.sleep(settings.eval_sleep_seconds)

    if question_ids:
        first_id = min(question_ids)
        last_id = max(question_ids)
        print(f"Created question IDs from {first_id} to {last_id}.")
        print(f"DELETE FROM questions WHERE id BETWEEN {first_id} AND {last_id};")
    else:
        print("No new question rows were created.")
    print(f"Wrote answers to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
