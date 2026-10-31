import csv
from pathlib import Path
from statistics import mean

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.document import Document
from app.services.chunker import clean_pages, chunk_pages
from app.services.pdf_parser import extract_pages
from scripts.eval_utils import ROOT, first_hit_rank, hit_at_k, load_eval_questions, mrr

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_CONFIGS = ((160, 25), (320, 50), (480, 75))
OUTPUT_PATH = ROOT / "data" / "results" / "chunk_size_comparison.csv"
CSV_COLUMNS = ["chunk_words", "overlap_words", "chunk_count", "Hit@1", "Hit@3", "Hit@6", "MRR"]


def main() -> None:
    with SessionLocal() as db:
        questions = load_eval_questions(db)
        document_ids = {item["document_id"] for item in questions}
        document_paths = dict(
            db.execute(
                select(Document.id, Document.storage_path).where(Document.id.in_(document_ids))
            ).all()
        )

    model = SentenceTransformer(MODEL_NAME)
    query_texts = [QUERY_PREFIX + item["question"] for item in questions]
    query_vectors = model.encode(query_texts, batch_size=32, normalize_embeddings=True)
    output_rows = []

    for chunk_words, overlap_words in CHUNK_CONFIGS:
        chunks_by_document = {}
        vectors_by_document = {}
        total_chunks = 0

        for document_id, storage_path in document_paths.items():
            path = Path(storage_path)
            if not path.is_absolute():
                path = ROOT / path
            pages = clean_pages(extract_pages(path))
            chunks = chunk_pages(pages, chunk_words, overlap_words)
            chunks_by_document[document_id] = [
                {"page_number": page_number, "content": content}
                for page_number, content in chunks
            ]
            if chunks:
                vectors_by_document[document_id] = np.asarray(
                    model.encode(
                        [content for _, content in chunks],
                        batch_size=32,
                        normalize_embeddings=True,
                    )
                )
            else:
                vectors_by_document[document_id] = np.empty((0, 0))
            total_chunks += len(chunks)

        answerable_ranks = []
        for index, item in enumerate(questions):
            chunks = chunks_by_document[item["document_id"]]
            vectors = vectors_by_document[item["document_id"]]
            if not item["answerable"]:
                continue
            if not chunks:
                answerable_ranks.append(None)
                continue

            scores = vectors @ query_vectors[index]
            order = np.argsort(-scores, kind="stable")
            ranked = [chunks[chunk_index] for chunk_index in order]
            answerable_ranks.append(first_hit_rank(ranked, set(item["expected_pages"])))

        metrics = [
            mean(hit_at_k(rank, cutoff) for rank in answerable_ranks) if answerable_ranks else 0.0
            for cutoff in (1, 3, 6)
        ]
        reciprocal_rank = mean(mrr(rank) for rank in answerable_ranks) if answerable_ranks else 0.0
        output_rows.append(
            {
                "chunk_words": chunk_words,
                "overlap_words": overlap_words,
                "chunk_count": total_chunks,
                "Hit@1": metrics[0],
                "Hit@3": metrics[1],
                "Hit@6": metrics[2],
                "MRR": reciprocal_rank,
            }
        )

    print("Hits are counted per chunk, so different chunk sizes cover different amounts of text at the same k.")
    print("chunk_words  overlap  chunks  Hit@1  Hit@3  Hit@6  MRR")
    for row in output_rows:
        print(
            f"{row['chunk_words']:<13}{row['overlap_words']:<9}{row['chunk_count']:<8}"
            f"{row['Hit@1']:<7.3f}{row['Hit@3']:<7.3f}{row['Hit@6']:<7.3f}{row['MRR']:.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nWrote results to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
