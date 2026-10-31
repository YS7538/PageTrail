import csv
from statistics import mean

import numpy as np
from sqlalchemy import select
from sentence_transformers import SentenceTransformer, models

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from scripts.eval_utils import ROOT, first_hit_rank, hit_at_k, load_eval_questions, mrr

MODEL_CONFIGS = [
    {"name": "bert-base-uncased", "query_prefix": ""},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": ""},
    {
        "name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
]
OUTPUT_PATH = ROOT / "data" / "results" / "encoder_comparison.csv"
CSV_COLUMNS = [
    "model",
    "max_seq_length",
    "truncated_chunk_pct",
    "answerable_count",
    "Hit@1",
    "Hit@3",
    "Hit@6",
    "MRR",
    "mean_unanswerable_top1_cosine",
]


def load_model(name: str) -> SentenceTransformer:
    if name == "bert-base-uncased":
        transformer = models.Transformer(name)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        return SentenceTransformer(modules=[transformer, pooling])
    return SentenceTransformer(name)


def main() -> None:
    with SessionLocal() as db:
        questions = load_eval_questions(db)
        document_ids = {item["document_id"] for item in questions}
        chunks_by_document: dict[int, list[dict]] = {document_id: [] for document_id in document_ids}
        rows = db.execute(
            select(Chunk.document_id, Chunk.content, Chunk.page_number)
            .where(Chunk.document_id.in_(document_ids))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
        for row in rows:
            chunks_by_document[row.document_id].append(
                {"content": row.content, "page_number": row.page_number}
            )

    output_rows = []
    for config in MODEL_CONFIGS:
        model = load_model(config["name"])
        max_seq_length = model.max_seq_length
        encoded_chunks = {}
        total_chunks = 0
        truncated_chunks = 0

        for document_id, chunks in chunks_by_document.items():
            texts = [chunk["content"] for chunk in chunks]
            if texts:
                vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
                encoded_chunks[document_id] = np.asarray(vectors)
                tokenized = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
                truncated_chunks += sum(len(tokens) > max_seq_length for tokens in tokenized)
            else:
                encoded_chunks[document_id] = np.empty((0, 0))
            total_chunks += len(chunks)

        query_texts = [config["query_prefix"] + item["question"] for item in questions]
        query_vectors = model.encode(query_texts, batch_size=32, normalize_embeddings=True)
        answerable_ranks = []
        unanswerable_top_scores = []

        for index, item in enumerate(questions):
            chunks = chunks_by_document[item["document_id"]]
            vectors = encoded_chunks[item["document_id"]]
            if not chunks:
                if item["answerable"]:
                    answerable_ranks.append(None)
                continue

            scores = vectors @ query_vectors[index]
            order = np.argsort(-scores, kind="stable")
            ranked = [chunks[chunk_index] for chunk_index in order]
            if item["answerable"]:
                answerable_ranks.append(first_hit_rank(ranked, set(item["expected_pages"])))
            else:
                unanswerable_top_scores.append(float(scores[order[0]]))

        answerable_count = sum(bool(item["answerable"]) for item in questions)
        truncated_pct = 100 * truncated_chunks / total_chunks if total_chunks else 0.0
        metrics = [
            mean(hit_at_k(rank, cutoff) for rank in answerable_ranks) if answerable_ranks else 0.0
            for cutoff in (1, 3, 6)
        ]
        reciprocal_rank = mean(mrr(rank) for rank in answerable_ranks) if answerable_ranks else 0.0
        mean_unanswerable = (
            mean(unanswerable_top_scores) if unanswerable_top_scores else None
        )
        output_rows.append(
            {
                "model": config["name"],
                "max_seq_length": max_seq_length,
                "truncated_chunk_pct": truncated_pct,
                "answerable_count": answerable_count,
                "Hit@1": metrics[0],
                "Hit@3": metrics[1],
                "Hit@6": metrics[2],
                "MRR": reciprocal_rank,
                "mean_unanswerable_top1_cosine": mean_unanswerable,
            }
        )
        print(
            f"{config['name']}: max_seq_length={max_seq_length}, "
            f"truncated_chunks={truncated_pct:.2f}%"
        )

    print("\nEncoder comparison")
    print("model                                           max_seq  truncated%  Hit@1  Hit@3  Hit@6  MRR    unanswerable_top1")
    for row in output_rows:
        print(
            f"{row['model']:<48}{row['max_seq_length']:<9}{row['truncated_chunk_pct']:<12.2f}"
            f"{row['Hit@1']:<7.3f}{row['Hit@3']:<7.3f}{row['Hit@6']:<7.3f}"
            f"{row['MRR']:<7.3f}{row['mean_unanswerable_top1_cosine']}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nWrote results to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
