import csv
from statistics import mean, median
from time import perf_counter

from app.core.config import settings
from app.db.session import SessionLocal
from app.services.reranker import rerank
from app.services.retriever import retrieve
from scripts.eval_utils import ROOT, first_hit_rank, hit_at_k, load_eval_questions, mrr

OUTPUT_PATH = ROOT / "data" / "results" / "retrieval_eval.csv"
SWEEP_PATH = ROOT / "data" / "results" / "candidate_sweep.csv"
CSV_COLUMNS = [
    "question_index",
    "config",
    "first_hit_rank",
    "top1_page",
    "top1_vector_score",
    "top1_rerank_score",
    "hit_at_6",
]
SWEEP_CSV_COLUMNS = ["candidate_count", "Hit@1", "Hit@3", "Hit@6", "MRR", "median_rerank_ms"]
SWEEP_CANDIDATES = (10, 15, 20)


def main() -> None:
    rows = []
    answerable_results = {"baseline": [], "rerank": []}
    sweep_results = {count: [] for count in SWEEP_CANDIDATES}
    sweep_times = {count: [] for count in SWEEP_CANDIDATES}
    retrieval_times = []
    rerank_times = []

    with SessionLocal() as db:
        questions = load_eval_questions(db)
        if questions:
            warmup = questions[0]
            warmup_candidates = retrieve(
                db,
                warmup["question"],
                [warmup["document_id"]],
                top_k=max(settings.candidates, 20),
            )
            rerank(warmup["question"], warmup_candidates)

        for question_index, item in enumerate(questions, start=1):
            started_at = perf_counter()
            candidates = retrieve(
                db,
                item["question"],
                [item["document_id"]],
                top_k=max(settings.candidates, 20),
            )
            retrieval_times.append((perf_counter() - started_at) * 1000)

            started_at = perf_counter()
            reranked = rerank(item["question"], candidates)
            rerank_times.append((perf_counter() - started_at) * 1000)
            rerank_scores = {candidate["chunk_id"]: candidate["rerank_score"] for candidate in reranked}
            main_candidates = candidates[: settings.candidates]
            baseline = [
                {**candidate, "rerank_score": rerank_scores[candidate["chunk_id"]]}
                for candidate in main_candidates
            ]
            main_reranked = sorted(
                baseline, key=lambda candidate: candidate["rerank_score"], reverse=True
            )

            expected_pages = set(item["expected_pages"])
            for config, results in (("baseline", baseline), ("rerank", main_reranked)):
                rank = first_hit_rank(results, expected_pages) if item["answerable"] else None
                if item["answerable"]:
                    answerable_results[config].append(rank)
                top = results[0] if results else {}
                rows.append(
                    {
                        "question_index": question_index,
                        "config": config,
                        "first_hit_rank": rank,
                        "top1_page": top.get("page_number"),
                        "top1_vector_score": top.get("score"),
                        "top1_rerank_score": top.get("rerank_score"),
                        "hit_at_6": hit_at_k(rank, 6) if item["answerable"] else "",
                    }
                )

            for count in SWEEP_CANDIDATES:
                vector_subset = candidates[:count]
                if item["answerable"]:
                    ranked_subset = sorted(
                        vector_subset,
                        key=lambda candidate: rerank_scores[candidate["chunk_id"]],
                        reverse=True,
                    )
                    sweep_results[count].append(first_hit_rank(ranked_subset, expected_pages))
                started_at = perf_counter()
                rerank(item["question"], vector_subset)
                sweep_times[count].append((perf_counter() - started_at) * 1000)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    answerable_count = sum(bool(item["answerable"]) for item in questions)
    print("Per-question first-hit rank (- means absent or unanswerable)")
    print(f"{'Question':<12}{'Baseline':<14}{'Rerank':<14}")
    for question_index in range(1, len(questions) + 1):
        baseline_rank = rows[(question_index - 1) * 2]["first_hit_rank"]
        rerank_rank = rows[(question_index - 1) * 2 + 1]["first_hit_rank"]
        print(f"{question_index:<12}{str(baseline_rank or '-'):<14}{str(rerank_rank or '-'):<14}")

    print("\nSummary")
    print("config    Hit@1  Hit@3  Hit@5  Hit@6  Hit@20  MRR    retrieval_mean  retrieval_median  rerank_mean  rerank_median")
    for config in ("baseline", "rerank"):
        ranks = answerable_results[config]
        metrics = [
            mean(hit_at_k(rank, cutoff) for rank in ranks) if ranks else 0.0
            for cutoff in (1, 3, 5, 6, 20)
        ]
        print(
            f"{config:<10}{metrics[0]:<7.3f}{metrics[1]:<7.3f}{metrics[2]:<7.3f}"
            f"{metrics[3]:<7.3f}{metrics[4]:<8.3f}"
            f"{mean(mrr(rank) for rank in ranks) if ranks else 0.0:<7.3f}"
            f"{mean(retrieval_times) if retrieval_times else 0.0:<16.2f}"
            f"{median(retrieval_times) if retrieval_times else 0.0:<17.2f}"
            f"{mean(rerank_times) if rerank_times else 0.0:<13.2f}"
            f"{median(rerank_times) if rerank_times else 0.0:.2f}"
        )

    sweep_rows = []
    print("\nCandidate sweep")
    print("N    Hit@1  Hit@3  Hit@6  MRR    median_rerank_ms")
    for count, ranks in sweep_results.items():
        metrics = [mean(hit_at_k(rank, cutoff) for rank in ranks) if ranks else 0.0 for cutoff in (1, 3, 6)]
        reciprocal_rank = mean(mrr(rank) for rank in ranks) if ranks else 0.0
        median_time = median(sweep_times[count]) if sweep_times[count] else 0.0
        sweep_rows.append(
            {
                "candidate_count": count,
                "Hit@1": metrics[0],
                "Hit@3": metrics[1],
                "Hit@6": metrics[2],
                "MRR": reciprocal_rank,
                "median_rerank_ms": median_time,
            }
        )
        print(
            f"{count:<5}{metrics[0]:<7.3f}{metrics[1]:<7.3f}{metrics[2]:<7.3f}"
            f"{reciprocal_rank:<7.3f}{median_time:.2f}"
        )

    with SWEEP_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SWEEP_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Wrote candidate sweep to {SWEEP_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
