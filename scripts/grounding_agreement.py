import csv
import json
from pathlib import Path

from app.core.config import settings
from scripts.eval_utils import ROOT

RESULTS_DIR = ROOT / "data" / "results"
SUFFIX = "rerank" if settings.use_reranker else "vector"
AUDIT_PATH = RESULTS_DIR / f"grounding_audit_{SUFFIX}.csv"
CLAIMS_PATH = RESULTS_DIR / f"grounding_claims_{SUFFIX}.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def print_agreement(label: str, labeled: list[tuple[bool, bool]]) -> None:
    matches = sum(judge == human for judge, human in labeled)
    total = len(labeled)
    percent = f"{matches / total * 100:.1f}%" if total else "n/a"
    judge_supported = [(judge, human) for judge, human in labeled if judge]
    judge_unsupported = [(judge, human) for judge, human in labeled if not judge]
    false_positive = sum(judge and not human for judge, human in labeled)
    false_negative = sum(not judge and human for judge, human in labeled)

    def subset_agreement(rows: list[tuple[bool, bool]]) -> str:
        if not rows:
            return "0/0 (n/a)"
        count = sum(judge == human for judge, human in rows)
        return f"{count}/{len(rows)} ({count / len(rows) * 100:.1f}%)"

    print(label)
    print(f"  labeled: {total}")
    print(f"  agreement overall: {matches}/{total} ({percent})")
    print(f"  agreement where judge said supported: {subset_agreement(judge_supported)}")
    print(f"  agreement where judge said unsupported: {subset_agreement(judge_unsupported)}")
    print(f"  judge yes / human no: {false_positive}")
    print(f"  judge no / human yes: {false_negative}")


def main() -> None:
    audit_rows = read_rows(AUDIT_PATH)
    claims_by_id = {row["claim_id"]: row for row in read_rows(CLAIMS_PATH)}
    support_labels = []
    citation_labels = []

    for audit in audit_rows:
        claim = claims_by_id.get(audit["claim_id"])
        if claim is None:
            continue

        judge_support = parse_bool(claim["supported_by_any"])
        human_support = parse_bool(audit["human_supported_by_any"])
        if judge_support is not None and human_support is not None:
            support_labels.append((judge_support, human_support))

        try:
            cited_ids = json.loads(claim["cited_ids"] or "[]")
        except json.JSONDecodeError:
            cited_ids = []
        judge_citation = parse_bool(claim["supported_by_cited"])
        human_citation = parse_bool(audit["human_citation_ok"])
        if cited_ids and judge_citation is not None and human_citation is not None:
            citation_labels.append((judge_citation, human_citation))

    print_agreement("Supported by any passage", support_labels)
    print_agreement("Citation correctness (cited claims only)", citation_labels)


if __name__ == "__main__":
    main()
