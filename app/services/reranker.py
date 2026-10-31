from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    pairs = [(question, candidate["content"]) for candidate in candidates]
    scores = _model().predict(pairs)
    ranked = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    return sorted(ranked, key=lambda candidate: candidate["rerank_score"], reverse=True)
