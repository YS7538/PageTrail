from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(texts, batch_size=32, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(question: str) -> list[float]:
    query = "Represent this sentence for searching relevant passages: " + question
    vector = _model().encode([query], normalize_embeddings=True)[0]
    return vector.tolist()
