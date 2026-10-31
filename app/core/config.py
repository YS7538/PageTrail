from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 20
    database_url: str
    app_api_key: str | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    gemini_api_key: str = ""
    generation_model: str = "gemini-2.5-flash"
    top_k: int = 6
    min_score: float = 0.55
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 20
    use_reranker: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
