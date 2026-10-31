from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class RetrievedChunk(Base):
    __tablename__ = "retrieved_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
