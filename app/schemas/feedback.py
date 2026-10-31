from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackOut(BaseModel):
    id: int
    question_id: int
    rating: int
    comment: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
