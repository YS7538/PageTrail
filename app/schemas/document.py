from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    page_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}
