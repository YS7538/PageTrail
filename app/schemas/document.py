from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    page_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkOut(BaseModel):
    id: int
    page_number: int
    chunk_index: int
    token_count: int
    content: str

    model_config = {"from_attributes": True}
