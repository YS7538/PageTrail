from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.db.session import Base, engine
from app.models.document import Document
from app.models.feedback import Feedback
from app.models.question import Question


class EchoRequest(BaseModel):
    message: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/echo", response_model=EchoRequest)
def echo(request: EchoRequest) -> EchoRequest:
    return request
