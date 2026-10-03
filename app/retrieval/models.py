from pydantic import BaseModel, Field

from app.chunking.models import DocumentChunk


class RetrievalResult(BaseModel):
    """A ranked chunk returned by a retrieval system."""

    rank: int = Field(ge=1)
    score: float
    chunk: DocumentChunk