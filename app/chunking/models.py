from pydantic import BaseModel, Field

from app.ingestion.models import DocumentMetadata


class DocumentChunk(BaseModel):
    """A retrievable chunk derived from an enterprise document."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    metadata: DocumentMetadata