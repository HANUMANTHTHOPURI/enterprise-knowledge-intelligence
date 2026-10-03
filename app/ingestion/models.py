from datetime import date

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an enterprise knowledge document."""

    document_id: str
    department: str
    document_type: str
    version: str
    effective_date: date
    access_level: str
    source: str


class EnterpriseDocument(BaseModel):
    """Normalized representation of a document in the knowledge base."""

    content: str = Field(min_length=1)
    metadata: DocumentMetadata