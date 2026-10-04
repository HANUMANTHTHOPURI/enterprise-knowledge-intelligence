from pydantic import BaseModel, Field


class ContextSource(BaseModel):
    """Source metadata exposed to the generation layer."""

    source_number: int = Field(ge=1)
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    department: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    source_file: str = Field(min_length=1)


class ContextBundle(BaseModel):
    """Structured evidence supplied to the language model."""

    context: str = Field(min_length=1)
    sources: list[ContextSource] = Field(min_length=1)
    