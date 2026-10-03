from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """Ground-truth query used for retrieval evaluation."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)


class RetrievalMetrics(BaseModel):
    """Aggregate metrics for a retrieval evaluation run."""

    query_count: int = Field(ge=1)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr_at_k: float = Field(ge=0.0, le=1.0)