from pydantic import BaseModel, Field


class RAGEvaluationQuery(BaseModel):
    """One question in the end-to-end RAG benchmark."""

    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    should_answer: bool

    expected_document_ids: list[str]
    expected_keywords: list[str]


class RAGEvaluationResult(BaseModel):
    """Evaluation result for one RAG response."""

    query_id: str

    decision_correct: bool
    citation_correct: bool
    abstention_correct: bool

    keyword_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )


class RAGEvaluationMetrics(BaseModel):
    """Aggregate end-to-end RAG evaluation metrics."""

    total_queries: int = Field(ge=0)

    supported_queries: int = Field(ge=0)
    unsupported_queries: int = Field(ge=0)

    decision_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    citation_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    abstention_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_keyword_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )