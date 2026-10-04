from pydantic import BaseModel, Field, field_validator

from app.generation.models import ContextSource


class AskRequest(BaseModel):
    """Question submitted to the enterprise knowledge agent."""

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Question cannot be empty"
            )

        return cleaned


class AskResponse(BaseModel):
    """HTTP response returned by the enterprise RAG API."""

    question: str
    answer: str
    sufficient_evidence: bool
    sources: list[ContextSource]
    latency_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    """Basic API health status."""

    status: str
    service: str


class ReadinessResponse(BaseModel):
    """Application readiness status."""

    status: str
    rag_agent_ready: bool