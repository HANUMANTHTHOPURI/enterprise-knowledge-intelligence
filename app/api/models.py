from pydantic import BaseModel, Field, field_validator


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


class HealthResponse(BaseModel):
    """Basic API health status."""

    status: str
    service: str