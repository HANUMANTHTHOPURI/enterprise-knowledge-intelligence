from fastapi import APIRouter, Request

from app.api.models import AskRequest, HealthResponse
from app.generation.models import RAGResponse


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return application health information."""

    return HealthResponse(
        status="ok",
        service="enterprise-knowledge-intelligence",
    )


@router.post(
    "/ask",
    response_model=RAGResponse,
)
def ask(
    payload: AskRequest,
    request: Request,
) -> RAGResponse:
    """Answer an enterprise knowledge question."""

    agent = request.app.state.rag_agent

    return agent.invoke(
        payload.question
    )