import logging
from time import perf_counter

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from app.api.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadinessResponse,
)


logger = logging.getLogger(
    "enterprise_knowledge_intelligence.api"
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return basic application health information."""

    return HealthResponse(
        status="ok",
        service="enterprise-knowledge-intelligence",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
)
def readiness(
    request: Request,
) -> ReadinessResponse:
    """Report whether the RAG agent is initialized."""

    rag_agent_ready = hasattr(
        request.app.state,
        "rag_agent",
    )

    return ReadinessResponse(
        status=(
            "ready"
            if rag_agent_ready
            else "not_ready"
        ),
        rag_agent_ready=rag_agent_ready,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    payload: AskRequest,
    request: Request,
) -> AskResponse:
    """Answer an enterprise knowledge question."""

    start_time = perf_counter()

    agent = getattr(
        request.app.state,
        "rag_agent",
        None,
    )

    if agent is None:
        logger.error(
            "rag_request_failed reason=agent_not_initialized"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The enterprise knowledge service "
                "is not ready."
            ),
        )

    try:
        response = agent.invoke(
            payload.question
        )

    except Exception:
        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        logger.exception(
            (
                "rag_request_failed "
                "question_length=%d "
                "latency_ms=%.2f"
            ),
            len(payload.question),
            latency_ms,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The enterprise knowledge service "
                "is temporarily unavailable."
            ),
        ) from None

    latency_ms = (
        perf_counter() - start_time
    ) * 1000

    logger.info(
        (
            "rag_request_complete "
            "question_length=%d "
            "sufficient_evidence=%s "
            "source_count=%d "
            "latency_ms=%.2f"
        ),
        len(payload.question),
        response.sufficient_evidence,
        len(response.sources),
        latency_ms,
    )

    return AskResponse(
        question=response.question,
        answer=response.answer,
        sufficient_evidence=response.sufficient_evidence,
        sources=response.sources,
        latency_ms=latency_ms,
    )