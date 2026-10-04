import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agents.rag_graph import RAGAgent
from app.api.routes import router
from app.core.bootstrap import build_rag_agent
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


def create_app(
    agent: RAGAgent | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        if agent is not None:
            app.state.rag_agent = agent
        else:
            app.state.rag_agent = build_rag_agent()

        yield

    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Evidence-gated enterprise knowledge intelligence "
            "platform using RAG, FAISS, reranking, and LangGraph."
        ),
        lifespan=lifespan,
    )

    application.include_router(
        router,
        prefix="/api/v1",
    )

    return application


app = create_app()