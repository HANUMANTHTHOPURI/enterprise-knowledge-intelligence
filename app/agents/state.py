from typing import TypedDict

from app.generation.models import (
    ContextBundle,
    GeneratedAnswer,
    RAGResponse,
)
from app.retrieval.models import RetrievalResult


class RAGAgentState(TypedDict, total=False):
    """Shared state passed between LangGraph RAG nodes."""

    question: str

    retrieval_results: list[RetrievalResult]

    context_bundle: ContextBundle

    generated_answer: GeneratedAnswer

    final_response: RAGResponse