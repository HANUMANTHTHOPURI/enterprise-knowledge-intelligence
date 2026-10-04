import pytest

from app.agents.rag_graph import RAGAgent
from app.chunking.models import DocumentChunk
from app.generation.context_builder import ContextBuilder
from app.generation.models import GeneratedAnswer
from app.ingestion.models import DocumentMetadata
from app.retrieval.models import RetrievalResult


def make_result() -> RetrievalResult:
    metadata = DocumentMetadata(
        document_id="HR-POL-001",
        department="Human Resources",
        document_type="Policy",
        version="1.0",
        effective_date="2026-01-01",
        access_level="Internal",
        source="employee_remote_work_policy.txt",
    )

    chunk = DocumentChunk(
        chunk_id="HR-POL-001::chunk-0005",
        document_id="HR-POL-001",
        chunk_index=5,
        content=(
            "REMOTE WORK POLICY\n\n"
            "6. INTERNATIONAL REMOTE WORK\n\n"
            "International work requires prior approval."
        ),
        metadata=metadata,
    )

    return RetrievalResult(
        rank=1,
        score=0.9,
        chunk=chunk,
    )


class FakeRetriever:
    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        return [make_result()]


class FakeGroundedGenerator:
    def generate(
        self,
        question,
        context_bundle,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=(
                "International remote work requires prior approval."
            ),
            sufficient_evidence=True,
            cited_source_numbers=[1],
        )


class FakeAbstainingGenerator:
    def generate(
        self,
        question,
        context_bundle,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=(
                "The available company documents do not provide "
                "enough information."
            ),
            sufficient_evidence=False,
            cited_source_numbers=[],
        )


def test_graph_routes_to_grounded_response():
    agent = RAGAgent(
        retriever=FakeRetriever(),
        context_builder=ContextBuilder(),
        generator=FakeGroundedGenerator(),
    )

    response = agent.invoke(
        "Can I work from another country?"
    )

    assert response.sufficient_evidence is True
    assert len(response.sources) == 1
    assert response.sources[0].document_id == "HR-POL-001"


def test_graph_routes_to_abstention():
    agent = RAGAgent(
        retriever=FakeRetriever(),
        context_builder=ContextBuilder(),
        generator=FakeAbstainingGenerator(),
    )

    response = agent.invoke(
        "How many vacation days do employees receive?"
    )

    assert response.sufficient_evidence is False
    assert response.sources == []


def test_empty_question_raises_error():
    agent = RAGAgent(
        retriever=FakeRetriever(),
        context_builder=ContextBuilder(),
        generator=FakeGroundedGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        agent.invoke("")


def test_invalid_retrieval_top_k_raises_error():
    with pytest.raises(
        ValueError,
        match="retrieval_top_k must be greater than zero",
    ):
        RAGAgent(
            retriever=FakeRetriever(),
            context_builder=ContextBuilder(),
            generator=FakeGroundedGenerator(),
            retrieval_top_k=0,
        )