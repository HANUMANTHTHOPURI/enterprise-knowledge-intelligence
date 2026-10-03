import pytest

from app.chunking.models import DocumentChunk
from app.ingestion.models import DocumentMetadata
from app.retrieval.models import RetrievalResult
from app.retrieval.reranked_retriever import RerankedRetriever


def make_chunk(
    chunk_id: str,
    chunk_index: int,
) -> DocumentChunk:
    metadata = DocumentMetadata(
        document_id="TEST-DOC",
        department="Testing",
        document_type="Policy",
        version="1.0",
        effective_date="2026-01-01",
        access_level="Internal",
        source="test.txt",
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="TEST-DOC",
        chunk_index=chunk_index,
        content=f"TEST POLICY\n\n{chunk_id}",
        metadata=metadata,
    )


class FakeDenseRetriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        chunks = [
            make_chunk("TEST-DOC::chunk-0000", 0),
            make_chunk("TEST-DOC::chunk-0001", 1),
        ]

        return [
            RetrievalResult(
                rank=index + 1,
                score=0.9 - index,
                chunk=chunk,
            )
            for index, chunk in enumerate(chunks[:top_k])
        ]


class FakeReranker:
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        reversed_results = list(reversed(results))

        if top_k is not None:
            reversed_results = reversed_results[:top_k]

        return [
            RetrievalResult(
                rank=index + 1,
                score=float(len(reversed_results) - index),
                chunk=result.chunk,
            )
            for index, result in enumerate(reversed_results)
        ]


def test_reranked_retriever_reorders_candidates():
    retriever = RerankedRetriever(
        FakeDenseRetriever(),
        FakeReranker(),
        candidate_k=2,
    )

    results = retriever.search(
        "test query",
        top_k=2,
    )

    assert results[0].chunk.chunk_id == (
        "TEST-DOC::chunk-0001"
    )

    assert results[0].rank == 1


def test_empty_query_raises_error():
    retriever = RerankedRetriever(
        FakeDenseRetriever(),
        FakeReranker(),
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.search("")


def test_invalid_top_k_raises_error():
    retriever = RerankedRetriever(
        FakeDenseRetriever(),
        FakeReranker(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.search(
            "test query",
            top_k=0,
        )


def test_invalid_candidate_k_raises_error():
    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than zero",
    ):
        RerankedRetriever(
            FakeDenseRetriever(),
            FakeReranker(),
            candidate_k=0,
        )