import pytest

from app.chunking.models import DocumentChunk
from app.evaluation.models import RetrievalQuery
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.models import DocumentMetadata
from app.retrieval.models import RetrievalResult


def make_chunk(chunk_id: str) -> DocumentChunk:
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
        chunk_index=0,
        content="TEST POLICY\n\n1. TEST SECTION\n\nTest content.",
        metadata=metadata,
    )


class FakeRetriever:
    """Simple deterministic retriever used for evaluation tests."""

    def __init__(
        self,
        results_by_query: dict[str, list[RetrievalResult]],
    ) -> None:
        self.results_by_query = results_by_query

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return self.results_by_query[query][:top_k]


def test_evaluator_perfect_ranking():
    relevant_chunk = make_chunk("TEST-DOC::chunk-0000")

    retriever = FakeRetriever(
        {
            "test query": [
                RetrievalResult(
                    rank=1,
                    score=0.9,
                    chunk=relevant_chunk,
                )
            ]
        }
    )

    evaluator = RetrievalEvaluator(retriever)

    queries = [
        RetrievalQuery(
            query_id="Q001",
            query="test query",
            relevant_chunk_ids=["TEST-DOC::chunk-0000"],
        )
    ]

    metrics = evaluator.evaluate(
        queries,
        top_k=5,
    )

    assert metrics.query_count == 1
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 1.0


def test_evaluator_reciprocal_rank():
    irrelevant_chunk = make_chunk(
        "TEST-DOC::chunk-0000"
    )
    relevant_chunk = make_chunk(
        "TEST-DOC::chunk-0001"
    )

    retriever = FakeRetriever(
        {
            "test query": [
                RetrievalResult(
                    rank=1,
                    score=0.9,
                    chunk=irrelevant_chunk,
                ),
                RetrievalResult(
                    rank=2,
                    score=0.8,
                    chunk=relevant_chunk,
                ),
            ]
        }
    )

    evaluator = RetrievalEvaluator(retriever)

    queries = [
        RetrievalQuery(
            query_id="Q001",
            query="test query",
            relevant_chunk_ids=["TEST-DOC::chunk-0001"],
        )
    ]

    metrics = evaluator.evaluate(
        queries,
        top_k=5,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 0.5


def test_evaluator_handles_missed_result():
    irrelevant_chunk = make_chunk(
        "TEST-DOC::chunk-0000"
    )

    retriever = FakeRetriever(
        {
            "test query": [
                RetrievalResult(
                    rank=1,
                    score=0.9,
                    chunk=irrelevant_chunk,
                )
            ]
        }
    )

    evaluator = RetrievalEvaluator(retriever)

    queries = [
        RetrievalQuery(
            query_id="Q001",
            query="test query",
            relevant_chunk_ids=["TEST-DOC::chunk-9999"],
        )
    ]

    metrics = evaluator.evaluate(
        queries,
        top_k=5,
    )

    assert metrics.recall_at_k == 0.0
    assert metrics.mrr_at_k == 0.0


def test_empty_evaluation_queries_raise_error():
    evaluator = RetrievalEvaluator(
        FakeRetriever({})
    )

    with pytest.raises(
        ValueError,
        match="At least one evaluation query is required",
    ):
        evaluator.evaluate([])


def test_invalid_top_k_raises_error():
    evaluator = RetrievalEvaluator(
        FakeRetriever({})
    )

    query = RetrievalQuery(
        query_id="Q001",
        query="test query",
        relevant_chunk_ids=["TEST-DOC::chunk-0000"],
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        evaluator.evaluate(
            [query],
            top_k=0,
        )