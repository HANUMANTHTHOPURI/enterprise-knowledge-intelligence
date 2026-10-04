import pytest

from app.chunking.models import DocumentChunk
from app.generation.context_builder import ContextBuilder
from app.ingestion.models import DocumentMetadata
from app.retrieval.models import RetrievalResult


def make_result(
    chunk_id: str,
    document_id: str,
    department: str,
    content: str,
    rank: int,
) -> RetrievalResult:
    metadata = DocumentMetadata(
        document_id=document_id,
        department=department,
        document_type="Policy",
        version="1.0",
        effective_date="2026-01-01",
        access_level="Internal",
        source="test_policy.txt",
    )

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=rank - 1,
        content=content,
        metadata=metadata,
    )

    return RetrievalResult(
        rank=rank,
        score=1.0 / rank,
        chunk=chunk,
    )


def test_context_builder_creates_sources():
    results = [
        make_result(
            chunk_id="TEST-001::chunk-0000",
            document_id="TEST-001",
            department="Testing",
            content="TEST POLICY\n\n1. PURPOSE\n\nTest content.",
            rank=1,
        )
    ]

    bundle = ContextBuilder().build(results)

    assert len(bundle.sources) == 1
    assert bundle.sources[0].source_number == 1
    assert bundle.sources[0].document_id == "TEST-001"
    assert bundle.sources[0].chunk_id == "TEST-001::chunk-0000"


def test_context_contains_source_boundaries_and_metadata():
    results = [
        make_result(
            chunk_id="TEST-001::chunk-0000",
            document_id="TEST-001",
            department="Testing",
            content="TEST POLICY\n\n1. PURPOSE\n\nTest content.",
            rank=1,
        )
    ]

    bundle = ContextBuilder().build(results)

    assert "[SOURCE 1]" in bundle.context
    assert "[/SOURCE 1]" in bundle.context
    assert "Document ID: TEST-001" in bundle.context
    assert "Department: Testing" in bundle.context
    assert "Chunk ID: TEST-001::chunk-0000" in bundle.context
    assert "Test content." in bundle.context


def test_context_preserves_retrieval_order():
    results = [
        make_result(
            chunk_id="TEST-001::chunk-0002",
            document_id="TEST-001",
            department="Testing",
            content="First ranked content.",
            rank=1,
        ),
        make_result(
            chunk_id="TEST-001::chunk-0001",
            document_id="TEST-001",
            department="Testing",
            content="Second ranked content.",
            rank=2,
        ),
    ]

    bundle = ContextBuilder().build(results)

    assert bundle.sources[0].chunk_id == "TEST-001::chunk-0002"
    assert bundle.sources[1].chunk_id == "TEST-001::chunk-0001"

    assert bundle.context.index("First ranked content.") < (
        bundle.context.index("Second ranked content.")
    )


def test_empty_results_raise_error():
    with pytest.raises(
        ValueError,
        match="At least one retrieval result is required",
    ):
        ContextBuilder().build([])