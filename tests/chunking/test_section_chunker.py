import pytest

from app.chunking.section_chunker import SectionChunker
from app.ingestion.corpus_loader import CorpusLoader
from app.ingestion.models import DocumentMetadata, EnterpriseDocument


def get_hr_document() -> EnterpriseDocument:
    documents = CorpusLoader().load("data/raw")

    return next(
        document
        for document in documents
        if document.metadata.document_id == "HR-POL-001"
    )


def test_hr_document_produces_seven_chunks():
    document = get_hr_document()

    chunks = SectionChunker().chunk(document)

    assert len(chunks) == 7


def test_chunk_ids_are_deterministic():
    document = get_hr_document()
    chunker = SectionChunker()

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    first_ids = [chunk.chunk_id for chunk in first_run]
    second_ids = [chunk.chunk_id for chunk in second_run]

    assert first_ids == second_ids

    assert first_ids[0] == "HR-POL-001::chunk-0000"
    assert first_ids[-1] == "HR-POL-001::chunk-0006"


def test_document_metadata_is_propagated_to_chunks():
    document = get_hr_document()

    chunks = SectionChunker().chunk(document)

    for chunk in chunks:
        assert chunk.document_id == "HR-POL-001"
        assert chunk.metadata.document_id == "HR-POL-001"
        assert chunk.metadata.department == "Human Resources"
        assert chunk.metadata.source == "employee_remote_work_policy.txt"


def test_document_title_is_preserved_in_every_chunk():
    document = get_hr_document()

    chunks = SectionChunker().chunk(document)

    for chunk in chunks:
        assert chunk.content.startswith("REMOTE WORK POLICY")


def test_section_heading_is_preserved():
    document = get_hr_document()

    chunks = SectionChunker().chunk(document)

    international_chunk = chunks[5]

    assert "6. INTERNATIONAL REMOTE WORK" in international_chunk.content
    assert "prior approval from Human Resources" in international_chunk.content


def test_entire_corpus_produces_35_chunks():
    documents = CorpusLoader().load("data/raw")
    chunker = SectionChunker()

    chunks = [
        chunk
        for document in documents
        for chunk in chunker.chunk(document)
    ]

    assert len(chunks) == 35


def test_unstructured_document_raises_error():
    metadata = DocumentMetadata(
        document_id="TEST-001",
        department="Testing",
        document_type="Policy",
        version="1.0",
        effective_date="2026-01-01",
        access_level="Internal",
        source="test.txt",
    )

    document = EnterpriseDocument(
        content="THIS DOCUMENT HAS NO NUMBERED SECTIONS",
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="does not contain recognizable numbered sections",
    ):
        SectionChunker().chunk(document)

def get_oversized_document() -> EnterpriseDocument:
    """Create a synthetic document with a deliberately oversized section."""

    metadata = DocumentMetadata(
        document_id="TEST-LARGE-001",
        department="Testing",
        document_type="Policy",
        version="1.0",
        effective_date="2026-01-01",
        access_level="Internal",
        source="large_test.txt",
    )

    body = " ".join(
        f"token{index:03d}"
        for index in range(80)
    )

    return EnterpriseDocument(
        content=(
            "TEST KNOWLEDGE POLICY\n\n"
            "1. LARGE SECTION\n\n"
            f"{body}"
        ),
        metadata=metadata,
    )


def test_oversized_section_is_split():
    document = get_oversized_document()

    chunker = SectionChunker(
        max_chunk_size=220,
        chunk_overlap=40,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 1


def test_oversized_chunks_respect_maximum_size():
    document = get_oversized_document()

    chunker = SectionChunker(
        max_chunk_size=220,
        chunk_overlap=40,
    )

    chunks = chunker.chunk(document)

    assert all(
        len(chunk.content) <= 220
        for chunk in chunks
    )


def test_oversized_chunks_preserve_context():
    document = get_oversized_document()

    chunker = SectionChunker(
        max_chunk_size=220,
        chunk_overlap=40,
    )

    chunks = chunker.chunk(document)

    for chunk in chunks:
        assert chunk.content.startswith(
            "TEST KNOWLEDGE POLICY"
        )

        assert "1. LARGE SECTION" in chunk.content


def test_oversized_chunks_preserve_overlap():
    document = get_oversized_document()

    chunker = SectionChunker(
        max_chunk_size=220,
        chunk_overlap=40,
    )

    chunks = chunker.chunk(document)

    prefix = (
        "TEST KNOWLEDGE POLICY\n\n"
        "1. LARGE SECTION\n\n"
    )

    first_body = chunks[0].content.removeprefix(prefix)
    second_body = chunks[1].content.removeprefix(prefix)

    first_tail = set(first_body.split()[-10:])
    second_start = set(second_body.split()[:10])

    assert first_tail.intersection(second_start)