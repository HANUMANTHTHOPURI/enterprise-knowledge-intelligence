import json
import numpy as np
import pytest

from app.chunking.models import DocumentChunk
from app.ingestion.models import DocumentMetadata
from app.vectorstore.faiss_index import FAISSVectorIndex


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


def test_faiss_index_adds_vectors():
    index = FAISSVectorIndex(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk("TEST-DOC::chunk-0000", 0),
        make_chunk("TEST-DOC::chunk-0001", 1),
    ]

    index.add(embeddings, chunks)

    assert index.size == 2
    assert len(index.chunks) == 2


def test_faiss_index_returns_best_match():
    index = FAISSVectorIndex(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk("TEST-DOC::chunk-0000", 0),
        make_chunk("TEST-DOC::chunk-0001", 1),
    ]

    index.add(embeddings, chunks)

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    results = index.search(query, top_k=1)

    assert results[0][0].chunk_id == (
        "TEST-DOC::chunk-0000"
    )

    assert results[0][1] == pytest.approx(1.0)


def test_dimension_mismatch_raises_error():
    index = FAISSVectorIndex(dimension=3)

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    chunks = [
        make_chunk("TEST-DOC::chunk-0000", 0)
    ]

    with pytest.raises(
        ValueError,
        match="Embedding dimension does not match",
    ):
        index.add(embeddings, chunks)


def test_search_empty_index_raises_error():
    index = FAISSVectorIndex(dimension=3)

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Cannot search an empty FAISS index",
    ):
        index.search(query)

def test_faiss_index_save_and_load(tmp_path):
    index = FAISSVectorIndex(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk("TEST-DOC::chunk-0000", 0),
        make_chunk("TEST-DOC::chunk-0001", 1),
    ]

    index.add(embeddings, chunks)

    index_path = tmp_path / "test.faiss"
    metadata_path = tmp_path / "chunks.json"

    index.save(
        index_path,
        metadata_path,
    )

    loaded_index = FAISSVectorIndex.load(
        index_path,
        metadata_path,
    )

    assert loaded_index.dimension == 3
    assert loaded_index.size == 2
    assert len(loaded_index.chunks) == 2

    assert loaded_index.chunks[0].chunk_id == (
        "TEST-DOC::chunk-0000"
    )

    assert loaded_index.chunks[1].chunk_id == (
        "TEST-DOC::chunk-0001"
    )


def test_loaded_faiss_index_preserves_search_results(tmp_path):
    index = FAISSVectorIndex(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk("TEST-DOC::chunk-0000", 0),
        make_chunk("TEST-DOC::chunk-0001", 1),
    ]

    index.add(embeddings, chunks)

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    original_results = index.search(
        query,
        top_k=2,
    )

    index_path = tmp_path / "test.faiss"
    metadata_path = tmp_path / "chunks.json"

    index.save(
        index_path,
        metadata_path,
    )

    loaded_index = FAISSVectorIndex.load(
        index_path,
        metadata_path,
    )

    loaded_results = loaded_index.search(
        query,
        top_k=2,
    )

    original_ids = [
        chunk.chunk_id
        for chunk, _ in original_results
    ]

    loaded_ids = [
        chunk.chunk_id
        for chunk, _ in loaded_results
    ]

    assert loaded_ids == original_ids


def test_save_empty_index_raises_error(tmp_path):
    index = FAISSVectorIndex(dimension=3)

    with pytest.raises(
        ValueError,
        match="Cannot save an empty FAISS index",
    ):
        index.save(
            tmp_path / "empty.faiss",
            tmp_path / "empty.json",
        )


def test_load_detects_vector_metadata_mismatch(tmp_path):
    index = FAISSVectorIndex(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk("TEST-DOC::chunk-0000", 0),
        make_chunk("TEST-DOC::chunk-0001", 1),
    ]

    index.add(embeddings, chunks)

    index_path = tmp_path / "test.faiss"
    metadata_path = tmp_path / "chunks.json"

    index.save(
        index_path,
        metadata_path,
    )

    stored_metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata_path.write_text(
        json.dumps(
            stored_metadata[:1],
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="FAISS vector count does not match stored chunk metadata",
    ):
        FAISSVectorIndex.load(
            index_path,
            metadata_path,
        )