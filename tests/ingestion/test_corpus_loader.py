from pathlib import Path

import pytest

from app.ingestion.corpus_loader import CorpusLoader

CORPUS_DIRECTORY = Path("data/raw")


def test_load_enterprise_corpus():
    loader = CorpusLoader()

    documents = loader.load(CORPUS_DIRECTORY)

    assert len(documents) == 5

    document_ids = {
        document.metadata.document_id
        for document in documents
    }

    assert document_ids == {
        "HR-POL-001",
        "SEC-POL-001",
        "FIN-POL-001",
        "IT-POL-001",
        "LEG-POL-001",
    }


def test_document_ids_are_unique():
    loader = CorpusLoader()

    documents = loader.load(CORPUS_DIRECTORY)

    document_ids = [
        document.metadata.document_id
        for document in documents
    ]

    assert len(document_ids) == len(set(document_ids))


def test_missing_corpus_directory_raises_error():
    loader = CorpusLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("data/directory_that_does_not_exist")


def test_empty_corpus_directory_raises_error(tmp_path):
    loader = CorpusLoader()

    with pytest.raises(ValueError, match="No supported documents found"):
        loader.load(tmp_path)