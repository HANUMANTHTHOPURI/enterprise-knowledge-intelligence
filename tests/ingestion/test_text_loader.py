from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ingestion.text_loader import TextDocumentLoader

TEST_DOCUMENT = Path("data/raw/employee_remote_work_policy.txt")


def test_load_valid_document():
    loader = TextDocumentLoader()

    document = loader.load(TEST_DOCUMENT)

    assert document.metadata.document_id == "HR-POL-001"
    assert document.metadata.department == "Human Resources"
    assert document.metadata.document_type == "Policy"
    assert document.metadata.version == "2.1"
    assert document.metadata.access_level == "Internal"

    assert "REMOTE WORK POLICY" in document.content
    assert "Eligible employees may work remotely up to three days per week." in (
        document.content
    )


def test_missing_document_raises_error():
    loader = TextDocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("data/raw/document_that_does_not_exist.txt")


def test_unsupported_file_type_raises_error(tmp_path):
    unsupported_file = tmp_path / "document.pdf"
    unsupported_file.write_text("Test document", encoding="utf-8")

    loader = TextDocumentLoader()

    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load(unsupported_file)


def test_empty_document_raises_error(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    loader = TextDocumentLoader()

    with pytest.raises(ValueError, match="Document is empty"):
        loader.load(empty_file)

def test_missing_required_metadata_raises_error(tmp_path):
    malformed_file = tmp_path / "malformed.txt"

    malformed_file.write_text(
        """
Document ID: HR-POL-999
Department: Human Resources

BROKEN POLICY

This document intentionally contains incomplete metadata.
""".strip(),
        encoding="utf-8",
    )

    loader = TextDocumentLoader()

    with pytest.raises(ValidationError):
        loader.load(malformed_file)