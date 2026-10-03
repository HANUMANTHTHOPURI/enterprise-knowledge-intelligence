from pathlib import Path

from app.ingestion.models import EnterpriseDocument
from app.ingestion.text_loader import TextDocumentLoader


class CorpusLoader:
    """Load a collection of enterprise documents from a directory."""

    def __init__(self) -> None:
        self.text_loader = TextDocumentLoader()

    def load(self, directory: str | Path) -> list[EnterpriseDocument]:
        """Load all supported documents from a directory."""

        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {directory_path}"
            )

        if not directory_path.is_dir():
            raise NotADirectoryError(
                f"Corpus path is not a directory: {directory_path}"
            )

        document_paths = sorted(directory_path.glob("*.txt"))

        if not document_paths:
            raise ValueError(
                f"No supported documents found in: {directory_path}"
            )

        documents = [
            self.text_loader.load(path)
            for path in document_paths
        ]

        document_ids = [
            document.metadata.document_id
            for document in documents]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Duplicate document IDs detected in corpus")

        return documents