from pathlib import Path

from app.ingestion.models import DocumentMetadata, EnterpriseDocument


class TextDocumentLoader:
    """Load structured enterprise text documents from disk."""

    METADATA_FIELDS = {
        "Document ID": "document_id",
        "Department": "department",
        "Document Type": "document_type",
        "Version": "version",
        "Effective Date": "effective_date",
        "Access Level": "access_level",
    }

    def load(self, file_path: str | Path) -> EnterpriseDocument:
        """Read, parse, and validate an enterprise text document."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower() != ".txt":
            raise ValueError(f"Unsupported file type: {path.suffix}")

        raw_text = path.read_text(encoding="utf-8")

        if not raw_text.strip():
            raise ValueError(f"Document is empty: {path}")

        metadata, content = self._parse_document(raw_text, path.name)

        return EnterpriseDocument(
            content=content,
            metadata=metadata,
        )

    def _parse_document(
        self,
        raw_text: str,
        source: str,
    ) -> tuple[DocumentMetadata, str]:
        """Separate document metadata from document content."""

        lines = raw_text.splitlines()

        metadata_values: dict[str, str] = {}
        content_start = 0

        for index, line in enumerate(lines):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            matched_field = False

            for label, field_name in self.METADATA_FIELDS.items():
                prefix = f"{label}:"

                if stripped_line.startswith(prefix):
                    value = stripped_line[len(prefix):].strip()
                    metadata_values[field_name] = value
                    matched_field = True
                    break

            if (
                not matched_field
                and metadata_values
                and len(metadata_values) == len(self.METADATA_FIELDS)
            ):
                content_start = index
                break

        metadata = DocumentMetadata(
            **metadata_values,
            source=source,
        )

        content = "\n".join(lines[content_start:]).strip()

        if not content:
            raise ValueError(f"No document content found in {source}")

        return metadata, content