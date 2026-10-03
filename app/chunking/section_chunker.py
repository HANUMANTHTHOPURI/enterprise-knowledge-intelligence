import re

from app.chunking.models import DocumentChunk
from app.ingestion.models import EnterpriseDocument


class SectionChunker:
    """Split structured enterprise documents using numbered section headings."""

    SECTION_PATTERN = re.compile(
        r"(?=^\d+\.\s+[A-Z][A-Z\s&/-]*$)",
        flags=re.MULTILINE,
    )

    def chunk(self, document: EnterpriseDocument) -> list[DocumentChunk]:
        """Split a document into section-aware chunks."""

        title, sections = self._split_sections(document.content)

        chunks = [
            DocumentChunk(
                chunk_id=(
                    f"{document.metadata.document_id}::chunk-{index:04d}"
                ),
                document_id=document.metadata.document_id,
                chunk_index=index,
                content=f"{title}\n\n{section}",
                metadata=document.metadata,
            )
            for index, section in enumerate(sections)
        ]

        return chunks

    def _split_sections(
        self,
        content: str,
    ) -> tuple[str, list[str]]:
        """Extract the document title and numbered sections."""

        parts = [
            part.strip()
            for part in self.SECTION_PATTERN.split(content)
            if part.strip()
        ]

        if len(parts) < 2:
            raise ValueError(
                "Document does not contain recognizable numbered sections"
            )

        title = parts[0]
        sections = parts[1:]

        return title, sections