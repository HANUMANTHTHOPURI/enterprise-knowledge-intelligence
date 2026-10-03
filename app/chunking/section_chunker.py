import re

from app.chunking.models import DocumentChunk
from app.core.config import settings
from app.ingestion.models import EnterpriseDocument


class SectionChunker:
    """Split enterprise documents while preserving semantic sections."""

    SECTION_PATTERN = re.compile(
        r"(?=^\d+\.\s+[A-Z][A-Z\s&/-]*$)",
        flags=re.MULTILINE,
    )

    def __init__(
        self,
        max_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.max_chunk_size = (
            max_chunk_size
            if max_chunk_size is not None
            else settings.chunk_size
        )

        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )

        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than max_chunk_size"
            )

    def chunk(
        self,
        document: EnterpriseDocument,
    ) -> list[DocumentChunk]:
        """Split a document into deterministic retrieval chunks."""

        title, sections = self._split_sections(document.content)

        chunk_contents: list[str] = []

        for section in sections:
            section_content = f"{title}\n\n{section}"

            if len(section_content) <= self.max_chunk_size:
                chunk_contents.append(section_content)
                continue

            chunk_contents.extend(
                self._split_oversized_section(
                    title=title,
                    section=section,
                )
            )

        return [
            DocumentChunk(
                chunk_id=(
                    f"{document.metadata.document_id}"
                    f"::chunk-{index:04d}"
                ),
                document_id=document.metadata.document_id,
                chunk_index=index,
                content=content,
                metadata=document.metadata,
            )
            for index, content in enumerate(chunk_contents)
        ]

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

    def _split_oversized_section(
        self,
        title: str,
        section: str,
    ) -> list[str]:
        """Split an oversized section while retaining its context."""

        heading, separator, body = section.partition("\n")

        if not separator or not body.strip():
            raise ValueError(
                "Oversized section does not contain a body"
            )

        prefix = f"{title}\n\n{heading}\n\n"

        available_size = self.max_chunk_size - len(prefix)

        if available_size <= self.chunk_overlap:
            raise ValueError(
                "Chunk size is too small for the document context "
                "and configured overlap"
            )

        body_chunks = self._split_text_with_overlap(
            text=body.strip(),
            max_size=available_size,
            overlap=self.chunk_overlap,
        )

        return [
            f"{prefix}{body_chunk}"
            for body_chunk in body_chunks
        ]

    @staticmethod
    def _split_text_with_overlap(
        text: str,
        max_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping, whitespace-aware windows."""

        chunks: list[str] = []
        start = 0

        while start < len(text):
            proposed_end = min(
                start + max_size,
                len(text),
            )

            end = proposed_end

            if proposed_end < len(text):
                boundary = text.rfind(
                    " ",
                    start,
                    proposed_end,
                )

                minimum_boundary = start + (max_size // 2)

                if boundary >= minimum_boundary:
                    end = boundary

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            next_start = max(
                0,
                end - overlap,
            )

            if next_start > start:
                next_space = text.find(
                    " ",
                    next_start,
                    end,
                )

                if next_space != -1:
                    next_start = next_space + 1

            start = next_start

        return chunks