from app.generation.models import ContextBundle, ContextSource
from app.retrieval.models import RetrievalResult


class ContextBuilder:
    """Convert ranked retrieval results into grounded LLM context."""

    def build(
        self,
        results: list[RetrievalResult],
    ) -> ContextBundle:
        """Build structured context from retrieved enterprise chunks."""

        if not results:
            raise ValueError(
                "At least one retrieval result is required"
            )

        context_blocks: list[str] = []
        sources: list[ContextSource] = []

        for source_number, result in enumerate(
            results,
            start=1,
        ):
            chunk = result.chunk
            metadata = chunk.metadata

            sources.append(
                ContextSource(
                    source_number=source_number,
                    document_id=metadata.document_id,
                    chunk_id=chunk.chunk_id,
                    department=metadata.department,
                    document_type=metadata.document_type,
                    source_file=metadata.source,
                )
            )

            context_blocks.append(
                "\n".join(
                    [
                        f"[SOURCE {source_number}]",
                        f"Document ID: {metadata.document_id}",
                        f"Department: {metadata.department}",
                        f"Document Type: {metadata.document_type}",
                        f"Chunk ID: {chunk.chunk_id}",
                        "",
                        chunk.content,
                        f"[/SOURCE {source_number}]",
                    ]
                )
            )

        return ContextBundle(
            context="\n\n".join(context_blocks),
            sources=sources,
        )