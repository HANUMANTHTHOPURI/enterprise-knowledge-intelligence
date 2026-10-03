import numpy as np

from app.chunking.models import DocumentChunk
from app.embeddings.encoder import EmbeddingEncoder
from app.retrieval.models import RetrievalResult


class DenseRetriever:
    """Retrieve enterprise chunks using dense embedding similarity."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        encoder: EmbeddingEncoder,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self.encoder = encoder

        self.embeddings = self.encoder.encode_texts(
            [chunk.content for chunk in chunks]
        )

        if self.embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                "Embedding count does not match chunk count"
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the highest-scoring chunks for a query."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.encoder.encode_query(query)

        scores = self.embeddings @ query_embedding

        result_count = min(
            top_k,
            len(self.chunks),
        )

        top_indices = np.argsort(scores)[::-1][:result_count]

        return [
            RetrievalResult(
                rank=rank,
                score=float(scores[index]),
                chunk=self.chunks[index],
            )
            for rank, index in enumerate(
                top_indices,
                start=1,
            )
        ]