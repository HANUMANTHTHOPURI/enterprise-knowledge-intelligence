from app.embeddings.encoder import EmbeddingEncoder
from app.retrieval.models import RetrievalResult
from app.vectorstore.faiss_index import FAISSVectorIndex


class FAISSRetriever:
    """Retrieve enterprise chunks using a FAISS vector index."""

    def __init__(
        self,
        vector_index: FAISSVectorIndex,
        encoder: EmbeddingEncoder,
    ) -> None:
        if vector_index.dimension != encoder.dimension:
            raise ValueError(
                "Vector index dimension does not match encoder dimension"
            )

        self.vector_index = vector_index
        self.encoder = encoder

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Encode a query and retrieve the nearest indexed chunks."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = self.encoder.encode_query(query)

        matches = self.vector_index.search(
            query_embedding,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                rank=rank,
                score=score,
                chunk=chunk,
            )
            for rank, (chunk, score) in enumerate(
                matches,
                start=1,
            )
        ]