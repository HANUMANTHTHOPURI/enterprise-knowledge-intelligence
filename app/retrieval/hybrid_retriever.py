from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.models import RetrievalResult


class HybridRetriever:
    """Fuse dense and BM25 rankings using weighted reciprocal rank fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError(
                "Retriever weights cannot be negative"
            )

        if dense_weight + bm25_weight == 0:
            raise ValueError(
                "At least one retriever weight must be positive"
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return fused dense and lexical retrieval results."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        dense_results = self.dense_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        fused_scores: dict[str, float] = {}
        chunks_by_id = {}

        for result in dense_results:
            chunk_id = result.chunk.chunk_id

            chunks_by_id[chunk_id] = result.chunk

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + self.dense_weight
                / (self.rrf_k + result.rank)
            )

        for result in bm25_results:
            chunk_id = result.chunk.chunk_id

            chunks_by_id[chunk_id] = result.chunk

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + self.bm25_weight
                / (self.rrf_k + result.rank)
            )

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                fused_scores[chunk_id],
                chunk_id,
            ),
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                rank=rank,
                score=fused_scores[chunk_id],
                chunk=chunks_by_id[chunk_id],
            )
            for rank, chunk_id in enumerate(
                ranked_chunk_ids,
                start=1,
            )
        ]
    