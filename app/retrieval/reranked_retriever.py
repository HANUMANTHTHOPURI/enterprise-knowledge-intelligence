from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.models import RetrievalResult


class RerankedRetriever:
    """Retrieve dense candidates and rerank them with a cross-encoder."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: CrossEncoderReranker,
        candidate_k: int = 5,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve candidates and return cross-encoder-ranked results."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        candidate_count = max(
            self.candidate_k,
            top_k,
        )

        candidates = self.dense_retriever.search(
            query,
            top_k=candidate_count,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )