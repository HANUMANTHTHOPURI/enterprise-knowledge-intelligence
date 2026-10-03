from sentence_transformers import CrossEncoder

from app.retrieval.models import RetrievalResult


class CrossEncoderReranker:
    """Rerank retrieved chunks using joint query-document relevance."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using cross-encoder relevance scores."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            raise ValueError(
                "At least one retrieval result is required"
            )

        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        pairs = [
            (query, result.chunk.content)
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            RetrievalResult(
                rank=rank,
                score=float(score),
                chunk=result.chunk,
            )
            for rank, (result, score) in enumerate(
                ranked,
                start=1,
            )
        ]