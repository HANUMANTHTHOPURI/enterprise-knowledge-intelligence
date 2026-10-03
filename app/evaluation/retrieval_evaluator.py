import json
from pathlib import Path

from app.evaluation.models import RetrievalMetrics, RetrievalQuery
from app.retrieval.dense_retriever import DenseRetriever


class RetrievalEvaluator:
    """Evaluate ranked retrieval against ground-truth chunk IDs."""

    def __init__(
        self,
        retriever: DenseRetriever,
    ) -> None:
        self.retriever = retriever

    @staticmethod
    def load_queries(
        file_path: str | Path,
    ) -> list[RetrievalQuery]:
        """Load and validate retrieval queries from JSON."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation file not found: {path}"
            )

        raw_data = json.loads(
            path.read_text(encoding="utf-8")
        )

        return [
            RetrievalQuery.model_validate(item)
            for item in raw_data
        ]

    def evaluate(
        self,
        queries: list[RetrievalQuery],
        top_k: int = 5,
    ) -> RetrievalMetrics:
        """Compute Recall@K and MRR@K."""

        if not queries:
            raise ValueError(
                "At least one evaluation query is required"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        hits = 0
        reciprocal_rank_sum = 0.0

        for evaluation_query in queries:
            results = self.retriever.search(
                evaluation_query.query,
                top_k=top_k,
            )

            relevant_ids = set(
                evaluation_query.relevant_chunk_ids
            )

            relevant_ranks = [
                result.rank
                for result in results
                if result.chunk.chunk_id in relevant_ids
            ]

            if relevant_ranks:
                hits += 1

                first_relevant_rank = min(
                    relevant_ranks
                )

                reciprocal_rank_sum += (
                    1.0 / first_relevant_rank
                )

        query_count = len(queries)

        return RetrievalMetrics(
            query_count=query_count,
            recall_at_k=hits / query_count,
            mrr_at_k=(
                reciprocal_rank_sum / query_count
            ),
        )