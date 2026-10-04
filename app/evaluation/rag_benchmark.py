import json
from pathlib import Path

from app.evaluation.rag_evaluator import RAGEvaluator
from app.evaluation.rag_models import (
    RAGEvaluationMetrics,
    RAGEvaluationQuery,
)
from app.generation.models import RAGResponse


class RAGBenchmarkRunner:
    """Run end-to-end RAG evaluation against a benchmark dataset."""

    def __init__(self, agent):
        self.agent = agent

    def run(
        self,
        queries: list[RAGEvaluationQuery],
    ) -> tuple[
        list[RAGResponse],
        RAGEvaluationMetrics,
    ]:
        """Run all benchmark queries through the RAG agent."""

        if not queries:
            raise ValueError(
                "At least one evaluation query is required"
            )

        responses: list[RAGResponse] = []
        results = []

        for query in queries:
            response = self.agent.invoke(
                query.question
            )

            evaluation_result = (
                RAGEvaluator.evaluate_response(
                    query=query,
                    response=response,
                )
            )

            responses.append(response)
            results.append(evaluation_result)

        metrics = RAGEvaluator.aggregate(
            queries=queries,
            results=results,
        )

        return responses, metrics

    @staticmethod
    def save_report(
        path: str | Path,
        queries: list[RAGEvaluationQuery],
        responses: list[RAGResponse],
        metrics: RAGEvaluationMetrics,
    ) -> None:
        """Persist benchmark metrics and query-level results."""

        if len(queries) != len(responses):
            raise ValueError(
                "Query and response counts must match"
            )

        report_path = Path(path)

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        query_results = []

        for query, response in zip(
            queries,
            responses,
            strict=True,
        ):
            result = RAGEvaluator.evaluate_response(
                query=query,
                response=response,
            )

            query_results.append(
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "should_answer": query.should_answer,
                    "answer": response.answer,
                    "sufficient_evidence": (
                        response.sufficient_evidence
                    ),
                    "source_document_ids": [
                        source.document_id
                        for source in response.sources
                    ],
                    "decision_correct": (
                        result.decision_correct
                    ),
                    "citation_correct": (
                        result.citation_correct
                    ),
                    "abstention_correct": (
                        result.abstention_correct
                    ),
                    "keyword_coverage": (
                        result.keyword_coverage
                    ),
                }
            )

        report = {
            "metrics": metrics.model_dump(
                mode="json"
            ),
            "query_results": query_results,
        }

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
            )