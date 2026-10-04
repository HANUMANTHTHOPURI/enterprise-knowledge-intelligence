import json
from pathlib import Path

from app.evaluation.rag_models import (
    RAGEvaluationMetrics,
    RAGEvaluationQuery,
    RAGEvaluationResult,
)
from app.generation.models import RAGResponse


class RAGEvaluator:
    """Evaluate final RAG responses against a curated benchmark."""

    @staticmethod
    def load_queries(
        path: str | Path,
    ) -> list[RAGEvaluationQuery]:
        query_path = Path(path)

        if not query_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {query_path}"
            )

        with query_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_queries = json.load(file)

        return [
            RAGEvaluationQuery.model_validate(item)
            for item in raw_queries
        ]

    @staticmethod
    def evaluate_response(
        query: RAGEvaluationQuery,
        response: RAGResponse,
    ) -> RAGEvaluationResult:
        """Score one RAG response."""

        decision_correct = (
            response.sufficient_evidence
            == query.should_answer
        )

        citation_correct = (
            RAGEvaluator._citation_correct(
                query=query,
                response=response,
            )
        )

        abstention_correct = (
            RAGEvaluator._abstention_correct(
                query=query,
                response=response,
            )
        )

        keyword_coverage = (
            RAGEvaluator._keyword_coverage(
                query=query,
                response=response,
            )
        )

        return RAGEvaluationResult(
            query_id=query.query_id,
            decision_correct=decision_correct,
            citation_correct=citation_correct,
            abstention_correct=abstention_correct,
            keyword_coverage=keyword_coverage,
        )

    @staticmethod
    def aggregate(
        queries: list[RAGEvaluationQuery],
        results: list[RAGEvaluationResult],
    ) -> RAGEvaluationMetrics:
        """Aggregate per-query results into benchmark metrics."""

        if not queries:
            raise ValueError(
                "At least one evaluation query is required"
            )

        if len(queries) != len(results):
            raise ValueError(
                "Query and result counts must match"
            )

        total_queries = len(queries)

        supported_queries = sum(
            query.should_answer
            for query in queries
        )

        unsupported_queries = (
            total_queries - supported_queries
        )

        decision_accuracy = (
            sum(
                result.decision_correct
                for result in results
            )
            / total_queries
        )

        supported_results = [
            result
            for query, result in zip(
                queries,
                results,
                strict=True,
            )
            if query.should_answer
        ]

        unsupported_results = [
            result
            for query, result in zip(
                queries,
                results,
                strict=True,
            )
            if not query.should_answer
        ]

        citation_accuracy = (
            sum(
                result.citation_correct
                for result in supported_results
            )
            / len(supported_results)
            if supported_results
            else 1.0
        )

        abstention_accuracy = (
            sum(
                result.abstention_correct
                for result in unsupported_results
            )
            / len(unsupported_results)
            if unsupported_results
            else 1.0
        )

        mean_keyword_coverage = (
            sum(
                result.keyword_coverage
                for result in supported_results
            )
            / len(supported_results)
            if supported_results
            else 1.0
        )

        return RAGEvaluationMetrics(
            total_queries=total_queries,
            supported_queries=supported_queries,
            unsupported_queries=unsupported_queries,
            decision_accuracy=decision_accuracy,
            citation_accuracy=citation_accuracy,
            abstention_accuracy=abstention_accuracy,
            mean_keyword_coverage=mean_keyword_coverage,
        )

    @staticmethod
    def _citation_correct(
        query: RAGEvaluationQuery,
        response: RAGResponse,
    ) -> bool:
        if not query.should_answer:
            return len(response.sources) == 0

        if not response.sufficient_evidence:
            return False

        if not response.sources:
            return False

        actual_document_ids = {
            source.document_id
            for source in response.sources
        }

        expected_document_ids = set(
            query.expected_document_ids
        )

        return (
            bool(
                actual_document_ids
                & expected_document_ids
            )
            and actual_document_ids.issubset(
                expected_document_ids
            )
        )

    @staticmethod
    def _abstention_correct(
        query: RAGEvaluationQuery,
        response: RAGResponse,
    ) -> bool:
        if query.should_answer:
            return True

        return (
            response.sufficient_evidence is False
            and len(response.sources) == 0
        )

    @staticmethod
    def _keyword_coverage(
        query: RAGEvaluationQuery,
        response: RAGResponse,
    ) -> float:
        if not query.should_answer:
            return 1.0

        if not query.expected_keywords:
            return 1.0

        if not response.sufficient_evidence:
            return 0.0

        answer_text = response.answer.lower()

        matched_keywords = sum(
            keyword.lower() in answer_text
            for keyword in query.expected_keywords
        )

        return (
            matched_keywords
            / len(query.expected_keywords)
        )