import json

import pytest

from app.evaluation.rag_benchmark import RAGBenchmarkRunner
from app.evaluation.rag_models import RAGEvaluationQuery
from app.generation.models import (
    ContextSource,
    RAGResponse,
)


class FakeBenchmarkAgent:
    """Deterministic agent used for benchmark tests."""

    def invoke(
        self,
        question: str,
    ) -> RAGResponse:
        if "vacation" in question.lower():
            return RAGResponse(
                question=question,
                answer=(
                    "The available company documents do not "
                    "provide enough information to answer "
                    "this question."
                ),
                sufficient_evidence=False,
                sources=[],
            )

        return RAGResponse(
            question=question,
            answer=(
                "International remote work requires "
                "Human Resources approval."
            ),
            sufficient_evidence=True,
            sources=[
                ContextSource(
                    source_number=1,
                    document_id="HR-POL-001",
                    chunk_id=(
                        "HR-POL-001::chunk-0005"
                    ),
                    department="Human Resources",
                    document_type="Policy",
                    source_file=(
                        "employee_remote_work_policy.txt"
                    ),
                )
            ],
        )


def build_queries():
    return [
        RAGEvaluationQuery(
            query_id="TEST001",
            question=(
                "Who approves international remote work?"
            ),
            should_answer=True,
            expected_document_ids=[
                "HR-POL-001"
            ],
            expected_keywords=[
                "Human Resources"
            ],
        ),
        RAGEvaluationQuery(
            query_id="TEST002",
            question=(
                "How many vacation days are provided?"
            ),
            should_answer=False,
            expected_document_ids=[],
            expected_keywords=[],
        ),
    ]


def test_benchmark_runner():
    runner = RAGBenchmarkRunner(
        agent=FakeBenchmarkAgent()
    )

    queries = build_queries()

    responses, metrics = runner.run(
        queries
    )

    assert len(responses) == 2

    assert metrics.total_queries == 2
    assert metrics.supported_queries == 1
    assert metrics.unsupported_queries == 1

    assert metrics.decision_accuracy == 1.0
    assert metrics.citation_accuracy == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.mean_keyword_coverage == 1.0


def test_benchmark_runner_rejects_empty_queries():
    runner = RAGBenchmarkRunner(
        agent=FakeBenchmarkAgent()
    )

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        runner.run([])


def test_benchmark_report_is_saved(
    tmp_path,
):
    runner = RAGBenchmarkRunner(
        agent=FakeBenchmarkAgent()
    )

    queries = build_queries()

    responses, metrics = runner.run(
        queries
    )

    output_path = (
        tmp_path
        / "rag_benchmark_report.json"
    )

    runner.save_report(
        path=output_path,
        queries=queries,
        responses=responses,
        metrics=metrics,
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    assert report["metrics"][
        "decision_accuracy"
    ] == 1.0

    assert len(
        report["query_results"]
    ) == 2

    assert report["query_results"][0][
        "query_id"
    ] == "TEST001"