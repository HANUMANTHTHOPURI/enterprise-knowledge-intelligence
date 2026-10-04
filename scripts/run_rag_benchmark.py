import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.core.bootstrap import build_rag_agent
from app.evaluation.rag_benchmark import RAGBenchmarkRunner
from app.evaluation.rag_evaluator import RAGEvaluator


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_queries.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rag_benchmark_report.json"
)


def main() -> None:
    print("Loading RAG evaluation queries...")

    queries = RAGEvaluator.load_queries(
        DATASET_PATH
    )

    print(
        f"Loaded {len(queries)} queries "
        f"from {DATASET_PATH}"
    )

    print("\nBuilding production RAG agent...")

    agent = build_rag_agent(
        retrieval_top_k=3
    )

    runner = RAGBenchmarkRunner(
        agent=agent
    )

    print("\nRunning end-to-end benchmark...")
    print(
        "Using real FAISS retrieval, reranking, "
        "evidence evaluation, LangGraph, and OpenAI.\n"
    )

    start_time = perf_counter()

    responses, metrics = runner.run(
        queries
    )

    elapsed_seconds = (
        perf_counter() - start_time
    )

    runner.save_report(
        path=REPORT_PATH,
        queries=queries,
        responses=responses,
        metrics=metrics,
    )

    print("=" * 70)
    print("PER-QUERY RESULTS")
    print("=" * 70)

    for query, response in zip(
        queries,
        responses,
        strict=True,
    ):
        result = RAGEvaluator.evaluate_response(
            query=query,
            response=response,
        )

        source_ids = [
            source.document_id
            for source in response.sources
        ]

        print(
            f"\n{query.query_id}"
            f"\n  Should answer:       {query.should_answer}"
            f"\n  System answered:     {response.sufficient_evidence}"
            f"\n  Decision correct:    {result.decision_correct}"
            f"\n  Citation correct:    {result.citation_correct}"
            f"\n  Abstention correct:  {result.abstention_correct}"
            f"\n  Keyword coverage:    {result.keyword_coverage:.4f}"
            f"\n  Sources:             {source_ids}"
        )

    print("\n" + "=" * 70)
    print("FINAL RAG BENCHMARK")
    print("=" * 70)

    print(
        f"Total queries:          "
        f"{metrics.total_queries}"
    )

    print(
        f"Supported queries:      "
        f"{metrics.supported_queries}"
    )

    print(
        f"Unsupported queries:    "
        f"{metrics.unsupported_queries}"
    )

    print(
        f"Decision accuracy:      "
        f"{metrics.decision_accuracy:.4f}"
    )

    print(
        f"Citation accuracy:      "
        f"{metrics.citation_accuracy:.4f}"
    )

    print(
        f"Abstention accuracy:    "
        f"{metrics.abstention_accuracy:.4f}"
    )

    print(
        f"Mean keyword coverage:  "
        f"{metrics.mean_keyword_coverage:.4f}"
    )

    print(
        f"Benchmark time:         "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        f"\nReport saved to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()