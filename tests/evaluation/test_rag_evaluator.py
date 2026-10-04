from app.evaluation.rag_evaluator import RAGEvaluator
from app.evaluation.rag_models import RAGEvaluationQuery
from app.generation.models import ContextSource, RAGResponse


def make_source(
    document_id: str,
) -> ContextSource:
    """Create a deterministic citation source for tests."""

    return ContextSource(
        source_number=1,
        document_id=document_id,
        chunk_id=f"{document_id}::chunk-0001",
        department="Test Department",
        document_type="Policy",
        source_file="test_policy.txt",
    )


def test_load_rag_evaluation_queries():
    queries = RAGEvaluator.load_queries(
        "data/evaluation/rag_queries.json"
    )

    assert len(queries) == 10

    assert sum(
        query.should_answer
        for query in queries
    ) == 5

    assert sum(
        not query.should_answer
        for query in queries
    ) == 5


def test_supported_response_is_scored_correctly():
    query = RAGEvaluationQuery(
        query_id="TEST001",
        question=(
            "Who must approve international remote work?"
        ),
        should_answer=True,
        expected_document_ids=[
            "HR-POL-001"
        ],
        expected_keywords=[
            "Human Resources",
            "Information Security",
            "Legal",
        ],
    )

    response = RAGResponse(
        question=query.question,
        answer=(
            "International remote work requires approval "
            "from Human Resources, Information Security, "
            "and Legal."
        ),
        sufficient_evidence=True,
        sources=[
            make_source(
                "HR-POL-001"
            )
        ],
    )

    result = RAGEvaluator.evaluate_response(
        query=query,
        response=response,
    )

    assert result.query_id == "TEST001"
    assert result.decision_correct is True
    assert result.citation_correct is True
    assert result.abstention_correct is True
    assert result.keyword_coverage == 1.0


def test_unsupported_response_is_scored_as_correct_abstention():
    query = RAGEvaluationQuery(
        query_id="TEST002",
        question=(
            "How many vacation days do employees receive?"
        ),
        should_answer=False,
        expected_document_ids=[],
        expected_keywords=[],
    )

    response = RAGResponse(
        question=query.question,
        answer=(
            "The available company documents do not provide "
            "enough information to answer this question."
        ),
        sufficient_evidence=False,
        sources=[],
    )

    result = RAGEvaluator.evaluate_response(
        query=query,
        response=response,
    )

    assert result.decision_correct is True
    assert result.citation_correct is True
    assert result.abstention_correct is True
    assert result.keyword_coverage == 1.0


def test_wrong_citation_is_detected():
    query = RAGEvaluationQuery(
        query_id="TEST003",
        question=(
            "Who must approve international remote work?"
        ),
        should_answer=True,
        expected_document_ids=[
            "HR-POL-001"
        ],
        expected_keywords=[
            "Human Resources"
        ],
    )

    response = RAGResponse(
        question=query.question,
        answer=(
            "Human Resources must approve the request."
        ),
        sufficient_evidence=True,
        sources=[
            make_source(
                "FIN-POL-001"
            )
        ],
    )

    result = RAGEvaluator.evaluate_response(
        query=query,
        response=response,
    )

    assert result.decision_correct is True
    assert result.citation_correct is False
    assert result.keyword_coverage == 1.0


def test_missing_expected_keywords_reduces_coverage():
    query = RAGEvaluationQuery(
        query_id="TEST004",
        question=(
            "Who must approve international remote work?"
        ),
        should_answer=True,
        expected_document_ids=[
            "HR-POL-001"
        ],
        expected_keywords=[
            "Human Resources",
            "Information Security",
            "Legal",
        ],
    )

    response = RAGResponse(
        question=query.question,
        answer=(
            "Human Resources must approve the request."
        ),
        sufficient_evidence=True,
        sources=[
            make_source(
                "HR-POL-001"
            )
        ],
    )

    result = RAGEvaluator.evaluate_response(
        query=query,
        response=response,
    )

    assert result.citation_correct is True

    assert result.keyword_coverage == (
        1 / 3
    )


def test_aggregate_rag_metrics():
    supported_query = RAGEvaluationQuery(
        query_id="TEST005",
        question="Supported question",
        should_answer=True,
        expected_document_ids=[
            "HR-POL-001"
        ],
        expected_keywords=[
            "approval"
        ],
    )

    unsupported_query = RAGEvaluationQuery(
        query_id="TEST006",
        question="Unsupported question",
        should_answer=False,
        expected_document_ids=[],
        expected_keywords=[],
    )

    supported_response = RAGResponse(
        question=supported_query.question,
        answer=(
            "The request requires approval."
        ),
        sufficient_evidence=True,
        sources=[
            make_source(
                "HR-POL-001"
            )
        ],
    )

    unsupported_response = RAGResponse(
        question=unsupported_query.question,
        answer=(
            "The available company documents do not provide "
            "enough information to answer this question."
        ),
        sufficient_evidence=False,
        sources=[],
    )

    queries = [
        supported_query,
        unsupported_query,
    ]

    results = [
        RAGEvaluator.evaluate_response(
            query=supported_query,
            response=supported_response,
        ),
        RAGEvaluator.evaluate_response(
            query=unsupported_query,
            response=unsupported_response,
        ),
    ]

    metrics = RAGEvaluator.aggregate(
        queries=queries,
        results=results,
    )

    assert metrics.total_queries == 2
    assert metrics.supported_queries == 1
    assert metrics.unsupported_queries == 1

    assert metrics.decision_accuracy == 1.0
    assert metrics.citation_accuracy == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.mean_keyword_coverage == 1.0