from fastapi.testclient import TestClient

from app.generation.models import ContextSource, RAGResponse
from app.main import create_app


class FakeAgent:
    """Deterministic agent used for API tests."""

    def invoke(
        self,
        question: str,
    ) -> RAGResponse:
        if "vacation" in question.lower():
            return RAGResponse(
                question=question,
                answer=(
                    "The available company documents do not provide "
                    "enough information to answer this question."
                ),
                sufficient_evidence=False,
                sources=[],
            )

        return RAGResponse(
            question=question,
            answer=(
                "International remote work requires prior approval."
            ),
            sufficient_evidence=True,
            sources=[
                ContextSource(
                    source_number=1,
                    document_id="HR-POL-001",
                    chunk_id="HR-POL-001::chunk-0005",
                    department="Human Resources",
                    document_type="Policy",
                    source_file="employee_remote_work_policy.txt",
                )
            ],
        )


def test_health_endpoint():
    app = create_app(agent=FakeAgent())

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health"
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "enterprise-knowledge-intelligence",
    }


def test_ask_endpoint_returns_grounded_answer():
    app = create_app(agent=FakeAgent())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": (
                    "Can I work remotely from another country?"
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["sufficient_evidence"] is True
    assert len(data["sources"]) == 1

    assert data["sources"][0]["document_id"] == (
        "HR-POL-001"
    )

    assert data["sources"][0]["chunk_id"] == (
        "HR-POL-001::chunk-0005"
    )


def test_ask_endpoint_returns_abstention():
    app = create_app(agent=FakeAgent())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": (
                    "How many vacation days do employees receive?"
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["sufficient_evidence"] is False
    assert data["sources"] == []


def test_ask_endpoint_rejects_blank_question():
    app = create_app(agent=FakeAgent())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": "   "
            },
        )

    assert response.status_code == 422