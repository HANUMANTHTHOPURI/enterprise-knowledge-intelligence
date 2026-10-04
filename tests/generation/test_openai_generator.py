from types import SimpleNamespace

import pytest

from app.generation.models import (
    ContextBundle,
    ContextSource,
    GeneratedAnswer,
)
from app.generation.openai_generator import OpenAIGenerator


def make_context_bundle() -> ContextBundle:
    return ContextBundle(
        context=(
            "[SOURCE 1]\n"
            "Document ID: HR-POL-001\n"
            "International work requires prior approval.\n"
            "[/SOURCE 1]"
        ),
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


class FakeResponses:
    def __init__(
        self,
        generated_answer: GeneratedAnswer,
    ) -> None:
        self.generated_answer = generated_answer

    def parse(self, **kwargs):
        return SimpleNamespace(
            output_parsed=self.generated_answer
        )


class FakeOpenAIClient:
    def __init__(
        self,
        generated_answer: GeneratedAnswer,
    ) -> None:
        self.responses = FakeResponses(
            generated_answer
        )


def test_generator_accepts_grounded_answer():
    generated_answer = GeneratedAnswer(
        answer=(
            "International remote work requires prior approval."
        ),
        sufficient_evidence=True,
        cited_source_numbers=[1],
    )

    generator = OpenAIGenerator(
        client=FakeOpenAIClient(generated_answer)
    )

    result = generator.generate(
        "Can I work from another country?",
        make_context_bundle(),
    )

    assert result.sufficient_evidence is True
    assert result.cited_source_numbers == [1]


def test_generator_accepts_insufficient_evidence_answer():
    generated_answer = GeneratedAnswer(
        answer=(
            "The available company documents do not provide "
            "enough information."
        ),
        sufficient_evidence=False,
        cited_source_numbers=[],
    )

    generator = OpenAIGenerator(
        client=FakeOpenAIClient(generated_answer)
    )

    result = generator.generate(
        "How many vacation days do employees receive?",
        make_context_bundle(),
    )

    assert result.sufficient_evidence is False
    assert result.cited_source_numbers == []


def test_invalid_source_number_is_rejected():
    generated_answer = GeneratedAnswer(
        answer="Grounded answer.",
        sufficient_evidence=True,
        cited_source_numbers=[99],
    )

    generator = OpenAIGenerator(
        client=FakeOpenAIClient(generated_answer)
    )

    with pytest.raises(
        ValueError,
        match="citations that are not present",
    ):
        generator.generate(
            "Test question",
            make_context_bundle(),
        )


def test_grounded_answer_requires_citation():
    generated_answer = GeneratedAnswer(
        answer="Grounded answer.",
        sufficient_evidence=True,
        cited_source_numbers=[],
    )

    generator = OpenAIGenerator(
        client=FakeOpenAIClient(generated_answer)
    )

    with pytest.raises(
        ValueError,
        match="Grounded answers must cite at least one source",
    ):
        generator.generate(
            "Test question",
            make_context_bundle(),
        )


def test_insufficient_answer_cannot_cite_sources():
    generated_answer = GeneratedAnswer(
        answer="Not enough information.",
        sufficient_evidence=False,
        cited_source_numbers=[1],
    )

    generator = OpenAIGenerator(
        client=FakeOpenAIClient(generated_answer)
    )

    with pytest.raises(
        ValueError,
        match="Insufficient-evidence answers must not cite sources",
    ):
        generator.generate(
            "Test question",
            make_context_bundle(),
        )