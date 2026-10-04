from openai import OpenAI

from app.core.config import settings
from app.generation.models import ContextBundle, GeneratedAnswer
from app.generation.prompt_builder import PromptBuilder


class OpenAIGenerator:
    """Generate grounded enterprise answers using OpenAI."""

    def __init__(
        self,
        model_name: str | None = None,
        client: OpenAI | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.model_name = (
            model_name
            if model_name is not None
            else settings.llm_model
        )

        if client is not None:
            self.client = client
        else:
            if settings.openai_api_key is None:
                raise ValueError(
                    "OPENAI_API_KEY is not configured"
                )

            self.client = OpenAI(
                api_key=(
                    settings.openai_api_key.get_secret_value()
                )
            )

        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else PromptBuilder()
        )

    def generate(
        self,
        question: str,
        context_bundle: ContextBundle,
    ) -> GeneratedAnswer:
        """Generate and validate an answer grounded in retrieved evidence."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        user_input = self.prompt_builder.build_input(
            question=question,
            context_bundle=context_bundle,
        )

        response = self.client.responses.parse(
            model=self.model_name,
            instructions=self.prompt_builder.SYSTEM_INSTRUCTIONS,
            input=user_input,
            text_format=GeneratedAnswer,
        )

        answer = response.output_parsed

        if answer is None:
            raise RuntimeError(
                "OpenAI response did not contain a parsed answer"
            )

        self._validate_answer(
            answer=answer,
            context_bundle=context_bundle,
        )

        return answer

    @staticmethod
    def _validate_answer(
        answer: GeneratedAnswer,
        context_bundle: ContextBundle,
    ) -> None:
        """Validate grounding and citation invariants."""

        if not answer.answer.strip():
            raise ValueError(
                "Generated answer cannot be empty"
            )

        available_source_numbers = {
            source.source_number
            for source in context_bundle.sources
        }

        cited_source_numbers = set(
            answer.cited_source_numbers
        )

        invalid_citations = (
            cited_source_numbers
            - available_source_numbers
        )

        if invalid_citations:
            raise ValueError(
                "Generated answer contains citations "
                "that are not present in the context"
            )

        if (
            answer.sufficient_evidence
            and not cited_source_numbers
        ):
            raise ValueError(
                "Grounded answers must cite at least one source"
            )

        if (
            not answer.sufficient_evidence
            and cited_source_numbers
        ):
            raise ValueError(
                "Insufficient-evidence answers must not cite sources"
            )