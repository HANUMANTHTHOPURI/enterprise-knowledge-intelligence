from app.generation.context_builder import ContextBuilder
from app.generation.models import RAGResponse
from app.generation.openai_generator import OpenAIGenerator
from app.retrieval.reranked_retriever import RerankedRetriever


class RAGService:
    """Coordinate retrieval, context construction, and grounded generation."""

    def __init__(
        self,
        retriever: RerankedRetriever,
        context_builder: ContextBuilder,
        generator: OpenAIGenerator,
        retrieval_top_k: int = 3,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero"
            )

        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.retrieval_top_k = retrieval_top_k

    def ask(
        self,
        question: str,
    ) -> RAGResponse:
        """Answer a question using grounded enterprise knowledge."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty"
            )

        retrieval_results = self.retriever.search(
            question,
            top_k=self.retrieval_top_k,
        )

        context_bundle = self.context_builder.build(
            retrieval_results
        )

        generated_answer = self.generator.generate(
            question=question,
            context_bundle=context_bundle,
        )

        cited_numbers = set(
            generated_answer.cited_source_numbers
        )

        cited_sources = [
            source
            for source in context_bundle.sources
            if source.source_number in cited_numbers
        ]

        return RAGResponse(
            question=question,
            answer=generated_answer.answer,
            sufficient_evidence=(
                generated_answer.sufficient_evidence
            ),
            sources=cited_sources,
        )