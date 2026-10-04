from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.state import RAGAgentState
from app.generation.context_builder import ContextBuilder
from app.generation.evidence_evaluator import EvidenceEvaluator
from app.generation.models import RAGResponse
from app.generation.openai_generator import OpenAIGenerator
from app.retrieval.reranked_retriever import RerankedRetriever


class RAGAgent:
    """Stateful LangGraph workflow for grounded enterprise RAG."""

    def __init__(
        self,
        retriever: RerankedRetriever,
        context_builder: ContextBuilder,
        evidence_evaluator: EvidenceEvaluator,
        generator: OpenAIGenerator,
        retrieval_top_k: int = 3,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero"
            )

        self.retriever = retriever
        self.context_builder = context_builder
        self.evidence_evaluator = evidence_evaluator
        self.generator = generator
        self.retrieval_top_k = retrieval_top_k

        self.graph = self._build_graph()

    def _build_graph(self):
        """Compile the LangGraph workflow."""

        workflow = StateGraph(RAGAgentState)

        workflow.add_node(
            "retrieve",
            self._retrieve_node,
        )

        workflow.add_node(
            "build_context",
            self._context_node,
        )

        workflow.add_node(
            "assess_evidence",
            self._assess_evidence_node,
        )

        workflow.add_node(
            "generate",
            self._generate_node,
        )

        workflow.add_node(
            "grounded_response",
            self._grounded_response_node,
        )

        workflow.add_node(
            "abstention",
            self._abstention_node,
        )

        workflow.add_edge(
            START,
            "retrieve",
        )

        workflow.add_edge(
            "retrieve",
            "build_context",
        )

        workflow.add_edge(
            "build_context",
            "assess_evidence",
        )

        workflow.add_conditional_edges(
            "assess_evidence",
            self._route_evidence,
            {
                "sufficient": "generate",
                "insufficient": "abstention",
            },
        )

        workflow.add_conditional_edges(
            "generate",
            self._route_generated_answer,
            {
                "grounded": "grounded_response",
                "abstain": "abstention",
            },
        )

        workflow.add_edge(
            "grounded_response",
            END,
        )

        workflow.add_edge(
            "abstention",
            END,
        )

        return workflow.compile()

    def _retrieve_node(
        self,
        state: RAGAgentState,
    ) -> dict:
        """Retrieve and rerank evidence for the question."""

        results = self.retriever.search(
            state["question"],
            top_k=self.retrieval_top_k,
        )

        return {
            "retrieval_results": results,
        }

    def _context_node(
        self,
        state: RAGAgentState,
    ) -> dict:
        """Convert retrieval results into structured context."""

        context_bundle = self.context_builder.build(
            state["retrieval_results"]
        )

        return {
            "context_bundle": context_bundle,
        }

    def _assess_evidence_node(
        self,
        state: RAGAgentState,
    ) -> dict:
        """Decide whether retrieved evidence can answer the question."""

        assessment = self.evidence_evaluator.evaluate(
            question=state["question"],
            context_bundle=state["context_bundle"],
        )

        return {
            "evidence_assessment": assessment,
        }

    @staticmethod
    def _route_evidence(
        state: RAGAgentState,
    ) -> Literal["sufficient", "insufficient"]:
        """Route based on retrieved evidence quality."""

        if state["evidence_assessment"].sufficient_evidence:
            return "sufficient"

        return "insufficient"

    def _generate_node(
        self,
        state: RAGAgentState,
    ) -> dict:
        """Generate an answer from approved evidence."""

        generated_answer = self.generator.generate(
            question=state["question"],
            context_bundle=state["context_bundle"],
        )

        return {
            "generated_answer": generated_answer,
        }

    @staticmethod
    def _route_generated_answer(
        state: RAGAgentState,
    ) -> Literal["grounded", "abstain"]:
        """Apply a second safety check after generation."""

        if state["generated_answer"].sufficient_evidence:
            return "grounded"

        return "abstain"

    @staticmethod
    def _grounded_response_node(
        state: RAGAgentState,
    ) -> dict:
        """Construct a response containing only cited evidence."""

        generated_answer = state["generated_answer"]
        context_bundle = state["context_bundle"]

        cited_numbers = set(
            generated_answer.cited_source_numbers
        )

        cited_sources = [
            source
            for source in context_bundle.sources
            if source.source_number in cited_numbers
        ]

        response = RAGResponse(
            question=state["question"],
            answer=generated_answer.answer,
            sufficient_evidence=True,
            sources=cited_sources,
        )

        return {
            "final_response": response,
        }

    @staticmethod
    def _abstention_node(
        state: RAGAgentState,
    ) -> dict:
        """Construct a controlled insufficient-evidence response."""

        generated_answer = state.get(
            "generated_answer"
        )

        if generated_answer is not None:
            answer_text = generated_answer.answer
        else:
            answer_text = (
                "The available company documents do not provide "
                "enough information to answer this question."
            )

        response = RAGResponse(
            question=state["question"],
            answer=answer_text,
            sufficient_evidence=False,
            sources=[],
        )

        return {
            "final_response": response,
        }

    def invoke(
        self,
        question: str,
    ) -> RAGResponse:
        """Run the complete enterprise RAG graph."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty"
            )

        final_state = self.graph.invoke(
            {
                "question": question,
            }
        )

        return final_state["final_response"]