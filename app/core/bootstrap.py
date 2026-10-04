from pathlib import Path
from app.core.config import settings

from app.agents.rag_graph import RAGAgent
from app.embeddings.encoder import EmbeddingEncoder
from app.generation.context_builder import ContextBuilder
from app.generation.evidence_evaluator import EvidenceEvaluator
from app.generation.openai_generator import OpenAIGenerator
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.faiss_retriever import FAISSRetriever
from app.retrieval.reranked_retriever import RerankedRetriever
from app.services.rag_service import RAGService
from app.vectorstore.faiss_index import FAISSVectorIndex


DEFAULT_INDEX_PATH = Path(
    "data/vectorstore/enterprise_knowledge.faiss"
)

DEFAULT_METADATA_PATH = Path(
    "data/vectorstore/enterprise_knowledge_chunks.json"
)


def build_retriever(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> RerankedRetriever:
    """Construct the production retrieval and reranking pipeline."""

    encoder = EmbeddingEncoder()

    vector_index = FAISSVectorIndex.load(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    faiss_retriever = FAISSRetriever(
        vector_index=vector_index,
        encoder=encoder,
    )

    reranker = CrossEncoderReranker()

    return RerankedRetriever(
        dense_retriever=faiss_retriever,
        reranker=reranker,
        candidate_k=settings.rerank_candidate_k,
    )


def build_rag_service(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    retrieval_top_k: int = 3,
) -> RAGService:
    """Construct the production enterprise RAG service."""

    retriever = build_retriever(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    return RAGService(
        retriever=retriever,
        context_builder=ContextBuilder(),
        generator=OpenAIGenerator(),
        retrieval_top_k=retrieval_top_k,
    )


def build_rag_agent(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    retrieval_top_k: int = 3,
) -> RAGAgent:
    """Construct the production evidence-gated LangGraph RAG agent."""

    retriever = build_retriever(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    return RAGAgent(
        retriever=retriever,
        context_builder=ContextBuilder(),
        evidence_evaluator=EvidenceEvaluator(),
        generator=OpenAIGenerator(),
        retrieval_top_k=retrieval_top_k,
    )