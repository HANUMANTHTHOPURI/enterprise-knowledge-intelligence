from pathlib import Path

from app.embeddings.encoder import EmbeddingEncoder
from app.generation.context_builder import ContextBuilder
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


def build_rag_service(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    retrieval_top_k: int = 3,
) -> RAGService:
    """Construct the production enterprise RAG pipeline."""

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

    retriever = RerankedRetriever(
        dense_retriever=faiss_retriever,
        reranker=reranker,
        candidate_k=5,
    )

    context_builder = ContextBuilder()

    generator = OpenAIGenerator()

    return RAGService(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
        retrieval_top_k=retrieval_top_k,
    )