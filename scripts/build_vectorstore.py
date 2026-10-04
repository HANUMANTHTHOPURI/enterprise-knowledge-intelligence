from pathlib import Path
from time import perf_counter

from app.chunking.section_chunker import SectionChunker
from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.corpus_loader import CorpusLoader
from app.vectorstore.faiss_index import FAISSVectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
INDEX_PATH = (
    PROJECT_ROOT / "data" / "vectorstore" / "enterprise_knowledge.faiss"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "enterprise_knowledge_chunks.json"
)


def main() -> None:
    start_time = perf_counter()

    print("=" * 70)
    print("BUILDING ENTERPRISE KNOWLEDGE VECTOR STORE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load enterprise documents
    # ---------------------------------------------------------
    corpus_loader = CorpusLoader()
    documents = corpus_loader.load(RAW_DATA_PATH)

    print(f"\nLoaded documents: {len(documents)}")

    # ---------------------------------------------------------
    # 2. Chunk documents
    # ---------------------------------------------------------
    chunker = SectionChunker()

    chunks = []

    for document in documents:
        document_chunks = chunker.chunk(document)
        chunks.extend(document_chunks)

        print(
            f"{document.metadata.document_id}: "
            f"{len(document_chunks)} chunks"
        )

    print(f"\nTotal chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 3. Generate embeddings
    # ---------------------------------------------------------
    encoder = EmbeddingEncoder()

    chunk_texts = [chunk.content for chunk in chunks]

    embeddings = encoder.encode_texts(chunk_texts)

    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # ---------------------------------------------------------
    # 4. Build FAISS index
    # ---------------------------------------------------------
    vector_index = FAISSVectorIndex(
        dimension=embeddings.shape[1]
    )

    vector_index.add(
        embeddings=embeddings,
        chunks=chunks,
    )

    print(f"Indexed vectors: {len(chunks)}")

    # ---------------------------------------------------------
    # 5. Persist vector store
    # ---------------------------------------------------------
    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_index.save(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    elapsed_time = perf_counter() - start_time

    print("\n" + "=" * 70)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 70)

    print(f"Documents:           {len(documents)}")
    print(f"Chunks:              {len(chunks)}")
    print(f"Vectors:             {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index file:          {INDEX_PATH}")
    print(f"Metadata file:       {METADATA_PATH}")
    print(f"Build time:          {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
