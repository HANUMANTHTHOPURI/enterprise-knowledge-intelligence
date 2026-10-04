import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.chunking.section_chunker import SectionChunker
from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.corpus_loader import CorpusLoader
from app.vectorstore.faiss_index import FAISSVectorIndex


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "enterprise_knowledge.faiss"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "vectorstore"
    / "enterprise_knowledge_chunks.json"
)


def main() -> None:
    print("=" * 70)
    print("ENTERPRISE KNOWLEDGE VECTOR STORE BUILD")
    print("=" * 70)

    start_time = perf_counter()

    # ---------------------------------------------------------
    # 1. Load enterprise documents
    # ---------------------------------------------------------

    print("\n[1/5] Loading enterprise documents...")

    corpus_loader = CorpusLoader()

    documents = corpus_loader.load(
        RAW_DATA_PATH
    )

    print(
        f"Loaded documents: {len(documents)}"
    )

    for document in documents:
        print(
            f"  - {document.metadata.document_id} "
            f"({document.metadata.source})"
        )

    # ---------------------------------------------------------
    # 2. Chunk documents
    # ---------------------------------------------------------

    print("\n[2/5] Creating section-aware chunks...")

    chunker = SectionChunker()

    chunks = []

    for document in documents:
        document_chunks = chunker.chunk(
            document
        )

        chunks.extend(
            document_chunks
        )

        print(
            f"  {document.metadata.document_id}: "
            f"{len(document_chunks)} chunks"
        )

    if not chunks:
        raise RuntimeError(
            "Chunking produced no document chunks"
        )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    # ---------------------------------------------------------
    # 3. Generate embeddings
    # ---------------------------------------------------------

    print("\n[3/5] Generating embeddings...")

    encoder = EmbeddingEncoder()

    texts = [
        chunk.content
        for chunk in chunks
    ]

    embeddings = encoder.encode_texts(
        texts
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding dimension: "
        f"{encoder.dimension}"
    )

    # ---------------------------------------------------------
    # 4. Build FAISS index
    # ---------------------------------------------------------

    print("\n[4/5] Building FAISS vector index...")

    vector_index = FAISSVectorIndex(
        dimension=encoder.dimension
    )

    vector_index.add(
        embeddings=embeddings,
        chunks=chunks,
    )

    print(
        f"Indexed vectors: "
        f"{vector_index.size}"
    )

    # ---------------------------------------------------------
    # 5. Persist vector store
    # ---------------------------------------------------------

    print("\n[5/5] Saving vector store...")

    vector_index.save(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    elapsed_seconds = (
        perf_counter() - start_time
    )

    print("\n" + "=" * 70)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Documents:             "
        f"{len(documents)}"
    )

    print(
        f"Chunks:                "
        f"{len(chunks)}"
    )

    print(
        f"Vectors:               "
        f"{vector_index.size}"
    )

    print(
        f"Embedding dimension:   "
        f"{encoder.dimension}"
    )

    print(
        f"Index file:            "
        f"{INDEX_PATH}"
    )

    print(
        f"Metadata file:         "
        f"{METADATA_PATH}"
    )

    print(
        f"Build time:            "
        f"{elapsed_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()