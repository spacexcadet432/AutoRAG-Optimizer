"""Precompute FAISS artifacts for Render deployment startup preload."""

from typing import List, Tuple

from core.chunking import chunk_text
from core.embedding import EMBEDDING_MODEL_NAME, get_embeddings
from core.pipeline import load_document
from core.retrieval import FaissRetriever, compute_index_id, save_index


SUPPORTED_CONFIGS: List[Tuple[int, int]] = [
    (200, 0),
    (200, 50),
    (200, 100),
    (300, 0),
    (300, 50),
    (300, 100),
    (500, 0),
    (500, 50),
    (500, 100),
]


def main() -> None:
    """Build and save index files for all UI chunk/overlap combinations."""

    document_path = "data/documents/sample.txt"
    document = load_document(document_path)

    for chunk_size, overlap in SUPPORTED_CONFIGS:
        config = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": 3,
            "prompt_style": "basic",
        }
        chunks = chunk_text(document, chunk_size=chunk_size, overlap=overlap)
        vectors = get_embeddings(chunks)
        retriever = FaissRetriever(embeddings=vectors, chunks=chunks)
        index_id = compute_index_id(
            config=config,
            document_path=document_path,
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        save_index(index_id=index_id, index=retriever.index, chunks=chunks)
        print(f"Prepared index for chunk_size={chunk_size}, overlap={overlap}")


if __name__ == "__main__":
    main()

