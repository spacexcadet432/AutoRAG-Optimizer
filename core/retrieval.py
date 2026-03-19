import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


INDEX_DIR = "indexes"


class FaissRetriever:
    """
    Simple FAISS-based retriever over in-memory chunks.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        chunks: Optional[List[str]] = None,
        index: Optional[faiss.Index] = None,
    ) -> None:
        if index is not None:
            # Construct from an existing FAISS index and a list of chunks.
            self.index = index
            self.chunks = chunks or []
            self.dimension = index.d
            return

        if embeddings is None or chunks is None:
            raise ValueError("embeddings and chunks must be provided when index is None")

        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        self.chunks = chunks
        self.dimension = embeddings.shape[1]

        index_ip = faiss.IndexFlatIP(self.dimension)
        index_ip.add(embeddings)
        self.index = index_ip

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k most similar chunks for a query embedding.
        """

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        _, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks: List[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks


def _ensure_index_dir() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)


def compute_index_id(config: Dict, document_path: str, embedding_model: str) -> str:
    """
    Compute a stable hash for a given configuration and document.

    This controls the cache key for persistent FAISS indexes.
    """

    payload = {
        "document_path": os.path.abspath(document_path),
        "chunk_size": config["chunk_size"],
        "overlap": config["overlap"],
        "embedding_model": embedding_model,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def index_paths(index_id: str) -> Tuple[str, str]:
    """
    Return (faiss_index_path, chunks_path) for a given index id.
    """

    _ensure_index_dir()
    faiss_path = os.path.join(INDEX_DIR, f"{index_id}.faiss")
    chunks_path = os.path.join(INDEX_DIR, f"{index_id}_chunks.json")
    return faiss_path, chunks_path


def save_index(index_id: str, index: faiss.Index, chunks: List[str]) -> None:
    """
    Persist FAISS index and associated chunks to disk.
    """

    faiss_path, chunks_path = index_paths(index_id)
    faiss.write_index(index, faiss_path)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def load_index(index_id: str) -> Optional[Tuple[faiss.Index, List[str]]]:
    """
    Load FAISS index and chunks if they exist, otherwise return None.
    """

    faiss_path, chunks_path = index_paths(index_id)
    if not (os.path.exists(faiss_path) and os.path.exists(chunks_path)):
        return None

    index = faiss.read_index(faiss_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[str] = json.load(f)

    return index, chunks

