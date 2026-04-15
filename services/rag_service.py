"""Service layer for production RAG execution."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
from openai import OpenAI

from core.chunking import chunk_text
from core.retrieval import compute_index_id, index_paths


EMBEDDING_MODEL_NAME = "text-embedding-3-small"


@dataclass
class RetrievedChunk:
    """Retrieved chunk payload with score."""

    text: str
    score: float


@dataclass
class CachedIndex:
    """In-memory representation of a FAISS index and aligned chunks."""

    index: faiss.Index
    chunks: List[str]


class RAGRuntime:
    """
    Runtime cache for deployment:
    - document text is loaded once at startup
    - FAISS indexes are preloaded once at startup when available
    - request path only embeds query/answer and runs retrieval/generation
    """

    def __init__(self, document_path: str, supported_configs: List[Tuple[int, int]]) -> None:
        self.document_path = document_path
        self.supported_configs = supported_configs
        self.document_text: Optional[str] = None
        self.index_cache: Dict[Tuple[int, int], CachedIndex] = {}

    def startup(self) -> None:
        """Load corpus and available FAISS indexes into memory."""

        self.document_text = _load_document(self.document_path)

        for chunk_size, overlap in self.supported_configs:
            config = {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": 3,
                "prompt_style": "basic",
            }
            index_id = compute_index_id(
                config=config,
                document_path=self.document_path,
                embedding_model=EMBEDDING_MODEL_NAME,
            )
            faiss_path, chunks_path = index_paths(index_id)

            if os.path.exists(faiss_path) and os.path.exists(chunks_path):
                index = faiss.read_index(faiss_path)
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                self.index_cache[(chunk_size, overlap)] = CachedIndex(index=index, chunks=chunks)

    def _build_and_cache_index(self, client: OpenAI, chunk_size: int, overlap: int) -> CachedIndex:
        """
        Build and cache index on first use when startup preload artifact is absent.

        This avoids per-request recomputation after first use.
        """

        if self.document_text is None:
            raise RuntimeError("Runtime not initialized.")

        chunks = chunk_text(self.document_text, chunk_size=chunk_size, overlap=overlap)
        chunk_vectors = _embed_texts(client, chunks)
        index = faiss.IndexFlatIP(chunk_vectors.shape[1])
        index.add(chunk_vectors)

        cached = CachedIndex(index=index, chunks=chunks)
        self.index_cache[(chunk_size, overlap)] = cached
        return cached

    def get_index(self, client: OpenAI, chunk_size: int, overlap: int) -> CachedIndex:
        """Get cached index for config; build once if missing."""

        key = (chunk_size, overlap)
        if key in self.index_cache:
            return self.index_cache[key]
        return self._build_and_cache_index(client=client, chunk_size=chunk_size, overlap=overlap)


def _load_document(document_path: str) -> str:
    """Load plain text document from disk."""

    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    with open(document_path, "r", encoding="utf-8") as f:
        return f.read()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity via inner product."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Embed texts using OpenAI embeddings."""

    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = np.asarray([item.embedding for item in response.data], dtype="float32")
    return _normalize(vectors)


def _build_prompt(question: str, retrieved_chunks: List[RetrievedChunk], prompt_style: str) -> str:
    """Build the generation prompt based on style."""

    context = "\n\n".join(
        [f"Context {idx + 1}:\n{chunk.text}" for idx, chunk in enumerate(retrieved_chunks)]
    )

    if prompt_style == "strict":
        return f"""
You are a strict AI assistant.

Use ONLY the provided context to answer the question.
If the answer is not explicitly in the context, say:
"Insufficient information in the provided context."

--------------------
{context}
--------------------

Question: {question}
Answer:
""".strip()

    return f"""
You are a helpful AI assistant.
Answer the question using the provided context.

--------------------
{context}
--------------------

Question: {question}
Answer:
""".strip()


def _generate_answer(client: OpenAI, prompt: str) -> str:
    """Generate answer text from prompt."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only using the supplied context when possible."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _safe_div(num: float, den: float) -> float:
    """Return num / den with zero-protection."""

    return num / den if den else 0.0


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used for lightweight coverage metrics."""

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def compute_metrics(
    client: OpenAI,
    query: str,
    answer: str,
    retrieved_chunks: List[RetrievedChunk],
    top_k: int,
) -> Dict[str, float]:
    """
    Compute lightweight online metrics for the playground.

    Notes:
    - Similarity: embedding similarity between query and answer.
    - Faithfulness: fraction of answer tokens found in retrieved context.
    - Precision@k: mean normalized retrieval score over top-k.
    - Retrieval coverage: fraction of query tokens present in retrieved context.
    """

    query_vec, answer_vec = _embed_texts(client, [query, answer])
    similarity = float(np.dot(query_vec, answer_vec.T))

    context_text = " ".join(chunk.text for chunk in retrieved_chunks).lower()
    answer_tokens = _tokenize(answer)
    query_tokens = _tokenize(query)

    answer_hits = sum(1 for t in answer_tokens if t in context_text)
    query_hits = sum(1 for t in query_tokens if t in context_text)

    faithfulness = _safe_div(answer_hits, len(answer_tokens))
    retrieval_coverage = _safe_div(query_hits, len(query_tokens))

    top_scores = [max(0.0, min(1.0, c.score)) for c in retrieved_chunks[:top_k]]
    precision_at_k = _safe_div(sum(top_scores), max(1, top_k))

    return {
        "similarity": round(similarity, 4),
        "faithfulness": round(faithfulness, 4),
        "precision_at_k": round(precision_at_k, 4),
        "retrieval_coverage": round(retrieval_coverage, 4),
    }


def run_rag_experiment(
    query: str,
    config: Dict[str, int | str],
    openai_api_key: str,
    document_path: str = "data/documents/sample.txt",
) -> Tuple[str, List[RetrievedChunk], Dict[str, float], float]:
    """
    Execute one end-to-end RAG run and return answer, chunks, metrics, latency.
    """

    if runtime is None:
        raise RuntimeError("RAG runtime is not initialized.")

    client = OpenAI(api_key=openai_api_key)
    start = time.perf_counter()

    chunk_size = int(config["chunk_size"])
    overlap = int(config["overlap"])
    cached = runtime.get_index(client=client, chunk_size=chunk_size, overlap=overlap)

    query_vector = _embed_texts(client, [query])
    distances, indices = cached.index.search(query_vector, int(config["top_k"]))

    retrieved: List[RetrievedChunk] = []
    for score, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(cached.chunks):
            retrieved.append(RetrievedChunk(text=cached.chunks[idx], score=float(score)))

    prompt = _build_prompt(query, retrieved, str(config["prompt_style"]))
    answer = _generate_answer(client, prompt)

    metrics = compute_metrics(
        client=client,
        query=query,
        answer=answer,
        retrieved_chunks=retrieved,
        top_k=int(config["top_k"]),
    )
    latency = round(time.perf_counter() - start, 3)

    return answer, retrieved, metrics, latency


runtime: Optional[RAGRuntime] = None


def initialize_runtime(
    document_path: str = "data/documents/sample.txt",
    supported_configs: Optional[List[Tuple[int, int]]] = None,
) -> RAGRuntime:
    """
    Initialize global runtime cache for FastAPI startup.
    """

    global runtime
    configs = supported_configs or [
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
    runtime = RAGRuntime(document_path=document_path, supported_configs=configs)
    runtime.startup()
    return runtime

