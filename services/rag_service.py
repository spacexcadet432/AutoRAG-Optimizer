"""Service layer for in-memory uploaded-dataset RAG execution."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
from openai import OpenAI

from core.chunking import chunk_text


EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MAX_FILE_SIZE_BYTES = 3 * 1024 * 1024
MAX_CHUNKS = 800
DEFAULT_UPLOAD_CHUNK_SIZE = 500
DEFAULT_UPLOAD_OVERLAP = 100


@dataclass
class RetrievedChunk:
    """Retrieved chunk payload with score."""

    text: str
    score: float


@dataclass
class UploadDatasetState:
    """Global in-memory state for the currently uploaded dataset."""

    file_name: str
    file_size_bytes: int
    chunks: List[str]
    embeddings: np.ndarray
    faiss_index: faiss.Index
    chunk_size: int
    overlap: int


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


def _clean_text(text: str) -> str:
    """Apply basic normalization to uploaded text."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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
) -> Tuple[str, List[RetrievedChunk], Dict[str, float], float]:
    """
    Execute one end-to-end RAG run and return answer, chunks, metrics, latency.
    """

    if upload_state is None:
        raise RuntimeError("Please upload a dataset first.")

    client = OpenAI(api_key=openai_api_key)
    start = time.perf_counter()

    query_vector = _embed_texts(client, [query])
    distances, indices = upload_state.faiss_index.search(query_vector, int(config["top_k"]))

    retrieved: List[RetrievedChunk] = []
    for score, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(upload_state.chunks):
            retrieved.append(RetrievedChunk(text=upload_state.chunks[idx], score=float(score)))

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


upload_state: Optional[UploadDatasetState] = None


def upload_dataset(
    file_name: str,
    file_bytes: bytes,
    openai_api_key: str,
    chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
    overlap: int = DEFAULT_UPLOAD_OVERLAP,
) -> UploadDatasetState:
    """Validate uploaded text, chunk it, embed it, and build FAISS index in memory."""

    global upload_state

    if not file_name.lower().endswith(".txt"):
        raise ValueError("Only .txt files are supported.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File too large. Max size is 3MB.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    try:
        decoded = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unable to decode file. Please upload UTF-8 text.") from exc

    cleaned = _clean_text(decoded)
    if not cleaned:
        raise ValueError("Empty file. Please upload a non-empty text file.")

    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError("Dataset too large")

    client = OpenAI(api_key=openai_api_key)
    embeddings = _embed_texts(client, chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    upload_state = UploadDatasetState(
        file_name=file_name,
        file_size_bytes=len(file_bytes),
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=index,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return upload_state


def upload_dataset_text(
    file_name: str,
    text: str,
    openai_api_key: str,
    chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
    overlap: int = DEFAULT_UPLOAD_OVERLAP,
) -> UploadDatasetState:
    """
    Upload dataset from raw text payload (JSON path, no multipart transport).
    """

    file_bytes = text.encode("utf-8")
    return upload_dataset(
        file_name=file_name,
        file_bytes=file_bytes,
        openai_api_key=openai_api_key,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def reset_dataset() -> None:
    """Clear the in-memory uploaded dataset."""

    global upload_state
    upload_state = None


def get_dataset_status() -> Dict[str, object]:
    """Return current in-memory dataset status for frontend sync."""

    if upload_state is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "file_name": upload_state.file_name,
        "file_size_bytes": upload_state.file_size_bytes,
        "chunk_count": len(upload_state.chunks),
        "chunk_size": upload_state.chunk_size,
        "overlap": upload_state.overlap,
    }

