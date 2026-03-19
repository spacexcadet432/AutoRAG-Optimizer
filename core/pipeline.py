import os
import time
from typing import Any, Dict

from core.chunking import chunk_text
from core.embedding import EMBEDDING_MODEL_NAME, get_embeddings
from core.generation import build_prompt, generate_from_prompt
from core.retrieval import FaissRetriever, compute_index_id, load_index, save_index


def load_document(file_path: str) -> str:
    """
    Load a UTF-8 encoded text document from disk.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _approx_token_count(text: str) -> int:
    """
    Lightweight token count proxy based on character length.
    """

    return max(1, int(len(text) / 4))


def run_rag_pipeline(document_path: str, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the RAG pipeline using the provided configuration.

    This implementation:
    - Computes corpus embeddings once per (document, chunking config, embedding model)
      and caches them in a persistent FAISS index on disk.
    - Returns both answer quality fields and cost proxy metrics.
    """

    chunk_size = config["chunk_size"]
    overlap = config["overlap"]
    top_k = config["top_k"]
    prompt_style = config["prompt_style"]

    # 1) Load document text
    text = load_document(document_path)

    # 2) Prepare index id for this corpus + chunking configuration
    index_id = compute_index_id(config, document_path=document_path, embedding_model=EMBEDDING_MODEL_NAME)

    # 3) Try to reuse cached FAISS index + chunks
    loaded = load_index(index_id)
    if loaded is not None:
        index, chunks = loaded
        num_chunks = len(chunks)
        retriever = FaissRetriever(index=index, chunks=chunks)
    else:
        # 3a) Chunk document
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        num_chunks = len(chunks)

        # 3b) Compute embeddings and build index
        chunk_embeddings = get_embeddings(chunks)
        retriever = FaissRetriever(chunk_embeddings, chunks)

        # 3c) Persist index and chunks for future runs
        save_index(index_id, retriever.index, retriever.chunks)

    # 4) Embed question
    question_embedding = get_embeddings([question])

    # 5) Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve(question_embedding, top_k=top_k)

    # 6) Build prompt and call LLM with simple cost proxies
    prompt = build_prompt(question, retrieved_chunks, prompt_style=prompt_style)
    approx_tokens = _approx_token_count(prompt)

    t0 = time.perf_counter()
    answer = generate_from_prompt(prompt)
    latency_s = time.perf_counter() - t0

    return {
        "question": question,
        "config": config,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "num_chunks": num_chunks,
        "cost": {
            "prompt_tokens_approx": approx_tokens,
            "latency_s": latency_s,
        },
    }


if __name__ == "__main__":
    default_config = {
        "chunk_size": 500,
        "overlap": 100,
        "top_k": 3,
        "prompt_style": "basic",
    }
    q = input("Enter your question: ")
    result = run_rag_pipeline("data/documents/sample.txt", q, default_config)
    print(result["answer"])

