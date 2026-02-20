import os

from rag.chunker import chunk_text
from rag.embedder import get_embeddings
from rag.retriever import FaissRetriever
from rag.generator import generate_answer


def load_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def run_rag_pipeline(document_path: str, question: str, config: dict):
    """
    Run RAG pipeline using provided configuration.
    """

    chunk_size = config["chunk_size"]
    overlap = config["overlap"]
    top_k = config["top_k"]

    # 1️⃣ Load document
    text = load_document(document_path)

    # 2️⃣ Chunk document
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    num_chunks = len(chunks)

    # 3️⃣ Generate embeddings for chunks
    chunk_embeddings = get_embeddings(chunks)

    # 4️⃣ Build FAISS retriever
    retriever = FaissRetriever(chunk_embeddings, chunks)

    # 5️⃣ Embed question
    question_embedding = get_embeddings([question])

    # 6️⃣ Retrieve relevant chunks
    retrieved_chunks = retriever.retrieve(question_embedding, top_k=top_k)

    # 7️⃣ Generate grounded answer
    answer = generate_answer(
    question,
    retrieved_chunks,
    prompt_style=config["prompt_style"]
)

    return {
        "question": question,
        "config": config,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer , 
        "num_chunks": num_chunks
    }


if __name__ == "__main__":
    document_path = "data/documents/sample.txt"

    config = {
        "chunk_size": 500,
        "overlap": 100,
        "top_k": 3,
        "prompt_style": "basic"
    }

    question = input("Enter your question: ")

    result = run_rag_pipeline(document_path, question, config)

    print("\n--- Config Used ---")
    print(result["config"])

    print("\n--- Retrieved Chunks ---")
    for chunk in result["retrieved_chunks"]:
        print("-", chunk[:200], "\n")

    print("\n--- Answer ---")
    print(result["answer"])