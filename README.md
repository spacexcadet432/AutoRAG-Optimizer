# AutoRAG-Optimizer
 A configurable Retrieval-Augmented Generation (RAG) experimentation framework for systematically evaluating chunking strategies, retrieval depth, and prompt styles using FAISS and embedding-based metrics.


# AutoRAG-Optimizer

A configurable Retrieval-Augmented Generation (RAG) experimentation framework for systematically evaluating chunking strategies, retrieval depth, and prompt styles using FAISS and embedding-based metrics.

---

## 🚀 Problem Statement

Most RAG systems are tuned manually through trial and error.  
There is no structured way to evaluate how chunk size, overlap, retrieval depth (top-k), and prompt strategies impact performance.

This project builds a reproducible evaluation framework to benchmark RAG configurations quantitatively.

---

## 🧠 What This Project Does

- Implements RAG from scratch (no LangChain)
- Uses FAISS for vector similarity search
- Supports configurable:
  - Chunk size
  - Chunk overlap
  - Retrieval depth (top-k)
  - Prompt strategy (basic vs strict grounding)
- Performs grid search over multiple configurations
- Evaluates performance using:
  - Embedding-based answer similarity
  - Retrieval coverage metric
  - Chunk count (computational cost proxy)
- Automatically identifies the best-performing configuration

---

## 🏗 Architecture


Document (.txt)
↓
Chunking (overlap-aware)
↓
OpenAI Embeddings
↓
FAISS Vector Index
↓
Top-k Retrieval
↓
Prompted Generation
↓
Evaluation:
- Answer Similarity
- Retrieval Coverage
- Chunk Count


---

## 📊 Evaluation Metrics

### 1️⃣ Average Similarity
Semantic similarity between generated answer and expected answer using embedding cosine similarity.

### 2️⃣ Retrieval Accuracy
Checks whether the expected answer appears inside retrieved chunks.

### 3️⃣ Number of Chunks
Tracks how chunk size affects index size and computational cost.

---

## 🔬 Example Experiment

Grid search over:

- chunk_size: [300, 500]
- overlap: [50, 100]
- top_k: [3, 5]
- prompt_style: ["basic", "strict"]

The framework automatically ranks configurations by:

1. Average similarity  
2. Retrieval accuracy  

---

## 🛠 Tech Stack

- Python
- FAISS (vector search)
- OpenAI Embeddings API
- NumPy
- python-dotenv

---

## 📂 Project Structure


auto-rag-optimizer/
│
├── data/
│ ├── documents/
│ └── test_questions.json
│
├── rag/
│ ├── chunker.py
│ ├── embedder.py
│ ├── retriever.py
│ ├── generator.py
│ └── pipeline.py
│
├── optimizer/
│ ├── config_generator.py
│ ├── evaluator.py
│ └── experiment_runner.py
│
├── utils/
│ └── metrics.py
│
└── README.md


---

## ▶️ How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt

Add your OpenAI API key in a .env file:

OPENAI_API_KEY=your_key_here

Run experiments:

python -m optimizer.experiment_runner
