"""Render-ready FastAPI backend entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import MetricsResponse, RetrievedChunkResponse, RunRAGRequest, RunRAGResponse
from services.rag_service import initialize_runtime, run_rag_experiment


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load corpus and available FAISS indexes at startup."""

    initialize_runtime(document_path="data/documents/sample.txt")
    yield


app = FastAPI(title="AutoRAG Optimizer API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Health endpoint for Render checks."""

    return {"status": "ok"}


@app.post("/run-rag", response_model=RunRAGResponse)
def run_rag(payload: RunRAGRequest) -> RunRAGResponse:
    """
    Run one RAG experiment. User key is accepted per request and not persisted.
    """

    if payload.config.overlap >= payload.config.chunk_size:
        raise HTTPException(status_code=400, detail="overlap must be smaller than chunk_size")

    try:
        answer, retrieved, metrics, latency = run_rag_experiment(
            query=payload.query,
            config=payload.config.model_dump(),
            openai_api_key=payload.openai_api_key,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG run failed: {exc}") from exc

    return RunRAGResponse(
        answer=answer,
        retrieved_chunks=[RetrievedChunkResponse(text=c.text, score=c.score) for c in retrieved],
        metrics=MetricsResponse(**metrics),
        latency=latency,
    )

