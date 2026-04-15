"""FastAPI application exposing the RAG experimentation API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import RunRAGRequest, RunRAGResponse, RetrievedChunkResponse, MetricsResponse
from services.rag_service import run_rag_experiment


app = FastAPI(title="AutoRAG Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Simple health endpoint for deployment checks."""

    return {"status": "ok"}


@app.post("/run-rag", response_model=RunRAGResponse)
def run_rag(payload: RunRAGRequest) -> RunRAGResponse:
    """
    Execute one RAG experiment run for an incoming playground request.

    The API key is only used for this request and is never persisted.
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

