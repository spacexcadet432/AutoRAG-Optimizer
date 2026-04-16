"""Render-ready FastAPI backend entrypoint."""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    DatasetStatusResponse,
    MetricsResponse,
    RetrievedChunkResponse,
    RunRAGRequest,
    RunRAGResponse,
    UploadResponse,
)
from services.rag_service import get_dataset_status, reset_dataset, run_rag_experiment, upload_dataset


app = FastAPI(title="AutoRAG Optimizer API", version="1.2.0")

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


@app.get("/dataset-status", response_model=DatasetStatusResponse)
def dataset_status() -> DatasetStatusResponse:
    """Get current in-memory upload status."""

    return DatasetStatusResponse(**get_dataset_status())


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    openai_api_key: str = Form(...),
    chunk_size: int = Form(500),
    overlap: int = Form(100),
) -> UploadResponse:
    """
    Upload a .txt dataset, build embeddings and FAISS index, and store in memory.
    """

    if not openai_api_key.strip():
        raise HTTPException(status_code=400, detail="Missing API key.")

    file_bytes = await file.read()
    try:
        state = upload_dataset(
            file_name=file.filename or "uploaded.txt",
            file_bytes=file_bytes,
            openai_api_key=openai_api_key,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return UploadResponse(
        message="Dataset uploaded and indexed successfully.",
        file_name=state.file_name,
        file_size_bytes=state.file_size_bytes,
        chunk_count=len(state.chunks),
        chunk_size=state.chunk_size,
        overlap=state.overlap,
    )


@app.post("/reset-dataset")
def reset_uploaded_dataset() -> dict:
    """Clear currently uploaded in-memory dataset."""

    reset_dataset()
    return {"message": "Dataset reset successfully."}


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
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG run failed: {exc}") from exc

    return RunRAGResponse(
        answer=answer,
        retrieved_chunks=[RetrievedChunkResponse(text=c.text, score=c.score) for c in retrieved],
        metrics=MetricsResponse(**metrics),
        latency=latency,
    )

