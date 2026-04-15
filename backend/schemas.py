"""Pydantic schemas for backend API contracts."""

from typing import Literal, List

from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Request config for one RAG run."""

    chunk_size: int = Field(default=300, ge=100, le=4000)
    overlap: int = Field(default=50, ge=0, le=2000)
    top_k: int = Field(default=3, ge=1, le=20)
    prompt_style: Literal["basic", "strict"] = "basic"


class RunRAGRequest(BaseModel):
    """POST /run-rag request."""

    query: str = Field(min_length=1)
    config: RAGConfig
    openai_api_key: str = Field(min_length=10)


class RetrievedChunkResponse(BaseModel):
    """Retrieved chunk for UI rendering."""

    text: str
    score: float


class MetricsResponse(BaseModel):
    """Online metric bundle."""

    similarity: float
    faithfulness: float
    precision_at_k: float
    retrieval_coverage: float


class RunRAGResponse(BaseModel):
    """POST /run-rag response."""

    answer: str
    retrieved_chunks: List[RetrievedChunkResponse]
    metrics: MetricsResponse
    latency: float

