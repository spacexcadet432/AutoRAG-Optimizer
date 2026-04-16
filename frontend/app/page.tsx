"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useState } from "react";

type PromptStyle = "basic" | "strict";

type RunRAGResponse = {
  answer: string;
  retrieved_chunks: Array<{ text: string; score: number }>;
  metrics: {
    similarity: number;
    faithfulness: number;
    precision_at_k: number;
    retrieval_coverage: number;
  };
  latency: number;
};

type DatasetStatus = {
  loaded: boolean;
  file_name?: string;
  file_size_bytes?: number;
  chunk_count?: number;
  chunk_size?: number;
  overlap?: number;
};

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "");

export default function HomePage() {
  const [query, setQuery] = useState("What is the refund policy?");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [chunkSize, setChunkSize] = useState<number>(300);
  const [overlap, setOverlap] = useState<number>(50);
  const [topK, setTopK] = useState<number>(3);
  const [promptStyle, setPromptStyle] = useState<PromptStyle>("basic");

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [result, setResult] = useState<RunRAGResponse | null>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>({ loaded: false });

  useEffect(() => {
    const saved = sessionStorage.getItem("autorag_openai_api_key");
    if (saved) {
      setOpenaiApiKey(saved);
    }
  }, []);

  useEffect(() => {
    if (openaiApiKey) {
      sessionStorage.setItem("autorag_openai_api_key", openaiApiKey);
    }
  }, [openaiApiKey]);

  useEffect(() => {
    async function loadStatus() {
      if (!BACKEND_URL) {
        return;
      }
      try {
        const response = await fetch(`${BACKEND_URL}/dataset-status`);
        if (!response.ok) {
          return;
        }
        const data = (await response.json()) as DatasetStatus;
        setDatasetStatus(data);
      } catch {
        // Keep UI usable even if backend health/status check fails.
      }
    }
    void loadStatus();
  }, []);

  const canRun = useMemo(
    () => query.trim().length > 0 && openaiApiKey.trim().length > 0 && datasetStatus.loaded,
    [query, openaiApiKey, datasetStatus.loaded]
  );

  function formatBytes(bytes: number | undefined): string {
    if (!bytes) {
      return "0 B";
    }
    const mb = bytes / (1024 * 1024);
    if (mb >= 1) {
      return `${mb.toFixed(1)} MB`;
    }
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  async function uploadFile(file: File) {
    if (!BACKEND_URL) {
      setUploadError("Backend URL is not configured. Set NEXT_PUBLIC_BACKEND_URL in Vercel project settings.");
      return;
    }
    if (!openaiApiKey.trim()) {
      setUploadError("Please enter your OpenAI API key before uploading.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".txt")) {
      setUploadError("Only .txt files are supported.");
      return;
    }

    setUploading(true);
    setUploadError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("openai_api_key", openaiApiKey);
    formData.append("chunk_size", String(chunkSize));
    formData.append("overlap", String(overlap));

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || "Upload failed");
      }
      setDatasetStatus({
        loaded: true,
        file_name: data.file_name,
        file_size_bytes: data.file_size_bytes,
        chunk_count: data.chunk_count,
        chunk_size: data.chunk_size,
        overlap: data.overlap
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setUploadError(message);
      setDatasetStatus({ loaded: false });
    } finally {
      setUploading(false);
    }
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      await uploadFile(file);
    }
    event.target.value = "";
  }

  async function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      await uploadFile(file);
    }
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
  }

  async function handleResetDataset() {
    if (!BACKEND_URL) {
      return;
    }
    setRunError(null);
    setUploadError(null);
    setResult(null);
    try {
      await fetch(`${BACKEND_URL}/reset-dataset`, { method: "POST" });
    } finally {
      setDatasetStatus({ loaded: false });
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canRun) {
      if (!datasetStatus.loaded) {
        setRunError("Please upload a dataset first.");
      }
      return;
    }

    if (!BACKEND_URL) {
      setRunError("Backend URL is not configured. Set NEXT_PUBLIC_BACKEND_URL in Vercel project settings.");
      return;
    }

    setLoading(true);
    setRunError(null);
    setResult(null);

    try {
      const response = await fetch(`${BACKEND_URL}/run-rag`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query,
          openai_api_key: openaiApiKey,
          config: {
            chunk_size: chunkSize,
            overlap,
            top_k: topK,
            prompt_style: promptStyle
          }
        })
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data?.detail || "Request failed");
      }

      const data = (await response.json()) as RunRAGResponse;
      setResult(data);
      setExpandedIndex(data.retrieved_chunks.length ? 0 : null);
    } catch (err) {
      const defaultMessage =
        "Unable to reach backend. Verify NEXT_PUBLIC_BACKEND_URL points to your Render API and CORS is enabled.";
      const rawMessage = err instanceof Error ? err.message : "";
      const message =
        rawMessage.toLowerCase().includes("failed to fetch") || rawMessage.trim().length === 0
          ? defaultMessage
          : rawMessage;
      setRunError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <h1 className="title">AutoRAG Optimizer Playground</h1>
      <p className="subtitle">Interactive RAG experimentation dashboard (not a chatbot)</p>

      <div className="grid">
        <section className="panel">
          <h2 className="section-title">Input Panel</h2>
          <form onSubmit={handleSubmit}>
            <label htmlFor="query">Query</label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question to evaluate RAG behavior..."
            />

            <label htmlFor="apiKey">OpenAI API Key (session only)</label>
            <input
              id="apiKey"
              type="password"
              value={openaiApiKey}
              onChange={(e) => setOpenaiApiKey(e.target.value)}
              placeholder="sk-..."
            />

            <h3 className="section-title" style={{ marginTop: 16 }}>
              Dataset Upload
            </h3>
            <div className="upload-hint">Recommended: files under 3MB.</div>
            <div className="upload-hint">Your data is processed in-memory and not stored.</div>

            <div className="upload-dropzone" onDrop={handleDrop} onDragOver={handleDragOver}>
              Drag and drop a .txt file here
            </div>
            <label htmlFor="datasetFile">Or Upload .txt File</label>
            <input id="datasetFile" type="file" accept=".txt,text/plain" onChange={handleFileChange} />
            {uploadError && <div className="error" style={{ marginTop: 10 }}>{uploadError}</div>}
            {uploading && <div className="upload-status">Uploading and building embeddings...</div>}
            {datasetStatus.loaded && (
              <div className="upload-status">
                Dataset loaded: {datasetStatus.file_name} ({formatBytes(datasetStatus.file_size_bytes)},{" "}
                {datasetStatus.chunk_count} chunks)
              </div>
            )}
            <button type="button" onClick={handleResetDataset} className="secondary-btn">
              Reset Dataset
            </button>

            <label htmlFor="chunkSize">Chunk Size</label>
            <select
              id="chunkSize"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
            >
              {[200, 300, 500].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>

            <label htmlFor="overlap">Overlap</label>
            <select id="overlap" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))}>
              {[0, 50, 100].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>

            <label htmlFor="topK">Top-k</label>
            <select id="topK" value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {[2, 3, 5].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>

            <label htmlFor="promptStyle">Prompt Style</label>
            <select
              id="promptStyle"
              value={promptStyle}
              onChange={(e) => setPromptStyle(e.target.value as PromptStyle)}
            >
              <option value="basic">basic</option>
              <option value="strict">strict</option>
            </select>

            <button type="submit" disabled={!canRun || loading || uploading}>
              {loading ? "Running experiment..." : "Run RAG Experiment"}
            </button>
          </form>
        </section>

        <section className="panel">
          <h2 className="section-title">Output Panel</h2>

          {loading && <div className="answer-box">Running pipeline, retrieving chunks, and scoring metrics...</div>}
          {runError && <div className="error">{runError}</div>}

          {result && (
            <>
              <h3 className="section-title">Answer</h3>
              <div className="answer-box">{result.answer}</div>

              <h3 className="section-title" style={{ marginTop: 16 }}>
                Metrics
              </h3>
              <div className="metrics-grid">
                <MetricCard label="Similarity" value={result.metrics.similarity} />
                <MetricCard label="Faithfulness" value={result.metrics.faithfulness} />
                <MetricCard label="Precision@k" value={result.metrics.precision_at_k} />
                <MetricCard label="Coverage" value={result.metrics.retrieval_coverage} />
                <MetricCard label="Latency (s)" value={result.latency} />
              </div>

              <h3 className="section-title" style={{ marginTop: 16 }}>
                Retrieved Chunks
              </h3>
              {result.retrieved_chunks.map((chunk, idx) => {
                const expanded = expandedIndex === idx;
                return (
                  <article className="chunk-card" key={idx}>
                    <button
                      type="button"
                      className="chunk-header"
                      onClick={() => setExpandedIndex(expanded ? null : idx)}
                    >
                      <span>Chunk #{idx + 1}</span>
                      <span>score: {chunk.score.toFixed(4)} {expanded ? "▲" : "▼"}</span>
                    </button>
                    {expanded && <div className="chunk-text">{chunk.text}</div>}
                  </article>
                );
              })}
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value.toFixed(4)}</div>
    </div>
  );
}

