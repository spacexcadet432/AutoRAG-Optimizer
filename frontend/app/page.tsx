"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

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

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "");

export default function HomePage() {
  const [query, setQuery] = useState("What is the refund policy?");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [chunkSize, setChunkSize] = useState<number>(300);
  const [overlap, setOverlap] = useState<number>(50);
  const [topK, setTopK] = useState<number>(3);
  const [promptStyle, setPromptStyle] = useState<PromptStyle>("basic");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunRAGResponse | null>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);

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

  const canRun = useMemo(
    () => query.trim().length > 0 && openaiApiKey.trim().length > 0,
    [query, openaiApiKey]
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canRun) {
      return;
    }

    if (!BACKEND_URL) {
      setError("Backend URL is not configured. Set NEXT_PUBLIC_BACKEND_URL in Vercel project settings.");
      return;
    }

    setLoading(true);
    setError(null);
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
      setError(message);
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

            <button type="submit" disabled={!canRun || loading}>
              {loading ? "Running experiment..." : "Run RAG Experiment"}
            </button>
          </form>
        </section>

        <section className="panel">
          <h2 className="section-title">Output Panel</h2>

          {loading && <div className="answer-box">Running pipeline, retrieving chunks, and scoring metrics...</div>}
          {error && <div className="error">{error}</div>}

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

