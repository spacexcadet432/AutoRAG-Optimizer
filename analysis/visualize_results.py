import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_PATH = os.path.join("experiments", "results.jsonl")


def load_results(path: str = RESULTS_PATH) -> List[Dict[str, Any]]:
    """
    Load experiment results from JSONL.
    """

    results: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run experiments first.")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    return results


def _results_to_dataframe(results: List[Dict[str, Any]]):
    """
    Convert raw result dicts into a Pandas DataFrame-like structure.

    We lazily import pandas so that the core library does not depend on it.
    """

    import pandas as pd

    rows = []
    for r in results:
        cfg = r["config"]
        cost = r["cost"]
        rows.append(
            {
                "chunk_size": cfg["chunk_size"],
                "top_k": cfg["top_k"],
                "overlap": cfg["overlap"],
                "prompt_style": cfg["prompt_style"],
                "average_similarity": r["average_similarity"],
                "precision_at_k": r["precision_at_k"],
                "faithfulness": r["faithfulness"],
                "avg_num_chunks": cost["avg_num_chunks"],
                "avg_prompt_tokens_approx": cost["avg_prompt_tokens_approx"],
                "avg_latency_s": cost["avg_latency_s"],
            }
        )

    return pd.DataFrame(rows)


def plot_chunk_size_vs_topk_heatmap(df) -> None:
    """
    Heatmap of quality (average similarity) as a function of chunk_size and top_k.
    """

    pivot = df.pivot_table(
        values="average_similarity",
        index="chunk_size",
        columns="top_k",
        aggfunc="mean",
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Average Similarity vs. Chunk Size and top_k")
    plt.ylabel("chunk_size")
    plt.xlabel("top_k")
    plt.tight_layout()


def plot_quality_vs_cost(df) -> None:
    """
    Scatter plot of quality vs cost proxy (tokens).
    """

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="avg_prompt_tokens_approx",
        y="average_similarity",
        hue="chunk_size",
        style="top_k",
        palette="viridis",
    )
    plt.title("Quality vs Cost (Approx Tokens)")
    plt.xlabel("Approx prompt tokens")
    plt.ylabel("Average similarity")
    plt.tight_layout()


def plot_prompt_style_comparison(df) -> None:
    """
    Bar chart comparing prompt styles by average similarity.
    """

    grouped = df.groupby("prompt_style")["average_similarity"].mean().reset_index()

    plt.figure(figsize=(4, 4))
    sns.barplot(data=grouped, x="prompt_style", y="average_similarity")
    plt.title("Prompt Style Comparison (Average Similarity)")
    plt.xlabel("Prompt style")
    plt.ylabel("Average similarity")
    plt.tight_layout()


def main() -> None:
    results = load_results()
    df = _results_to_dataframe(results)

    sns.set_theme(style="whitegrid")

    plot_chunk_size_vs_topk_heatmap(df)
    plot_quality_vs_cost(df)
    plot_prompt_style_comparison(df)

    plt.show()


if __name__ == "__main__":
    main()

