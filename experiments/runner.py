import json
import os
from typing import Any, Dict, List

from evaluation.evaluator import evaluate_config
from experiments.config_generator import generate_configs


RESULTS_PATH = os.path.join("experiments", "results.jsonl")


def _append_result(result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def run_experiments(document_path: str, test_questions_path: str) -> Dict[str, Any]:
    """
    Run a grid of RAG configurations and log results to JSONL.
    """

    configs = generate_configs()
    all_results: List[Dict[str, Any]] = []

    print(f"\nTotal configurations to test: {len(configs)}\n")

    for idx, config in enumerate(configs, start=1):
        print(f"Running config {idx}/{len(configs)}: {config}")

        result = evaluate_config(document_path, test_questions_path, config)

        print(f"Average similarity: {result['average_similarity']:.4f}")
        print(f"Precision@k:       {result['precision_at_k']:.4f}")
        print(f"Faithfulness:      {result['faithfulness']:.4f}")
        print(f"Cost (chunks):     {result['cost']['avg_num_chunks']:.2f}")
        print(f"Cost (tokens~):    {result['cost']['avg_prompt_tokens_approx']:.2f}")
        print(f"Latency (s):       {result['cost']['avg_latency_s']:.3f}")
        print("-" * 60)

        all_results.append(result)
        _append_result(result)

    # Rank configs by similarity then precision@k
    all_results.sort(
        key=lambda x: (x["average_similarity"], x["precision_at_k"]),
        reverse=True,
    )

    best_result = all_results[0]

    print("\n==============================")
    print("BEST CONFIG FOUND")
    print(best_result["config"])
    print(f"Average similarity: {best_result['average_similarity']:.4f}")
    print(f"Precision@k:        {best_result['precision_at_k']:.4f}")
    print(f"Faithfulness:       {best_result['faithfulness']:.4f}")
    print("==============================\n")

    return best_result


if __name__ == "__main__":
    document_path = "data/documents/sample.txt"
    test_questions_path = "data/test_questions.json"
    run_experiments(document_path, test_questions_path)

