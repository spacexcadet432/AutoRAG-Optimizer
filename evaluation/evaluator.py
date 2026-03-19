import json
from typing import Any, Dict, List

from core.pipeline import run_rag_pipeline
from evaluation.metrics import answer_relevance, faithfulness_score, precision_at_k


def load_test_questions(file_path: str) -> List[Dict[str, str]]:
    """
    Load evaluation questions and reference answers from JSON.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_config(document_path: str, test_questions_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the RAG pipeline over a set of test questions for a single config.

    Returns a dictionary with:
        - average_similarity (answer relevance)
        - precision_at_k
        - faithfulness
        - cost metrics (avg chunks, tokens, latency)
        - detailed per-question results
    """

    test_data = load_test_questions(test_questions_path)

    total_relevance = 0.0
    total_faithfulness = 0.0
    total_precision = 0.0

    total_chunks = 0
    total_tokens = 0
    total_latency = 0.0

    detailed_results: List[Dict[str, Any]] = []

    for item in test_data:
        question = item["question"]
        expected_answer = item["expected_answer"]

        result = run_rag_pipeline(document_path, question, config)

        generated_answer = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        num_chunks = result["num_chunks"]
        cost = result["cost"]

        # Quality metrics
        relevance = answer_relevance(generated_answer, expected_answer)
        faithfulness = faithfulness_score(generated_answer, retrieved_chunks)

        # With a single-document corpus, "correct document in top-k"
        # is equivalent to "we retrieved anything at all".
        correct_in_top_k = len(retrieved_chunks) > 0
        p_at_k = precision_at_k(correct_in_top_k, k=config["top_k"])

        # Accumulate
        total_relevance += relevance
        total_faithfulness += faithfulness
        total_precision += p_at_k

        total_chunks += num_chunks
        total_tokens += cost["prompt_tokens_approx"]
        total_latency += cost["latency_s"]

        detailed_results.append(
            {
                "question": question,
                "generated_answer": generated_answer,
                "expected_answer": expected_answer,
                "answer_relevance": relevance,
                "faithfulness": faithfulness,
                "precision_at_k": p_at_k,
                "num_chunks": num_chunks,
                "cost": cost,
            }
        )

    n = len(test_data)
    avg_relevance = total_relevance / n
    avg_faithfulness = total_faithfulness / n
    avg_precision = total_precision / n

    avg_chunks = total_chunks / n
    avg_tokens = total_tokens / n
    avg_latency = total_latency / n

    return {
        "config": config,
        "average_similarity": avg_relevance,
        "precision_at_k": avg_precision,
        "faithfulness": avg_faithfulness,
        "cost": {
            "avg_num_chunks": avg_chunks,
            "avg_prompt_tokens_approx": avg_tokens,
            "avg_latency_s": avg_latency,
        },
        "details": detailed_results,
    }

