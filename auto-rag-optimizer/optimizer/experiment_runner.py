from optimizer.evaluator import evaluate_config
from optimizer.config_generator import generate_configs


def run_experiments(document_path: str, test_questions_path: str):
    configs = generate_configs()

    all_results = []

    print(f"\nTotal Configurations to Test: {len(configs)}\n")

    for idx, config in enumerate(configs, start=1):
        print(f"Running config {idx}/{len(configs)}: {config}")

        result = evaluate_config(document_path, test_questions_path, config)

        print(f"Average Similarity: {result['average_similarity']:.4f}")
        print(f"Retrieval Accuracy: {result['retrieval_accuracy']:.4f}")
        print(f"Number of Chunks: {result['num_chunks']}")
        print("-" * 50)

        all_results.append(result)

    # Sort by similarity first, then retrieval accuracy
    all_results.sort(
        key=lambda x: (x["average_similarity"], x["retrieval_accuracy"]),
        reverse=True
    )

    best_result = all_results[0]

    print("\n==============================")
    print("🏆 BEST CONFIG FOUND:")
    print(best_result["config"])
    print(f"Average Similarity: {best_result['average_similarity']:.4f}")
    print(f"Retrieval Accuracy: {best_result['retrieval_accuracy']:.4f}")
    print(f"Number of Chunks: {best_result['num_chunks']}")
    print("==============================\n")

    print("📊 TOP 3 CONFIGS:\n")
    for i in range(min(3, len(all_results))):
        r = all_results[i]
        print(f"Rank {i+1}:")
        print("Config:", r["config"])
        print(f"Average Similarity: {r['average_similarity']:.4f}")
        print(f"Retrieval Accuracy: {r['retrieval_accuracy']:.4f}")
        print(f"Number of Chunks: {r['num_chunks']}")
        print("-" * 50)

    return best_result


if __name__ == "__main__":
    document_path = "data/documents/sample.txt"
    test_questions_path = "data/test_questions.json"

    run_experiments(document_path, test_questions_path)