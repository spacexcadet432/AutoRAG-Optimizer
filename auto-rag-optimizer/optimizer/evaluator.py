import json
from rag.pipeline import run_rag_pipeline
from utils.metrics import score_answer


def load_test_questions(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_config(document_path: str, test_questions_path: str, config: dict):
    test_data = load_test_questions(test_questions_path)

    total_similarity = 0
    retrieval_success_count = 0
    total_chunks = None

    detailed_results = []

    for item in test_data:
        question = item["question"]
        expected_answer = item["expected_answer"]

        result = run_rag_pipeline(document_path, question, config)

        generated_answer = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        num_chunks = result["num_chunks"]

        # Save total chunks once
        total_chunks = num_chunks

        # Similarity score
        similarity = score_answer(generated_answer, expected_answer)
        total_similarity += similarity

        # Retrieval coverage check
        retrieval_success = any(
            expected_answer.lower() in chunk.lower()
            for chunk in retrieved_chunks
        )

        if retrieval_success:
            retrieval_success_count += 1

        detailed_results.append({
            "question": question,
            "similarity_score": similarity,
            "retrieval_success": retrieval_success
        })

    avg_similarity = total_similarity / len(test_data)
    retrieval_accuracy = retrieval_success_count / len(test_data)

    return {
        "config": config,
        "average_similarity": avg_similarity,
        "retrieval_accuracy": retrieval_accuracy,
        "num_chunks": total_chunks,
        "details": detailed_results
    }