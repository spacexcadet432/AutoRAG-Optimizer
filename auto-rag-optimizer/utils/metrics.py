import numpy as np
from rag.embedder import get_embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    """
    Compute cosine similarity between two normalized vectors.
    """
    return float(np.dot(vec1, vec2.T))


def score_answer(generated_answer: str, expected_answer: str):
    """
    Compute similarity score between generated and expected answers
    using embedding cosine similarity.
    """

    embeddings = get_embeddings([generated_answer, expected_answer])

    gen_vec = embeddings[0]
    exp_vec = embeddings[1]

    similarity = cosine_similarity(gen_vec, exp_vec)

    return similarity



if __name__ == "__main__":
    score = score_answer(
        "The refund policy is 14 days.",
        "Customers can get a refund within 14 days of purchase."
    )

    print("Similarity Score:", score)