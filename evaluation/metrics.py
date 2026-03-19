from typing import List

import numpy as np

from core.embedding import get_embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalized vectors.
    """

    return float(np.dot(vec1, vec2.T))


def answer_relevance(generated_answer: str, expected_answer: str) -> float:
    """
    Embed generated and expected answers and compute cosine similarity.
    """

    embeddings = get_embeddings([generated_answer, expected_answer])
    gen_vec = embeddings[0]
    exp_vec = embeddings[1]
    return cosine_similarity(gen_vec, exp_vec)


def faithfulness_score(generated_answer: str, retrieved_chunks: List[str]) -> float:
    """
    Check whether the generated answer text appears in any retrieved chunk.

    Returns:
        1.0 if the answer is fully contained in at least one chunk (case-insensitive),
        otherwise 0.0. This can be averaged across examples.
    """

    answer_lower = generated_answer.strip().lower()
    if not answer_lower:
        return 0.0

    for chunk in retrieved_chunks:
        if answer_lower in chunk.lower():
            return 1.0

    return 0.0


def precision_at_k(correct_in_top_k: bool, k: int) -> float:
    """
    Simple precision@k for binary "correct document found in top-k" setting.

    Args:
        correct_in_top_k: Whether the gold document appeared in the retrieved set.
        k: The number of retrieved items.
    """

    if k <= 0:
        return 0.0
    return 1.0 / float(k) if correct_in_top_k else 0.0

