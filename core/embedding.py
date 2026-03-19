import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL_NAME = "text-embedding-3-small"


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Compute normalized embeddings for a list of texts.

    Args:
        texts: List of input strings.

    Returns:
        Array of shape (len(texts), dim) with L2-normalized vectors.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]
    embeddings_array = np.asarray(embeddings, dtype="float32")

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    return embeddings_array


def get_single_embedding(text: str) -> np.ndarray:
    """
    Convenience wrapper to get a single embedding.
    """

    return get_embeddings([text])[0]

