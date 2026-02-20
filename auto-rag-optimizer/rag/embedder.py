import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts: list):
    """
    Takes a list of strings and returns normalized embeddings as numpy array.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]

    embeddings_array = np.array(embeddings).astype("float32")

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    return embeddings_array


def get_single_embedding(text: str):
    """
    Returns normalized embedding for a single string.
    """

    embedding = get_embeddings([text])
    return embedding




if __name__ == "__main__":
    sample_texts = ["Refund policy is 14 days.", "Neural networks use backpropagation."]
    vectors = get_embeddings(sample_texts)

    print("Shape:", vectors.shape)
    print("First vector (first 5 values):", vectors[0][:5])