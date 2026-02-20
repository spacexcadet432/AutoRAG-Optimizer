import faiss
import numpy as np


class FaissRetriever:
    def __init__(self, embeddings: np.ndarray, chunks: list):
        """
        Initialize FAISS retriever.

        Args:
            embeddings (np.ndarray): Normalized embeddings (num_chunks, dimension)
            chunks (list): List of chunk texts (same order as embeddings)
        """

        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        self.chunks = chunks
        self.dimension = embeddings.shape[1]

        # Create FAISS index using Inner Product
        # Since embeddings are normalized, this equals cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)

        # Add embeddings to FAISS index
        self.index.add(embeddings)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Retrieve top-k most similar chunks for a query.

        Args:
            query_embedding (np.ndarray): Normalized query embedding (1, dimension)
            top_k (int): Number of results to retrieve

        Returns:
            List[str]: Retrieved chunk texts
        """

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks = []

        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])

        return retrieved_chunks