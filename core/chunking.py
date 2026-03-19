from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: Full document text.
        chunk_size: Number of characters per chunk.
        overlap: Number of characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

