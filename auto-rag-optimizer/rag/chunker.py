def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): Full document text
        chunk_size (int): Number of characters per chunk
        overlap (int): Overlap size between chunks

    Returns:
        List[str]: List of text chunks
    """

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks



if __name__ == "__main__":
    sample_text = "This is a test document. " * 100
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)

    print(f"Total chunks: {len(chunks)}")
    print("First chunk:\n", chunks[0])