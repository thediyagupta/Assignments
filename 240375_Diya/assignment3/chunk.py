from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 50, chunk_overlap: int = 10) -> dict:
    """
    Split input text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text: The input text to split.
        chunk_size: Maximum size of each chunk (default 50).
        chunk_overlap: Overlap between consecutive chunks (default 10).

    Returns:
        A dict with 'chunks' (list of str) and 'total_chunks' (int).

    Raises:
        ValueError: If chunk_size < 1 or chunk_overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return {"chunks": chunks, "total_chunks": len(chunks)}
