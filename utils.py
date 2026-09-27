"""
utils.py — Document loading and text chunking utilities.
Supports PDF, TXT, and Markdown files.
"""

from pypdf import PdfReader


def load_text(filepath: str) -> str:
    """Read a plain-text or Markdown file as UTF-8."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        print(f"Warning: {filepath} appears to be empty.")
    return text


def load_pdf(filepath: str) -> str:
    """Extract all text from a PDF, page by page."""
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)
    return "\n".join(pages)


def load_document(filepath: str) -> str:
    """
    Load a document from disk and return its text content.
    Supports: .pdf, .txt, .md
    """
    if filepath.endswith(".pdf"):
        return load_pdf(filepath)
    elif filepath.endswith((".txt", ".md")):
        return load_text(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:       Full document text.
        chunk_size: Max characters per chunk.
        overlap:    Characters of overlap between adjacent chunks.

    Returns:
        List of text chunks (non-empty only).
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    step = chunk_size - overlap
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]
    # Drop empty or whitespace-only trailing chunks
    return [c for c in chunks if c.strip()]


if __name__ == "__main__":
    import os
    sample = os.path.join("docs", "office_guide.txt")
    doc_text = load_document(sample)
    doc_chunks = chunk_text(doc_text)
    print(f"Total chunks: {len(doc_chunks)}")
    for i, chunk in enumerate(doc_chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")