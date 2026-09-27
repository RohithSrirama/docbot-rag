"""
retrieval.py — Hybrid retrieval (ChromaDB vector + BM25) with cross-encoder reranking.

The module-level objects (reranker, BM25 index) are initialised once at import time
so they are shared across all requests handled by the Flask server.
"""

import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

DB_PATH = "./db"

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection    = chroma_client.get_or_create_collection(name="docs")

# Cross-encoder reranker — loaded once at startup
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# BM25 index built over the entire corpus — rebuilt if collection changes
_all_data       = collection.get()
_all_chunks     = _all_data["documents"]
_tokenized      = [chunk.lower().split() for chunk in _all_chunks]
bm25            = BM25Okapi(_tokenized)


def search_and_rerank(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 3,
    retrieve_count: int = 10,
    alpha: float = 0.5,
) -> list[str]:
    """
    Retrieve the top-k most relevant chunks for a query using hybrid scoring.

    Pipeline:
        1. ChromaDB semantic search → retrieve_count candidates.
        2. BM25 keyword scoring over those candidates.
        3. Cross-encoder reranking for precision.
        4. Weighted combination of reranker + BM25 scores.

    Args:
        query:          The user's (possibly rewritten) question.
        collection:     ChromaDB collection to search.
        top_k:          Number of final chunks to return.
        retrieve_count: Initial candidate pool size from ChromaDB.
        alpha:          Weight for reranker score (1-alpha for BM25).

    Returns:
        List of up to top_k chunk strings, best first.
    """
    if not query.strip():
        return []

    # ── 1. Semantic retrieval ──────────────────────────────────────────────
    chroma_results = collection.query(
        query_texts=[query],
        n_results=min(retrieve_count, collection.count()),
    )
    chroma_chunks = chroma_results["documents"][0]

    if not chroma_chunks:
        return []

    # ── 2. BM25 scores for the candidate set ─────────────────────────────
    tokenized_query   = query.lower().split()
    bm25_all_scores   = bm25.get_scores(tokenized_query)

    chroma_bm25_scores = []
    for chunk in chroma_chunks:
        try:
            idx = _all_chunks.index(chunk)
            chroma_bm25_scores.append(float(bm25_all_scores[idx]))
        except ValueError:
            chroma_bm25_scores.append(0.0)

    # ── 3. Cross-encoder reranking ────────────────────────────────────────
    pairs         = [(query, chunk) for chunk in chroma_chunks]
    rerank_scores = reranker.predict(pairs).tolist()

    # ── 4. Min-max normalise both score vectors ───────────────────────────
    def normalise(scores: list[float]) -> list[float]:
        lo, hi = min(scores), max(scores)
        span = hi - lo + 1e-10
        return [(s - lo) / span for s in scores]

    rerank_norm = normalise(rerank_scores)
    bm25_norm   = normalise(chroma_bm25_scores)

    # ── 5. Combine and rank ───────────────────────────────────────────────
    combined = [
        alpha * rerank_norm[i] + (1 - alpha) * bm25_norm[i]
        for i in range(len(chroma_chunks))
    ]

    scored = sorted(zip(combined, chroma_chunks), reverse=True)
    return [chunk for _, chunk in scored[:top_k]]