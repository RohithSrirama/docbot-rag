"""
evaluation.py — Lightweight grounding evaluation for the RAG pipeline.

Usage:
    from evaluation import run_evaluation
    results = run_evaluation(["What is the refund policy?", ...])

Note: The grounding check is a simple word-overlap heuristic, not a full
semantic evaluation (e.g. RAGAS). It is useful for a quick sanity check.
"""

import chromadb
from retrieval import search_and_rerank
from generation import generate_answer

DB_PATH = "./db"

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection    = chroma_client.get_or_create_collection(name="docs")


def evaluate_answer(question: str, answer: str, chunks: list[str]) -> dict:
    """
    Score a single answer against its source chunks.

    Grounding heuristic: at least one word (>=4 chars) in the answer
    must appear in at least one retrieved chunk.

    Returns:
        {
            "grounded":      bool,   # answer is traceable to source chunks
            "answer_length": int,    # word count of the answer
            "used_chunks":   int,    # number of chunks that contributed content
        }
    """
    answer_words  = set(w for w in answer.lower().split() if len(w) >= 4)
    grounded      = False
    used_chunks   = 0

    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in answer_words):
            grounded    = True
            used_chunks += 1

    return {
        "grounded":      grounded,
        "answer_length": len(answer.split()),
        "used_chunks":   used_chunks,
    }


def run_evaluation(questions: list[str]) -> list[dict]:
    """
    Run the full RAG pipeline on a list of questions and evaluate each answer.

    Prints a grounding-rate summary to stdout.

    Returns:
        List of result dicts: [{question, answer, scores}]
    """
    results = []

    for question in questions:
        chunks  = search_and_rerank(question, collection)
        context = "\n\n".join(chunks)
        answer  = generate_answer(question, context)
        scores  = evaluate_answer(question, answer, chunks)
        results.append({"question": question, "answer": answer, "scores": scores})

    total    = len(results)
    grounded = sum(1 for r in results if r["scores"]["grounded"])
    rate     = (grounded / total * 100) if total else 0.0

    print(f"\n{'='*40}")
    print(f"Evaluation Summary")
    print(f"{'='*40}")
    print(f"  Total questions : {total}")
    print(f"  Grounded answers: {grounded}")
    print(f"  Grounding rate  : {rate:.1f}%")
    print(f"{'='*40}\n")

    return results


if __name__ == "__main__":
    sample_questions = [
        "What is the refund policy?",
        "How many sick days are employees allowed?",
        "What are the remote work rules?",
        "Can a visitor enter the office without an escort?",
        "What happens if a chargeback is filed without contacting billing first?",
    ]
    run_evaluation(sample_questions)