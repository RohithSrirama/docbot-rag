"""
main.py — CLI interface for the DocBot RAG chatbot.

Usage:
    python main.py

Commands:
    /quit   — exit the chatbot
    /clear  — reset conversation history for this session
"""

import os
import sys

# Force UTF-8 output on Windows to prevent UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chromadb
from dotenv import load_dotenv
from groq import Groq

from generation import generate_answer
from memory import add_message, get_history, rewrite_query
from retrieval import search_and_rerank

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./db")
collection    = chroma_client.get_or_create_collection(name="docs")
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))

messages: list = []


def run_cli() -> None:
    print("\n=== DocBot — Document Assistant ===")
    print("Commands: /quit to exit, /clear to reset history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            print("Conversation history cleared.\n")
            continue

        # 1. Rewrite query using conversation context
        history = get_history(messages)
        search_query = rewrite_query(user_input, history, groq_client) if history else user_input

        # 2. Retrieve and rerank
        top_chunks = search_and_rerank(search_query, collection)

        if not top_chunks:
            reply = "I don't have information about that in the company documents."
            print(f"\nDocBot: {reply}\n")
            add_message(messages, "user",      user_input)
            add_message(messages, "assistant", reply)
            continue

        context = "\n\n".join(top_chunks)

        # 3. Generate answer (streams to stdout token by token)
        print("\nDocBot: ", end="", flush=True)
        answer = generate_answer(search_query, context)

        # 4. Save exchange
        add_message(messages, "user",      user_input)
        add_message(messages, "assistant", answer)

        # 5. Show sources
        raw = collection.query(query_texts=[search_query], n_results=3)["metadatas"][0]
        sources = list(dict.fromkeys(m["source"] for m in raw))
        print(f"Sources: {', '.join(sources)}\n")


if __name__ == "__main__":
    run_cli()