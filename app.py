import os
import sys

# Force UTF-8 output on Windows to prevent UnicodeEncodeError from cp1252
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import chromadb
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

from memory import add_message, get_history, rewrite_query
from retrieval import search_and_rerank
from generation import generate_answer_stream

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(name="docs")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Per-session conversation history stored in memory
# For a production app, use a proper session store
conversation_store: dict = {}


def get_session_messages(session_id: str) -> list:
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    messages = get_session_messages(session_id)
    history = get_history(messages)

    # Query rewriting when context exists
    if history:
        rewritten = rewrite_query(user_input, history, groq_client)
        search_query = rewritten.strip() if rewritten and rewritten.strip() else user_input
    else:
        search_query = user_input

    # Retrieve and rerank
    top_chunks = search_and_rerank(search_query, collection)

    if not top_chunks:
        fallback = "I don't have information about that in my documents."
        add_message(messages, "user", user_input)
        add_message(messages, "assistant", fallback)

        def no_docs():
            yield f"data: {json.dumps({'token': fallback})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"

        return Response(no_docs(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    context = "\n\n".join(top_chunks)

    # Fetch sources
    raw_sources = collection.query(
        query_texts=[search_query], n_results=3
    )["metadatas"][0]
    sources = list(dict.fromkeys(m["source"] for m in raw_sources))

    def stream_response():
        full_response = ""
        for chunk in generate_answer_stream(search_query, context):
            full_response += chunk
            yield f"data: {json.dumps({'token': chunk})}\n\n"

        add_message(messages, "user", user_input)
        add_message(messages, "assistant", full_response)
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return Response(stream_response(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    conversation_store.pop(session_id, None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
