# DocBot — RAG Document Q&A Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents. Built with a hybrid retrieval pipeline, cross-encoder reranking, multi-turn conversation memory, and a ChatGPT-like streaming web interface.

---

## Features

- **Hybrid Retrieval** — combines ChromaDB vector search with BM25 keyword search for better recall
- **Cross-Encoder Reranking** — re-scores retrieved chunks using a cross-encoder model for higher precision
- **Query Rewriting** — rewrites follow-up questions into standalone queries for accurate multi-turn retrieval
- **Streaming Responses** — answers stream token-by-token via Server-Sent Events (SSE), no waiting
- **Source Attribution** — each answer shows which document it came from
- **ChatGPT-like Web UI** — dark-themed interface with sidebar, suggestion cards, and real-time typing
- **Evaluation Module** — lightweight grounding check to measure answer quality

---

## Architecture

```
User (Browser)
     |
     | HTTP / SSE
     v
Flask Web Server  (app.py)
     |
     |-- Query Rewriting  (memory.py)
     |-- Hybrid Retrieval (retrieval.py)
     |       |-- ChromaDB vector search
     |       |-- BM25 keyword search
     |       |-- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
     |
     |-- Answer Generation (generation.py)
             |-- Groq API  (openai/gpt-oss-120b)
             |-- Streaming output
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM API | [Groq](https://groq.com) |
| Vector Store | [ChromaDB](https://www.trychroma.com) |
| Embeddings | sentence-transformers (default ChromaDB) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Keyword Search | BM25 (rank-bm25) |
| Backend | Flask + Flask-CORS |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Document Parsing | pypdf, plain text |

---

## Project Structure

```
.
├── app.py            # Flask backend — serves UI and /chat SSE endpoint
├── ingestion.py      # Load and chunk documents into ChromaDB
├── retrieval.py      # Hybrid search + cross-encoder reranking
├── generation.py     # Answer generation with Groq streaming API
├── memory.py         # Conversation history + query rewriting
├── evaluation.py     # Lightweight grounding evaluation
├── utils.py          # Document loading (PDF, TXT) and text chunking
├── main.py           # CLI interface (alternative to web UI)
├── static/
│   └── index.html    # ChatGPT-like web interface
├── docs/             # Place your documents here (PDF, TXT, MD)
├── db/               # ChromaDB persistent store (auto-created)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docbot-rag.git
cd docbot-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your [Groq API key](https://console.groq.com/keys):

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add your documents

Place your `.pdf`, `.txt`, or `.md` files inside the `docs/` folder.

### 6. Ingest documents into the vector store

```bash
python ingestion.py
```

### 7. Run the web interface

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## CLI Mode

If you prefer the terminal instead of the web UI:

```bash
python main.py
```

---

## Evaluation

Run a basic grounding evaluation against a set of questions:

```python
from evaluation import run_evaluation

questions = [
    "What is the refund policy?",
    "How many sick days are allowed?",
    "What are the remote work rules?",
]

results = run_evaluation(questions)
```

---

## How It Works

1. **Ingestion** — Documents are loaded, chunked (1000 chars, 200 overlap), and stored in ChromaDB with metadata.
2. **Query Rewriting** — If there is prior conversation context, the follow-up question is rewritten into a standalone query using the LLM.
3. **Hybrid Retrieval** — The rewritten query is searched using both ChromaDB (semantic) and BM25 (keyword). Scores are combined with a weighted alpha parameter.
4. **Reranking** — A cross-encoder model rescores the top candidates and returns the highest-quality chunks.
5. **Generation** — The context chunks are passed to the LLM with a strict system prompt. The answer streams back token-by-token.
6. **Source Display** — The source filenames are attached to each response so the user knows where the information came from.

---

## Notes

- The `db/` folder (ChromaDB) is excluded from git. Run `ingestion.py` after cloning to rebuild it.
- The `.env` file is excluded from git. Never commit API keys.
- The web UI session history is stored in memory and resets when the server restarts.
