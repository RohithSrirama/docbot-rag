"""
ingestion.py — Load documents from the docs/ folder and store them in ChromaDB.

Run once (or whenever documents change):
    python ingestion.py
"""

import os
import chromadb
from utils import load_document, chunk_text

DOCS_DIR = "docs"
DB_PATH  = "./db"

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="docs")

SUPPORTED = {".pdf", ".txt", ".md"}


def ingest_all(docs_dir: str = DOCS_DIR) -> None:
    files = [f for f in os.listdir(docs_dir)
             if os.path.splitext(f)[1].lower() in SUPPORTED]

    if not files:
        print(f"No supported documents found in '{docs_dir}'.")
        return

    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        try:
            text   = load_document(filepath)
            chunks = chunk_text(text)
            ids       = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename} for _ in chunks]

            collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
            print(f"  Ingested {len(chunks):>4} chunks  <-  {filename}")
        except Exception as exc:
            print(f"  ERROR ingesting {filename}: {exc}")

    print(f"\nTotal chunks in collection: {collection.count()}")


if __name__ == "__main__":
    print(f"Ingesting documents from '{DOCS_DIR}'...\n")
    ingest_all()
