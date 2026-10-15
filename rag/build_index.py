#!/usr/bin/env python3
"""
build_index.py — Build a Chroma vector store from the TB corpus for RAG retrieval.

Mirrors Marine_RAG/build_index.py from the PLVis project, adapted to:
  - run entirely locally (no OpenAI dependency) via a sentence-transformers
    biomedical retrieval model instead of OpenAIEmbeddings
  - index the full v5 combined/deduped corpus (489,423 chunks)

Run this once, and re-run (after deleting CHROMA_PERSIST_DIR) whenever the
corpus changes.

Usage:
    python build_index.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_PATH        = Path("/uss/skavlak/version_5/corpus_v5_train.jsonl")
CHROMA_PERSIST_DIR = Path("/uss/skavlak/tb_rag_chroma_db")
COLLECTION_NAME    = "tb_rag_v1"
EMBEDDING_MODEL    = "pritamdeka/S-PubMedBert-MS-MARCO"  # biomedical retrieval-tuned
BATCH_SIZE         = 256  # local GPU encoding handles much larger batches than an API


# ---------------------------------------------------------------------------
# Local embedding wrapper (langchain_core.embeddings.Embeddings interface)
# ---------------------------------------------------------------------------

class LocalSentenceTransformerEmbeddings(Embeddings):
    """Wraps sentence-transformers so Chroma can use a local, free model in
    place of OpenAIEmbeddings. No API calls, no cost, runs on GPU if available."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cuda")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )
        return emb[0].tolist()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_documents(jsonl_path: Path) -> List[Document]:
    docs: List[Document] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = (item.get("text") or "").strip()
            if not text:
                continue
            # Each chunk's text already starts with a self-citing
            # "[Document: X | Chapter: Y | Section: Z | Page: N]" header
            # (see extract_corpus.py), so no separate sources block is needed
            # at generation time - the retrieved text is already attributable.
            metadata = {k: v for k, v in item.items() if k != "text"}
            metadata.setdefault("content_type", "text")
            for key in list(metadata.keys()):
                if isinstance(metadata[key], (dict, list)):
                    metadata[key] = json.dumps(metadata[key])
                elif metadata[key] is None:
                    metadata[key] = ""
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_index() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} not found.")

    print(f"Loading documents from {CORPUS_PATH}...")
    docs = load_documents(CORPUS_PATH)
    print(f"  {len(docs)} chunks loaded")

    print(f"\nInitialising local embedding model ({EMBEDDING_MODEL})...")
    embeddings = LocalSentenceTransformerEmbeddings(EMBEDDING_MODEL)
    test = embeddings.embed_query("test")
    print(f"  Embedding dim: {len(test)}")

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nIndexing into Chroma ({CHROMA_PERSIST_DIR} / {COLLECTION_NAME})...")
    start = time.time()
    vectorstore = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_PERSIST_DIR),
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)
        done = min(i + BATCH_SIZE, len(docs))
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(docs) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(docs)}]  {rate:.0f} chunks/s  ETA {eta/60:.1f} min", end="\r", flush=True)

    elapsed = time.time() - start
    total = vectorstore._collection.count()
    print(f"\n  Done in {elapsed/60:.1f} min  —  {total} vectors stored")
    print(f"\nChroma store ready at: {CHROMA_PERSIST_DIR.resolve()}")


if __name__ == "__main__":
    build_index()
