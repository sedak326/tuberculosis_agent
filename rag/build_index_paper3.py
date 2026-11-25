#!/usr/bin/env python3
"""
build_index_paper3.py — Build a Chroma vector store replicating the retrieval
methodology from Bal & Puhan, "Benchmarking Retrieval Strategies for Biomedical
Retrieval-Augmented Generation: A Controlled Empirical Study" (arXiv:2605.02520).

Differs from build_index.py (our own recipe) and build_index_medimind.py
(MediMind recipe) only in the ingestion/embedding configuration, matching the
paper's disclosed setup exactly:
  - Re-chunks our existing corpus passages (already document/section/page-aware
    from extract_corpus.py) using LangChain's RecursiveCharacterTextSplitter at
    512 tokens (cl100k_base encoding) with 50-token overlap — mirroring the
    paper's own workflow, which likewise started from pre-existing passages and
    applied this same splitter on top (yielding their reported ~1.13 chunks per
    source passage).
  - Embeds with OpenAI's text-embedding-3-small (1536-dim), not a local
    sentence-transformers model — the one deliberately OpenAI-billed step in
    this project's RAG pipeline, since it is the paper's own disclosed choice.
  - Stores in ChromaDB with cosine similarity, same as the paper.

The generation side (query_rag in rag_pipeline_paper3.py) still uses one of
our own fine-tuned checkpoints, not the paper's GPT-4o-mini — only the
retrieval methodology is being replicated here.

Usage:
    export OPENAI_API_KEY=sk-...
    python build_index_paper3.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration — matches Bal & Puhan (arXiv:2605.02520) Section 3.3/3.4
# ---------------------------------------------------------------------------

CORPUS_PATH        = Path("/uss/skavlak/version_5/corpus_v5_train.jsonl")
CHROMA_PERSIST_DIR = Path("/uss/skavlak/tb_rag_chroma_paper3_db")
COLLECTION_NAME    = "tb_rag_paper3"
EMBEDDING_MODEL    = "text-embedding-3-small"  # OpenAI, 1536-dim
CHUNK_SIZE_TOKENS  = 512   # cl100k_base encoding, per paper Sec. 3.3
CHUNK_OVERLAP      = 50    # ~10%, per paper Sec. 3.3
BATCH_SIZE         = 256   # documents per Chroma.add_documents call


# ---------------------------------------------------------------------------
# Load source passages (same pre-chunked corpus as build_index.py)
# ---------------------------------------------------------------------------

def load_source_passages(jsonl_path: Path) -> List[Document]:
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

    print(f"Loading source passages from {CORPUS_PATH}...")
    passages = load_source_passages(CORPUS_PATH)
    print(f"  {len(passages)} source passages loaded")

    print(
        f"\nRe-chunking with RecursiveCharacterTextSplitter "
        f"(cl100k_base, chunk_size={CHUNK_SIZE_TOKENS} tokens, overlap={CHUNK_OVERLAP})..."
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.split_documents(passages)
    print(f"  {len(docs)} chunks produced ({len(docs) / len(passages):.3f} chunks per source passage)")

    print(f"\nInitialising embeddings ({EMBEDDING_MODEL}, OpenAI API)...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
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
                collection_metadata={"hnsw:space": "cosine"},
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
