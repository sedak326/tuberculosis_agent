#!/usr/bin/env python3
"""
build_index_paper3_full.py — "Upper bound" variant of build_index_paper3.py:
indexes the corpus PLUS the 3,477 held-out eval chunks (200 papers) that
aligned960's MCQ questions were generated from. See build_index_full.py for
the full rationale (same lower/upper-bound logic applies to every RAG
configuration, this one included). Everything else (re-chunking, embedding
model, Chroma settings) is identical to build_index_paper3.py.

Usage:
    export OPENAI_API_KEY=sk-...
    python build_index_paper3_full.py
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

CORPUS_PATH        = Path("/uss/skavlak/version_5/corpus_v5_train_full.jsonl")
CHROMA_PERSIST_DIR = Path("/uss/skavlak/tb_rag_chroma_paper3_full_db")
COLLECTION_NAME    = "tb_rag_paper3_full"
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

    # Resume support: Chroma persists each add_documents() call to disk, so a
    # job killed by the SLURM time limit mid-run still has its prior batches
    # safely on disk. Open the existing collection (if any) and skip however
    # many chunks it already reports, instead of re-embedding from scratch.
    already_done = 0
    vectorstore = None
    if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
        try:
            existing = Chroma(
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            )
            already_done = existing._collection.count()
            if already_done > 0:
                vectorstore = existing
                print(f"  Resuming: {already_done} chunks already indexed, skipping them")
        except Exception as e:
            print(f"  Could not open existing collection for resume ({e}); starting fresh")

    print(f"\nIndexing into Chroma ({CHROMA_PERSIST_DIR} / {COLLECTION_NAME})...")
    start = time.time()

    for i in range(already_done, len(docs), BATCH_SIZE):
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
        rate = (done - already_done) / elapsed if elapsed > 0 else 0
        eta = (len(docs) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(docs)}]  {rate:.0f} chunks/s  ETA {eta/60:.1f} min", end="\r", flush=True)

    elapsed = time.time() - start
    total = vectorstore._collection.count()
    print(f"\n  Done in {elapsed/60:.1f} min  —  {total} vectors stored")
    print(f"\nChroma store ready at: {CHROMA_PERSIST_DIR.resolve()}")


if __name__ == "__main__":
    build_index()
