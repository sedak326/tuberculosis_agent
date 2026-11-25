#!/usr/bin/env python3
"""
rag_pipeline_paper3.py — Retrieve-then-generate pipeline replicating the
Cross-Encoder Reranking strategy from Bal & Puhan (arXiv:2605.02520), the
best-performing retrieval strategy in their controlled benchmark (composite
score 0.827, beating their Dense and Hybrid BM25+Dense baselines).

Retrieval matches the paper exactly: dense search retrieves an initial pool
of 3x the final k (paper's own ratio: initial_k=30 for a final k=10), then a
cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2, the same model the paper
uses) rescores every (query, passage) pair and the top k by that score are
kept. Only the generation side differs from the paper: we use one of our own
fine-tuned checkpoints instead of their GPT-4o-mini.

Importable as a module, or run via evaluate_emcqa_rag_paper3.py for eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from build_index_paper3 import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL

INITIAL_K = 30   # dense candidates fetched before reranking (paper: 3 x final k)
FINAL_K   = 10   # chunks kept after cross-encoder reranking (paper's top-k)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SYSTEM_PROMPT = """\
You are an expert tuberculosis (TB) research assistant. You have access to a curated corpus \
of TB research literature (context chunks below). Each chunk starts with its source location \
(Document, Chapter, Section, Page) — cite it naturally using that source location when you use \
it (e.g. "According to document 10027986, page 3..."). Prioritise answering from the context. \
The retrieved text may itself contain inline numeric or bracketed citation markers from the \
original paper's own reference list (e.g. "(39)", "[12]") — you were NOT given that paper's \
reference list, so these numbers are meaningless to you and to the reader. Never reproduce a \
bare numeric/bracketed citation marker from the retrieved text; if you need to refer to the \
claim it was attached to, cite the document/section it came from instead, or drop the marker \
entirely. If the context chunks are not relevant to the question, answer from your general \
TB/biomedical knowledge but clearly note: "This is from general knowledge, not the specific \
literature." Never fabricate citations.

Format your answer for readability using Markdown: keep paragraphs short, use a bullet list \
("- item") when enumerating multiple options, steps, or causes, use **bold** for key terms you \
are defining or emphasising, and use ## or ### headers only for longer answers with multiple \
distinct parts. Do not force structure (headers/bullets) onto a short, single-point answer that \
doesn't need it."""

_vectorstore: Optional[Chroma] = None
_reranker: Optional[CrossEncoder] = None
_model = None
_tokenizer = None


def load_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = _vectorstore._collection.count()
        print(f"Chroma vectorstore (paper3) loaded — {count} chunks indexed")
    return _vectorstore


def load_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
    return _reranker


def load_generator(model_path: str, tokenizer_path: Optional[str] = None, adapter_path: Optional[str] = None):
    """Loads the generation model. Identical loading pattern to rag_pipeline.py —
    only the retrieval side differs between the two conditions."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    tok_path = tokenizer_path or model_path
    _tokenizer = AutoTokenizer.from_pretrained(tok_path, padding_side="left", truncation_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if adapter_path:
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto",
        )
        _model = PeftModel.from_pretrained(base, adapter_path)
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    _model.eval()
    return _model, _tokenizer


def retrieve_rerank(question: str, initial_k: int = INITIAL_K, final_k: int = FINAL_K) -> list:
    """Two-stage retrieval matching the paper's Cross-Encoder Reranking strategy:
    dense search for `initial_k` candidates, then cross-encoder rescoring down
    to the top `final_k`. Returns a list of langchain Document objects, most
    relevant (highest cross-encoder score) first."""
    vs = load_vectorstore()
    candidates = vs.similarity_search(question, k=initial_k)
    if not candidates:
        return []

    reranker = load_reranker()
    pairs = [(question, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:final_k]]


def build_context(retrieved: list) -> str:
    if not retrieved:
        return ""
    return "\n\n---\n\n".join(d.page_content for d in retrieved)


def query_rag(question: str, model_path: str, tokenizer_path: Optional[str] = None,
              adapter_path: Optional[str] = None, initial_k: int = INITIAL_K,
              final_k: int = FINAL_K, max_new_tokens: int = 512) -> tuple[str, list]:
    """Retrieve relevant TB corpus chunks (paper3 methodology) and generate a
    grounded answer with one of our own checkpoints.

    Returns (answer, retrieved_documents).
    """
    model, tokenizer = load_generator(model_path, tokenizer_path, adapter_path)
    retrieved = retrieve_rerank(question, initial_k=initial_k, final_k=final_k)
    context = build_context(retrieved)

    format_reminder = (
        "(Format this answer in Markdown: short paragraphs, a bullet list for "
        "multiple options/steps/causes, **bold** for key terms, ## headers only "
        "if it's a long multi-part answer.)"
    )
    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n{format_reminder}\n\nAnswer:"
        if context else
        f"Question: {question}\n\n{format_reminder}\n\nAnswer:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return answer, retrieved
