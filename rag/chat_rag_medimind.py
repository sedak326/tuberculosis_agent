#!/usr/bin/env python3
"""
chat_rag_medimind.py — CLI for the TB RAG agent, paired with the MediMind-
style index (jina-embeddings-v3 late chunking, FAISS/HNSW) instead of the
original Chroma/PubMedBERT index (see chat_rag.py).

Must run inside /uss/skavlak/envs/jina_rag_env:
    /uss/skavlak/envs/jina_rag_env/bin/python3 chat_rag_medimind.py --question "..."

Generation model defaults to v5 (SFT-from-Instruct, 100k data, best
standalone condition on the aligned-960 eval: 0.806 accuracy).
"""

import argparse

from rag_pipeline_medimind import query_rag

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = "/uss/skavlak/tb_sft_instruct_v5_100k_qlora_checkpoints/final"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="Single question; omit for an interactive loop")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retrieval-mode", choices=["hybrid", "dense", "bm25"], default="hybrid",
                         help="hybrid = dense (jina-v3) + BM25 keyword search fused via RRF (default)")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks before the answer")
    args = parser.parse_args()

    def ask(question: str):
        answer, retrieved = query_rag(
            question,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            adapter_path=args.adapter,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            mode=args.retrieval_mode,
        )
        if args.show_context:
            print("\n--- Retrieved chunks ---")
            for r in retrieved:
                print(f"  [{r['document_name']}] {r['text'][:120]}...")
            print("---\n")
        print(f"\n{answer}\n")

    if args.question:
        ask(args.question)
    else:
        print("TB RAG agent (MediMind-style index) ready. Type a question (Ctrl+D or 'quit' to exit).\n")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question or question.lower() in ("quit", "exit"):
                break
            ask(question)


if __name__ == "__main__":
    main()
