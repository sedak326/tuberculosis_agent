#!/usr/bin/env python3
"""
chat_rag.py — CLI for the TB RAG agent.

Single question:
    python chat_rag.py --question "What is the role of ESAT-6 in TB virulence?"

Interactive loop (no --question):
    python chat_rag.py

Defaults to the best standalone fine-tuned checkpoint so far (Run 4
SFT-from-Instruct, 0.801 accuracy on the aligned-960 eval) - swap via
--model/--tokenizer/--adapter once the RAG-paired comparison has its own
eval and a different backbone wins.
"""

import argparse

from rag_pipeline import query_rag

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = "/uss/skavlak/tb_sft_instruct_run4_qlora_checkpoints/final"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="Single question; omit for an interactive loop")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
        )
        if args.show_context:
            print("\n--- Retrieved chunks ---")
            for d in retrieved:
                src = d.metadata.get("document_name", "?")
                print(f"  [{src}] {d.page_content[:120]}...")
            print("---\n")
        print(f"\n{answer}\n")

    if args.question:
        ask(args.question)
    else:
        print("TB RAG agent ready. Type a question (Ctrl+D or 'quit' to exit).\n")
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
