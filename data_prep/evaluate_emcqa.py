#!/usr/bin/env python3
"""
evaluate_emcqa.py — Run a model on the E-MCQA dataset and record answers + explanations.

Usage:
    python evaluate_emcqa.py \
        --mcq     /uss/skavlak/tb_corpus_v3/mcq_eval.jsonl \
        --model   meta-llama/Llama-3.1-8B-Instruct \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_base.jsonl

    # For a full fine-tuned checkpoint (8B SFT):
    python evaluate_emcqa.py \
        --mcq     /uss/skavlak/tb_corpus_v3/mcq_eval.jsonl \
        --model   /uss/skavlak/tb_llm_v3_checkpoints/final \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_sft.jsonl

    # For a QLoRA adapter checkpoint (70B SFT):
    python evaluate_emcqa.py \
        --mcq     /uss/skavlak/tb_corpus_v3/mcq_eval.jsonl \
        --model   meta-llama/Llama-3.1-70B \
        --adapter /uss/skavlak/tb_llm_70b_v3_checkpoints/final \
        --tokenizer meta-llama/Llama-3.1-70B-Instruct \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/70b_sft.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = """\
You are an expert in tuberculosis (TB) research. Answer the multiple-choice question below \
by selecting the best answer and providing a detailed explanation of your reasoning."""

QUESTION_TEMPLATE = """\
Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

Respond in this exact format:
Answer: [letter]
Explanation: [your explanation of why the chosen answer is correct and why the others are wrong]"""


def load_mcqs(path: str) -> list[dict]:
    mcqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            mcqs.append(json.loads(line))
    return mcqs



def shuffle_choices(mcq: dict, rng: random.Random) -> tuple[dict, str]:
    """
    Randomly reassign answer letters, returning the shuffled MCQ and the new
    letter that corresponds to the original correct answer.
    """
    letters = ["A", "B", "C", "D"]
    original_correct_text = mcq["choices"][mcq["correct"]]
    texts = [mcq["choices"][l] for l in letters]
    rng.shuffle(texts)
    shuffled_choices = dict(zip(letters, texts))
    new_correct = next(l for l, t in shuffled_choices.items() if t == original_correct_text)
    return shuffled_choices, new_correct


def build_prompt(tokenizer, mcq: dict, choices: dict) -> str:
    user_content = QUESTION_TEMPLATE.format(
        question=mcq["question"],
        A=choices["A"],
        B=choices["B"],
        C=choices["C"],
        D=choices["D"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq", default="/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path if different from model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--randomize-answers", action="store_true",
                        help="Shuffle answer order per question to control for position bias (Balepur et al., 2025)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter", default=None,
                        help="Path to PEFT/LoRA adapter dir. When set, --model is the base model "
                             "and the adapter is loaded on top in 4-bit (NF4) mode.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load base model in 4-bit NF4 (for large models without an adapter).")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-evaluated IDs
    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already evaluated")

    mcqs = [m for m in load_mcqs(args.mcq) if m["id"] not in done_ids]
    print(f"Evaluating {len(mcqs)} MCQs with {args.model}")

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter:
        from peft import PeftModel
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.adapter)
    elif args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()

    total = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(mcqs), args.batch_size):
            batch = mcqs[i : i + args.batch_size]

            batch_choices = []
            batch_correct = []
            for m in batch:
                if args.randomize_answers:
                    choices, correct_letter = shuffle_choices(m, rng)
                else:
                    choices, correct_letter = m["choices"], m["correct"]
                batch_choices.append(choices)
                batch_correct.append(correct_letter)

            prompts = [build_prompt(tokenizer, m, c) for m, c in zip(batch, batch_choices)]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j, (mcq, output_ids) in enumerate(zip(batch, outputs)):
                input_len = inputs["input_ids"].shape[1]
                new_tokens = output_ids[input_len:]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

                record = {
                    "id": mcq["id"],
                    "question": mcq["question"],
                    "correct_original": mcq["correct"],
                    "correct_presented": batch_correct[j],
                    "raw_output": raw_output,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                total += 1

            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  [{total}/{len(mcqs)}]")

    print(f"\nDone. {total} questions saved to {out_path}")


if __name__ == "__main__":
    main()
