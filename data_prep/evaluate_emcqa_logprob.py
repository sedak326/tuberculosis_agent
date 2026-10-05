#!/usr/bin/env python3
"""
evaluate_emcqa_logprob.py — Score a model on the E-MCQA dataset via log-likelihood
comparison instead of free-text generation + GPT-4o judging.

Rationale: evaluate_emcqa.py lets the model generate a full free-text response,
then has GPT-4o infer which answer it supports. That protocol is sensitive to
response style/verbosity/hedging, not just underlying knowledge — a model that's
been SFT'd toward long, exploratory research-consultant prose can score worse
under it even with equal or better knowledge, simply by being less likely to
land on an unambiguous, gradeable answer. Most published fine-tuning papers'
MCQ evals instead compare the model's own next-token probability on "A"/"B"/
"C"/"D" directly — a protocol that's nearly immune to style, since it never
generates free text at all. This script implements that comparison so the two
protocols can be checked against each other on the same 9 checkpoints.

No GPT-4o / API calls needed — correctness is a deterministic argmax over the
model's own logits at inference time.

Usage: identical CLI to evaluate_emcqa.py (same --model/--adapter/--tokenizer/
--randomize-answers/--seed conventions), e.g.:
    python evaluate_emcqa_logprob.py \
        --model   meta-llama/Llama-3.1-8B-Instruct \
        --out     /uss/skavlak/tb_corpus_v3/eval_results/8b_base_r1_logprob.jsonl \
        --randomize-answers --seed 42 --resume
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = """\
You are an expert in tuberculosis (TB) research. Answer the multiple-choice question below \
by selecting the best answer."""

QUESTION_TEMPLATE = """\
Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

Respond with only the letter of the correct answer."""

LETTERS = ["A", "B", "C", "D"]


def load_mcqs(path: str) -> list[dict]:
    mcqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            mcqs.append(json.loads(line))
    return mcqs


def shuffle_choices(mcq: dict, rng: random.Random) -> tuple[dict, str]:
    """Mirrors evaluate_emcqa.py exactly, so results are directly comparable
    question-for-question against the generation+judge protocol."""
    letters = LETTERS
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
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prefix + "Answer:"


def get_letter_token_ids(tokenizer) -> dict:
    """Token id for each letter as it would appear right after 'Answer:' —
    i.e. with a leading space, as a standalone token. Verified single-token
    per letter for the Llama-3 tokenizer family; asserts if that ever changes."""
    ids = {}
    for letter in LETTERS:
        token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        assert len(token_ids) == 1, (
            f"Expected ' {letter}' to be a single token, got {token_ids} "
            f"({[tokenizer.decode([t]) for t in token_ids]}) — logit comparison "
            f"needs adjustment for this tokenizer."
        )
        ids[letter] = token_ids[0]
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq", default="/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path if different from model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
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

    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} already evaluated")

    mcqs = [m for m in load_mcqs(args.mcq) if m["id"] not in done_ids]
    print(f"Evaluating {len(mcqs)} MCQs with {args.model} (log-likelihood scoring)")

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    letter_token_ids = get_letter_token_ids(tokenizer)
    letter_id_tensor = torch.tensor([letter_token_ids[l] for l in LETTERS])
    print(f"Letter token ids: {letter_token_ids}")

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

    letter_id_tensor = letter_id_tensor.to(model.device if hasattr(model, "device") else "cuda")

    total = 0
    correct = 0

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
                outputs = model(**inputs)
                # padding_side="left" -> last position is always the real last token
                last_logits = outputs.logits[:, -1, :]
                candidate_logits = last_logits[:, letter_id_tensor]  # [batch, 4]
                probs = F.softmax(candidate_logits.float(), dim=-1)

            for j, mcq in enumerate(batch):
                letter_probs = {LETTERS[k]: probs[j, k].item() for k in range(4)}
                predicted = max(letter_probs, key=letter_probs.get)
                is_correct = predicted == batch_correct[j]

                record = {
                    "id": mcq["id"],
                    "question": mcq["question"],
                    "correct_original": mcq["correct"],
                    "correct_presented": batch_correct[j],
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "probs": letter_probs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                total += 1
                if is_correct:
                    correct += 1

            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  [{total}/{len(mcqs)}]  acc={correct/total:.3f}")

    if total > 0:
        print(f"\nDone. {total} questions saved to {out_path}")
        print(f"  Accuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()
