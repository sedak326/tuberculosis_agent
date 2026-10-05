#!/usr/bin/env python3
"""Generate raw outputs from base Instruct and the Run 4 (fixed-seeds) adapter
on the same held-out, analysis-category free-form prompts, for side-by-side
reading. No scoring/judging here - just paired raw generations."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "/uss/skavlak/tb_sft_instruct_run4_qlora_checkpoints/final"
PROMPTS_PATH = "/home/skavlak/finetuning/data_prep/eyeball_prompts_filtered.json"
OUT_PATH = "/home/skavlak/finetuning/data_prep/eyeball_comparison.json"
MAX_NEW_TOKENS = 512

prompts = json.load(open(PROMPTS_PATH))
user_turns = []
for r in prompts:
    user_msg = next(m["content"] for m in r["messages"] if m["role"] == "user")
    user_turns.append({"category": r["category"], "user_content": user_msg})

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate(model, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


results = []

print("Loading base Instruct model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
base_model.eval()

print("Generating base outputs...")
for i, t in enumerate(user_turns):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": t["user_content"]}], tokenize=False, add_generation_prompt=True
    )
    out = generate(base_model, chat)
    results.append({"category": t["category"], "prompt": t["user_content"], "base_output": out})
    print(f"  [{i+1}/{len(user_turns)}] base done")

del base_model
torch.cuda.empty_cache()

print("Loading Run 4 (adapter) model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_for_adapter = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
run4_model = PeftModel.from_pretrained(base_for_adapter, ADAPTER)
run4_model.eval()

print("Generating Run 4 outputs...")
for i, t in enumerate(user_turns):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": t["user_content"]}], tokenize=False, add_generation_prompt=True
    )
    out = generate(run4_model, chat)
    results[i]["run4_output"] = out
    print(f"  [{i+1}/{len(user_turns)}] run4 done")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {len(results)} paired comparisons to {OUT_PATH}")
