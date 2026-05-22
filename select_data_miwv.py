"""
MIWV data selection (Jiang et al., AAAI-26).
Scores each training sample by how much a similar one-shot example
helps the base model respond to it. High score = sample is hard and
valuable. Selects top TOP_FRACTION of the dataset.

Run: python select_data_miwv.py
Or via SLURM: sbatch run_miwv.slurm
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH   = "mtubercolosis/output/training_data_clean3.jsonl"
BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
SCORES_PATH = "mtubercolosis/output/miwv_scores.npy"       # saved as we go, supports resume
OUTPUT_PATH = "mtubercolosis/output/training_data_miwv_top10.jsonl"
TOP_FRACTION = 0.10
MAX_SEQ_LEN  = 2048


def load_data(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def compute_response_loss(model, tokenizer, messages, device):
    """Cross-entropy loss on the last assistant turn only."""
    full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(device)
    prefix_len = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)

    labels = full_ids.clone()
    labels[0, :prefix_len] = -100  # mask everything before the response

    with torch.no_grad():
        loss = model(input_ids=full_ids, labels=labels).loss.item()
    return loss


def main():
    print("Loading data...")
    data = load_data(DATA_PATH)
    n = len(data)
    print(f"{n} samples loaded")

    # --- Step 1: embed instructions ---
    print("Embedding instructions...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    instructions = [s["messages"][0]["content"] for s in data]
    embeddings = embed_model.encode(
        instructions, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    # --- Step 2: nearest-neighbor index ---
    print("Finding nearest neighbors...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, I = index.search(embeddings, 2)   # k=2: self is always rank-0
    neighbors = I[:, 1]                  # closest other sample

    # --- Step 3: load base model ---
    print("Loading base model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # --- Step 4: compute MIWV scores (with resume) ---
    scores = np.full(n, np.nan)
    if os.path.exists(SCORES_PATH):
        saved = np.load(SCORES_PATH)
        scores[:len(saved)] = saved
        start = int(np.sum(~np.isnan(scores)))
        print(f"Resuming from sample {start}")
    else:
        start = 0

    for i in tqdm(range(start, n), initial=start, total=n):
        sample   = data[i]
        neighbor = data[neighbors[i]]

        no_shot = [
            {"role": "user",      "content": sample["messages"][0]["content"]},
            {"role": "assistant", "content": sample["messages"][1]["content"]},
        ]
        one_shot = [
            {"role": "user",      "content": neighbor["messages"][0]["content"]},
            {"role": "assistant", "content": neighbor["messages"][1]["content"]},
            {"role": "user",      "content": sample["messages"][0]["content"]},
            {"role": "assistant", "content": sample["messages"][1]["content"]},
        ]

        loss_no_shot  = compute_response_loss(model, tokenizer, no_shot,  device)
        loss_one_shot = compute_response_loss(model, tokenizer, one_shot, device)
        scores[i] = loss_one_shot - loss_no_shot

        # checkpoint every 500 samples
        if (i + 1) % 500 == 0:
            np.save(SCORES_PATH, scores)

    np.save(SCORES_PATH, scores)
    print("Scores saved.")

    # --- Step 5: select top fraction ---
    n_select   = int(n * TOP_FRACTION)
    top_indices = np.argsort(scores)[::-1][:n_select]
    selected    = [data[i] for i in sorted(top_indices)]

    with open(OUTPUT_PATH, "w") as f:
        for s in selected:
            f.write(json.dumps(s) + "\n")

    print(f"Saved {len(selected)} samples to {OUTPUT_PATH}")
    print(f"Score stats — mean: {np.nanmean(scores):.4f}  std: {np.nanstd(scores):.4f}  "
          f"max: {np.nanmax(scores):.4f}  min: {np.nanmin(scores):.4f}")


if __name__ == "__main__":
    main()
