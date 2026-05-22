"""
Reselect top MIWV samples after excluding contaminated ones.
No recomputation needed — uses existing miwv_scores.npy.
"""

import json
import numpy as np

DATA_PATH    = "mtubercolosis/output/training_data_clean3.jsonl"
SCORES_PATH  = "mtubercolosis/output/miwv_scores.npy"
OUTPUT_PATH  = "mtubercolosis/output/training_data_miwv_top10_clean.jsonl"
TOP_FRACTION = 0.10

BAD_PATTERNS = [
    "response training examples",
    "## step",
    "step 1:",
    "step 2:",
    "step 3:",
    "first training example",
    "second training example",
    "instruction-response pair",
]

def is_contaminated(sample):
    text = (sample["messages"][0]["content"] + sample["messages"][1]["content"]).lower()
    return any(p in text for p in BAD_PATTERNS)

data = []
with open(DATA_PATH) as f:
    for line in f:
        data.append(json.loads(line))

scores = np.load(SCORES_PATH)

# Mask contaminated samples so they can't be selected
clean_scores = scores.copy()
n_bad = 0
for i, s in enumerate(data):
    if is_contaminated(s):
        clean_scores[i] = -np.inf
        n_bad += 1

n_clean = len(data) - n_bad
n_select = int(n_clean * TOP_FRACTION)
top_indices = np.argsort(clean_scores)[::-1][:n_select]
selected = [data[i] for i in sorted(top_indices)]

with open(OUTPUT_PATH, "w") as f:
    for s in selected:
        f.write(json.dumps(s) + "\n")

valid_scores = clean_scores[clean_scores != -np.inf]
print(f"Total: {len(data)}  Contaminated: {n_bad}  Clean: {n_clean}")
print(f"Selected: {len(selected)} samples -> {OUTPUT_PATH}")
print(f"Score stats — mean: {valid_scores.mean():.4f}  std: {valid_scores.std():.4f}  max: {valid_scores.max():.4f}")
