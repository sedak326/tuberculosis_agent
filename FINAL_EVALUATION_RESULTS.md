# Final Evaluation Results — Aligned-960 MCQA Set

All conditions below were evaluated on the same 960-question MCQA set
(`aligned_mcq_eval_scaled.jsonl`), built to be more aligned with the SFT data's
style/content than the original `mcq_eval.jsonl`. Answer order was randomized
per question (seed 42) to control for position bias. Each condition was scored
by GPT-4o (`score_explanations.py`), which both extracts the predicted letter
from the model's free-text response and rates the explanation on four 1-5
dimensions against a gold explanation.

- **Accuracy** — fraction of questions answered correctly.
- **Penalized accuracy** — `+1.0` correct, `-0.5` wrong, `0.0` no parseable
  answer; punishes confident wrong answers more than it rewards correct ones.
- **Factual accuracy / Coherence / Naturalness / Completeness** — GPT-4o's
  1-5 rating of the explanation text, independent of whether the letter
  answer was correct.

## Results

| Condition | Accuracy | Penalized Acc | Factual | Coherence | Natural | Complete |
|---|---|---|---|---|---|---|
| Base (Instruct, untouched) | 0.7698 | 0.6578 | 4.740 | 4.801 | 4.831 | 4.609 |
| SFT-Instruct (Run 4, 40k data) | 0.8010 | 0.7042 | 4.795 | 4.845 | 4.877 | 4.538 |
| CPT+SFT (Run 4) | 0.7094 | 0.5745 | 4.650 | 4.604 | 4.682 | 4.352 |
| Merged-noSFT (Run 4) | 0.7490 | 0.6245 | 4.776 | 4.802 | 4.832 | 4.649 |
| Merged+SFT (Run 4) | 0.7115 | 0.5698 | 4.794 | 4.848 | 4.871 | 4.550 |
| Smoothed-merge-noSFT (Run 4) | 0.7521 | 0.6307 | 4.784 | 4.861 | 4.890 | 4.650 |
| Smoothed-merge+SFT (Run 4) | 0.7531 | 0.6307 | 4.802 | 4.856 | 4.879 | 4.593 |
| **SFT-Instruct (v5, 100k data)** | **0.8063** | **0.7104** | 4.804 | 4.855 | 4.881 | 4.568 |
| Merged-noSFT (v5, full corpus + faithful BioMistral schedule) | 0.7760 | 0.6667 | 4.726 | 4.772 | 4.802 | 4.571 |
| Merged+SFT100k (v5) | 0.7479 | 0.6240 | 4.782 | 4.840 | 4.868 | 4.533 |

Bold = best result overall.

## What changed between Run 4 and v5

- **CPT corpus**: combined + deduplicated across all prior fetch runs, no
  post-cutoff restriction — 489,423 chunks / ~173M tokens (4.1x Run 4's
  corpus), vs Run 4's 119,325 chunks / ~42M tokens.
- **CPT**: full-parameter (not QLoRA), starting from Llama-3.1-8B-**Instruct**
  (not raw base) — matches the BioMistral paper's own CPT recipe.
- **Merge schedule**: the literal, unsmoothed paper schedule (self_attn/mlp
  opposite depth ramps, `embed_tokens`/`lm_head` left in the catch-all
  `value: 0.5`) — not the smoothed variant used for Run 4's "smoothed" rows.
- **SFT data**: scaled from 40k to 100,536 examples.

## Key findings

1. **SFT-Instruct remains the best standalone condition**, and scaling its
   data from 40k to 100k gave a small additional lift (0.801 → 0.806
   accuracy). This is the only condition that reliably beats untouched base
   across both runs.
2. **The bigger corpus + full-parameter CPT + faithful merge schedule
   meaningfully improved the merge-by-itself condition**: v5's
   Merged-noSFT (0.776) is the first merge condition in the whole project
   that doesn't lose ground to base (0.770) — every Run 4 merge variant
   (plain or smoothed, with or without SFT) scored below base on accuracy.
   (Statistical note: this improvement over base is *not* significant by
   McNemar's test at n=960, p≈0.63 — a real step up from prior merges, not
   yet proof of beating base.)
3. **SFT-after-merge is still harmful**, even in v5: Merged+SFT100k (0.748)
   scores *below* Merged-noSFT (0.776) — the same qualitative failure
   pattern seen in Run 4, just milder. QLoRA SFT (targeting only
   `q/k/v/o/gate/up/down_proj`) still cannot repair whatever the SLERP merge
   does to the checkpoint, even when the merge-by-itself result has improved.
4. **CPT+SFT (full-parameter, Run 4) is the worst condition tested on every
   single dimension** — consistent with the catastrophic-forgetting
   diagnosis that motivated the corpus-expansion and merging work in the
   first place.

## Source files

Raw eval outputs and GPT-4o-scored results live in
`/uss/skavlak/version_3/eval_results/*_scored.jsonl` (not tracked in this
repo — large data files live on `/uss`, see `CLAUDE.md`).
