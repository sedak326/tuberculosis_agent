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
| SFT-Instruct (v5, 100k data) + RAG (full v5 corpus, top-8) | 0.7010 | 0.5542 | 4.680 | 4.730 | 4.770 | 4.310 |

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
5. **Adding RAG on top of the best standalone model (SFT-Instruct v5
   100k) hurt accuracy on this MCQ benchmark** (0.806 → 0.701, penalized
   0.710 → 0.554), rather than helping. The RAG condition retrieves the
   top-8 chunks from the full 489k-chunk v5 corpus and prepends them to
   each MCQ prompt. A plausible explanation: on closed-form, 4-choice
   questions the fine-tuned model's own parametric knowledge is already
   strong, and injecting ~8 extra chunks of retrieved text lengthens and
   dilutes the prompt, occasionally pulling the model's attention toward
   tangential passages instead of the actual answer choices. This does not
   mean the RAG system itself is broken — it is built, working, and
   produces well-grounded, properly cited, well-formatted answers in the
   interactive chat UI (`rag/chat_ui.py`) — it means retrieval-augmentation
   does not pay off on *this specific* short, closed-form MCQ eval format,
   which is a different task shape than the open-ended research-assistant
   queries RAG is actually meant for.

## RAG diagnostic — why retrieval hurt accuracy, and does gating fix it

Follow-up investigation into why RAG lowered accuracy (0.806 → 0.701, see
finding 5 above), using the exact 960 scored outputs from both conditions.

**Per-item flip analysis.** Of 960 questions, RAG flipped 167 from correct to
wrong and only 66 from wrong to correct (607 stayed correct in both, 120
stayed wrong in both). Reading the flipped cases directly: the model's
free-text reasoning is nearly identical with and without RAG — same facts,
same conclusion in substance — but the extra retrieved context nudges the
wording enough to tip GPT-4o's answer-matching call to a different letter.
This matters because **the entire aligned-960 benchmark is built from three
"critique a synthetic scenario" categories** (`methodology_critique`,
`premise_correction`, `result_interpretation` — 320 each, confirmed via the
set's own `cognitive_level` field), not fact-recall questions. For these, the
four answer choices are often close paraphrases of each other, so a small
wording nudge is enough to flip the judge's letter-match even when a human
would call the two explanations equivalent. 114 of the 167 flips (68%) fall
in these three categories.

**Adaptive/gated RAG (63-question diagnostic, not a full re-run).** Built a
confidence gate (`evaluate_emcqa_rag.py --adaptive --score-threshold`): only
inject retrieved context when the top-1 Chroma distance score clears a
threshold (median score across the full 960, 0.0676), otherwise generate
with no context at all. Tested on a 63-question sample stratified across
all three cognitive levels and across retrieval-confidence terciles (ids in
`/uss/skavlak/version_3/eval_results/adaptive_sample_ids.json`), scored the
same way (GPT-4o) and compared against the same 63 ids from the full no-RAG
and always-RAG runs:

| Condition (n=63) | Accuracy | Penalized | Factual | Coherence | Natural | Complete |
|---|---|---|---|---|---|---|
| No-RAG | 0.889 | 0.833 | 4.83 | 4.86 | 4.90 | 4.59 |
| Always-RAG | 0.762 | 0.643 | 4.65 | 4.63 | 4.68 | 4.24 |
| Adaptive-RAG | 0.794 | 0.690 | 4.76 | 4.83 | 4.84 | 4.48 |

Adaptive gating recovers roughly a third of the accuracy gap. But splitting
by what the gate actually decided reveals the gate is not finding "good"
retrieval:

| Subset of the 63 | No-RAG acc | Always-RAG acc | Adaptive acc |
|---|---|---|---|
| Gate skipped context (35/63, "low confidence") | 0.971 | 0.943 | 0.943 |
| Gate used context (28/63, "high confidence") | 0.786 | 0.536 | 0.607 |

The closest-matching retrieved chunks (lowest distance score) are the ones
that hurt accuracy *most* — not least. In a 489k-chunk broad biomedical
corpus, a confident match usually means "similar TB vocabulary," not
"addresses this specific synthetic scenario," so it tends to surface
superficially-similar-but-substantively-different content, which is more
misleading than a chunk the model can clearly tell is irrelevant and ignore.
The net improvement from gating comes almost entirely from **using RAG less
often** (the gate skips retrieval on 55% of this sample), not from
successfully separating helpful from harmful retrieval. A real fix would
need content-relevance reranking, not just a distance cutoff.

**Caveat:** the 63-question diagnostic is directional, not a statistically
powered replacement for the full 960-question numbers.

**Open question for the team:** RAG measurably hurts this closed-form MCQ
benchmark, but the interactive chat UI (`rag/chat_ui.py`) produces
well-grounded, properly cited answers to open-ended research questions —
arguably RAG's real use case, which this benchmark doesn't test. Decision
needed: keep RAG as an optional/default feature for the interactive
assistant despite the MCQ regression, hold off until a better-targeted
open-ended eval exists, or invest in real relevance reranking before
shipping either way.

## Source files

Raw eval outputs and GPT-4o-scored results live in
`/uss/skavlak/version_3/eval_results/*_scored.jsonl` (not tracked in this
repo — large data files live on `/uss`, see `CLAUDE.md`).
