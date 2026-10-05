# TB LLM Project — Plan & Status

## Repo

Pushed to GitHub: https://github.com/sedak326/tuberculosis_agent (`main`). Large files (checkpoints, corpora, raw papers, SLURM logs) are gitignored and live only on `/uss` or locally — see "Data paths" below for where everything actually is.

## Storage

Large files (model checkpoints, large datasets, raw papers) must be stored on `/uss`, not in `/home/skavlak`. The home directory is on a 4T network filesystem shared across the lab. Always point SLURM output dirs, checkpoint dirs, and new paper downloads at `/uss` before launching a job.

## Cluster

- `gpu4`: 8× RTX PRO 6000 (96GB each). Keep individual jobs under 6 hours to avoid blocking others — safe to resubmit, scripts have auto-resume logic.
- `gpu3`: 2× RTX 6000 Ada (48GB each). Usable for smaller 8B jobs (~40-50GB/GPU need on a 2-way FSDP split) when gpu4 is occupied.
- `gpu` / `gpu2`: 2× RTX 4090 (24GB each) per node. Too small for full-parameter 8B FSDP training; fine for lighter workloads.

## Goal

Build a domain-specific LLM that serves as a research assistant for tuberculosis, combining all existing TB research and literature. Focus on protein/omics-level interactions. The LLM should assist researchers with questions, reasoning, and hypothesis generation — like Claude Code but for TB research.

## Status (updated 2026-10-05)

Four distinct rounds of work now. **Do not conflate them.**

- **Run 1** — the original v3-corpus pipeline. COMPLETE, all 9 conditions scored. Results were disappointing and diagnosed as full-parameter-fine-tuning catastrophic forgetting on a severely data-starved 8B model. **Kept frozen as a reference point — do not delete or overwrite these results/checkpoints.**
- **Run 2** — corpus expansion effort. Corpus (148.4M tokens) and SFT data (636,520 examples) are both DONE. **CPT and SFT training were never actually run** — CPT got blocked on gpu4 availability and the run stalled after data prep. No eval results exist for Run 2. Not actively being pursued; superseded in spirit by Run 3/4's contamination-controlled design, but the data is sitting there if anyone wants to pick it back up.
- **Run 3** — COMPLETE. A standalone, contamination-controlled replication built only from papers published after Llama's pretraining cutoff (Dec 2023), so CPT can't re-expose the model to memorized text and the eval set is guaranteed unseen. QLoRA for both CPT and SFT. All 6 conditions evaluated and scored (see "Run 3" section below).
- **Run 4** — COMPLETE. Isolates seed-pool quality as the only changed variable vs Run 3: same corpus, same CPT checkpoint (unused by this condition), same eval set — only the SFT seed pool was fixed (see "Seed pool audit" below) and SFT data regenerated from it. Result: no detectable change on any eval metric vs Run 3 (see below). Led to a diagnosis that the MCQ eval format may not match what the SFT data is actually training.

---

## Run 1 (COMPLETE — frozen reference, v3 corpus, original size)

### Conditions evaluated — 9 total, all scored

| Condition | CPT | SFT data | Accuracy |
|-----------|-----|----------|----------|
| Base 8B (Instruct) | no | — | 0.561 |
| 8B SFT | no | full data (48,782 ex) | 0.489 |
| 8B SFT + MIWV | no | MIWV top 10% (4,878 ex) | 0.318 |
| 8B CPT + SFT | yes | full data | 0.505 |
| 8B CPT + SFT + MIWV | yes | MIWV top 10% | 0.605 |
| Base 70B (raw, non-instruct) | no | — | 0.020 |
| Base 70B (Instruct) | no | — | 0.905 |
| 70B SFT (QLoRA) | no | full data | 0.910 |
| 70B SFT + MIWV (QLoRA) | no | MIWV top 10% | 0.770 |

70B CPT skipped — too expensive (~80-100 hrs) and corpus too small to meaningfully shift a 70B model.

### Diagnosis

Full-parameter 8B SFT scored *worse* than the untouched base model, and SFT+MIWV collapsed further (57% of outputs failed to produce a parseable answer — a format/instruction-following collapse, not just factual errors). Root cause: **tokens-per-trainable-parameter starvation**. Full 8B fine-tuning exposes ~8B trainable params to only ~50M tokens of data (0.006 tokens/param) — enough freedom to overfit/memorize without needing to generalize, and enough magnitude of update to damage the base model's instruction-following. 70B's QLoRA setup (~828M trainable params, same data) has a 10x better ratio, avoiding damage — but still too data-starved to gain much, landing close to its own untouched Instruct baseline (0.910 vs 0.905). This is what motivated Run 2's corpus expansion and QLoRA exploration.

### Original pipeline steps (as executed for Run 1)

1. Fetch new papers — broader TB queries → `/uss/skavlak/tb_corpus_v3/`
2. Extract corpus — `data_prep/extract_corpus.py` → corpus_v3.jsonl
3. Merge + split — combine with proteomics corpus, hold out 200 papers for MCQ eval (`data_prep/split_corpus.py`, fixed seed=42)
4. CPT — `8b/run_cpt.slurm`
5. Generate SFT data — `data_prep/generate_training_data.py` → training_data_v3.jsonl (now archived, see Run 2)
6. MIWV scoring — `data_prep/select_data_miwv.py`
7. SFT — 3 conditions (SFT, SFT-MIWV, CPT+SFT)
8. 70B SFT — QLoRA, full + MIWV data
9. Build MCQ dataset — GPT-4o, from held-out 20% corpus
10. Evaluate — all 9 conditions against MCQ dataset

Note: step 1's "broader corpus" fetch mostly failed silently — of the 6,328 papers fetched into `tb_corpus_v3/`, 6,238 (98.6%) turned out to already exist in the old proteomics corpus. Root cause: narrow NCBI MeSH-term queries largely overlapped with what the original corpus already covered. This is why Run 2 needed a real corpus expansion (see below) rather than just rerunning Run 1's fetch.

### Checkpoints (Run 1, v3 corpus — do not overwrite)

| Run | Location | Status |
|-----|----------|--------|
| 8B CPT v3 | `/uss/skavlak/tb_cpt_v3_checkpoints` | COMPLETE |
| 8B SFT v3 | `/uss/skavlak/tb_llm_v3_checkpoints` | COMPLETE |
| 8B SFT MIWV v3 | `/uss/skavlak/tb_llm_miwv_v3_checkpoints` | COMPLETE |
| 8B CPT+SFT v3 | `/uss/skavlak/tb_sft_cpt_v3_checkpoints` | COMPLETE |
| 8B CPT+SFT MIWV v3 | `/uss/skavlak/tb_sft_cpt_miwv_v3_checkpoints` | COMPLETE |
| 70B SFT v3 | `/uss/skavlak/tb_llm_70b_v3_checkpoints` | COMPLETE |
| 70B SFT MIWV v3 | `/uss/skavlak/tb_llm_70b_miwv_v3_checkpoints` | COMPLETE |
| Llama-3.1-70B weights | `/uss/skavlak/hf_models/hub/models--meta-llama--Llama-3.1-70B` | downloaded (132GB) |

Eval results (scored): `/uss/skavlak/tb_corpus_v3/eval_results/*_scored.jsonl` — 9 files, 800/800 each.

### Known bugs found during Run 2 investigation (relevant if retrying Run 1-style full-parameter jobs)

- An attempt to rerun 8B training from `Llama-3.1-8B-Instruct` (instead of raw base) crashed on all 5 conditions. Root cause: `CUDA_HOME` env var missing → `accelerate`'s FSDP model-save path imports `deepspeed`, which fails immediately (`MissingCUDAException`). Made worse by a second bug: the `.slurm` scripts never check the training command's exit code, so SLURM/`sacct` reported these as `COMPLETED` despite total failure and zero checkpoint output. **Fix planned but never applied**: add `export CUDA_HOME=...` / `export TRITON_CACHE_DIR=...` (already present in `70b/*.slurm`, missing from `8b/*.slurm`) and add exit-code checking after the `torchrun` call in each script.
- `data_prep/fetch_europmc_batch.py` originally had no retry logic on transient search-API errors — a single 503 would silently truncate the rest of a query's results. Fixed with exponential backoff (now in the script).

---

## Run 3 (COMPLETE — contamination-controlled replication)

Standalone pipeline, independent of Run 1/2 — no shared corpus, no comparison assumed. Built entirely from papers verified to have been published after Llama 3.1/3.3's pretraining cutoff (Dec 2023), so CPT contamination is structurally impossible and the MCQ eval set is guaranteed leak-free.

- **Corpus**: `/uss/skavlak/version_3/` — 119,325 train chunks + 3,477 eval chunks, ~32.5M words (roughly 40-45M BPE tokens, estimated).
- **CPT**: QLoRA (added support to `train_cpt.py` for this), ~4 GPUs, completed in ~3h41m.
- **SFT data**: 24,000-chunk sample → 39,996 examples, generated from `seeds.json` (pre-fix, at the time).
- **Conditions evaluated** (6, all scored):

| Condition | Accuracy |
|---|---|
| Base (raw, non-instruct) | 0.800 |
| Base (Instruct, untouched) | 0.873 |
| SFT from raw base | 0.838 |
| CPT + SFT | 0.839 |
| SFT from Instruct base | 0.881 |
| SFT from Instruct base, 200K ex | 0.886 |

Notable: SFT-from-raw-base and CPT+SFT both underperform the untouched baseline; only the SFT-from-Instruct variants beat it. Full GPT-4o-judged breakdown (factual_accuracy/coherence/naturalness/completeness, not just accuracy) lives in `/uss/skavlak/version_3/eval_results/*_scored.jsonl`.

This whole pipeline (CPT → SFT → eval → scoring) ran to completion via unattended auto-resubmit chains (`8b/launch_run3_*_chain.sh`).

## Seed pool audit (this session, 4 rounds)

`data_prep/seeds.json` went through four rounds of bug-fixing this session, prompted by repeated external audits (several of which turned out to be stale or wrong when checked against the actual file — **always independently verify scientific claims against primary literature via web search before trusting an audit or applying a fix; do not trust citations at face value**).

- Fixed: instruction/output mismatches, facts leaking into instruction fields, wrong scientific claims (reversed TAP/MHC-I mechanism, fabricated temperature figures, wrong enzyme assignments, missing/incomplete facts), miscategorized seeds, corrupted notes fields (bad string concatenation), a correction-preamble stylistic tic, editorial "explaining the edit" language leaking into outputs, ungrammatical clauses, and one content duplication (swapped for a verified alternative).
- Quarantined (not deleted — moved to `data_prep/seeds_quarantine.json` with a reason): 9 seeds, mostly unverifiable or contradicted-by-literature claims.
- One item (`gen-rc-011`, WhiB4/biofilm claim) deliberately left unresolved — no verifiable evidence either way; needs Adrian or further literature work.
- Final state: **50 active seeds + 9 quarantined = 59 total**, nothing lost. The `notes` field was stripped from the active pool (`generate_training_data.py` never reads it — it was purely a human-review artifact); still present in the quarantine file for reference.
- Still open, not yet addressed (design/systemic, not bugs): `ambiguous: false` on every single record; 2 near-duplicate SSB seeds; the generic "understand a concept or gene" instruction repeated ~9 times in `closed_book_qa`.
- Backups of each fix round: `data_prep/seeds.json.bak` through `.bak4`.

## Run 4 (COMPLETE — isolated seed-quality test)

Purpose-built to answer one question: does fixing the seeds actually change anything downstream? Reused Run 3's corpus, MCQ eval set, and (implicitly, since this condition doesn't use it) CPT checkpoint untouched — only regenerated SFT data from the fixed seeds and retrained the one condition that had beaten baseline in Run 3 (SFT-from-Instruct).

- **SFT data**: same 24,000-chunk sample as Run 3, sharded across up to 8 GPUs this time → **40,252 examples**. Output: `/uss/skavlak/version_3/training_data_run4.jsonl`. (Gotcha hit along the way: `generate_training_data.py` ignores a custom `--output` basename when sharded and always writes `training_data_shard_{id}.jsonl` — concatenate from that pattern, not your intended output name.)
- **Training**: identical hyperparameters to Run 3 (rank-64 QLoRA, lr 2e-5, 2 epochs, effective batch 32), run at both 2 GPUs (2h41m) and 8 GPUs (51 min, grad-accum dropped 16→4 to hold effective batch constant) — a clean same-recipe GPU-scaling data point.
- **Checkpoint**: `/uss/skavlak/tb_sft_instruct_run4_qlora_checkpoints/final`.
- **Result, full comparison vs Run 3 and baseline:**

| Metric | Base | Run 3 (pre-fix seeds) | Run 4 (fixed seeds) |
|---|---|---|---|
| Accuracy | 0.8725 | 0.8812 | 0.8825 |
| Factual accuracy | 4.555 | 4.491 | 4.490 |
| Coherence | 4.753 | 4.731 | 4.714 |
| Naturalness | 4.834 | 4.830 | 4.835 |
| Completeness | 4.524 | 4.476 | 4.494 |

**No detectable movement on any metric** — all differences are within noise at n=800.

### Diagnosis: eval/SFT-data misalignment (unresolved, worth revisiting)

Comparing real MCQ eval questions (closed-form, 4-choice, grounded in one specific passage — tests recall/comprehension) against real SFT training examples (open-ended consultation/critique/hypothesis-generation — roughly half the categories aren't grounded in any passage at all) shows they're testing genuinely different skills. This is a plausible explanation for Run 4's flat result that's independent of whether the seed fixes were real improvements — the eval may simply be insensitive to this kind of change.

Proposed but not built: a category-aligned, rubric-graded held-out eval mirroring the SFT categories (GPT-4o judge scoring against category-specific rubrics — "did it catch the false premise," "did it propose a real distinguishing experiment" — rather than correctness-against-answer-key). Would need either curated gold eval items or generated-and-verified ones.

**Interim cheap check (done)**: `data_prep/eyeball_comparison.md` — 30 held-out, analysis-heavy-category prompts (premise_correction, result_interpretation, hypothesis_generation, methodology_critique, gene_target_prioritization) pulled from the same held-out corpus split as the MCQ set, generated via `data_prep/run_eyeball_compare.py` (Base Instruct vs Run 4, greedy decoding, no scoring) for direct human reading. Built via `data_prep/run_eyeball_generation.slurm` (generates prompts by running the real generation pipeline against `corpus_v3_eval.jsonl` instead of the train split) + `run_eyeball_compare.slurm`.

---

## Corpus v3 expansion (Run 2, COMPLETE)

- Diagnosed that EuropePMC's free-text search covers far more open-access TB literature (93,843 papers for bare "Mycobacterium tuberculosis") than the NCBI MeSH-term queries originally used (~7,400 total).
- First expanded fetch attempt (23 queries, existing 15 + 8 new protein/omics-specific ones) had a precision bug: unrestricted full-text search matched papers that merely *mentioned* TB in passing (e.g. one citation in a 350-page review about MRSA) — audit found only 47% of matches had TB in the title/abstract, 27% mentioned it ≤2 times total. Fixed by restricting the TB term to `TITLE:`/`ABSTRACT:` fields and excluding retracted publications (`NOT PUB_TYPE:"Retracted Publication"`).
- Clean rerun: **12,976 new papers**, verified on-topic (spot-checked: dozens to hundreds of genuine TB mentions per paper). Landed in `/uss/skavlak/tb_corpus_v3_extra/`.
- Extracted via `data_prep/extract_corpus.py` → 244,951 chunks, 91.66M tokens, with `pub_year` metadata added per chunk (pulled from JATS `<pub-date>`, useful for future recency-aware filtering/weighting — not currently used downstream).
- Merged into `corpus_v3_train.jsonl`, excluding 19 papers whose document *names* collided with the held-out eval set (despite different PMC IDs — likely republished/duplicate records) to protect eval-set integrity.
- **Result: 148.4M CPT tokens total** (up from 56.9M), 382,700 train chunks. Eval split untouched (2,950 chunks / 200 papers).
- Query fixes and full query list live in `data_prep/fetch_europmc_batch.py`. Raw extra-corpus papers + extraction output also live in `/uss/skavlak/tb_corpus_v3_extra/`.

## SFT categories (Run 2, in progress)

Original seed pool (`data_prep/seeds.json`) had 8 categories, heavily skewed (research_consult 40.7% + closed_book_qa 29.1% of generated output ≈ 70% combined, estimated via embedding classification since the old generated data had no category tag).

Changes made:
- Cut `custom` (was a single leftover example, not a real category).
- Added `methodology_critique` and `structured_extraction` — 6 new seed examples (3 each), authored by Claude, explicitly flagged `"AI-drafted seed — needs Adrian/domain review"` in their `notes` field. **Not yet reviewed.**
- Rewrote `data_prep/generate_training_data.py`: each chunk now gets a category assigned round-robin (shuffled first, so assignment isn't correlated with corpus file order) rather than the old random-mixed-seed-sampling approach. `format_seeds()` only shows seeds from the assigned category. Output records now carry an explicit `"category"` field, so balance is verifiable directly instead of needing an estimate.
- Smoke-tested: 72 examples, exactly 8 per category across all 9 categories, 0 failures, quality spot-checked as good (grounded, on-topic, correctly following each category's intent).

The seed pool grew beyond what's described above via other contributors (Sarah, adriana, Ricardo, Nate, Melina) adding seeds independently, then went through a 4-round audit/repair this session — see "Seed pool audit" above for the current, settled state (50 active + 9 quarantined). Check `data_prep/seeds.json` directly if this doc goes stale again.

Generator model: `meta-llama/Llama-3.3-70B-Instruct` via vLLM (not GPT-4o) — deliberate choice to keep the SFT data generator independent from GPT-4o, which is the eval judge (`score_explanations.py`). Using the same model for both would risk self-preference bias (the fine-tuned model would learn to imitate GPT-4o's style, which GPT-4o would then rate favorably as judge). Estimated cost to switch to GPT-4o: ~$3,300 (654M input + 168M output tokens at standard rates) — not pursued, for the above reason, independent of cost.

Generation infra: 16-shard SLURM array (`data_prep/run_generation.slurm`, 2 GPUs/shard via vLLM tensor-parallel, gpu4 only — needs ~192GB to hold Llama-3.3-70B-Instruct in bf16 comfortably). 4-hour SLURM time limit per task; `data_prep/auto_resubmit_generation.sh` handles resubmission automatically (optionally time-boxed, e.g. 24h, to avoid indefinitely monopolizing the shared 8-GPU node — edit `MAX_HOURS` in the script). Known port-collision gotcha: an earlier run had two shards silently fail (port already in use, health-check falsely passed) — ports were moved to the 9100+ range in `run_generation.slurm` to avoid this.

## Evaluation pipeline

Two-step process — no regex parsing anywhere:

**Step 1 — Inference** (`data_prep/evaluate_emcqa.py`):
Runs the model on 800 MCQs and saves `id`, `question`, `correct_original`, `correct_presented`, `raw_output` per record. That's it — no answer extraction, no accuracy computation.
- Base models: `--model <hf_id>` (uses bf16 + device_map=auto)
- QLoRA fine-tuned models: `--model meta-llama/Llama-3.1-70B --adapter <checkpoint/final>` (loads 4-bit base + PEFT adapter)
- Always pass `--tokenizer meta-llama/Llama-3.1-Xb-Instruct` for the instruct chat template
- Use `--resume` so partial runs can be continued

**Step 2 — GPT-4o scoring** (`data_prep/score_explanations.py`):
GPT-4o reads each `raw_output`, extracts the predicted answer letter, and scores explanation quality on 4 dimensions (factual accuracy, coherence, naturalness, completeness). This is the only place answer extraction happens — do NOT add regex-based answer parsing to the inference script.
- Needs `OPENAI_API_KEY` in environment
- Always use `--resume` — it's idempotent and safe to rerun

MCQ eval set: 800 questions, exactly balanced across 4 cognitive levels (200 each: recall, analysis, understanding, application), enforced at generation time. Topic tags are *not* balanced (drug_resistance=194 down to clinical=9).

## Data paths

### Corpus v1/v2 (proteomics only, DO NOT USE FOR RETRAINING)
- 7,780 papers in `mtubercolosis/`
- corpus.jsonl — 118,899 chunks, 260MB
- training_data_v2.jsonl — 48,896 SFT examples (advisory style)
- training_data_miwv_v2_top10.jsonl — 4,889 examples (MIWV top 10%)
- Superseded, kept for reference only

### Corpus v3 (Run 1 + Run 2 expansion)
- **Raw papers**: original fetch in `/uss/skavlak/tb_corpus_v3/`, the 12,976-paper expansion in `/uss/skavlak/tb_corpus_v3_extra/` (both PDFs/XMLs plus extracted JSONL).
- `/uss/skavlak/tb_corpus_v3/corpus_v3_train.jsonl` — **live CPT training corpus, 382,700 chunks, 148.4M tokens**
- `/uss/skavlak/tb_corpus_v3/corpus_v3_eval.jsonl` — held-out eval split, 2,950 chunks / 200 papers, protected
- `/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl` — 800-question MCQ eval set
- `/uss/skavlak/tb_corpus_v3/archive/` — old unbalanced SFT data + shards (Run 1), not in use
- `/uss/skavlak/tb_corpus_v3/training_data_shard_*.jsonl` — Run 2 SFT generation output, **COMPLETE: 636,520 examples** (not "in progress" — CPT/SFT training on this data was never run, see Status)
- MIWV scores/filtered data (`miwv_scores_v3.npy`, `training_data_miwv_v3_top10.jsonl`) — stale, computed against old data, need rescoring if Run 2 is ever picked back up

### Corpus v3 (Run 3 + Run 4, post-cutoff-only, current main pipeline)
- **Raw papers**: `/uss/skavlak/version_3/papers/` (organized by year: `2024/`, `2025/`, `2026/`), metadata in `papers.csv` / `papers.jsonl` at `/uss/skavlak/version_3/`.
- `/uss/skavlak/version_3/corpus_v3_train.jsonl` — CPT training corpus, 119,325 chunks (~32.5M words / ~40-45M BPE tokens estimated)
- `/uss/skavlak/version_3/corpus_v3_eval.jsonl` — held-out eval split, 3,477 chunks / 200 papers — this is what both the MCQ set and the eyeball-comparison prompts are grounded in
- `/uss/skavlak/version_3/corpus_v3_train_shuffled.jsonl` / `_shuffled_24k.jsonl` — pre-shuffled train corpus and its fixed 24,000-chunk sample, used for SFT data generation
- `/uss/skavlak/version_3/mcq_eval.jsonl` — 800-question MCQ eval set (Run 3 + Run 4 share this)
- `/uss/skavlak/version_3/training_data.jsonl` — Run 3's SFT data (39,996 ex, pre-fix seeds)
- `/uss/skavlak/version_3/training_data_run4.jsonl` — Run 4's SFT data (40,252 ex, fixed seeds)
- `/uss/skavlak/version_3/eval_results/*_scored.jsonl` — all Run 3 + Run 4 eval results, GPT-4o-judged

### Seeds
`data_prep/seeds.json` — 50 active seeds, audited and repaired (see "Seed pool audit" above). `data_prep/seeds_quarantine.json` — 9 removed-but-recoverable seeds with reasons. Backups of each fix round: `seeds.json.bak` through `.bak4`.

## Checkpoints — Run 2 (v3 expanded corpus — never created)

| Run | Location | Status |
|-----|----------|--------|
| 8B CPT (Run 2) | TBD, don't overwrite `tb_cpt_v3_checkpoints` (Run 1) | not started, stalled on gpu4 availability, never resumed |
| 8B SFT (method TBD: full-param or QLoRA) | TBD | not started |
| 70B SFT (QLoRA, if pursued) | TBD | not started |

## Checkpoints — Run 3 / Run 4

| Run | Location | Status |
|-----|----------|--------|
| 8B CPT (Run 3, QLoRA) | `/uss/skavlak/tb_cpt_run3_qlora_checkpoints` (+ merged at `tb_cpt_run3_qlora_merged`) | COMPLETE |
| 8B SFT, from raw base (Run 3) | `/uss/skavlak/tb_llm_run3_qlora_checkpoints` | COMPLETE |
| 8B CPT+SFT (Run 3) | `/uss/skavlak/tb_sft_cpt_run3_qlora_checkpoints` | COMPLETE |
| 8B SFT from Instruct base (Run 3) | `/uss/skavlak/tb_sft_instruct_run3_qlora_checkpoints` | COMPLETE |
| 8B SFT from Instruct base, 200K (Run 3) | `/uss/skavlak/tb_sft_instruct_200k_run3_qlora_checkpoints` | COMPLETE |
| 8B SFT from Instruct base (Run 4, fixed seeds) | `/uss/skavlak/tb_sft_instruct_run4_qlora_checkpoints` | COMPLETE |

## Folder structure

```
finetuning/
  8b/                      — all 8B SLURM scripts: Run 1 (full-parameter), Run 2/3/4 (QLoRA),
                             plus auto-resubmit/launch chains (launch_run3_*, launch_run4_chain.sh)
  8b_small_batch/          — Marek et al. batch-size ablation, built, never run
  70b/                     — 70B SLURM scripts (QLoRA)
  eval/                    — early/misc eval SLURM scripts
  data_prep/               — corpus, generation, seeds, eval pipeline, seed-audit scripts
    fetch_europmc_batch.py      — main literature fetch (Run 2), TITLE/ABSTRACT-restricted + retraction filter
    fetch_postcutoff_papers.py  — Run 3/4's contamination-controlled fetch (post-Dec-2023-cutoff only)
    generate_training_data.py  — SFT generation, category-balanced; SEED_PATH fixed to seeds.json
    evaluate_emcqa.py / score_explanations.py — eval pipeline (inference, then GPT-4o judge)
    generate_mcq.py             — builds the MCQ eval set from held-out corpus chunks
    seeds.json                  — active seed pool (50), audited this session
    seeds_quarantine.json       — 9 quarantined seeds, with reasons
    apply_seed_fixes*.py        — the 4 seed-audit fix-batch scripts (one-off, not a repeatable pipeline)
    run_eyeball_generation.slurm / run_eyeball_compare.py — the held-out qualitative comparison (Run 4 diagnosis)
  refs/                    — reference papers/PDFs
  figures/                 — generated plots
  train_cpt.py             — supports --bits/--lora-rank (QLoRA, added for Run 3) and full-parameter FSDP
  train_sft.py             — supports --bits/--lora-rank (QLoRA) and --adam-beta1/--adam-beta2/--lr-scheduler-type
  mtubercolosis/           — old proteomics corpus (v1/v2, keep for reference, gitignored — too large/raw for git)
/uss/skavlak/tb_corpus_v3/       — v3 corpus + raw papers, all large files (Run 1 + Run 2)
/uss/skavlak/tb_corpus_v3_extra/ — Run 2 newly-fetched papers, already merged into corpus_v3_train.jsonl
/uss/skavlak/version_3/          — Run 3/4 corpus + raw papers + SFT data + eval results (current main pipeline)
```

## MIWV Data Selection

Reference: "Importance-Aware Data Selection for Efficient LLM Instruction Tuning" (Jiang et al., AAAI-26), PDF at `refs/40396-Article Text-44487-1-2-20260314.pdf`

For each training sample: compute the base model's loss zero-shot and with a nearest-neighbor example prepended (one-shot). MIWV = one_shot_loss - zero_shot_loss. High score = model struggles even with a hint = high-value sample. Select top 10%.

`data_prep/select_data_miwv.py` — scores all samples, saves incrementally, selects top 10%
`data_prep/reselect_miwv.py` — reuses saved scores, filters contaminated samples, reselects

Current MIWV scores/filtered set are stale (computed against Run 1's smaller SFT data) — rescore once Run 2's SFT set is finalized.

## CPT details (recipe used in Run 1, and planned for Run 2 rerun)

- Base model: `meta-llama/Llama-3.1-8B` (weights in ~/.cache/huggingface/hub)
- Objective: causal LM on all tokens, sequence packing via TRL SFTTrainer with packing=True
- Hyperparams: lr 1e-5 cosine, 2 epochs, batch 4/GPU, grad accum 4, max_seq_len 2048, 4 GPUs, full-parameter FSDP (no LoRA support in `train_cpt.py` currently)
- Resume: SLURM scripts auto-detect valid checkpoints — safe to resubmit
- Known issue: FSDP doesn't write a `final/` dir — use the last `checkpoint-NNNN` directly
- Decided (2026-08-24): stays full-parameter for Run 2, not QLoRA — see "Run 2 — current state" above for reasoning

## Marek et al. batch-size/optimizer scaling (explored for `8b_small_batch/`, not yet run)

Reference: "Small Batch Size Training for Language Models" (Marek et al., NeurIPS 2025) — recommends avoiding gradient accumulation (wasteful for single-node data-parallel training) and scaling Adam's β2 to hold the second-moment half-life constant in tokens: `β2_new = β2_ref^(B_new/B_ref)`, not fixing β2 across batch sizes. LR scales with batch as an empirical power law (`batch^0.1585`), much more slowly than the commonly-used √batch rule.

`8b_small_batch/` applies this: effective batch 16 (grad-accum=1, no accumulation), β1=0.9 (unchanged), β2=0.95 (scaled), LR=8e-6 (scaled from originals via the power law), constant LR schedule (no warmup). 3 conditions only: CPT, SFT, CPT+SFT (no MIWV — kept as a clean 3-condition test). Built and smoke-verified (round-robin/config logic, not an actual training run), never submitted. Note: as of Run 2's corpus expansion, these scripts point at the now-larger `corpus_v3_train.jsonl`, so running them now would no longer be an isolated "same data, different batch/LR" test against Run 1's original results — corpus size becomes a confound too, unless a frozen copy of the old corpus is used instead.

## SFT results — v2 corpus (oldest round, historical reference only)

All runs: 3 epochs, lr 2e-5 cosine, batch 4/GPU, grad accum 8, max_seq_len 2048, 2 GPUs

| Run | Best eval loss | Best epoch | Final eval loss | Train acc |
|-----|---------------|------------|-----------------|-----------|
| 8B SFT MIWV (4,889 ex) | 0.764 | 1.37 | 0.832 | 95.3% |
| 8B CPT+SFT MIWV (4,889 ex) | 0.769 | 1.37 | 0.858 | 95.7% |
| 8B SFT full (48,896 ex) | 0.675 | 1.93 | 0.794 | 94.6% |
| 8B CPT+SFT full (48,896 ex) | 0.677 | 1.93 | 0.819 | 95.0% |

### Interpretation (for thesis methodology)

**Overfitting:** All runs overfit after epoch ~2. Best checkpoints are at epoch ~1.93 for full-data runs — use those for eval, not the final checkpoint.

**CPT makes almost no difference in SFT eval loss:** This is expected — token-level cross-entropy measures fluency and format matching, not factual domain knowledge. CPT improves what the model *knows* about TB biology, which only shows up in MCQ evaluation.

**Why eval loss is a weak metric here:** A model that generates fluent but factually wrong answers looks identical to a correct model by this metric. The MCQ evaluation is the real test.

### Why 70B CPT was skipped (Run 1)

1. **Compute:** ~80-100 hours, 13-17 SLURM resubmissions. Not practical.
2. **Diminishing returns:** Llama 3.1 70B already has strong biomedical knowledge. Our corpus is too small to meaningfully shift its domain distribution. (Note: corpus is now 148.4M tokens post-expansion — this reasoning hasn't been explicitly revisited against the new size.)
