# TB LLM Project — Plan & Status

## Storage

Large files (model checkpoints, large datasets, raw papers) must be stored on `/uss`, not in `/home/skavlak`. The home directory is on a 4T network filesystem shared across the lab. Always point SLURM output dirs, checkpoint dirs, and new paper downloads at `/uss` before launching a job.

## Cluster

- `gpu4`: 8× RTX PRO 6000 (96GB each). Keep individual jobs under 6 hours to avoid blocking others — safe to resubmit, scripts have auto-resume logic.
- `gpu3`: 2× RTX 6000 Ada (48GB each). Usable for smaller 8B jobs (~40-50GB/GPU need on a 2-way FSDP split) when gpu4 is occupied.
- `gpu` / `gpu2`: 2× RTX 4090 (24GB each) per node. Too small for full-parameter 8B FSDP training; fine for lighter workloads.

## Goal

Build a domain-specific LLM that serves as a research assistant for tuberculosis, combining all existing TB research and literature. Focus on protein/omics-level interactions. The LLM should assist researchers with questions, reasoning, and hypothesis generation — like Claude Code but for TB research.

## Status (updated 2026-08-24)

There are two distinct rounds of work. **Do not conflate them.**

- **Run 1** — the original v3-corpus pipeline below. COMPLETE, all 9 conditions scored. Results were disappointing (see "Run 1 results" below) and diagnosed as full-parameter-fine-tuning catastrophic forgetting on a severely data-starved 8B model. **Kept frozen as a reference point — do not delete or overwrite these results/checkpoints.**
- **Run 2** — a fresh effort, in progress, not a patch on Run 1. Two changes from Run 1: (1) a much larger literature corpus, (2) exploring QLoRA instead of full-parameter fine-tuning to fix the data-starvation problem. Approach still being finalized as of this writing.

### Run 2 — current state

- **Corpus**: expanded from 56.9M → **148.4M CPT tokens** (382,700 chunks) by fetching 12,976 additional verified on-topic papers via EuropePMC (see "Corpus v3 expansion" below). Live at `corpus_v3_train.jsonl`. The 200-paper/2,950-chunk held-out eval set (`corpus_v3_eval.jsonl`) was protected during the merge — untouched.
- **SFT data**: being regenerated from scratch with rebalanced categories (see "SFT categories" below). Old Run 1 SFT data (48,782 examples, unbalanced) archived at `archive/training_data_v3_unbalanced_20260807.jsonl` — not deleted, not in use. New generation in progress: **258,722 of ~637,700 target examples** as of 2026-08-24 (paused/resumed multiple times; see `data_prep/auto_resubmit_generation.sh`).
- **CPT**: decided to keep full-parameter (not QLoRA) for Run 2 — CPT's job is broad knowledge injection, which full-parameter updates likely handle better than LoRA's low-rank constraint; also keeps the CPT+SFT-vs-SFT ablation clean (adding QLoRA-CPT would confound "more data" with "different method"). Same recipe as Run 1 (see "CPT details"), rerun once against the new corpus. **Blocked on gpu4 availability** (fully occupied by SFT generation as of this writing).
- **SFT training method**: **not yet decided.** Leaning QLoRA for 8B — rank-64 LoRA has ~168M trainable params (48x fewer than full 8B), which takes tokens/trainable-param from 0.006 (the value that caused Run 1's catastrophic forgetting) to ~0.30 using the *same* data already in hand. This is the most important open decision — nothing else in Run 2's SFT plan is finalized until this is settled.
- **Model sizes**: reconsidering whether 70B is worth including. In Run 1, 70B (QLoRA) and 8B (full-parameter) were confounded — couldn't tell if 70B's better behavior was scale or training method. If 8B also moves to QLoRA, 70B QLoRA becomes a cleaner "does scale matter, holding method constant" comparison — genuinely interesting, but not necessary to validate the fix (70B was never the broken one in Run 1). Current lean: prioritize nailing 8B QLoRA first (cheap/fast to iterate — needs far fewer GPU-hours than 70B), treat 70B QLoRA as a secondary addition once 8B is working, not something to build in parallel from the start.
- **MIWV**: stays in the pipeline conceptually, but existing MIWV scores are stale (computed against the old, smaller data) and would need rescoring once the new SFT set is finalized.
- **Not decided yet**: exact LoRA rank/alpha/target-modules for 8B SFT, whether to also retry the instruct-base experiment (see "Known bugs" below), whether/when to run the `8b_small_batch` hyperparameter ablation (built, never run — see below).

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

As of this writing, the seed pool has grown beyond what's described above via other contributors (Sarah, adriana, Ricardo, Nate, Melina) adding seeds independently — **check `data_prep/seeds.json` directly for current category counts rather than trusting this doc**, it's a live, actively-edited file.

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

### Corpus v3 (Run 1 + Run 2 expansion, current)
- `/uss/skavlak/tb_corpus_v3/corpus_v3_train.jsonl` — **live CPT training corpus, 382,700 chunks, 148.4M tokens**
- `/uss/skavlak/tb_corpus_v3/corpus_v3_eval.jsonl` — held-out eval split, 2,950 chunks / 200 papers, protected
- `/uss/skavlak/tb_corpus_v3/mcq_eval.jsonl` — 800-question MCQ eval set
- `/uss/skavlak/tb_corpus_v3/archive/` — old unbalanced SFT data + shards (Run 1), not in use
- `/uss/skavlak/tb_corpus_v3/training_data_shard_*.jsonl` — Run 2 SFT generation output, in progress
- `/uss/skavlak/tb_corpus_v3_extra/` — raw newly-fetched papers + their extracted corpus (already merged into corpus_v3_train.jsonl)
- MIWV scores/filtered data (`miwv_scores_v3.npy`, `training_data_miwv_v3_top10.jsonl`) — stale, computed against old data, need rescoring once Run 2 SFT data is final

### Seeds
`data_prep/seeds.json` — actively growing, multiple contributors. Check directly for current state.

## Checkpoints — Run 2 (v3 expanded corpus — to be created)

| Run | Location | Status |
|-----|----------|--------|
| 8B CPT (Run 2) | TBD, don't overwrite `tb_cpt_v3_checkpoints` (Run 1) | not started, blocked on gpu4 |
| 8B SFT (method TBD: full-param or QLoRA) | TBD | not started |
| 70B SFT (QLoRA, if pursued) | TBD | not started |

## Folder structure

```
finetuning/
  8b/                      — Run 1 8B SLURM scripts (full-parameter)
    run_cpt.slurm
    run_training.slurm
    run_sft_cpt.slurm
    run_training_miwv.slurm
    run_cpt_instruct.slurm, run_training_instruct.slurm, etc. — instruct-base attempt, broken (see Known bugs)
  8b_large_batch/          — batch~8000 ablation, misreading of Marek et al. paper, built, never run
  8b_small_batch/          — corrected batch-size ablation (batch 16, paper-scaled beta2/LR), built, never run
  70b/                     — 70B SLURM scripts (QLoRA)
    run_training.slurm     — 70B SFT (8 GPUs, 2 epochs, QLoRA)
  data_prep/               — corpus, generation, MIWV, seeds
    fetch_europmc_batch.py — main literature fetch script, TITLE/ABSTRACT-restricted + retraction filter
    fetch_pmc_batch.py     — legacy NCBI MeSH-query fetch, superseded by fetch_europmc_batch.py
    generate_training_data.py — SFT generation, category-balanced (Run 2)
    auto_resubmit_generation.sh — handles SLURM 4hr time-limit resubmission for generation shards
    seeds.json              — seed task pool, actively growing
  refs/                    — papers and docs
  train_cpt.py             — supports --adam-beta1/--adam-beta2/--lr-scheduler-type (added for Marek et al. experiments); no LoRA/QLoRA support yet
  train_sft.py             — supports --bits/--lora-rank (QLoRA) and --adam-beta1/--adam-beta2/--lr-scheduler-type
  mtubercolosis/           — old proteomics corpus (v1/v2, keep for reference)
/uss/skavlak/tb_corpus_v3/       — v3 corpus, all large files (Run 1 + Run 2)
/uss/skavlak/tb_corpus_v3_extra/ — Run 2 newly-fetched papers, already merged into corpus_v3_train.jsonl
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
