# Tuberculosis Research Assistant LLM

A domain-specific LLM research assistant for tuberculosis (TB) research, built as part of a
Master's thesis. The goal is a model that can help TB researchers with questions, reasoning,
and hypothesis generation over the TB literature — grounded in real TB/*Mycobacterium
tuberculosis* research, not just general biomedical knowledge.

## Approach

The project adapts `Llama-3.1-8B-Instruct` (and, for one comparison recipe, `Qwen2.5-7B-Instruct`)
to the TB domain through a four-stage pipeline:

1. **Corpus construction** — TB/Mtb-specific research literature fetched from EuropePMC,
   restricted to papers that genuinely discuss TB in the title/abstract (not just a passing
   mention). The current corpus is contamination-controlled: built only from papers published
   after Llama 3.1's pretraining cutoff (Dec 2023), so continued pretraining can't re-expose the
   model to text it may have memorized, and the held-out evaluation set is guaranteed unseen.
2. **Continued pretraining (CPT)** — further pretraining on the TB corpus to inject domain
   knowledge. Three CPT recipes are compared side by side on the *same* corpus: a custom recipe,
   a recipe following Doll et al. (2026)'s DeFineMed setup, and a recipe following
   Labrak et al. (2024)'s BioMistral setup.
3. **Model merging (SLERP)** — CPT on raw text tends to degrade instruction-following. SLERP
   merging between the CPT checkpoint and the original Instruct model recovers most of that
   ability while keeping the domain knowledge gained during CPT.
4. **Supervised fine-tuning (QLoRA SFT)** — fine-tuning on TB-specific instruction data
   (consultation-style Q&A, critique, hypothesis generation) to turn the knowledge-adapted model
   into something that follows a research-assistant interaction style.

On top of the fine-tuned model sits a **retrieval-augmented generation (RAG) layer**, which
retrieves relevant passages from the TB corpus at query time instead of (or in addition to)
relying on the model's parametric knowledge. Three retrieval backends are implemented and
compared: a custom Chroma/PubMedBERT index, a MediMind-style dense index (jina-embeddings-v3),
and an index built following Bal & Puhan's retrieval setup.

## Key findings

Every condition (CPT alone, CPT+SFT, RAG) is evaluated against two differently-composed
multiple-choice QA sets — one balanced across recall/understanding/application/analysis
cognitive levels, one weighted entirely toward reasoning/critique-style questions — scored by
GPT-4o on both answer correctness and explanation quality (factual accuracy, coherence,
naturalness, completeness), with paired t-tests against the untouched base model.

The central finding, validated across CPT-only, CPT+SFT, and RAG conditions: **domain adaptation
reliably improves performance on knowledge-recall questions but not on reasoning/analysis
questions.** Continued pretraining teaches the model TB facts (next-token prediction on raw text
shapes what the model knows), but doesn't teach it to reason better over that knowledge — the
largest and most consistent accuracy gains are concentrated specifically in the recall-type
question category, while reasoning-heavy categories stay flat. RAG shows the mirror image: it
*hurts* accuracy on reasoning-heavy questions (retrieved context dilutes/distracts from the
actual reasoning task) while *helping* recall-type questions for two of the three retrieval
backends tested.

Full numeric results, statistical tests, and the RAG diagnostic (including an adaptive-retrieval
confidence-gating experiment) are in [`FINAL_EVALUATION_RESULTS.md`](FINAL_EVALUATION_RESULTS.md).

## Repository structure

| Path | Contents |
|---|---|
| `train_cpt.py`, `train_sft.py` | Core training scripts (continued pretraining, QLoRA SFT) |
| `data_prep/` | Corpus fetching/extraction, synthetic instruction-data generation, MCQ eval set construction, the evaluation pipeline (`evaluate_emcqa.py` → `score_explanations.py`) |
| `rag/` | RAG pipelines (indexing, retrieval, generation) for all three retrieval backends, plus the interactive chat UI (`chat_ui.py`) |
| `merge_configs/` | SLERP merge configuration files |
| `8b/` | SLURM job scripts for the 8B-parameter model pipeline (CPT, merging, SFT, evaluation) |
| `70b/`, `8b_small_batch/` | Exploratory/comparison SLURM scripts (larger model, batch-size scaling) |
| `eval/` | Early/miscellaneous evaluation scripts |
| `refs/` | Reference papers cited in the methodology |
| `figures/` | Generated plots |
| `CLAUDE.md` | Detailed internal project log: experiment history, diagnoses, data paths, cluster setup |
| `FINAL_EVALUATION_RESULTS.md` | Full results tables, statistical tests, and the RAG diagnostic writeup |

Model checkpoints, the full corpus, and raw papers are not stored in this repo (too large) —
see `CLAUDE.md` for where they live.

## Status

Active thesis project. See `CLAUDE.md` for the full experiment log (what's been run, what's
pending, known issues) and `FINAL_EVALUATION_RESULTS.md` for the latest scored results.
