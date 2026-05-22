# TB LLM Project — Plan & Status

## Goal

Build a domain-specific LLM that serves as a research assistant for tuberculosis, combining all existing TB research and literature. It is supposed to serve as a hub of all tuberculosis knowledge eventually, but for no we want to focus on protein level (omics level) interactions. The LLM should be able to assist future tuberculosis researchers with questions, help reason about certain problems or phenomena and assist with generating hypotheses. Basically like claude-code but for tuberculosis research instead of coding. 

## Approach

CPT (continued pre-training) followed by SFT (supervised fine-tuning). This is the standard approach used in biomedical NLP (BioGPT, BioMedLM, etc.). This is mainly for me to learn how to do these things. The outcome of the project is secondary, but still important. 

CPT stage: train a base model on raw TB literature text using causal language modelling objective.
SFT stage: fine-tune the CPT checkpoint on instruction-response pairs to teach it to answer questions and follow instructions in the TB domain. Eventually we also want to use reinforcement learning to better align the LLM wit expectations and also provide tool use for databases etc. 

## Data

### Current corpus
- ~80 TB research papers in mtubercolosis/ (PubMed IDs as filenames, mix of PDFs and Elsevier XMLs)
- extract_corpus.py extracts text, tables into structured chunks with section/chapter metadata
- Output: mtubercolosis/output/corpus.jsonl
- Extracted corpus lives at mtubercolosis/output/corpus.jsonl — 24,238 chunks (23,102 text + 1,136 table)

This is just the initial corpus until we know what the exact pipeline is gonna be. I am aware similar LLM CPT projects used millions of literature (papers, books).

### Planned expansion
- Pull a larger TB-domain subset from PubMed to supplement the 80 papers
- More volume is needed for CPT to be effective

### Seed tasks
- seed_tasks.md contains the seed examples used to guide synthetic data generation
- I am gonna write 80 seeds with my Prof who is a tuberculosis specialist. 
- Examples for how they are gonna look can be found in seed_tasks.md
- Human-written seeds matter because LLM-written seeds introduce uniformity that degrades generation diversity 
- Once we have those seeds we will use LLama to come up with more questions 

## Pipeline

### Stage 1 — Corpus extraction
Script: extract_corpus.py
Input: mtubercolosis/*.pdf and mtubercolosis/*.xml
Output: mtubercolosis/output/corpus.jsonl
Environment: autorag conda env (needs fitz/PyMuPDF, pdfplumber, Pillow, openai)
Status: not yet run on current setup

### Stage 2 — Synthetic data generation
Script: generate_training_data.py
Input: corpus.jsonl + seed_tasks.md
Output: mtubercolosis/output/training_data.jsonl (sharded during SLURM runs)
Model: Llama-3.3-70B-Instruct via local vLLM server
Generates 2 instruction-response pairs per corpus chunk
Runs as 16-shard SLURM array job (run_generation.slurm) on gpu4
Supports resume on interruption
Status: was completed in a previous run — 22,293 training pairs generated (training_data_clean3.jsonl)

### Stage 3 — SFT
Script: train_sft.py
Input: training_data_clean3.jsonl
Model: meta-llama/Llama-3.1-8B-Instruct
FSDP full_shard on 2 GPUs, 3 epochs, bf16, lr 2e-5 cosine
Status: completed. Final train loss 0.0825, eval loss ~0.845, token accuracy ~78.9%
Checkpoints: mtubercolosis/output/tb_llm_checkpoints/ (600, 800, 993 steps + final/)
Known issue: final/ directory only contains tokenizer files, no model weights — likely an FSDP merge issue

### Stage 0 — CPT
Scripts: train_cpt.py, run_cpt.slurm, run_cpt_smoke.slurm
Input: mtubercolosis/output/corpus.jsonl (text fields only, metadata headers stripped)
Base model: meta-llama/Llama-3.1-8B (base, not instruct) — downloaded to ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
Output: mtubercolosis/output/tb_cpt_checkpoints/
Objective: causal language modelling on all tokens (no response masking), sequence packing via TRL SFTTrainer with packing=True
Hyperparameters: lr 1e-5 cosine, 2 epochs, batch 4 per GPU, grad accum 4, max seq len 2048, 4 GPUs, 6-hour SLURM time limit
Resume: SLURM script auto-detects valid checkpoints (checks for trainer_state.json) — safe to resubmit repeatedly
Status: smoke test passed on gpu4. Real job submitted but failed on first run due to incomplete checkpoint-59 from smoke test (now deleted). Fixed by adding checkpoint validation to SLURM script.

Reference implementation: /home/skavlak/PMC-LLaMA (the paper this CPT approach is based on)
Key differences vs PMC-LLaMA:
- PMC-LLaMA pre-tokenizes papers to .npy files for efficiency; we tokenize on-the-fly (fine for 80 papers, worth switching when corpus scales to thousands)
- PMC-LLaMA uses 512-token random windows; we use 2048-token packed sequences (better token utilisation)
- PMC-LLaMA runs 30 epochs on millions of papers; we run 2 epochs on ~80 papers to avoid overfitting
- PMC-LLaMA uses lr 2e-5; we use 1e-5 (more conservative to limit catastrophic forgetting)
- PMC-LLaMA uses plain Trainer; we use SFTTrainer with packing=True (functionally identical for CPT)

## Key files

- extract_corpus.py — corpus extraction from PDFs and XMLs
- generate_training_data.py — synthetic instruction-response pair generation
- train_sft.py — SFT training script
- seed_tasks.md — seed examples for data generation
- run_generation.slurm — 16-shard SLURM array job for generation
- run_training.slurm — SLURM job for SFT
- run_training_miwv.slurm — SLURM job variant (MIWV data selection)
- select_data_miwv.py / reselect_miwv.py — MIWV-based data selection scripts (see below)
- deepspeed_zero2.json — DeepSpeed ZeRO stage 2 config
- setup_scratch.sh — copies model/corpus to /scratch before SLURM jobs
- ocae122_Supplementary_Data/appendix.tex — BioInstruct paper appendix with seed task format reference

## MIWV Data Selection

Reference paper: "Importance-Aware Data Selection for Efficient LLM Instruction Tuning" (Jiang et al., AAAI-26), PDF at finetuning/40396-Article Text-44487-1-2-20260314.pdf

The core idea is that not all generated training samples are equally useful. Some samples the base model can already handle well — training on those wastes compute and may introduce noise. MIWV (Model Instruction Weakness Value) is a metric that quantifies how much a given sample actually challenges the model and improves its capabilities.

### How MIWV is computed

For each training sample i:
1. Find its nearest neighbor in the dataset by embedding similarity (using BAAI/bge-large-en-v1.5 + FAISS).
2. Compute the base model's cross-entropy loss on sample i's response with NO extra context (zero-shot loss).
3. Compute the base model's cross-entropy loss on sample i's response WITH the neighbor prepended as a one-shot example (one-shot loss).
4. MIWV(i) = one_shot_loss - zero_shot_loss

A high positive MIWV means: the model's loss got worse (or didn't improve) even when given a helpful example — i.e. the model fundamentally lacks the capability to handle this type of sample. These are the most valuable samples for training. A low or negative MIWV means: the model already handles this well with minimal context, so it's less informative.

The paper showed that selecting only the top 10% of samples by MIWV score can outperform training on the full dataset.

### Implementation

select_data_miwv.py:
- Loads training_data_clean3.jsonl (22,293 pairs)
- Embeds all instructions, builds a FAISS index, finds each sample's nearest neighbor
- Loads Llama-3.1-8B-Instruct as the base model (bfloat16, device_map=auto)
- For each sample, computes zero-shot and one-shot cross-entropy loss on the assistant response only (user turn is masked in labels)
- Saves scores incrementally to miwv_scores.npy every 500 steps (supports resume)
- Selects top 10% by score and writes to training_data_miwv_top10.jsonl
- Run via: sbatch run_miwv.slurm

reselect_miwv.py:
- Runs after select_data_miwv.py, no recomputation needed — reuses miwv_scores.npy
- Before selecting, masks out "contaminated" samples where the LLM leaked meta-instructions into its output (patterns like "step 1:", "instruction-response pair", "first training example", etc.)
- Recalculates top 10% from the clean subset only
- Output: training_data_miwv_top10_clean.jsonl

### Status
Was run on the previous training_data_clean3.jsonl dataset. If the corpus and generation steps are rerun with the new seed tasks and expanded corpus, MIWV scoring will need to be rerun on the new dataset.
