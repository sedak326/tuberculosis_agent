#!/bin/bash
#SBATCH --job-name=tb_eval_cpt_sft_run2
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_cpt_sft_run2_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_cpt_sft_run2_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
OUT_DIR=/uss/skavlak/tb_corpus_v3/eval_results
mkdir -p "$OUT_DIR"

echo "=== CPT + SFT QLoRA ==="
$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --model      meta-llama/Llama-3.1-8B \
    --adapter    /uss/skavlak/tb_sft_cpt_v3_run2_qlora_checkpoints/final \
    --tokenizer  meta-llama/Llama-3.1-8B-Instruct \
    --out        "$OUT_DIR/8b_sft_cpt_qlora_run2.jsonl" \
    --resume
status=$?
if [ $status -ne 0 ]; then
    echo "EVAL FAILED (exit $status)"
    exit $status
fi
echo "eval_cpt_sft_run2 complete."
