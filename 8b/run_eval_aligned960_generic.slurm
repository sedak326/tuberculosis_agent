#!/bin/bash
#SBATCH --job-name=tb_eval_aligned960
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eval_aligned960_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eval_aligned960_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

# COND_NAME, MODEL, TOKENIZER, ADAPTER (optional) passed in via sbatch --export
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python3
MCQ=/uss/skavlak/version_3/aligned_mcq_eval_scaled.jsonl
OUT=/uss/skavlak/version_3/eval_results/aligned960_${COND_NAME}.jsonl

echo "=== $COND_NAME on aligned-960 set ==="
echo "model=$MODEL tokenizer=$TOKENIZER adapter=${ADAPTER:-none}"

ADAPTER_ARG=""
if [ -n "$ADAPTER" ]; then
    ADAPTER_ARG="--adapter $ADAPTER"
fi

$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq "$MCQ" --model "$MODEL" --tokenizer "$TOKENIZER" $ADAPTER_ARG \
    --out "$OUT" \
    --randomize-answers --seed 42 --resume
status=$?

if [ $status -ne 0 ]; then
    echo "EVAL FAILED ($COND_NAME, exit $status)"
    exit $status
fi
echo "eval_aligned960_${COND_NAME} complete."
