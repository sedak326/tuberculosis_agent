#!/bin/bash
#SBATCH --job-name=tb_oldeval_v10_cpt_sft
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/oldeval_v10_cpt_sft_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/oldeval_v10_cpt_sft_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

PYTHON=/home/skavlak/venv/bin/python3
MODEL=/uss/skavlak/tb_cpt_v10_biomistral_full_instruct_slerp_merged
ADAPTER=/uss/skavlak/tb_sft_v10_biomistral_merged_100k_qlora_checkpoints/final
MCQ=/uss/skavlak/version_3/mcq_eval.jsonl
OUT_DIR=/uss/skavlak/version_3/eval_results
RAW_OUT=$OUT_DIR/oldeval_v10_biomistral_sft100k.jsonl
SCORED_OUT=$OUT_DIR/oldeval_v10_biomistral_sft100k_scored.jsonl

echo "=== oldeval BioMistral/Labrak CPT+SFT (v10) on old 800-Q mcq_eval.jsonl ==="
echo "model=$MODEL tokenizer=$MODEL adapter=$ADAPTER"

$PYTHON /home/skavlak/finetuning/data_prep/evaluate_emcqa.py \
    --mcq        "$MCQ" \
    --model      "$MODEL" \
    --tokenizer  "$MODEL" \
    --adapter    "$ADAPTER" \
    --out        "$RAW_OUT" \
    --randomize-answers --seed 42 --resume
status=$?
if [ $status -ne 0 ]; then
    echo "INFERENCE FAILED, exit $status"
    exit $status
fi

echo "=== scoring with GPT-4o ==="
$PYTHON /home/skavlak/finetuning/data_prep/score_explanations.py \
    --mcq "$MCQ" \
    --results "$RAW_OUT" \
    --out "$SCORED_OUT" \
    --randomize-answers --seed 42 --resume
status=$?
if [ $status -ne 0 ]; then
    echo "SCORING FAILED, exit $status"
    exit $status
fi
echo "oldeval_v10_cpt_sft complete."
