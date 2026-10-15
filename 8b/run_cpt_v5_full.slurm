#!/bin/bash
#SBATCH --job-name=tb_cpt_v5_full
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_v5_full_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_v5_full_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_v5_full_checkpoints

echo "Starting full-parameter CPT (v5: combined, deduped corpus across all runs," \
     "no post-cutoff restriction, CPT starting from Llama-3.1-8B-INSTRUCT not raw" \
     "base - matches the SLERP-merge paper's own recipe) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same hyperparameters as the Run 4 full-parameter CPT recipe - only the corpus
# (489,423 deduped chunks / ~173M tokens, combined across tb_corpus_v3 + version_3,
# no cutoff-year filter) and the base model (Instruct, not raw) changed.
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29522 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus      /uss/skavlak/version_5/corpus_v5_train.jsonl \
    --model       meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          1e-5 \
    --batch-size  4 \
    --grad-accum  4 \
    --max-seq-len 2048 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT (v5, full-parameter) FAILED (exit $status)"
    exit $status
fi
echo "CPT (v5, full-parameter) complete."
