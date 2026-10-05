#!/bin/bash
#SBATCH --job-name=tb_sft_cpt_sb
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_cpt_sb_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_cpt_sb_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache

CPT_DIR=/uss/skavlak/tb_cpt_v3_smallbatch_checkpoints
CPT_FINAL=$(ls -td "$CPT_DIR"/checkpoint-* 2>/dev/null | while read ckpt; do [ -f "$ckpt/trainer_state.json" ] && echo "$ckpt" && break; done)
OUTPUT_DIR=/uss/skavlak/tb_sft_cpt_v3_smallbatch_checkpoints

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

RESUME_FROM=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | while read ckpt; do
        [ -f "$ckpt/trainer_state.json" ] && echo "$ckpt" && break
    done)
    [ -n "$LATEST" ] && RESUME_FROM="--resume-from $LATEST"
fi

echo "Starting SFT-from-CPT (small batch, Marek et al. 2025 scaling) on $(hostname)"
echo "Base model : $CPT_FINAL"
echo "Output dir : $OUTPUT_DIR"
[ -n "$RESUME_FROM" ] && echo "Resuming   : $RESUME_FROM"

if [ -z "$CPT_FINAL" ]; then
    echo "ERROR: no CPT checkpoint found in $CPT_DIR — run run_cpt.slurm first."
    exit 1
fi

# batch-size 4 x grad-accum 1 x 4 GPUs = 16 effective batch (no accumulation)
# Matches CPT's GPU count and LR so batch size, beta2, and LR are all consistent across the whole pipeline.
# beta2 = 0.95^(16/16) = 0.95
# lr    = shared with CPT (see run_cpt.slurm) rather than scaled separately from SFT's old 2e-5
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29543 \
    /home/skavlak/finetuning/train_sft.py \
    --data         /uss/skavlak/tb_corpus_v3/training_data_v3.jsonl \
    --model        "$CPT_FINAL" \
    --tokenizer    meta-llama/Llama-3.1-8B-Instruct \
    --output-dir   "$OUTPUT_DIR" \
    --epochs       3 \
    --lr           8e-6 \
    --batch-size   4 \
    --grad-accum   1 \
    --adam-beta1   0.9 \
    --adam-beta2   0.95 \
    --lr-scheduler-type constant \
    --max-seq-len  2048 \
    $RESUME_FROM
status=$?

if [ $status -ne 0 ]; then
    echo "Training FAILED (exit $status)"
    exit $status
fi
echo "Training complete."
