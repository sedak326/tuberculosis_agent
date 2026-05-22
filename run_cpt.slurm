#!/bin/bash
#SBATCH --job-name=tb_cpt
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/home/skavlak/finetuning/mtubercolosis/output/tb_cpt_v2_checkpoints

echo "Starting CPT on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Only resume if a valid checkpoint exists (has trainer_state.json).
# Prevents crashing on the first run or after an incomplete checkpoint.
RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29501 \
    /home/skavlak/finetuning/train_cpt.py \
    --model       meta-llama/Llama-3.1-8B \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          1e-5 \
    --batch-size  4 \
    --grad-accum  4 \
    --max-seq-len 2048 \
    $RESUME_ARG

echo "CPT complete."
