#!/bin/bash
#SBATCH --job-name=tb_sft_cpt
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_cpt_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_cpt_%j.err

mkdir -p /home/skavlak/finetuning/logs

CPT_FINAL=/home/skavlak/finetuning/mtubercolosis/output/tb_cpt_checkpoints/final
OUTPUT_DIR=/home/skavlak/finetuning/mtubercolosis/output/tb_sft_cpt_checkpoints

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

# Auto-resume: pick the latest valid checkpoint (has trainer_state.json).
RESUME_FROM=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | while read ckpt; do
        [ -f "$ckpt/trainer_state.json" ] && echo "$ckpt" && break
    done)
    [ -n "$LATEST" ] && RESUME_FROM="--resume-from $LATEST"
fi

echo "Starting SFT-from-CPT on $(hostname)"
echo "Base model : $CPT_FINAL"
echo "Output dir : $OUTPUT_DIR"
[ -n "$RESUME_FROM" ] && echo "Resuming   : $RESUME_FROM"

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29501 \
    /home/skavlak/finetuning/train_sft.py \
    --model       "$CPT_FINAL" \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  "$OUTPUT_DIR" \
    --epochs      3 \
    --lr          2e-5 \
    --batch-size  4 \
    --grad-accum  8 \
    --max-seq-len 2048 \
    $RESUME_FROM

echo "Training complete."
