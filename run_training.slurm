#!/bin/bash
#SBATCH --job-name=tb_sft
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

echo "Starting SFT training on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29500 \
    /home/skavlak/finetuning/train_sft.py \
    --model       meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  /home/skavlak/finetuning/mtubercolosis/output/tb_llm_checkpoints \
    --epochs      3 \
    --batch-size  4 \
    --grad-accum  8 \
    --max-seq-len 2048 \
    --resume-from /home/skavlak/finetuning/mtubercolosis/output/tb_llm_checkpoints/checkpoint-600

echo "Training complete."
