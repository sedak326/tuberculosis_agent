#!/bin/bash
#SBATCH --job-name=tb_sft_miwv
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_miwv_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_miwv_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

echo "Starting MIWV SFT on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29501 \
    /home/skavlak/finetuning/train_sft.py \
    --data      /home/skavlak/finetuning/mtubercolosis/output/training_data_miwv_top10_clean.jsonl \
    --model     meta-llama/Llama-3.1-8B-Instruct \
    --output-dir /home/skavlak/finetuning/mtubercolosis/output/tb_llm_miwv_checkpoints \
    --epochs    2 \
    --batch-size  4 \
    --grad-accum  8 \
    --max-seq-len 2048

echo "Training complete."
