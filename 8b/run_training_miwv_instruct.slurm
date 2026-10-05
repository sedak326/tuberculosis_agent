#!/bin/bash
#SBATCH --job-name=tb_sft_miwv_instruct
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_miwv_instruct_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_miwv_instruct_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

echo "Starting MIWV SFT (instruct base) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29521 \
    /home/skavlak/finetuning/train_sft.py \
    --data      /uss/skavlak/tb_corpus_v3/training_data_miwv_v3_top10.jsonl \
    --model     meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --output-dir /uss/skavlak/tb_llm_miwv_v3_instruct_checkpoints \
    --epochs    2 \
    --batch-size  4 \
    --grad-accum  8 \
    --max-seq-len 2048

echo "Training complete."
