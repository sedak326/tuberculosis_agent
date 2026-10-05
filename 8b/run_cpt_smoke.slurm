#!/bin/bash
#SBATCH --job-name=tb_cpt_smoke
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_smoke_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_smoke_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

echo "Starting CPT smoke test on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29501 \
    /home/skavlak/finetuning/train_cpt.py \
    --epochs      1 \
    --batch-size  1 \
    --grad-accum  1 \
    --sample      200 \
    --max-seq-len 512

echo "Smoke test complete."
