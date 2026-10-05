#!/bin/bash
#SBATCH --job-name=tb_sft_sb
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_sb_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_sb_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

echo "Starting SFT training (small batch, Marek et al. 2025 scaling) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# batch-size 4 x grad-accum 1 x 4 GPUs = 16 effective batch (no accumulation)
# Matches CPT's GPU count and LR so batch size, beta2, and LR are all consistent across the whole pipeline.
# beta2 = 0.95^(16/16) = 0.95
# lr    = shared with CPT (see run_cpt.slurm) rather than scaled separately from SFT's old 2e-5
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29542 \
    /home/skavlak/finetuning/train_sft.py \
    --data         /uss/skavlak/tb_corpus_v3/training_data_v3.jsonl \
    --model        meta-llama/Llama-3.1-8B \
    --tokenizer    meta-llama/Llama-3.1-8B-Instruct \
    --output-dir   /uss/skavlak/tb_llm_v3_smallbatch_checkpoints \
    --epochs       3 \
    --lr           8e-6 \
    --batch-size   4 \
    --grad-accum   1 \
    --adam-beta1   0.9 \
    --adam-beta2   0.95 \
    --lr-scheduler-type constant \
    --max-seq-len  2048
status=$?

if [ $status -ne 0 ]; then
    echo "Training FAILED (exit $status)"
    exit $status
fi
echo "Training complete."
