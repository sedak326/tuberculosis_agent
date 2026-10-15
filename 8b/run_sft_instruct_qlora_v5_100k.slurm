#!/bin/bash
#SBATCH --job-name=tb_sft_instruct_qlora_v5_100k
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_instruct_qlora_v5_100k_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_instruct_qlora_v5_100k_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_instruct_v5_100k_qlora_checkpoints

echo "Starting SFT-from-Instruct QLoRA (v5, 100k data, no merge) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe as run_sft_qlora_v5_merged_100k.slurm (rank 64 QLoRA, lr 2e-5,
# 2 epochs, effective batch 32, same 100k data) - only the base model differs:
# plain Llama-3.1-8B-Instruct instead of the CPT x Instruct SLERP merge.
# Isolates whether the 100k dataset itself (vs run4's 40k) moves the needle,
# independent of the merge.
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29624 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_5/training_data_v5_100k.jsonl \
    --model       meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          2e-5 \
    --batch-size  1 \
    --grad-accum  8 \
    --max-seq-len 2048 \
    --bits        4 \
    --lora-rank   64 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "SFT-from-Instruct QLoRA (v5, 100k) FAILED (exit $status)"
    exit $status
fi
echo "SFT-from-Instruct QLoRA (v5, 100k) complete."
