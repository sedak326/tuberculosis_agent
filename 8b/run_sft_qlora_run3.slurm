#!/bin/bash
#SBATCH --job-name=tb_sft_qlora_run3
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_qlora_run3_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_qlora_run3_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_llm_run3_qlora_checkpoints

echo "Starting SFT-only QLoRA (Run 3, no CPT) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe as Run 2's SFT-only QLoRA (rank 64, alpha 128, lr 2e-5, 2 epochs) —
# kept identical on purpose so corpus contamination status is the only
# manipulated variable vs. the established pipeline. Raw (non-instruct) base,
# same as Run 2.
$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29612 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_3/training_data.jsonl \
    --model       meta-llama/Llama-3.1-8B \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          2e-5 \
    --batch-size  1 \
    --grad-accum  16 \
    --max-seq-len 2048 \
    --bits        4 \
    --lora-rank   64 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "SFT-only QLoRA FAILED (exit $status)"
    exit $status
fi
echo "SFT-only QLoRA complete."
