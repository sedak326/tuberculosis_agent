#!/bin/bash
#SBATCH --job-name=tb_sft_qlora_run2
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_qlora_run2_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_qlora_run2_%j.err

mkdir -p /home/skavlak/finetuning/logs

# Same env vars needed for FSDP jobs — harmless here too, kept for consistency.
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_llm_v3_run2_qlora_checkpoints

echo "Starting SFT-only QLoRA (Run 2, no CPT) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Matches Run 1's 70B QLoRA recipe exactly (rank 64, alpha 128 via train_sft.py's
# default LoraConfig, lr 2e-5, 2 epochs, adamw_torch) — just scaled to fewer GPUs
# since 8B's 4-bit footprint is far smaller than 70B's. Raw (non-instruct) base,
# same as the 70B run, not the Instruct model — see CLAUDE.md Run 2 notes.
$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29610 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/tb_corpus_v3/training_data_v3_run2.jsonl \
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
