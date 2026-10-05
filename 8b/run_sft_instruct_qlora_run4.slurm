#!/bin/bash
#SBATCH --job-name=tb_sft_instruct_qlora_run4
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_instruct_qlora_run4_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_instruct_qlora_run4_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_instruct_run4_qlora_checkpoints

echo "Starting SFT-from-Instruct QLoRA (Run 4) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe/hyperparams as run_sft_instruct_qlora_run3.slurm (rank 64,
# alpha 128, lr 2e-5, 2 epochs) — --data points at Run 4's
# training_data_run4.jsonl (fixed seeds) instead of Run 3's training_data.jsonl
# (pre-fix seeds), and a separate output dir so Run 3's checkpoint is
# untouched. gpu4 is fully free right now, so this uses 8 GPUs instead of
# Run 3's 2 (pure data-parallel under QLoRA — each GPU just holds its own
# 4-bit copy + LoRA adapter, no model parallelism needed for an 8B model).
# grad-accum dropped from 16 to 4 (2->8 GPUs = 4x) to hold the effective
# batch size at 32, identical to Run 3 — only parallelism changed, not the
# actual optimization trajectory, so this stays a clean seeds-only comparison.
$TORCHRUN \
    --nproc_per_node=8 \
    --master_port=29615 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_3/training_data_run4.jsonl \
    --model       meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          2e-5 \
    --batch-size  1 \
    --grad-accum  4 \
    --max-seq-len 2048 \
    --bits        4 \
    --lora-rank   64 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "SFT-from-Instruct QLoRA (Run 4) FAILED (exit $status)"
    exit $status
fi
echo "SFT-from-Instruct QLoRA (Run 4) complete."
