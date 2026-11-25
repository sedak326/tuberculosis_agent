#!/bin/bash
#SBATCH --job-name=tb_sft_qlora_v7_medimind
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=7
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_qlora_v7_medimind_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_qlora_v7_medimind_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_v7_medimind_qlora_checkpoints

echo "Starting SFT-from-Instruct QLoRA (v7, MediMind SFT recipe, 100k data, no merge) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same base model and data as run_sft_instruct_qlora_v5_100k.slurm / v6. Recipe
# swapped to match MediMind's published Table IV: rank 32, alpha 32 (scaling
# factor 1), dropout 0.0, lr 2e-4, linear schedule, AdamW-8bit, weight decay
# 0.01, 3 epochs, effective batch originally 64 (their 32 x grad-accum 2 on 1
# GPU; initially matched here via batch 1 x grad-accum 16 x 4 GPUs). Sequence
# length kept at our own 2048 (not their 8192) since our SFT data was generated
# around that length and our other runs (v5/v6) use it too - keeps this an
# apples-to-apples rank/lr/schedule comparison rather than also varying context
# length.
#
# Resubmitted mid-training (resuming from checkpoint-1800) on 7 GPUs instead
# of 4, after temporarily pausing the parallel RAG index-build shards to free
# up gpu4 capacity, since faster wall-clock completion was wanted. grad-accum
# dropped 16->9 to land effective batch at 63 (7*1*9), the closest integer
# multiple of 7 to the original target of 64 - a small deliberate deviation
# traded for speed, worth noting if citing this run's exact effective batch.
$TORCHRUN \
    --nproc_per_node=7 \
    --master_port=29626 \
    /home/skavlak/finetuning/train_sft.py \
    --data          /uss/skavlak/version_5/training_data_v5_100k.jsonl \
    --model         meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer     meta-llama/Llama-3.1-8B-Instruct \
    --output-dir    $OUTPUT_DIR \
    --epochs        3 \
    --lr            2e-4 \
    --lr-scheduler-type linear \
    --optim         adamw_bnb_8bit \
    --weight-decay  0.01 \
    --batch-size    1 \
    --grad-accum    9 \
    --max-seq-len   2048 \
    --bits          4 \
    --lora-rank     32 \
    --lora-alpha    32 \
    --lora-dropout  0.0 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "SFT QLoRA (v7, MediMind recipe) FAILED (exit $status)"
    exit $status
fi
echo "SFT QLoRA (v7, MediMind recipe) complete."
