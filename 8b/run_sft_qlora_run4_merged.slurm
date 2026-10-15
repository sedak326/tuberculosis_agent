#!/bin/bash
#SBATCH --job-name=tb_sft_qlora_run4_merged
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_qlora_run4_merged_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_qlora_run4_merged_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_run4_merged_qlora_checkpoints
MERGED=/uss/skavlak/tb_cpt_run4_full_instruct_slerp_merged

echo "Starting SFT QLoRA on top of the full-CPT x Instruct SLERP merge (Run 4) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe as every other Run 4 SFT condition (rank 64 QLoRA, lr 2e-5, 2
# epochs, effective batch 32) and the fixed-seed training data - only the
# starting point differs: the SLERP-merged (full CPT x Instruct) model
# instead of raw Instruct or the QLoRA-CPT-merged checkpoint.
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29617 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_3/training_data_run4.jsonl \
    --model       "$MERGED" \
    --tokenizer   "$MERGED" \
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
    echo "SFT QLoRA on merged model FAILED (exit $status)"
    exit $status
fi
echo "SFT QLoRA on merged model (Run 4) complete."
