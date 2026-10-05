#!/bin/bash
#SBATCH --job-name=tb_cpt_run3_qlora
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=16:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_run3_qlora_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_run3_qlora_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_run3_qlora_checkpoints

echo "Starting Run 3 CPT (QLoRA, post-cutoff-only corpus) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# QLoRA CPT: added to train_cpt.py this run (previously full-parameter only).
# Rank 64/alpha 128 matches SFT's established recipe — same target modules.
# Batch 1 x grad-accum 4 x 4 GPUs = 16 effective batch (each GPU holds the full
# quantized model under DDP, not FSDP-sharded, so per-GPU batch stays small).
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29520 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus      /uss/skavlak/version_3/corpus_v3_train.jsonl \
    --model       meta-llama/Llama-3.1-8B \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          1e-5 \
    --batch-size  1 \
    --grad-accum  4 \
    --max-seq-len 2048 \
    --bits        4 \
    --lora-rank   64 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT FAILED (exit $status)"
    exit $status
fi
echo "CPT complete."
