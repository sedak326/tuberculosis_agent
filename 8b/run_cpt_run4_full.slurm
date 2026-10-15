#!/bin/bash
#SBATCH --job-name=tb_cpt_run4_full
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_run4_full_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_run4_full_%j.err

mkdir -p /home/skavlak/finetuning/logs

# Fixes the known-but-never-applied bug from Run 1 (see CLAUDE.md "Known bugs"):
# missing CUDA_HOME makes accelerate's FSDP model-save path import deepspeed and
# crash; missing exit-code checking then lets SLURM report that as COMPLETED.
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_run4_full_checkpoints

echo "Starting full-parameter CPT (Run 4, post-cutoff corpus) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Full-parameter FSDP - no --bits/--lora-rank, so train_cpt.py takes the
# use_fsdp path. Same hyperparameters as Run 1's original CPT recipe, only the
# corpus changed: the post-cutoff-only Run 3/4 corpus instead of the old v3
# corpus. Lets us directly compare full-parameter vs QLoRA CPT on an identical
# corpus, holding everything else constant - something Run 1 vs Run 3 never
# could, since they also differed in corpus.
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29521 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus      /uss/skavlak/version_3/corpus_v3_train.jsonl \
    --model       meta-llama/Llama-3.1-8B \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          1e-5 \
    --batch-size  4 \
    --grad-accum  4 \
    --max-seq-len 2048 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT (Run 4, full-parameter) FAILED (exit $status)"
    exit $status
fi
echo "CPT (Run 4, full-parameter) complete."
