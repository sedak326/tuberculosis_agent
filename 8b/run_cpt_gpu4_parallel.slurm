#!/bin/bash
#SBATCH --job-name=tb_cpt_run2
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_run2_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_run2_%j.err

mkdir -p /home/skavlak/finetuning/logs

# Needed by accelerate/deepspeed's is_peft_model() check during FSDP model save —
# missing this crashes the save step silently (see CLAUDE.md "Known bugs").
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_v3_run2_checkpoints

echo "Starting CPT (Run 2, expanded corpus) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# batch-size 4 x grad-accum 8 x 2 GPUs = 64 effective batch (matches Run 1's recipe,
# just spread over 2 GPUs instead of 4 since gpu4 is occupied by SFT generation)
$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29602 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus      /uss/skavlak/tb_corpus_v3/corpus_v3_train.jsonl \
    --model       meta-llama/Llama-3.1-8B \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          1e-5 \
    --batch-size  4 \
    --grad-accum  8 \
    --max-seq-len 2048 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT FAILED (exit $status)"
    exit $status
fi
echo "CPT complete."
