#!/bin/bash
#SBATCH --job-name=tb_cpt_70b
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=6
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_70b_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_70b_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_70b_checkpoints
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

echo "Starting 70B CPT on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

$TORCHRUN \
    --nproc_per_node=6 \
    --master_port=29502 \
    /home/skavlak/finetuning/train_cpt.py \
    --model       meta-llama/Llama-3.1-70B \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          1e-5 \
    --batch-size  1 \
    --grad-accum  16 \
    --max-seq-len 2048 \
    --optim       adamw_bnb_8bit \
    $RESUME_ARG

echo "70B CPT complete."
