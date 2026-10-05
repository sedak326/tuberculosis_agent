#!/bin/bash
#SBATCH --job-name=tb_sft_cpt_70b
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_cpt_70b_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_cpt_70b_%j.err

mkdir -p /home/skavlak/finetuning/logs

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_cpt_70b_checkpoints
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

# Find latest valid 70B CPT checkpoint
CPT_DIR=/uss/skavlak/tb_cpt_70b_checkpoints
CPT_CKPT=$(ls -td "$CPT_DIR"/checkpoint-* 2>/dev/null | while read ckpt; do
    [ -f "$ckpt/trainer_state.json" ] && echo "$ckpt" && break
done)

if [ -z "$CPT_CKPT" ]; then
    echo "ERROR: No valid 70B CPT checkpoint found in $CPT_DIR — run 70B CPT first"
    exit 1
fi

echo "Starting 70B CPT+SFT on $(hostname)"
echo "CPT checkpoint : $CPT_CKPT"
echo "Output dir     : $OUTPUT_DIR"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
fi

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29504 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /home/skavlak/finetuning/mtubercolosis/output/training_data_v2.jsonl \
    --model       "$CPT_CKPT" \
    --tokenizer   meta-llama/Llama-3.1-70B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          2e-5 \
    --batch-size  1 \
    --grad-accum  16 \
    --max-seq-len 2048 \
    --optim       adamw_bnb_8bit \
    $RESUME_ARG

echo "Training complete."
