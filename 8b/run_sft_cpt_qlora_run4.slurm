#!/bin/bash
#SBATCH --job-name=tb_sft_cpt_qlora_run4
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_cpt_qlora_run4_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_cpt_qlora_run4_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_cpt_run4_qlora_checkpoints
# CPT itself is reused from Run 3 untouched - it's pure corpus pretraining with
# zero dependency on seeds.json, so re-running it would just reproduce the same
# checkpoint statistically. Only --data changed vs run_sft_cpt_qlora_run3.slurm:
# training_data_run4.jsonl (fixed seeds) instead of training_data.jsonl (pre-fix).
CPT_MERGED=/uss/skavlak/tb_cpt_run3_qlora_merged

echo "Starting SFT-from-CPT QLoRA (Run 4, fixed seeds) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe as Run 3's SFT-from-CPT condition (rank 64 QLoRA, lr 2e-5, 2
# epochs) - 4 GPUs (grad-accum 16->8) to hold effective batch at 32. Picked 4
# over 8 deliberately: 4 GPUs were free immediately, 8 would have meant
# queuing for the full node with no known wait time, for a correctness-neutral
# speed difference (pure data-parallel QLoRA, no GPU-count requirement).
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29616 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_3/training_data_run4.jsonl \
    --model       "$CPT_MERGED" \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
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
    echo "SFT-from-CPT QLoRA (Run 4) FAILED (exit $status)"
    exit $status
fi
echo "SFT-from-CPT QLoRA (Run 4) complete."
