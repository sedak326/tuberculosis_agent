#!/bin/bash
#SBATCH --job-name=tb_sft_instruct_200k_run3
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_instruct_200k_run3_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_instruct_200k_run3_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_instruct_200k_run3_qlora_checkpoints

echo "Starting SFT-from-Instruct QLoRA on FULL 200K dataset (Run 3) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe as the 40K SFT-from-Instruct condition (rank 64, alpha 128,
# lr 2e-5, 2 epochs), but on the full archived 200,622-example set instead of
# the balanced 40K subsample — tests whether broader corpus coverage helps
# generalization to held-out eval content, or erodes the alignment parity
# the 40K run showed. All 8 GPUs free right now; grad-accum dropped to 4 to
# hold effective batch at 32 (1 x 4 x 8), same as the 2-GPU/16-accum runs.
$TORCHRUN \
    --nproc_per_node=8 \
    --master_port=29615 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_3/training_data_full_200k.jsonl \
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
    echo "SFT-from-Instruct-200K QLoRA FAILED (exit $status)"
    exit $status
fi
echo "SFT-from-Instruct-200K QLoRA complete."
