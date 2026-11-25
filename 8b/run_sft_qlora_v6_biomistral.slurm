#!/bin/bash
#SBATCH --job-name=tb_sft_qlora_v6_biomistral
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_qlora_v6_biomistral_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_qlora_v6_biomistral_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_v6_biomistral_qlora_checkpoints

echo "Starting SFT-from-Instruct QLoRA (v6, BioMistral SFT recipe, 100k data, no merge) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same base model, data, lr, schedule, dropout and target modules as
# run_sft_instruct_qlora_v5_100k.slurm. Only the SFT recipe itself is swapped
# to match BioMistral's own published QLoRA table (Table 6): rank 16 instead
# of 64, alpha = rank (scaling factor 1) instead of 2x, 3 epochs instead of 2.
# Effective batch size 64 (matching BioMistral's 8-GPU x batch4 x grad-accum2)
# reached here via grad-accum 16 on our available 4 GPUs, since raw GPU count
# wasn't the thing being tested.
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29625 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_5/training_data_v5_100k.jsonl \
    --model       meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      3 \
    --lr          2e-5 \
    --batch-size  1 \
    --grad-accum  16 \
    --max-seq-len 2048 \
    --bits        4 \
    --lora-rank   16 \
    --lora-alpha  16 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "SFT QLoRA (v6, BioMistral recipe) FAILED (exit $status)"
    exit $status
fi
echo "SFT QLoRA (v6, BioMistral recipe) complete."
