#!/bin/bash
#SBATCH --job-name=tb_sft_qlora_v10_merged_100k
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_qlora_v10_merged_100k_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_qlora_v10_merged_100k_%j.err

mkdir -p /home/skavlak/finetuning/logs
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_v10_biomistral_merged_100k_qlora_checkpoints
MERGED=/uss/skavlak/tb_cpt_v10_biomistral_full_instruct_slerp_merged

echo "Starting SFT QLoRA (100k data, our own v5 recipe) on top of the v10" \
     "BioMistral-recipe CPT x Instruct SLERP merge on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe/data as run_sft_qlora_v8_merged_100k.slurm (our own best-performing
# SFT recipe: rank 64 QLoRA, lr 2e-5, 2 epochs, effective batch 32, 100k data) -
# only the merge source changed (v10's BioMistral-recipe CPT x Instruct merge).
# Effective batch 32 held constant via grad-accum8x4GPU -> grad-accum4x8GPU.
$TORCHRUN \
    --nproc_per_node=8 \
    --master_port=29534 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_5/training_data_v5_100k.jsonl \
    --model       "$MERGED" \
    --tokenizer   "$MERGED" \
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
    echo "SFT QLoRA on v10 merged model FAILED (exit $status)"
    exit $status
fi
echo "SFT QLoRA on v10 merged model complete."
