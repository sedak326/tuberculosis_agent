#!/bin/bash
#SBATCH --job-name=tb_sft_cpt_qlora_run3
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_cpt_qlora_run3_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_cpt_qlora_run3_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_sft_cpt_run3_qlora_checkpoints
# CPT was QLoRA this run (not full-parameter like Run 2), so its checkpoint was
# a LoRA adapter, not a full model — merged into a real base first via
# merge_cpt_adapter.py, since train_sft.py has no --adapter flag to stack a
# second LoRA on top of a first one directly.
CPT_MERGED=/uss/skavlak/tb_cpt_run3_qlora_merged

echo "Starting SFT-from-CPT QLoRA (Run 3) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same recipe as the SFT-only condition — only the starting model differs
# (merged CPT weights vs raw base), so this is a clean CPT-vs-no-CPT ablation.
$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29613 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/version_3/training_data.jsonl \
    --model       "$CPT_MERGED" \
    --tokenizer   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir  $OUTPUT_DIR \
    --epochs      2 \
    --lr          2e-5 \
    --batch-size  1 \
    --grad-accum  16 \
    --max-seq-len 2048 \
    --bits        4 \
    --lora-rank   64 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "SFT-from-CPT QLoRA FAILED (exit $status)"
    exit $status
fi
echo "SFT-from-CPT QLoRA complete."
