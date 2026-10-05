#!/bin/bash
#SBATCH --job-name=tb_sft_cpt_qlora_run2
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/sft_cpt_qlora_run2_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/sft_cpt_qlora_run2_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun

CPT_DIR=/uss/skavlak/tb_cpt_v3_run2_checkpoints
CPT_FINAL=$(ls -td "$CPT_DIR"/checkpoint-* 2>/dev/null | while read ckpt; do [ -f "$ckpt/trainer_state.json" ] && echo "$ckpt" && break; done)
OUTPUT_DIR=/uss/skavlak/tb_sft_cpt_v3_run2_qlora_checkpoints

echo "Starting SFT-from-CPT QLoRA (Run 2) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPT checkpoint: $CPT_FINAL"

if [ -z "$CPT_FINAL" ]; then
    echo "ERROR: no CPT checkpoint found in $CPT_DIR — run CPT first."
    exit 1
fi

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same QLoRA recipe as the SFT-only run (rank 64, alpha 128, lr 2e-5, 2 epochs) —
# only difference is --model points at the finished CPT checkpoint instead of the
# raw base, and --tokenizer is still the Instruct tokenizer since CPT checkpoints
# don't carry a chat template.
$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29611 \
    /home/skavlak/finetuning/train_sft.py \
    --data        /uss/skavlak/tb_corpus_v3/training_data_v3_run2.jsonl \
    --model       "$CPT_FINAL" \
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
