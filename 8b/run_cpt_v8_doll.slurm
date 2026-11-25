#!/bin/bash
#SBATCH --job-name=tb_cpt_v8_doll
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_v8_doll_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_v8_doll_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_v8_doll_full_checkpoints

echo "Starting full-parameter CPT (v8: same corpus + base model as v5's CPT," \
     "only the CPT recipe swapped to match Doll et al. 2026 (arXiv:2604.19394," \
     "Table 7/8, Appendix A.3) - lr 2e-5 linear w/ 500-step warmup, weight decay" \
     "0.01, seq len 8192, effective batch 1024 - on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Same corpus (489,423 deduped chunks / ~173M tokens) and base model
# (Llama-3.1-8B-Instruct) as run_cpt_v5_full.slurm - only the recipe itself
# changes, to match Doll et al.'s own published CPT hyperparameters:
#   lr 2e-5, linear schedule, 500-step warmup, AdamW weight decay 0.01,
#   seq len 8192, 2 epochs, effective batch 1024 (paper: micro-batch 1,
#   64-128 GPUs; reached here via batch-size 1 x grad-accum 256 on our
#   4 available GPUs, since raw GPU/node count wasn't the thing being tested).
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29531 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus            /uss/skavlak/version_5/corpus_v5_train.jsonl \
    --model             meta-llama/Llama-3.1-8B-Instruct \
    --output-dir        $OUTPUT_DIR \
    --epochs            2 \
    --lr                2e-5 \
    --lr-scheduler-type linear \
    --warmup-steps      500 \
    --weight-decay      0.01 \
    --batch-size        1 \
    --grad-accum        256 \
    --max-seq-len       8192 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT (v8, Doll et al. recipe) FAILED (exit $status)"
    exit $status
fi
echo "CPT (v8, Doll et al. recipe) complete."
