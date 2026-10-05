#!/bin/bash
#SBATCH --job-name=tb_cpt_sb
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_sb_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_sb_%j.err

mkdir -p /home/skavlak/finetuning/logs

# Needed by accelerate/deepspeed's is_peft_model() check during FSDP model save
# (see 8b/run_cpt_instruct.slurm postmortem — missing CUDA_HOME crashes the save step).
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_v3_smallbatch_checkpoints

echo "Starting CPT (small batch, Marek et al. 2025 scaling) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# batch-size 4 x grad-accum 1 x 4 GPUs = 16 effective batch (no accumulation — paper argues
# it's wasteful for single-node data parallel training; micro-batch alone already gives good MFU)
# beta2 = 0.95^(16/16) = 0.95   (half-life-preserving scaling, anchored at their own beta2=0.95 @ batch=16)
# lr    = 1e-5 / (64/16)^0.1585 ~= 8e-6   (their empirical LR~batch^0.1585 power law, not sqrt)
$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=29541 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus       /uss/skavlak/tb_corpus_v3/corpus_v3_train.jsonl \
    --model        meta-llama/Llama-3.1-8B \
    --output-dir   $OUTPUT_DIR \
    --epochs       2 \
    --lr           8e-6 \
    --batch-size   4 \
    --grad-accum   1 \
    --adam-beta1   0.9 \
    --adam-beta2   0.95 \
    --lr-scheduler-type constant \
    --max-seq-len  2048 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT FAILED (exit $status)"
    exit $status
fi
echo "CPT complete."
