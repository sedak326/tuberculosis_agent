#!/bin/bash
#SBATCH --job-name=tb_cpt_v10_biomistral
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/cpt_v10_biomistral_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/cpt_v10_biomistral_%j.err

mkdir -p /home/skavlak/finetuning/logs

export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

TORCHRUN=/home/skavlak/miniconda3/envs/autorag/bin/torchrun
OUTPUT_DIR=/uss/skavlak/tb_cpt_v10_biomistral_full_checkpoints

echo "Starting full-parameter CPT (v10: same TB corpus/base model as v5/v8's" \
     "CPT, Llama-3.1-8B-Instruct - recipe swapped to BioMistral's own" \
     "further-pretraining config, Labrak et al. 2024 (arXiv:2402.10373v3)" \
     "Section 3.2: lr 2e-5 cosine with NO warmup, weight decay 0.01, seq len" \
     "2048, effective batch 1024, full-parameter FSDP - on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

RESUME_ARG=""
if compgen -G "$OUTPUT_DIR/checkpoint-*/trainer_state.json" > /dev/null 2>&1; then
    RESUME_ARG="--resume-from latest"
    echo "Resuming from latest checkpoint in $OUTPUT_DIR"
else
    echo "No valid checkpoint found, starting from scratch"
fi

# Epochs kept at 2 (not BioMistral's own ~1.5) for comparability with every
# other CPT run in this project - only the optimization recipe itself (lr,
# schedule, warmup, weight decay, batch, seq len) is matched to the paper.
# Effective batch 1024 reached via batch1 x grad-accum128 x 8 GPUs (restarted
# from scratch on the full node instead of 4 GPUs/grad-accum256, since no
# checkpoint existed yet at the time of the switch - same effective batch,
# same reconciliation pattern already used for v7's SFT GPU-count change).
$TORCHRUN \
    --nproc_per_node=8 \
    --master_port=29533 \
    /home/skavlak/finetuning/train_cpt.py \
    --corpus            /uss/skavlak/version_5/corpus_v5_train.jsonl \
    --model             meta-llama/Llama-3.1-8B-Instruct \
    --output-dir        $OUTPUT_DIR \
    --epochs            2 \
    --lr                2e-5 \
    --lr-scheduler-type cosine \
    --warmup-steps      0 \
    --weight-decay      0.01 \
    --batch-size        1 \
    --grad-accum        128 \
    --max-seq-len       2048 \
    $RESUME_ARG
status=$?

if [ $status -ne 0 ]; then
    echo "CPT (v10, BioMistral recipe) FAILED (exit $status)"
    exit $status
fi
echo "CPT (v10, BioMistral recipe) complete."
