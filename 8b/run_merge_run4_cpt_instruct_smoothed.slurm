#!/bin/bash
#SBATCH --job-name=tb_merge_run4_smoothed
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/merge_run4_smoothed_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/merge_run4_smoothed_%j.err

mkdir -p /home/skavlak/finetuning/logs
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

MERGEKIT=/home/skavlak/miniconda3/envs/autorag/bin/mergekit-yaml
CONFIG=/home/skavlak/finetuning/merge_configs/run4_cpt_instruct_slerp_smoothed.yaml
OUT=/uss/skavlak/tb_cpt_run4_full_instruct_slerp_smoothed_merged

echo "Starting SMOOTHED SLERP merge (Run 4 full-param CPT x Instruct) on $(hostname)"

$MERGEKIT "$CONFIG" "$OUT" --cuda --lazy-unpickle -v
status=$?

if [ $status -ne 0 ]; then
    echo "MERGE FAILED (exit $status)"
    exit $status
fi
echo "merge_run4_smoothed complete. Output at $OUT"
