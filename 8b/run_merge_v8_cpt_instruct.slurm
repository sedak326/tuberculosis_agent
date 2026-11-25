#!/bin/bash
#SBATCH --job-name=tb_merge_v8_cpt_instruct
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/merge_v8_cpt_instruct_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/merge_v8_cpt_instruct_%j.err

mkdir -p /home/skavlak/finetuning/logs
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

MERGEKIT=/home/skavlak/miniconda3/envs/autorag/bin/mergekit-yaml
CONFIG=/home/skavlak/finetuning/merge_configs/v8_cpt_instruct_slerp.yaml
OUT=/uss/skavlak/tb_cpt_v8_doll_full_instruct_slerp_merged

echo "Starting SLERP merge (v8 Doll-recipe full-param CPT x Llama-3.1-8B-Instruct) on $(hostname)"

$MERGEKIT "$CONFIG" "$OUT" --cuda --lazy-unpickle -v
status=$?

if [ $status -ne 0 ]; then
    echo "MERGE FAILED (exit $status)"
    exit $status
fi
echo "merge_v8_cpt_instruct complete. Output at $OUT"
