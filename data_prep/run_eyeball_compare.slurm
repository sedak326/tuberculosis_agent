#!/bin/bash
#SBATCH --job-name=tb_eyeball_compare
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eyeball_compare_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eyeball_compare_%j.err

mkdir -p /home/skavlak/finetuning/logs
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=$(cat /home/skavlak/.cache/huggingface/token)

/home/skavlak/miniconda3/envs/autorag/bin/python /home/skavlak/finetuning/data_prep/run_eyeball_compare.py
status=$?
if [ $status -ne 0 ]; then
    echo "EYEBALL COMPARE FAILED (exit $status)"
    exit $status
fi
echo "eyeball_compare complete."
