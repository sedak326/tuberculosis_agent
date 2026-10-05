#!/bin/bash
#SBATCH --job-name=miwv_select
#SBATCH --partition=compute-gpu4
#SBATCH --nodelist=gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/miwv_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/miwv_%j.err

cd /home/skavlak/finetuning

/home/skavlak/miniconda3/envs/autorag/bin/python data_prep/select_data_miwv.py
