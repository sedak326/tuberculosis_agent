#!/bin/bash
# Run this once before submitting the generation jobs.
# Copies model weights and corpus to /scratch for faster I/O during generation.

SCRATCH=/scratch/skavlak/tb_generation

echo "Creating scratch directories..."
mkdir -p $SCRATCH/hf_hub $SCRATCH/output

echo "Copying model weights (214GB — this will take a few minutes)..."
cp -r ~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct \
    $SCRATCH/hf_hub/

echo "Copying corpus..."
cp /home/skavlak/finetuning/mtubercolosis/output/corpus.jsonl \
    $SCRATCH/corpus.jsonl

echo ""
echo "Done. Files in scratch:"
du -sh $SCRATCH/hf_hub/models--meta-llama--Llama-3.3-70B-Instruct
du -sh $SCRATCH/corpus.jsonl
echo ""
echo "Now submit the job: sbatch /home/skavlak/finetuning/run_generation.slurm"
