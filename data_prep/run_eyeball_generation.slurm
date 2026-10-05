#!/bin/bash
#SBATCH --job-name=tb_eyeball_gen
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/eyeball_gen_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/eyeball_gen_%j.err

mkdir -p /home/skavlak/finetuning/logs
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
PORT=9400

echo "Small one-off generation pass against the HELD-OUT eval corpus split"
echo "(corpus_v3_eval.jsonl, same one the MCQ set was built from) to produce"
echo "genuinely held-out, passage-grounded free-form prompts across the SFT"
echo "categories for the eyeball comparison, using the current fixed seeds."

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port $PORT &
VLLM_PID=$!

until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

$PYTHON /home/skavlak/finetuning/data_prep/generate_training_data.py \
    --corpus      /uss/skavlak/version_3/corpus_v3_eval.jsonl \
    --sample      30 \
    --output      /home/skavlak/finetuning/data_prep/eyeball_prompts_raw.jsonl \
    --base-url    "http://localhost:$PORT/v1" \
    --concurrency 8
status=$?

kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

if [ $status -ne 0 ]; then
    echo "EYEBALL GENERATION FAILED (exit $status)"
    exit $status
fi
echo "eyeball_generation complete."
