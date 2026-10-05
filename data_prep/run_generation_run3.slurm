#!/bin/bash
#SBATCH --job-name=tb_data_gen_run3
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/skavlak/finetuning/logs/generation_run3_%j.log
#SBATCH --error=/home/skavlak/finetuning/logs/generation_run3_%j.err

mkdir -p /home/skavlak/finetuning/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRATCH=/scratch/skavlak/tb_generation_run3_${SLURM_JOB_ID}
PORT=9200

echo "Run 3 SFT generation (single job, no sharding needed — 24,000-chunk target"
echo "is close to one Run 2 shard's workload of ~23,919 chunks)"
echo "Port    : $PORT"
echo "Scratch : $SCRATCH"

mkdir -p $SCRATCH
echo "Copying shuffled corpus to scratch..."
cp /uss/skavlak/version_3/corpus_v3_train_shuffled.jsonl $SCRATCH/corpus.jsonl
echo "Corpus ready."

OUTPUT_HOME=/uss/skavlak/version_3
mkdir -p $OUTPUT_HOME

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port $PORT &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

echo "Waiting for vLLM server to start..."
until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

# --sample truncates to the FIRST N chunks in file order — corpus was already
# shuffled (seed 42) before being copied here, so this is a genuine random
# subsample, not order-biased. Resuming (--resume implicit via generate_training_data.py's
# own done_ids skip logic) is safe since output is appended incrementally.
$PYTHON /home/skavlak/finetuning/data_prep/generate_training_data.py \
    --corpus      $SCRATCH/corpus.jsonl \
    --sample      24000 \
    --output      $OUTPUT_HOME/training_data.jsonl \
    --base-url    "http://localhost:$PORT/v1" \
    --concurrency 24
status=$?

echo "Generation step exited with status $status. Shutting down vLLM server."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

if [ $status -ne 0 ]; then
    echo "GENERATION FAILED (exit $status)"
    exit $status
fi
echo "generation_run3 complete."
