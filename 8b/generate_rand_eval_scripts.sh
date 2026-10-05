#!/bin/bash
# Generates the 9 re-eval SLURM scripts (all non-MIWV Run 1 + Run 2 conditions,
# re-run with --randomize-answers to fix the 98%-answer-is-A dataset bias).
set -e
OUT=/home/skavlak/finetuning/8b
LOG=/home/skavlak/finetuning/logs
RESULTS=/uss/skavlak/tb_corpus_v3/eval_results
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRIPT=/home/skavlak/finetuning/data_prep/evaluate_emcqa.py

mkgen() {
  local name=$1 gpus=$2 model=$3 tokenizer=$4 adapter=$5

  # Build the full argument block as one pre-assembled string so there's no
  # fragile blank-line-in-the-middle-of-a-backslash-continuation problem.
  local args="    --model      ${model} \\
"
  if [ -n "$adapter" ]; then
    args="${args}    --adapter    ${adapter} \\
"
  fi
  args="${args}    --tokenizer  ${tokenizer} \\
    --out        ${RESULTS}/${name}_rand.jsonl \\
    --randomize-answers \\
    --seed       42 \\
    --resume"

  cat > "$OUT/run_eval_${name}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=tb_eval_${name}
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=${gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=${LOG}/eval_${name}_%j.log
#SBATCH --error=${LOG}/eval_${name}_%j.err

mkdir -p ${LOG}
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=\$(cat /home/skavlak/.cache/huggingface/token)

mkdir -p ${RESULTS}

echo "=== ${name} (randomized answers, seed 42) ==="
${PYTHON} ${SCRIPT} \\
${args}
status=\$?
if [ \$status -ne 0 ]; then
    echo "EVAL FAILED (exit \$status)"
    exit \$status
fi
echo "eval_${name} complete."
EOF
  echo "wrote run_eval_${name}.slurm"
}

# name                    gpus  model                            tokenizer                          adapter
mkgen 8b_base_r1           1 meta-llama/Llama-3.1-8B-Instruct   meta-llama/Llama-3.1-8B-Instruct   ""
mkgen 8b_sft_r1            1 /uss/skavlak/tb_llm_v3_checkpoints/final          meta-llama/Llama-3.1-8B-Instruct ""
mkgen 8b_cpt_sft_r1        1 /uss/skavlak/tb_sft_cpt_v3_checkpoints/final     meta-llama/Llama-3.1-8B-Instruct ""
mkgen 70b_base_r1          2 meta-llama/Llama-3.1-70B            meta-llama/Llama-3.1-70B-Instruct  ""
mkgen 70b_instruct_r1      2 meta-llama/Llama-3.1-70B-Instruct   meta-llama/Llama-3.1-70B-Instruct  ""
mkgen 70b_sft_r1           1 meta-llama/Llama-3.1-70B            meta-llama/Llama-3.1-70B-Instruct  /uss/skavlak/tb_llm_70b_v3_checkpoints/final
mkgen 8b_base_r2           1 meta-llama/Llama-3.1-8B             meta-llama/Llama-3.1-8B-Instruct   ""
mkgen 8b_sft_qlora_r2      1 meta-llama/Llama-3.1-8B             meta-llama/Llama-3.1-8B-Instruct   /uss/skavlak/tb_llm_v3_run2_qlora_checkpoints/final
mkgen 8b_sft_cpt_qlora_r2  1 meta-llama/Llama-3.1-8B             meta-llama/Llama-3.1-8B-Instruct   /uss/skavlak/tb_sft_cpt_v3_run2_qlora_checkpoints/final

echo "Done generating 9 scripts."
