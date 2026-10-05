#!/bin/bash
# Generates 9 log-likelihood MCQ eval SLURM scripts (same 9 conditions as the
# randomized generation+judge re-eval), using evaluate_emcqa_logprob.py instead
# of evaluate_emcqa.py+score_explanations.py. No GPT-4o calls, no generation
# decoding loop — just one forward pass per question comparing P(A/B/C/D).
set -e
OUT=/home/skavlak/finetuning/8b
LOG=/home/skavlak/finetuning/logs
RESULTS=/uss/skavlak/tb_corpus_v3/eval_results
PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
SCRIPT=/home/skavlak/finetuning/data_prep/evaluate_emcqa_logprob.py

mkgen() {
  local name=$1 gpus=$2 model=$3 tokenizer=$4 adapter=$5

  local args="    --model      ${model} \\
"
  if [ -n "$adapter" ]; then
    args="${args}    --adapter    ${adapter} \\
"
  fi
  args="${args}    --tokenizer  ${tokenizer} \\
    --out        ${RESULTS}/${name}_logprob.jsonl \\
    --randomize-answers \\
    --seed       42 \\
    --resume"

  cat > "$OUT/run_logprob_${name}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=tb_logprob_${name}
#SBATCH --nodelist=gpu4
#SBATCH --partition=compute-gpu4
#SBATCH --ntasks=1
#SBATCH --gpus=${gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=${LOG}/logprob_${name}_%j.log
#SBATCH --error=${LOG}/logprob_${name}_%j.err

mkdir -p ${LOG}
export CUDA_HOME=/home/skavlak/miniconda3/envs/autorag
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/uss/skavlak/hf_models
export HF_TOKEN=\$(cat /home/skavlak/.cache/huggingface/token)

mkdir -p ${RESULTS}

echo "=== ${name} (log-likelihood scoring, randomized answers, seed 42) ==="
${PYTHON} ${SCRIPT} \\
${args}
status=\$?
if [ \$status -ne 0 ]; then
    echo "LOGPROB EVAL FAILED (exit \$status)"
    exit \$status
fi
echo "logprob_${name} complete."
EOF
  echo "wrote run_logprob_${name}.slurm"
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
