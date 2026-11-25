#!/usr/bin/env python3
"""
significance_tests_rag_base.py — same methodology as significance_tests.py, applied to the
base Llama-3.1-8B-Instruct (no SFT adapter) + RAG conditions (lower-bound index), compared
against the SFT-3 no-RAG baseline. Family of tests for Holm adjustment is this table alone
(3 conditions x 6 metrics = 18 tests), matching how Table 10/12 were adjusted.

Usage:
    python significance_tests_rag_base.py aligned960
    python significance_tests_rag_base.py recall800
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from significance_tests import load, compare, add_adjustments, METRICS, PRIMARY

DATA_DIR = Path("/uss/skavlak/version_3")

CONFIGS = {
    "aligned960": {
        "question_file": "aligned_mcq_eval_scaled.jsonl",
        "baseline": "aligned960_v5_sft_instruct_100k",
        "conditions": {
            "RAG-1": "aligned960_base_medimind_rag",
            "RAG-2": "aligned960_base_paper3_rag",
            "RAG-3": "aligned960_base_rag",
        },
    },
    "recall800": {
        "question_file": "mcq_eval.jsonl",
        "baseline": "oldeval_sft3_only",
        "conditions": {
            "RAG-1": "oldeval_base_medimind_rag",
            "RAG-2": "oldeval_base_paper3_rag",
            "RAG-3": "oldeval_base_rag",
        },
    },
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "aligned960"
    cfg = CONFIGS[name]
    res_dir = DATA_DIR / "eval_results"
    qs = load(DATA_DIR / cfg["question_file"])
    docs = {i: r["document_name"] for i, r in qs.items()}
    scored = {"SFT-3": load(res_dir / f"{cfg['baseline']}_scored.jsonl")}
    scored.update({k: load(res_dir / f"{v}_scored.jsonl") for k, v in cfg["conditions"].items()})
    rng = np.random.default_rng(0)
    rows = []
    for cond in cfg["conditions"]:
        for r in compare("SFT-3", cond, scored, docs, 10000, rng):
            rows.append(r)
    add_adjustments(rows, lambda r: True, "table")

    print(f"\n=== {name}: base Llama-3.1-8B-Instruct + RAG (lower-bound index) vs SFT-3 (no RAG) ===\n")
    header = f"{'Condition':8} {'Metric':20} {'n':>5} {'SFT-3':>8} {'Base+RAG':>8} {'diff':>8} {'p_ttest':>12} {'p_holm':>12} {'p_mcnemar':>12} {'ci_lo':>9} {'ci_hi':>9}"
    print(header)
    for r in rows:
        print(f"{r['condition']:8} {r['metric']:20} {r['n']:5d} {r['ref_mean']:8.4f} {r['cond_mean']:8.4f} "
              f"{r['diff']:+8.4f} {r['p_ttest']:12.4g} {r['p_holm_table']:12.4g} "
              f"{(r['p_mcnemar'] if not np.isnan(r['p_mcnemar']) else float('nan')):12.4g} "
              f"{(r['ci_lo'] if not np.isnan(r['ci_lo']) else float('nan')):9.4f} "
              f"{(r['ci_hi'] if not np.isnan(r['ci_hi']) else float('nan')):9.4f}")


if __name__ == "__main__":
    main()
