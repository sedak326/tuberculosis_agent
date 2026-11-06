#!/usr/bin/env python3
"""
significance_tests.py — reproducible significance testing with multiple-comparison
correction for the thesis tables (Tables 6-12).

Reads the GPT-4o-scored per-question files (`<name>_scored.jsonl`) and, for every
comparison reported in the thesis, computes:

  * the paired two-sided t-test used in the thesis (raw p),
  * an exact McNemar test for accuracy (binary outcome; robustness check),
  * a paper-level cluster bootstrap CI for the accuracy / penalized-accuracy
    difference (questions from the same paper are not strictly independent),
  * Bonferroni, Holm (family-wise) and Benjamini-Hochberg (FDR) adjusted p-values
    under three definitions of the "family" of tests:
        within_table   - the 18 tests (3 conditions x 6 metrics) of one table
        global         - all tests of the dataset (Tables 6-10 / Tables 11-12)
        primary        - accuracy + penalized accuracy only, across the dataset

The choice of family is a judgment call; report which one you use. The model
conditions are mapped to scored-file names in CONDITIONS below. That mapping was
inferred by matching the thesis numbers, and --check verifies it reproduces known
p-values from the thesis text.

Usage:
    python significance_tests.py            # writes CSVs + summary.md to ./results
    python significance_tests.py --check    # also verify against thesis p-values
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA_DIR = Path("/uss/skavlak/version_3")
METRICS = ["accuracy", "penalized_accuracy", "factual_accuracy", "coherence",
           "naturalness", "completeness"]
PRIMARY = {"accuracy", "penalized_accuracy"}

# thesis condition -> scored-file stem (in <data-dir>/eval_results/)
DATASETS = {
    "aligned960": {
        "question_file": "aligned_mcq_eval_scaled.jsonl",
        "conditions": {
            "Base": "aligned960_base",
            "CPT-1": "aligned960_v8_doll_merged_nosft",
            "CPT-2": "aligned960_v10_biomistral_merged_nosft",
            "CPT-3": "aligned960_v5_merged_nosft",
            "SFT-1": "aligned960_v6_biomistral_qlora",
            "SFT-2": "aligned960_v7_medimind_qlora",
            "SFT-3": "aligned960_v5_sft_instruct_100k",
            "CPT-1+SFT-3": "aligned960_v8_doll_merged_sft100k",
            "CPT-2+SFT-3": "aligned960_v10_biomistral_sft100k",
            "CPT-3+SFT-3": "aligned960_v5_merged_sft100k",
            "RAG-1": "aligned960_v5_medimind_rag",
            "RAG-2": "aligned960_v5_paper3_rag",
            "RAG-3": "aligned960_v5_sft_instruct_100k_rag",
        },
        "tables": {
            "Table 6 (CPT vs base)": [("Base", c) for c in ("CPT-1", "CPT-2", "CPT-3")],
            "Table 7 (SFT vs base)": [("Base", c) for c in ("SFT-1", "SFT-2", "SFT-3")],
            "Table 8 (CPT+SFT vs base)": [("Base", c) for c in ("CPT-1+SFT-3", "CPT-2+SFT-3", "CPT-3+SFT-3")],
            "Table 9 (CPT+SFT vs SFT-3)": [("SFT-3", c) for c in ("CPT-1+SFT-3", "CPT-2+SFT-3", "CPT-3+SFT-3")],
            "Table 10 (RAG vs SFT-3)": [("SFT-3", c) for c in ("RAG-1", "RAG-2", "RAG-3")],
        },
        # extra pairwise SFT comparisons quoted in the text (reported, not in 'global')
        "extra_tables": {
            "Text: SFT pairwise": [("SFT-1", "SFT-3"), ("SFT-2", "SFT-3"), ("SFT-1", "SFT-2")],
        },
    },
    "recall800": {
        "question_file": "mcq_eval.jsonl",
        "conditions": {
            "Base": "run3_base",
            "CPT-1": "oldeval_v8_doll_merged_nosft",
            "CPT-2": "oldeval_v10_biomistral_merged_nosft",
            "CPT-3": "oldeval_v5_merged_nosft",
            "SFT-3": "oldeval_sft3_only",
            "RAG-1": "oldeval_v5_medimind_rag",
            "RAG-2": "oldeval_v5_paper3_rag",
            "RAG-3": "oldeval_v5_sft_instruct_100k_rag",
        },
        "tables": {
            "Table 11 (CPT vs base)": [("Base", c) for c in ("CPT-1", "CPT-2", "CPT-3")],
            "Table 12 (RAG vs SFT-3)": [("SFT-3", c) for c in ("RAG-1", "RAG-2", "RAG-3")],
        },
        "extra_tables": {},
    },
}

# (dataset, reference, condition, metric) -> p-value stated in the thesis text
THESIS_P = {
    ("aligned960", "Base", "CPT-2", "accuracy"): 0.21,
    ("aligned960", "Base", "CPT-1", "completeness"): 0.0003,
    ("aligned960", "Base", "SFT-3", "penalized_accuracy"): 5.68e-3,
    ("aligned960", "Base", "SFT-3", "factual_accuracy"): 1.91e-5,
    ("aligned960", "SFT-3", "CPT-3+SFT-3", "accuracy"): 6.7e-6,
    ("aligned960", "SFT-1", "SFT-3", "penalized_accuracy"): 2.16e-3,
}


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def values(rec, metric):
    if metric == "accuracy":
        return float(rec["is_correct"])
    return float(rec[metric])


# ---- multiple-comparison adjustments -------------------------------------------------
def bonferroni(p):
    return np.minimum(1.0, np.asarray(p) * len(p))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adj[idx] = prev
    return adj


def add_adjustments(rows, key, label):
    """Adjust p over the rows selected by key(row) -> bool, in place, into three columns."""
    sel = [r for r in rows if key(r)]
    if not sel:
        return
    p = np.array([r["p_ttest"] for r in sel])
    for name, fn in (("bonf", bonferroni), ("holm", holm), ("bh", benjamini_hochberg)):
        for r, v in zip(sel, fn(p)):
            r[f"p_{name}_{label}"] = float(v)


# ---- per-comparison statistics -------------------------------------------------------
def cluster_bootstrap(diff, groups, reps, rng):
    """CI for mean(diff) resampling whole papers (groups) with replacement."""
    k = groups.max() + 1
    sums = np.bincount(groups, weights=diff, minlength=k)
    ns = np.bincount(groups, minlength=k).astype(float)
    est = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, k, k)
        est[i] = sums[pick].sum() / ns[pick].sum()
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(lo), float(hi)


def compare(ref, cond, scored, docs, reps, rng):
    A, B = scored[ref], scored[cond]
    ids = sorted(set(A) & set(B) & set(docs))
    names = sorted({d for d in docs.values()})
    gidx = {d: k for k, d in enumerate(names)}
    groups = np.array([gidx[docs[i]] for i in ids])
    out = []
    for m in METRICS:
        x = np.array([values(A[i], m) for i in ids])
        y = np.array([values(B[i], m) for i in ids])
        d = y - x
        t = stats.ttest_rel(y, x)
        row = dict(reference=ref, condition=cond, metric=m, n=len(ids),
                   ref_mean=float(x.mean()), cond_mean=float(y.mean()),
                   diff=float(d.mean()), p_ttest=float(t.pvalue),
                   p_mcnemar=np.nan, ci_lo=np.nan, ci_hi=np.nan)
        if m == "accuracy":
            b, c = int(((y == 1) & (x == 0)).sum()), int(((x == 1) & (y == 0)).sum())
            row["p_mcnemar"] = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
        if m in PRIMARY:
            row["ci_lo"], row["ci_hi"] = cluster_bootstrap(d, groups, reps, rng)
        out.append(row)
    return out


def run_dataset(name, cfg, data_dir, reps):
    res_dir = data_dir / "eval_results"
    qs = load(data_dir / cfg["question_file"])
    docs = {i: r["document_name"] for i, r in qs.items()}
    scored = {k: load(res_dir / f"{v}_scored.jsonl") for k, v in cfg["conditions"].items()}
    rng = np.random.default_rng(0)
    rows = []
    for group, tabs in (("main", cfg["tables"]), ("extra", cfg["extra_tables"])):
        for tname, pairs in tabs.items():
            for ref, cond in pairs:
                for r in compare(ref, cond, scored, docs, reps, rng):
                    r.update(dataset=name, table=tname, group=group)
                    rows.append(r)
    for tname in {r["table"] for r in rows}:
        add_adjustments(rows, lambda r, t=tname: r["table"] == t, "table")
    add_adjustments(rows, lambda r: r["group"] == "main", "global")
    add_adjustments(rows, lambda r: r["group"] == "main" and r["metric"] in PRIMARY, "primary")
    return rows


# ---- output ---------------------------------------------------------------------------
COLS = ["dataset", "table", "reference", "condition", "metric", "n", "ref_mean", "cond_mean",
        "diff", "ci_lo", "ci_hi", "p_ttest", "p_mcnemar",
        "p_bonf_table", "p_holm_table", "p_bh_table",
        "p_bonf_global", "p_holm_global", "p_bh_global",
        "p_bonf_primary", "p_holm_primary", "p_bh_primary"]


def write_csv(rows, path):
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.6g}"
        return "" if v is None else str(v)
    with open(path, "w") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(",".join('"%s"' % fmt(r.get(c)) if c in ("table",) else fmt(r.get(c)) for c in COLS) + "\n")


def summarize(rows, dataset, alpha):
    main = [r for r in rows if r["group"] == "main"]
    lines = [f"### {dataset}: {len(main)} tests (Tables), alpha = {alpha}", ""]
    for label, col, sub in (
        ("raw (paired t-test)", "p_ttest", main),
        ("Holm, within table", "p_holm_table", main),
        ("Holm, global", "p_holm_global", main),
        ("BH/FDR, global", "p_bh_global", main),
        ("Holm, primary metrics", "p_holm_primary", [r for r in main if r["metric"] in PRIMARY]),
        ("BH/FDR, primary metrics", "p_bh_primary", [r for r in main if r["metric"] in PRIMARY]),
    ):
        k = sum(1 for r in sub if r[col] < alpha)
        lines.append(f"- {label}: {k} / {len(sub)} significant")
    lines += ["", "Raw-significant results that do **not** survive Holm (global):", ""]
    lost = [r for r in main if r["p_ttest"] < alpha and r["p_holm_global"] >= alpha]
    if not lost:
        lines.append("- none")
    for r in lost:
        lines.append(f"- {r['condition']} vs {r['reference']}, {r['metric']}: diff {r['diff']:+.3f}, "
                     f"raw p={r['p_ttest']:.3g}, Holm={r['p_holm_global']:.3g}, BH={r['p_bh_global']:.3g}")
    lines.append("")
    return "\n".join(lines)


def check(all_rows):
    ok = True
    for (ds, ref, cond, metric), expected in THESIS_P.items():
        got = [r["p_ttest"] for r in all_rows
               if r["dataset"] == ds and r["reference"] == ref and r["condition"] == cond and r["metric"] == metric]
        if not got:
            print(f"CHECK MISSING {ds} {cond} vs {ref} {metric}")
            ok = False
            continue
        rel = abs(got[0] - expected) / expected
        flag = "ok " if rel < 0.15 else "MISMATCH"
        ok &= rel < 0.15
        print(f"CHECK {flag} {ds} {cond} vs {ref} {metric}: thesis p={expected:g}, reproduced p={got[0]:.3g}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05, help="alpha used in the printed summary")
    ap.add_argument("--check", action="store_true", help="verify reproduction of p-values quoted in the thesis")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []
    for name, cfg in DATASETS.items():
        rows = run_dataset(name, cfg, args.data_dir, args.bootstrap_reps)
        write_csv(rows, args.out_dir / f"significance_{name}.csv")
        summary.append(summarize(rows, name, args.alpha))
        all_rows += rows
    text = "\n".join(summary)
    (args.out_dir / "summary.md").write_text(text)
    print(text)
    if args.check:
        print("\n" + ("All checks passed." if check(all_rows) else "Some checks FAILED - verify the condition->file mapping."))


if __name__ == "__main__":
    main()
