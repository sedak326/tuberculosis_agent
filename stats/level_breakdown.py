#!/usr/bin/env python3
"""
level_breakdown.py — accuracy difference per question type (cognitive level / category).

For every comparison in the thesis tables (Tables 6-12, as defined in significance_tests.py),
report the accuracy difference and an exact McNemar p-value separately for each
`cognitive_level` of the eval set, plus the reference model's accuracy in that level
(to show how much headroom each level leaves).

  recall800   : recall / understanding / application / analysis   (n = 200 each)
  aligned960  : methodology_critique / premise_correction / result_interpretation (n = 320 each)

The p-values here are NOT corrected for multiple comparisons and each cell has few questions,
so treat them as descriptive. Used for the per-level table in the CPT Discussion paragraph.

Usage:
    python level_breakdown.py        # writes results/level_breakdown_<dataset>.csv and prints tables
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from significance_tests import DATA_DIR, DATASETS, load


def breakdown(name, cfg, data_dir):
    qs = load(data_dir / cfg["question_file"])
    level = {i: r["cognitive_level"] for i, r in qs.items()}
    levels = sorted(set(level.values()))
    res_dir = data_dir / "eval_results"
    scored = {k: load(res_dir / f"{v}_scored.jsonl") for k, v in cfg["conditions"].items()}
    pairs = [p for tab in cfg["tables"].values() for p in tab]
    rows = []
    for ref, cond in pairs:
        for lv in levels + ["ALL"]:
            ids = [i for i in scored[ref] if i in scored[cond] and (lv == "ALL" or level.get(i) == lv)]
            x = np.array([float(scored[ref][i]["is_correct"]) for i in ids])
            y = np.array([float(scored[cond][i]["is_correct"]) for i in ids])
            b, c = int(((y == 1) & (x == 0)).sum()), int(((x == 1) & (y == 0)).sum())
            p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
            rows.append(dict(dataset=name, reference=ref, condition=cond, level=lv, n=len(ids),
                             ref_acc=float(x.mean()), cond_acc=float(y.mean()),
                             diff=float((y - x).mean()), p_mcnemar=p))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in DATASETS.items():
        rows = breakdown(name, cfg, args.data_dir)
        path = args.out_dir / f"level_breakdown_{name}.csv"
        cols = ["dataset", "reference", "condition", "level", "n", "ref_acc", "cond_acc", "diff", "p_mcnemar"]
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
        print(f"\n=== {name}: accuracy difference (points) vs reference, exact McNemar p (uncorrected) ===")
        for ref, cond in dict.fromkeys((r["reference"], r["condition"]) for r in rows):
            print(f"  {cond} vs {ref}")
            for r in rows:
                if r["reference"] == ref and r["condition"] == cond:
                    print(f"    {r['level']:22s} n={r['n']:4d}  ref acc {r['ref_acc']:.3f}  diff {r['diff'] * 100:+5.1f}  p={r['p_mcnemar']:.2f}")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
