# stats/

Reproducible significance testing for the thesis tables (6-12), with multiple-comparison
correction. The thesis reports paired two-sided t-tests only; this adds the corrections and
robustness checks that were missing from the repo.

```bash
/home/skavlak/miniconda3/envs/autorag/bin/python stats/significance_tests.py --check
```

Outputs (in `stats/results/`): `significance_aligned960.csv`, `significance_recall800.csv`,
`summary.md`. Each CSV row is one comparison x metric with:

- `p_ttest` - the paired t-test as in the thesis (reproduced exactly; `--check` verifies this)
- `p_mcnemar` - exact McNemar test for accuracy (binary outcome)
- `ci_lo`, `ci_hi` - 95% paper-level cluster-bootstrap CI for the accuracy / penalized-accuracy difference
- `p_bonf_*`, `p_holm_*`, `p_bh_*` - Bonferroni / Holm (family-wise) / Benjamini-Hochberg (FDR) adjusted p,
  for three definitions of the family: `table` (the 18 tests of one table), `global` (all tests of
  the eval set), `primary` (accuracy + penalized accuracy only)

Notes:

- The choice of family is a judgment call. State which one the thesis uses. Pre-specifying
  accuracy / penalized accuracy as primary and the four GPT-4o quality scores as secondary is the
  cleanest option.
- The condition -> scored-file mapping (`DATASETS` in the script) was inferred by matching the
  thesis numbers. `--check` reproduces six p-values quoted in the thesis text; confirm the mapping
  anyway.
- SFT-3 was chosen as the best of three recipes on the same test set. That selection is not covered
  by these corrections; multiplying SFT-3's p-values by 3 is a minimal adjustment for it.
- The Holm/BH adjustments use the t-test p-values. Paper-level clustering barely changes the
  standard errors here (inflation about 0.94-1.13x).
