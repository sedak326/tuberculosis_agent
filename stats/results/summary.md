### aligned960: 90 tests (Tables), alpha = 0.05

- raw (paired t-test): 49 / 90 significant
- Holm, within table: 44 / 90 significant
- Holm, global: 37 / 90 significant
- BH/FDR, global: 47 / 90 significant
- Holm, primary metrics: 10 / 30 significant
- BH/FDR, primary metrics: 12 / 30 significant

Raw-significant results that do **not** survive Holm (global):

- SFT-1 vs Base, factual_accuracy: diff +0.048, raw p=0.00189, Holm=0.0984, BH=0.0043
- SFT-1 vs Base, naturalness: diff +0.042, raw p=0.000985, Holm=0.0522, BH=0.00233
- SFT-1 vs Base, completeness: diff -0.059, raw p=0.00218, Holm=0.109, BH=0.00474
- SFT-3 vs Base, accuracy: diff +0.036, raw p=0.00409, Holm=0.196, BH=0.00856
- SFT-3 vs Base, penalized_accuracy: diff +0.053, raw p=0.00568, Holm=0.267, BH=0.0116
- SFT-3 vs Base, completeness: diff -0.042, raw p=0.037, Holm=1, BH=0.0694
- CPT-3+SFT-3 vs Base, factual_accuracy: diff +0.043, raw p=0.0137, Holm=0.602, BH=0.0262
- CPT-3+SFT-3 vs Base, coherence: diff +0.039, raw p=0.0119, Holm=0.537, BH=0.0234
- CPT-3+SFT-3 vs Base, naturalness: diff +0.036, raw p=0.00921, Holm=0.424, BH=0.0184
- CPT-3+SFT-3 vs Base, completeness: diff -0.076, raw p=0.00191, Holm=0.0984, BH=0.0043
- CPT-1+SFT-3 vs SFT-3, naturalness: diff +0.029, raw p=0.00221, Holm=0.109, BH=0.00474
- CPT-2+SFT-3 vs SFT-3, naturalness: diff +0.021, raw p=0.0476, Holm=1, BH=0.0875

### recall800: 36 tests (Tables), alpha = 0.05

- raw (paired t-test): 12 / 36 significant
- Holm, within table: 4 / 36 significant
- Holm, global: 4 / 36 significant
- BH/FDR, global: 8 / 36 significant
- Holm, primary metrics: 2 / 12 significant
- BH/FDR, primary metrics: 2 / 12 significant

Raw-significant results that do **not** survive Holm (global):

- CPT-1 vs Base, factual_accuracy: diff +0.054, raw p=0.0247, Holm=0.667, BH=0.0889
- CPT-1 vs Base, coherence: diff +0.037, raw p=0.0142, Holm=0.398, BH=0.0569
- CPT-1 vs Base, naturalness: diff +0.034, raw p=0.00374, Holm=0.12, BH=0.0231
- CPT-2 vs Base, accuracy: diff +0.019, raw p=0.0356, Holm=0.89, BH=0.107
- CPT-2 vs Base, naturalness: diff +0.034, raw p=0.00658, Holm=0.197, BH=0.0338
- CPT-3 vs Base, naturalness: diff +0.036, raw p=0.00385, Holm=0.12, BH=0.0231
- RAG-1 vs SFT-3, coherence: diff -0.045, raw p=0.0296, Holm=0.768, BH=0.0967
- RAG-1 vs SFT-3, naturalness: diff -0.037, raw p=0.00894, Holm=0.259, BH=0.0402
