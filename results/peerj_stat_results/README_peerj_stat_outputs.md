# PeerJ Results statistical report summary

Use the generated Markdown files to revise the Results section.

Important reporting rule:
- Wilcoxon signed-rank test has no conventional degrees of freedom; report `df = NA`.
- Prefer exact p-values from the CSV.
- Report both the unadjusted exact p-value and the Holm-adjusted p-value when multiple tests are discussed together.
- Use rank-biserial r as the main effect size for Wilcoxon tests.

Generated files:
- peerj_overall_baseline_tests.csv
- peerj_ablation_tests.csv
- peerj_clean_fpr_tests.csv
- peerj_length_group_tests.csv

Suggested Results wording pattern:

NacgVuln significantly improved A@10 compared with the Self-Attention baseline (mean ± SD: xx ± xx vs xx ± xx; paired mean difference = xx, 95% CI [xx, xx]; Wilcoxon signed-rank test, W = xx, n = 10, df = NA, exact p = xx, Holm-adjusted p = xx, rank-biserial r = xx).
