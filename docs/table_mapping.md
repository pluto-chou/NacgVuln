# Table and Figure Mapping

This document maps manuscript tables and figures to files in the repository.

## Manuscript Tables

| Table | Content | Input files | Script / source |
|---|---|---|---|
| Table 1 | LocVul paper result vs NacgVuln | `results/main/compare_with_paper.csv`, `results/main/compare_with_paper.md` | `scripts/run_seq2seq_eval.py` or `scripts/run_10seed_compare_with_paper.py` |
| Table 2 | NacgVuln 10-seed mean ± std | `results/main/Nacg_per_seed_metrics.csv`, `results/main/Nacg_mean_std_metrics.csv` | `scripts/run_seq2seq_eval.py` |
| Table 3 | Self-Attention baseline significance test | `results/selfattention_baseline/baseline_per_seed_metrics.csv`, `results/main/Nacg_per_seed_metrics.csv`, `results/selfattention_baseline/wilcoxon_results.md` | `scripts/run_baseline_and_wilcoxon.py` |
| Table 4 | Ablation results | `results/ablation/ablation_all_mean_std_metrics.csv`, `results/ablation/ablation_all_per_seed_metrics.csv` | `ablation/run_seq2seq_ablation_suite.py` and `scripts/collect_ablation_perseed_and_wilcoxon.py` |
| Table 5 | Long-function analysis | `results/length_group/length_group_mean_std_metrics.csv`, `results/length_group/length_group_per_seed_metrics.csv` | `scripts/length_group_analysis.py` |
| Table 6 | Ablation Wilcoxon tests | `results/ablation/ablation_wilcoxon_tests.csv` | `ablation/ablation_wilcoxon.py` or `scripts/collect_ablation_perseed_and_wilcoxon.py` |
| Table 7 | Case studies | Not included as a fixed table in the current archive | select from evaluation `*_details.csv` if released separately |

## Additional Result Groups

| Analysis | Files |
|---|---|
| Clean-function false positive rate | `results/clean_fpr/clean_fpr_per_seed.csv`, `results/clean_fpr/clean_fpr_mean_std.csv` |
| CWE-level analysis | `results/cwe_analysis/*.csv`, `results/cwe_analysis/*.md` |
| Cppcheck comparison | `results/cppcheck/cppcheck_eval_summary.csv`, `results/cppcheck/cppcheck_eval_details.csv` |
| LineVul local logs | `results/linevul/*.log` |

## Figure Scripts

| Script | Purpose |
|---|---|
| `figures/make_chapter5_figures_with_compare.py` | main comparison figures |
| `figures/make_cwe_level_table.py` | CWE-level table generation |
| `figures/make_cwe_paper_tables.py` | paper-ready CWE tables |
| `figures/make_fig1_from_candidate.py` | candidate-based illustrative figure |
| `figures/make_fig2.py` | additional method or result figure |

Generated figure files are not committed by default; regenerate them from the scripts and archived CSV summaries.

## Main Result Values

The current archived main result file reports:

```text
A@10              0.9440 ± 0.0126
P@10              0.3386 ± 0.0084
R@10              0.8485 ± 0.0149
MRR@10            0.8339 ± 0.0192
MAP@10            0.8599 ± 0.0174
Median IFA        0.0000 ± 0.0000
Effort@20%Recall  0.0128 ± 0.0006
Recall@1%LOC      0.1543 ± 0.0091
```
