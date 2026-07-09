# 原始 LocVul baseline vs 改进版：Wilcoxon 显著性检验

## 1. baseline 10-seed 均值

| Metric | mean | std | n_success |
|---|---:|---:|---:|
| A@10 | 64.86% | 1.52% | 10 |
| P@10 | 16.76% | 0.42% | 10 |
| R@10 | 49.78% | 1.18% | 10 |
| MRR@10 | 34.77% | 0.89% | 10 |
| MAP@10 | 32.61% | 1.00% | 10 |
| Median IFA | 4.6500 | 0.4743 | 10 |
| Effort@20%Recall | 93.00% | 4.83% | 10 |
| Recall@1%LOC | 20.41% | 0.35% | 10 |

## 2. Wilcoxon 结果

| Metric | Alternative | Improved mean | Baseline mean | Delta | p-value | Significant |
|---|---|---:|---:|---:|---:|---|
| A@10 | greater | 94.40% | 64.86% | 29.54% | 0.000976562 | yes |
| P@10 | greater | 33.86% | 16.76% | 17.10% | 0.000976562 | yes |
| R@10 | greater | 84.85% | 49.78% | 35.07% | 0.000976562 | yes |
| MRR@10 | greater | 83.39% | 34.77% | 48.62% | 0.000976562 | yes |
| MAP@10 | greater | 85.99% | 32.61% | 53.38% | 0.000976562 | yes |
| Median IFA | less | 0.0000 | 4.6500 | -4.6500 | 0.000976562 | yes |
| Effort@20%Recall | less | 1.28% | 93.00% | -91.72% | 0.000976562 | yes |
| Recall@1%LOC | greater | 15.43% | 20.41% | -4.98% | 1 | no |
