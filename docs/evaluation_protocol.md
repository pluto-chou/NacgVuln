# Evaluation Protocol

## Evaluation Scope

The main line-level localization metrics are computed on vulnerable functions only:

```text
EVAL_ONLY_VULN=yes
REMOVE_MISSING_LINE_LABELS=yes
```

This means `target==1` functions with valid line-level labels are used for A@K, P@K, R@K, MRR@K, MAP@K, IFA, Effort@20%Recall, and Recall@1%LOC. Clean functions are evaluated separately through Clean FPR.

## NacgVuln Prediction Handling

NacgVuln introduces a clean-function target token:

```text
<NO_VULN>
```

During evaluation:

1. `<NO_VULN>` and empty outputs are treated as predicting no vulnerable line;
2. no similarity replacement is applied to `<NO_VULN>`;
3. generated lines are mapped back to concrete function lines when similarity replacement is enabled;
4. duplicated candidate lines are removed when `DEDUP=yes`;
5. the final LocVul-style ranked list is constructed by placing predicted lines first and appending the remaining function lines in source order.

## Main Metrics

| Metric | Meaning | Direction |
|---|---|---|
| `A@10` | whether at least one vulnerable line appears in Top-10 | higher is better |
| `P@10` | fraction of Top-10 lines that are vulnerable | higher is better |
| `R@10` | fraction of ground-truth vulnerable lines recovered in Top-10 | higher is better |
| `MRR@10` | reciprocal rank of the first correct vulnerable line | higher is better |
| `MAP@10` | mean average precision in Top-10 | higher is better |
| `Median IFA` | median initial false alarms before first hit | lower is better |
| `Effort@20%Recall` | inspection effort to reach 20% recall | lower is better |
| `Recall@1%LOC` | recall achieved after inspecting 1% LOC | higher is better |

## Clean FPR Protocol

Clean FPR is computed on clean functions only (`target==0`). A clean function is counted as a false positive if the model generates any non-empty vulnerable-line prediction. `<NO_VULN>` or empty output is counted as correct rejection.

Archived files:

```text
results/clean_fpr/clean_fpr_per_seed.csv
results/clean_fpr/clean_fpr_mean_std.csv
```

## Statistical Testing

Paired Wilcoxon signed-rank tests use seed-level paired results. For metrics where higher is better, the alternative hypothesis is `improved > baseline`. For `Median IFA` and `Effort@20%Recall`, the alternative hypothesis is `improved < baseline`.

Archived files:

```text
results/selfattention_baseline/wilcoxon_results.md
results/ablation/ablation_wilcoxon_tests.csv
```
