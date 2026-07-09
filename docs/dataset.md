# Dataset

## Dataset Source

NacgVuln uses the Big-Vul dataset under a LocVul-style line-level vulnerability localization protocol. The raw dataset is not redistributed in this GitHub repository.

## Dataset Construction

Run:

```bash
python data/data_mining.py
```

The script downloads the component files, concatenates them, and writes:

```text
data/dataset.csv
```

## Expected Columns

The training and evaluation scripts expect at least these columns:

| Column | Description |
|---|---|
| `processed_func` | preprocessed C/C++ function body |
| `target` | function-level binary label; `1` vulnerable, `0` clean |
| `flaw_line` | vulnerable source line text or line-text list |
| `flaw_line_index` | vulnerable line index or index list |
| `project` | project identifier when available |
| `CWE ID` | CWE identifier when available |

## Split Protocol

The repository uses a fixed-order split:

1. last 10% of `dataset.csv`: test set;
2. remaining 90%: split again, with its last 10% as validation;
3. remaining samples: training set.

All 10 random seeds share the same test set. Seeds only affect training initialization, shuffling, and stochastic optimization.

## Missing Line Labels

For line-level localization metrics, samples with missing `flaw_line` or `flaw_line_index` are removed when `REMOVE_MISSING_LINE_LABELS=yes`.

## Data Redistribution Policy

The raw Big-Vul dataset is not committed to this repository. The generated `data/dataset.csv` is also ignored by `.gitignore` because it can be regenerated and may be large. Public releases should include the data construction script and processed result summaries, not the raw dataset, unless redistribution rights are confirmed.

## Result Files Related to the Dataset

```text
results/main/Nacg_per_seed_metrics.csv
results/main/Nacg_mean_std_metrics.csv
results/length_group/length_group_per_seed_metrics.csv
results/cwe_analysis/*.csv
```
