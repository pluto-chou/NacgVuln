# Dataset and Third-Party Data

NacgVuln uses the Big-Vul C/C++ vulnerability dataset introduced by Fan et al. (2020).

## Original Big-Vul Source

- Repository: `https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset`
- Article: Fan J, Li Y, Wang S, Nguyen TN. 2020. *A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries*. Proceedings of the 17th International Conference on Mining Software Repositories, 508-512.
- DOI: `10.1145/3379597.3387501`

## Files Used by NacgVuln

For consistency with previous line-level vulnerability localization work, NacgVuln uses the preprocessed Big-Vul `train.csv`, `val.csv`, and `test.csv` files distributed by the official LineVul replication package:

- LineVul repository: `https://github.com/awsm-research/LineVul`

The Google Drive identifiers in `data/data_mining.py` are the identifiers documented by LineVul. The script downloads the three files, validates the required columns, concatenates them in the order `train.csv`, `val.csv`, `test.csv`, and writes `data/dataset.csv`.

## Construction Command

```bash
python data/data_mining.py
```

The script is idempotent. If a valid `data/dataset.csv` already exists, it is reused and its row count and SHA-256 checksum are printed.

Force a clean rebuild:

```bash
python data/data_mining.py --force
```

Keep the downloaded components after assembly:

```bash
python data/data_mining.py --force --keep-components
```

The third-party CSV files are not committed to this repository.

## Expected Columns

The assembled file must contain at least:

- `processed_func`
- `target`
- `flaw_line`
- `flaw_line_index`

Optional metadata columns, including project and CWE identifiers, are retained when present.

## Fixed-Order Split

NacgVuln applies the following fixed-order split to the assembled file:

1. the last 10% of samples are used for testing;
2. the last 10% of the remaining samples are used for validation;
3. all earlier samples are used for training.

This produces an approximate 81%/9%/10% train-validation-test split. All random seeds use the same partition.

The data sources were accessed on 21 July 2026.
