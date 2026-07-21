## Dataset and Third-Party Data

NacgVuln uses the Big-Vul C/C++ vulnerability dataset introduced by Fan et al. (2020). The original Big-Vul dataset and its construction scripts are publicly available from:

`https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset`

The original dataset article is:

Fan J, Li Y, Wang S, Nguyen TN. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. Proceedings of the 17th International Conference on Mining Software Repositories, 508–512. DOI: 10.1145/3379597.3387501.

For consistency with previous line-level vulnerability localization studies, this repository uses the preprocessed Big-Vul `train.csv`, `val.csv`, and `test.csv` files distributed through the official LineVul replication package:

`https://github.com/awsm-research/LineVul`

The Google Drive identifiers used in `data/data_mining.py` are the same identifiers documented by the LineVul repository. The script downloads the three files, concatenates them in the order `train.csv`, `val.csv`, and `test.csv`, and writes the combined dataset to `data/dataset.csv`.

NacgVuln subsequently applies a fixed-order split to the combined file: the last 10% of samples are used for testing; the last 10% of the remaining samples are used for validation; and the remaining samples are used for training. This results in an approximate 81%/9%/10% train-validation-test split.

The third-party Big-Vul and LineVul CSV files are not stored in this repository. They can be reconstructed using:

```bash
python data/data_mining.py
```

The data sources were accessed on 21 July 2026.
