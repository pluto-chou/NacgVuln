# Code and Data Availability Statement

The source code, scripts, configuration files, documentation, and processed experimental outputs supporting this study are available in the NacgVuln replication package archived on Zenodo at `https://doi.org/10.5281/zenodo.21273222` and mirrored on GitHub at `https://github.com/pluto-chou/NacgVuln`. The repository includes scripts for third-party dataset reconstruction, public base-model download, negative-aware Seq2Seq training, line-level vulnerability localization, ablation analysis, statistical testing, clean-function false-positive analysis, long-function analysis, and figure generation.

The original Big-Vul dataset is third-party data. The original dataset is available from `https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset`, and the preprocessed train, validation, and test files used by this study are obtained from the official LineVul replication package at `https://github.com/awsm-research/LineVul`. These third-party CSV files are not duplicated in the NacgVuln archive.

Fine-tuned NacgVuln and baseline checkpoints are not distributed because of their size. The repository provides commands to download the public CodeT5-base and CodeBERT-base models and regenerate all checkpoints from scratch. Processed seed-level and aggregate result tables used in the manuscript are included for independent inspection.
