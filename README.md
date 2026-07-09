# NacgVuln

**Paper:** *Negative-aware and Chunk-enhanced Generative Line-level Vulnerability Localization*  
**Repository:** https://github.com/pluto-chou/NacgVuln  
**Author:** Shuai Huang (`pluto-chou`)  
**Affiliation:** School of Computer Science and Engineering, Sichuan University of Science & Engineering, Yibin 644000, China  
**License:** MIT  
**Zenodo DOI:** not assigned yet; archive the first public GitHub release on Zenodo and then update this field

NacgVuln is a negative-aware and chunk-enhanced generative framework for line-level vulnerability localization in C/C++ source code. It extends a LocVul-style Sequence-to-Sequence localization pipeline by explicitly modeling clean functions with a `<NO_VULN>` target, applying chunk-based training and inference to long functions, and mapping generated snippets back to concrete source-code lines through similarity replacement and ranking.

## Main Features

- **Negative-aware Seq2Seq training:** non-vulnerable functions are included during training and mapped to `<NO_VULN>`.
- **Chunk-enhanced long-function handling:** long functions are split during training and evaluated with sliding-window chunk inference.
- **Generation-to-source alignment:** generated lines are aligned back to original source-code lines through similarity replacement.
- **Audit-oriented ranked list:** generated candidates are ranked first, and remaining source lines are appended in original order.
- **10-seed evaluation:** reported results are aggregated as mean ± standard deviation.
- **Ablation suite:** negative-aware training, chunk training, chunk inference, and similarity replacement are evaluated separately.

## Repository Structure

```text
NacgVuln/
├── README.md
├── LICENSE
├── .gitignore
├── CITATION.cff
├── .zenodo.json
├── environment.yml
├── requirements.txt
├── data/
│   └── data_mining.py
├── src/
│   ├── train_func_classifier.py
│   ├── train_seq2seq_nacgvuln.py
│   └── eval_seq2seq_nacgvuln.py
├── scripts/
│   ├── run_10seed_compare_with_paper.py
│   ├── run_seq2seq_eval.py
│   ├── run_baseline_and_wilcoxon.py
│   ├── collect_ablation_perseed_and_wilcoxon.py
│   ├── length_group_analysis.py
│   └── download_hf_models.py
├── ablation/
│   ├── run_seq2seq_ablation_suite.py
│   ├── ablation_wilcoxon.py
│   ├── run_ablation_examples.sh
│   └── ablation_README.md
├── baselines/
│   ├── vulnDet_pipeline.py
│   ├── Seq2Seq_vulnDet.py
│   ├── seq2seq_eval.py
│   └── visualize.py
├── figures/
│   └── *.py
├── results/
│   ├── main/
│   ├── baselines/
│   ├── selfattention_baseline/
│   ├── ablation/
│   ├── clean_fpr/
│   ├── length_group/
│   ├── cwe_analysis/
│   ├── cppcheck/
│   └── linevul/
├── environment_records/
│   ├── conda_env.yml
│   ├── pip_freeze.txt
│   ├── python_version.txt
│   └── hardware_nvidia_smi.txt
└── docs/
    ├── reproduction.md
    ├── commands.md
    ├── dataset.md
    ├── evaluation_protocol.md
    ├── model_parameters.md
    ├── table_mapping.md
    ├── code_availability_statement.md
    └── zenodo_release.md
```

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate vulnloc
```

Alternatively, install Python dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The raw environment records used during the archived experiments are stored under `environment_records/`.

## Dataset

This project uses the Big-Vul dataset in a LocVul-style fixed-order split protocol. The raw dataset is not redistributed in this repository. To construct `data/dataset.csv`, run:

```bash
python data/data_mining.py
```

The generated `dataset.csv` is expected to include at least:

- `processed_func`
- `target`
- `flaw_line`
- `flaw_line_index`
- `project` if available
- `CWE ID` if available

See [`docs/dataset.md`](docs/dataset.md) for details.

## Quick Reproduction

### 1. Prepare data

```bash
python data/data_mining.py
```

### 2. Evaluate NacgVuln with existing checkpoints

If pretrained checkpoints are available locally or restored from the archived release assets, run:

```bash
python scripts/run_seq2seq_eval.py   --project_root ./   --output_root ./runs_seq2seq_eval_v3_batch_resilient   --dataset_path ./data/dataset.csv   --eval_script ./src/eval_seq2seq_nacgvuln.py   --existing_ckpt_root ./checkpoints_seq2seq_10seed/checkpoints   --model_variation_seq2seq ./hf_cache/codet5-base   --seeds 0,1,2,3,4,5,6,7,8,9   --resume_if_exists yes   --max_input_len 512   --max_target_len 256   --num_beams 1   --num_return_seqs 1   --fallback_sampling no   --sample_return_seqs 1   --chunk_long_funcs yes   --chunk_stride_lines_eval 20   --similarity_replacement yes   --dedup yes   --rerank no   --remove_missing_line_labels yes   --eval_only_vuln yes   --sort_by_lines yes   --continue_on_error yes   --max_retries_per_seed 2   --retry_disable_similarity yes   --retry_disable_rerank yes   --strict_failure no
```

### 3. Train and evaluate all 10 seeds

```bash
python scripts/run_10seed_compare_with_paper.py   --project_root ./   --output_root ./runs_10seed_compare_with_paper   --dataset_path ./data/dataset.csv   --train_script ./src/train_seq2seq_nacgvuln.py   --eval_script ./src/eval_seq2seq_nacgvuln.py   --model_variation Salesforce/codet5-base   --eval_model_variation_seq2seq Salesforce/codet5-base   --seeds 0,1,2,3,4,5,6,7,8,9   --resume_if_exists yes   --include_negatives yes   --negative_ratio 0.5   --chunk_training yes   --chunk_stride_lines_train 40   --chunk_long_funcs yes   --chunk_stride_lines_eval 20   --similarity_replacement yes   --rerank no
```

## Archived Results

The repository includes processed result summaries used for the manuscript:

```text
results/main/Nacg_mean_std_metrics.csv
results/main/Nacg_per_seed_metrics.csv
results/main/compare_with_paper.csv
results/main/compare_with_paper.md
results/ablation/ablation_all_mean_std_metrics.csv
results/ablation/ablation_all_per_seed_metrics.csv
results/ablation/ablation_wilcoxon_tests.csv
results/clean_fpr/clean_fpr_mean_std.csv
results/clean_fpr/clean_fpr_per_seed.csv
results/length_group/length_group_mean_std_metrics.csv
results/length_group/length_group_per_seed_metrics.csv
```

The current archived main results report the following 10-seed mean performance:

| Metric | Mean | Std |
|---|---:|---:|
| A@10 | 0.9440 | 0.0126 |
| P@10 | 0.3386 | 0.0084 |
| R@10 | 0.8485 | 0.0149 |
| MRR@10 | 0.8339 | 0.0192 |
| MAP@10 | 0.8599 | 0.0174 |
| Median IFA | 0.0000 | 0.0000 |
| Effort@20%Recall | 0.0128 | 0.0006 |
| Recall@1%LOC | 0.1543 | 0.0091 |

## Documentation

- [`docs/reproduction.md`](docs/reproduction.md): full replication workflow.
- [`docs/commands.md`](docs/commands.md): repository-relative command list.
- [`docs/dataset.md`](docs/dataset.md): dataset construction and split protocol.
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md): evaluation subsets and metrics.
- [`docs/model_parameters.md`](docs/model_parameters.md): model, training, and inference parameters.
- [`docs/table_mapping.md`](docs/table_mapping.md): mapping between manuscript tables and repository files.
- [`docs/code_availability_statement.md`](docs/code_availability_statement.md): PeerJ-ready code and data availability wording.
- [`docs/zenodo_release.md`](docs/zenodo_release.md): DOI archiving instructions.

## Code and Data Availability

The source code, scripts, configuration files, and processed experimental outputs supporting this study are provided in this repository. The raw Big-Vul dataset is not redistributed; instructions for obtaining and preprocessing it are provided in the documentation. For manuscript submission, a Zenodo DOI is recommended after creating the first public GitHub release.

## Citation

Please cite the software using [`CITATION.cff`](CITATION.cff). After Zenodo archives the first public release, add the generated DOI to `README.md`, `CITATION.cff`, `.zenodo.json`, and `docs/code_availability_statement.md`.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
