# NacgVuln

**Paper:** *Negative-aware and Chunk-enhanced Generative Line-level Vulnerability Localization*  
**Repository:** https://github.com/pluto-chou/NacgVuln  
**Software maintainer:** Shuai Huang (`pluto-chou`)  
**Affiliation:** School of Computer Science and Engineering, Sichuan University of Science & Engineering, Yibin 644000, China  
**License:** MIT  
**Zenodo DOI:** [10.5281/zenodo.21466071](https://doi.org/10.5281/zenodo.21466071)

NacgVuln is a negative-aware and chunk-enhanced generative framework for line-level vulnerability localization in C/C++ source code. It extends a LocVul-style Sequence-to-Sequence pipeline by modeling clean functions with a `<NO_VULN>` target, applying chunk-based training and inference to long functions, and aligning generated snippets with concrete source-code lines.

## Main Features

- **Negative-aware Seq2Seq training:** non-vulnerable functions are included during training and mapped to `<NO_VULN>`.
- **Chunk-enhanced long-function handling:** overlapping chunks are used during training and sliding windows are used during inference.
- **Generation-to-source alignment:** generated statements are mapped back to original source-code lines through similarity replacement.
- **Audit-oriented ranked output:** generated candidates are ranked first and unmatched source lines are appended in source order.
- **10-seed evaluation and ablation analysis:** processed seed-level and aggregate results are included.

## Repository Contents

```text
NacgVuln/
├── data/                  # Third-party data download and assembly
├── src/                   # NacgVuln training and evaluation code
├── scripts/               # Reproduction, analysis, and model-download scripts
├── baselines/             # LocVul-style and Self-Attention baselines
├── ablation/              # Ablation experiment runners
├── figures/               # Figure and table generation scripts
├── results/               # Processed manuscript result tables
├── environment_records/   # Full records from the original experiment machine
└── docs/                  # Dataset, protocol, command, and availability documentation
```


## Implementation File Mapping

The public repository renames the original experiment files as follows:

| Original experiment file | Public repository file |
|---|---|
| `vulnDet_pipeline.py` | `src/train_func_classifier.py` |
| `Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py` | `src/train_seq2seq_nacgvuln.py` |
| `seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py` | `src/eval_seq2seq_nacgvuln.py` |
| `run_10seed_func_classifier_train.py` | `scripts/run_10seed_func_classifier_train.py` |
| `run_10seed_seq2seq_train.py` | `scripts/run_10seed_seq2seq_train.py` |

See [`docs/script_mapping.md`](docs/script_mapping.md) for details. The function-level CodeBERT component and the line-level CodeT5 component are trained separately. The recommended PeerJ reproduction command uses `scripts/run_10seed_compare_with_paper.py` to train and evaluate the line-level component in one run.

## Requirements

- Python 3.9
- Linux is recommended
- An NVIDIA GPU is recommended for training, but model download and result inspection do not require a GPU
- Sufficient disk space for CodeT5-base, CodeBERT-base, the reconstructed dataset, and newly generated checkpoints

The concise direct dependencies are listed in `requirements.txt`. The complete environment snapshot from the original experiment machine remains under `environment_records/`.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate vulnloc
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch wheels depend on the local CPU/CUDA environment. If the default installation is unsuitable for the machine, install a compatible PyTorch build first and then install the remaining requirements.

## Download the Public Base Models

NacgVuln does not redistribute Hugging Face model files. Download the two public base models into repository-local directories:

```bash
python scripts/download_hf_models.py --target-dir ./hf_cache
```

Expected directories:

```text
hf_cache/codet5-base
hf_cache/codebert-base
```

For an alternative Hugging Face endpoint:

```bash
python scripts/download_hf_models.py \
  --target-dir ./hf_cache \
  --hf-endpoint https://hf-mirror.com
```

All reproduction commands below use these local paths.

## Dataset and Third-Party Data

The original Big-Vul dataset was introduced by Fan et al. (2020):

- Original Big-Vul repository: `https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset`
- Original article DOI: `10.1145/3379597.3387501`

For consistency with prior line-level localization work, NacgVuln uses the preprocessed Big-Vul `train.csv`, `val.csv`, and `test.csv` files distributed through the official LineVul replication package:

- LineVul repository: `https://github.com/awsm-research/LineVul`

The third-party CSV files are not committed to this repository. Construct the combined file with:

```bash
python data/data_mining.py
```

Expected output:

```text
data/dataset.csv
```

The script validates the required columns, reports the final row count and SHA-256 checksum, and reuses an existing valid `dataset.csv`. Use `--force` to redownload and rebuild it, or `--keep-components` to retain the three downloaded split files.

See [`docs/dataset.md`](docs/dataset.md) for the full provenance and split protocol.

## Quick Reproduction from Public Inputs

### 1. Install the environment

```bash
conda env create -f environment.yml
conda activate vulnloc
```

### 2. Download CodeT5-base and CodeBERT-base

```bash
python scripts/download_hf_models.py --target-dir ./hf_cache
```

### 3. Construct the dataset

```bash
python data/data_mining.py
```

### 4. Train the function-level CodeBERT component

```bash
python scripts/run_10seed_func_classifier_train.py \
  --project_root ./ \
  --output_root ./checkpoints_func_10seed \
  --train_script ./src/train_func_classifier.py \
  --model_variation ./hf_cache/codebert-base \
  --sampling no \
  --remove_missing_line_labels yes \
  --explainer ATTENTION \
  --explain_only_tp no \
  --sort_by_lines yes
```

This stage produces `checkpoints_func_10seed/checkpoints/seed*/best_weights.pt` for complete two-stage deployment and Self-Attention baseline reproduction.

### 5. Train and evaluate the NacgVuln line-level model for all 10 seeds

```bash
python scripts/run_10seed_compare_with_paper.py \
  --project_root ./ \
  --output_root ./runs_10seed_compare_with_paper \
  --dataset_path ./data/dataset.csv \
  --train_script ./src/train_seq2seq_nacgvuln.py \
  --eval_script ./src/eval_seq2seq_nacgvuln.py \
  --model_variation ./hf_cache/codet5-base \
  --eval_model_variation_seq2seq ./hf_cache/codet5-base \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --resume_if_exists yes \
  --skip_train no \
  --skip_eval no \
  --include_negatives yes \
  --negative_ratio 0.5 \
  --no_vuln_token '<NO_VULN>' \
  --rouge_on_vuln_only yes \
  --max_input_len 512 \
  --max_target_len_cap 128 \
  --max_target_len_eval 128 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --num_epochs 10 \
  --patience 5 \
  --lr 5e-5 \
  --chunk_training yes \
  --chunk_max_tokens 512 \
  --chunk_stride_lines_train 40 \
  --chunk_prefix_lines 8 \
  --chunk_prefix_max_tokens 192 \
  --neg_chunks_per_func 1 \
  --num_beams 1 \
  --num_return_seqs 1 \
  --fallback_sampling no \
  --sample_return_seqs 1 \
  --chunk_long_funcs yes \
  --chunk_stride_lines_eval 20 \
  --similarity_replacement yes \
  --dedup yes \
  --rerank no \
  --eval_mode locvul \
  --remove_missing_line_labels yes \
  --sort_by_lines yes
```

### 6. Re-evaluate checkpoints generated in Step 5

```bash
python scripts/run_seq2seq_eval.py \
  --project_root ./ \
  --output_root ./runs_seq2seq_eval \
  --dataset_path ./data/dataset.csv \
  --eval_script ./src/eval_seq2seq_nacgvuln.py \
  --existing_ckpt_root ./runs_10seed_compare_with_paper/checkpoints \
  --model_variation_seq2seq ./hf_cache/codet5-base \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --resume_if_exists yes \
  --max_input_len 512 \
  --max_target_len 128 \
  --num_beams 1 \
  --num_return_seqs 1 \
  --fallback_sampling no \
  --sample_return_seqs 1 \
  --chunk_long_funcs yes \
  --chunk_stride_lines_eval 20 \
  --similarity_replacement yes \
  --dedup yes \
  --rerank no \
  --remove_missing_line_labels yes \
  --eval_only_vuln yes \
  --sort_by_lines yes \
  --continue_on_error yes
```

The combined runner writes checkpoints under `runs_10seed_compare_with_paper/checkpoints`. A training-only alternative is available as `scripts/run_10seed_seq2seq_train.py`; its checkpoints are written under `checkpoints_seq2seq_10seed/checkpoints`. Neither checkpoint set is distributed.

## Checkpoint Availability

Fine-tuned NacgVuln and baseline checkpoints are **not uploaded** to GitHub or Zenodo. They are excluded because of their size. Reproduction therefore starts from the public CodeT5-base and CodeBERT-base models and regenerates checkpoints with the documented training commands.

Processed seed-level and aggregate result tables used in the manuscript are included under `results/`, so the reported analyses can be inspected without downloading model weights.

## Archived Results

Key processed result files include:

```text
results/main/Nacg_mean_std_metrics.csv
results/main/Nacg_per_seed_metrics.csv
results/main/compare_with_paper.csv
results/ablation/ablation_all_mean_std_metrics.csv
results/ablation/ablation_all_per_seed_metrics.csv
results/ablation/ablation_wilcoxon_tests.csv
results/clean_fpr/clean_fpr_mean_std.csv
results/clean_fpr/clean_fpr_per_seed.csv
results/length_group/length_group_mean_std_metrics.csv
results/length_group/length_group_per_seed_metrics.csv
```

The archived 10-seed means reported for NacgVuln are:

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

- [`docs/reproduction.md`](docs/reproduction.md): end-to-end reproduction workflow
- [`docs/script_mapping.md`](docs/script_mapping.md): mapping from original experiment filenames to public repository filenames
- [`docs/commands.md`](docs/commands.md): standardized repository-relative commands
- [`docs/dataset.md`](docs/dataset.md): third-party data provenance and split protocol
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md): evaluation subsets and metrics
- [`docs/model_parameters.md`](docs/model_parameters.md): model, training, and inference parameters
- [`docs/table_mapping.md`](docs/table_mapping.md): manuscript table-to-file mapping
- [`docs/code_availability_statement.md`](docs/code_availability_statement.md): PeerJ-ready availability statement

## Code and Data Availability

The source code, scripts, configuration files, documentation, and processed experimental outputs are archived on Zenodo and mirrored on GitHub. The third-party Big-Vul/LineVul CSV files and fine-tuned model checkpoints are not redistributed. Dataset reconstruction and model training instructions are provided above and in `docs/`.

## Citation

Cite the archived software record using [`CITATION.cff`](CITATION.cff) or DOI `10.5281/zenodo.21273222`.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
