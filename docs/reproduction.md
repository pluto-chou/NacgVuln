# Reproduction Guide

This guide describes how to reproduce the NacgVuln experiments from the released repository.

## 1. Repository and Environment

Clone the repository:

```bash
git clone https://github.com/pluto-chou/NacgVuln.git
cd NacgVuln
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate vulnloc
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

Raw environment records from the archived experiment machine are stored in:

```text
environment_records/conda_env.yml
environment_records/pip_freeze.txt
environment_records/python_version.txt
environment_records/hardware_nvidia_smi.txt
```

The archived hardware record reports Python 3.9.19 and an NVIDIA GeForce RTX 3090 Ti with CUDA 12.2 shown by `nvidia-smi`.

## 2. Dataset Construction

Run:

```bash
python data/data_mining.py
```

This creates:

```text
data/dataset.csv
```

The raw Big-Vul files are not redistributed in this repository. See [`dataset.md`](dataset.md).

## 3. Split Protocol

The repository follows a LocVul-style fixed-order split:

1. the last 10% of `dataset.csv` is the test set;
2. the remaining 90% is split again, with its last 10% used as validation;
3. the remaining samples are used for training;
4. all 10 seeds share the same test set.

## 4. Main NacgVuln Training

For one seed:

```bash
python src/train_seq2seq_nacgvuln.py   --seed 0   --FINE_TUNE yes   --model_variation Salesforce/codet5-base   --checkpoint_dir ./checkpoints_seq2seq_10seed/checkpoints/seed0   --INCLUDE_NEGATIVES yes   --NEGATIVE_RATIO 0.5   --NO_VULN_TOKEN '<NO_VULN>'   --ROUGE_ON_VULN_ONLY yes   --MAX_INPUT_LEN 512   --MAX_TARGET_LEN_CAP 128   --BATCH_SIZE 2   --GRAD_ACCUM_STEPS 4   --NUM_EPOCHS 10   --PATIENCE 5   --LR 5e-5   --CHUNK_TRAINING yes   --CHUNK_MAX_TOKENS 512   --CHUNK_STRIDE_LINES 40   --CHUNK_PREFIX_LINES 8   --CHUNK_PREFIX_MAX_TOKENS 192   --NEG_CHUNKS_PER_FUNC 1   --QUICK_TEST no
```

For 10-seed training and evaluation, use:

```bash
python scripts/run_10seed_compare_with_paper.py   --project_root ./   --output_root ./runs_10seed_compare_with_paper   --dataset_path ./data/dataset.csv   --train_script ./src/train_seq2seq_nacgvuln.py   --eval_script ./src/eval_seq2seq_nacgvuln.py   --model_variation Salesforce/codet5-base   --eval_model_variation_seq2seq Salesforce/codet5-base   --seeds 0,1,2,3,4,5,6,7,8,9
```

## 5. Evaluation with Existing Checkpoints

If checkpoints are available locally, run:

```bash
python scripts/run_seq2seq_eval.py   --project_root ./   --output_root ./runs_seq2seq_eval_v3_batch_resilient   --dataset_path ./data/dataset.csv   --eval_script ./src/eval_seq2seq_nacgvuln.py   --existing_ckpt_root ./checkpoints_seq2seq_10seed/checkpoints   --model_variation_seq2seq Salesforce/codet5-base   --seeds 0,1,2,3,4,5,6,7,8,9   --resume_if_exists yes   --max_input_len 512   --max_target_len 256   --num_beams 1   --num_return_seqs 1   --fallback_sampling no   --chunk_long_funcs yes   --chunk_stride_lines_eval 20   --similarity_replacement yes   --dedup yes   --rerank no   --remove_missing_line_labels yes   --eval_only_vuln yes   --sort_by_lines yes   --continue_on_error yes
```

## 6. Archived Main Results

The release includes processed result summaries:

```text
results/main/Nacg_per_seed_metrics.csv
results/main/Nacg_mean_std_metrics.csv
results/main/compare_with_paper.csv
results/main/compare_with_paper.md
```

## 7. Baseline and Statistical Testing

The original baseline scripts are under `baselines/`. Self-Attention baseline summary files are under:

```text
results/selfattention_baseline/
```

To rerun the baseline comparison and Wilcoxon test:

```bash
python scripts/run_baseline_and_wilcoxon.py   --project_root ./   --output_root ./runs_baseline_sigtest   --baseline_script ./baselines/vulnDet_pipeline.py   --improved_per_seed_csv ./results/main/Nacg_per_seed_metrics.csv   --seeds 0,1,2,3,4,5,6,7,8,9   --fine_tune no   --existing_ckpt_root ./checkpoints_func_10seed/checkpoints   --model_variation microsoft/codebert-base   --sampling no   --remove_missing_line_labels yes   --explainer ATTENTION   --explain_only_tp no   --sort_by_lines yes   --resume_if_exists yes
```

## 8. Ablation Experiments

The ablation runner is:

```text
ablation/run_seq2seq_ablation_suite.py
```

Examples are provided in:

```text
ablation/run_ablation_examples.sh
ablation/ablation_README.md
```

Archived ablation summaries are stored in:

```text
results/ablation/
```

## 9. Clean FPR, Long-Function, and CWE Analyses

Clean-function false-positive results:

```text
results/clean_fpr/clean_fpr_per_seed.csv
results/clean_fpr/clean_fpr_mean_std.csv
```

Long-function analysis results:

```text
results/length_group/length_group_per_seed_metrics.csv
results/length_group/length_group_mean_std_metrics.csv
```

CWE-level analysis results:

```text
results/cwe_analysis/
```

The long-function analysis can be regenerated with:

```bash
python scripts/length_group_analysis.py   --ablation_root ./ablation_runs   --variants full_improved,chunk_infer_off,chunk_infer_on,chunk_train_off,chunk_train_on   --out_dir ./results/length_group
```

## 10. Checkpoint Availability

Large model checkpoints are excluded from GitHub by `.gitignore`. They should be regenerated with the training commands or, if the authors choose to distribute them, attached as Zenodo release files. The processed CSV results are included so that paper tables can be inspected without downloading model weights.
