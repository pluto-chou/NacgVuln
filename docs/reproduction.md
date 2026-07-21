# Complete Reproduction Guide

This guide reproduces the NacgVuln implementation from the public CodeBERT and CodeT5-base models, the documented third-party Big-Vul/LineVul data files, and the public repository source code.

## Checkpoint policy

Fine-tuned function-level and line-level checkpoints are not uploaded to GitHub or Zenodo. They must be regenerated from the public base models. Processed seed-level and aggregate result tables used by the manuscript remain under `results/` so that the reported analyses can be inspected without model weights.

## 1. Script-name mapping

The public repository uses shorter, role-oriented file names. See [`script_mapping.md`](script_mapping.md) for the full mapping. The three central replacements are:

```text
vulnDet_pipeline.py
  -> src/train_func_classifier.py

Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py
  -> src/train_seq2seq_nacgvuln.py

seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py
  -> src/eval_seq2seq_nacgvuln.py
```

## 2. Clone and install

```bash
git clone https://github.com/pluto-chou/NacgVuln.git
cd NacgVuln
conda env create -f environment.yml
conda activate vulnloc
```

Alternative virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The concise direct dependencies are in `requirements.txt`. The full environment records from the original experiment machine remain under `environment_records/`.

## 3. Configure Hugging Face storage

Current `huggingface_hub` versions use `HF_HOME` and `HF_HUB_CACHE`. `HUGGINGFACE_HUB_CACHE` is a deprecated alias, so new commands use `HF_HUB_CACHE`.

### Library machine

Set the repository path to the actual clone location:

```bash
export PROJECT_ROOT=/home/user505/hs_vulnloc/NacgVuln
cd "$PROJECT_ROOT"

export HF_HOME=/home/user505/hs_vulnloc/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT=https://hf-mirror.com
```

### Laboratory machine

If the public repository has been placed in the former `LocVul-main` directory, the original path can remain unchanged:

```bash
export PROJECT_ROOT=/media/com/Elements1/deeplearning/LocVul-main
cd "$PROJECT_ROOT"

export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT=https://hf-mirror.com
```

`HF_ENDPOINT` is optional. Remove it when the official Hugging Face endpoint is directly accessible. The reproduction commands use explicit local model directories after download, so training does not depend on a remote model identifier.

## 4. Download the public base models

```bash
python scripts/download_hf_models.py \
  --target-dir "$PROJECT_ROOT/hf_cache" \
  --hf-endpoint "$HF_ENDPOINT"
```

Expected directories:

```text
$PROJECT_ROOT/hf_cache/codet5-base
$PROJECT_ROOT/hf_cache/codebert-base
```

Verify them:

```bash
test -s "$PROJECT_ROOT/hf_cache/codet5-base/config.json"
test -s "$PROJECT_ROOT/hf_cache/codebert-base/config.json"
```

## 5. Construct the dataset

```bash
cd "$PROJECT_ROOT"
python data/data_mining.py
```

Expected output:

```text
data/dataset.csv
```

The script downloads the LineVul-preprocessed Big-Vul `train.csv`, `val.csv`, and `test.csv` files, validates the required columns, concatenates them in that order, and prints the final SHA-256 checksum. Use `--force` to rebuild and `--keep-components` to retain the downloaded component files.

## 6. Train the function-level CodeBERT component

The function-level component performs candidate-function screening and is also used to reproduce the Self-Attention baseline.

### Ten seeds

```bash
nohup python -u scripts/run_10seed_func_classifier_train.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/checkpoints_func_10seed" \
  --train_script ./src/train_func_classifier.py \
  --model_variation "$PROJECT_ROOT/hf_cache/codebert-base" \
  --sampling no \
  --remove_missing_line_labels yes \
  --explainer ATTENTION \
  --explain_only_tp no \
  --sort_by_lines yes \
  --resume_if_exists yes \
  > "$PROJECT_ROOT/output_func_10seed.log" 2>&1 &
```

Expected checkpoints:

```text
checkpoints_func_10seed/checkpoints/seed0/best_weights.pt
...
checkpoints_func_10seed/checkpoints/seed9/best_weights.pt
```

Monitor training:

```bash
tail -f "$PROJECT_ROOT/output_func_10seed.log"
```

## 7. Train the NacgVuln line-level CodeT5 component

### Ten-seed training only

This command is the direct public-repository replacement for the original `run_10seed_seq2seq_train.py` command:

```bash
nohup python -u scripts/run_10seed_seq2seq_train.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/checkpoints_seq2seq_10seed" \
  --train_script ./src/train_seq2seq_nacgvuln.py \
  --model_variation "$PROJECT_ROOT/hf_cache/codet5-base" \
  --fine_tune yes \
  --include_negatives yes \
  --negative_ratio 0.5 \
  --no_vuln_token '<NO_VULN>' \
  --rouge_on_vuln_only yes \
  --max_input_len 512 \
  --max_target_len_cap 128 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --num_epochs 10 \
  --patience 5 \
  --lr 5e-5 \
  --chunk_training yes \
  --chunk_max_tokens 512 \
  --chunk_stride_lines 40 \
  --chunk_prefix_lines 8 \
  --chunk_prefix_max_tokens 192 \
  --neg_chunks_per_func 1 \
  --quick_test no \
  --resume_if_exists yes \
  > "$PROJECT_ROOT/output_seq2seq_10seed.log" 2>&1 &
```

Expected checkpoints:

```text
checkpoints_seq2seq_10seed/checkpoints/seed0/best_weights.pt
...
checkpoints_seq2seq_10seed/checkpoints/seed9/best_weights.pt
```

### Single seed

The original seed-1 command maps directly to the renamed training file:

```bash
python src/train_seq2seq_nacgvuln.py \
  --seed 1 \
  --FINE_TUNE yes \
  --model_variation "$PROJECT_ROOT/hf_cache/codet5-base" \
  --checkpoint_dir "$PROJECT_ROOT/checkpoints_seq2seq_10seed/checkpoints/seed1" \
  --INCLUDE_NEGATIVES yes \
  --NEGATIVE_RATIO 0.5 \
  --NO_VULN_TOKEN '<NO_VULN>' \
  --ROUGE_ON_VULN_ONLY yes \
  --MAX_INPUT_LEN 512 \
  --MAX_TARGET_LEN_CAP 128 \
  --BATCH_SIZE 2 \
  --GRAD_ACCUM_STEPS 4 \
  --NUM_EPOCHS 10 \
  --PATIENCE 5 \
  --LR 5e-5 \
  --CHUNK_TRAINING yes \
  --CHUNK_MAX_TOKENS 512 \
  --CHUNK_STRIDE_LINES 40 \
  --CHUNK_PREFIX_LINES 8 \
  --CHUNK_PREFIX_MAX_TOKENS 192 \
  --NEG_CHUNKS_PER_FUNC 1 \
  --QUICK_TEST no
```

## 8. Recommended one-command 10-seed reproduction

For PeerJ reviewers, the recommended command trains and evaluates every NacgVuln seed and writes the aggregate tables in one run:

```bash
nohup python -u scripts/run_10seed_compare_with_paper.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/runs_10seed_compare_with_paper" \
  --dataset_path ./data/dataset.csv \
  --train_script ./src/train_seq2seq_nacgvuln.py \
  --eval_script ./src/eval_seq2seq_nacgvuln.py \
  --model_variation "$PROJECT_ROOT/hf_cache/codet5-base" \
  --eval_model_variation_seq2seq "$PROJECT_ROOT/hf_cache/codet5-base" \
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
  --sort_by_lines yes \
  > "$PROJECT_ROOT/output_nacgvuln_10seed_train_eval.log" 2>&1 &
```

Expected outputs:

```text
runs_10seed_compare_with_paper/checkpoints/seed*/best_weights.pt
runs_10seed_compare_with_paper/eval_outputs/*_details.csv
runs_10seed_compare_with_paper/eval_outputs/*_summary.csv
runs_10seed_compare_with_paper/per_seed_metrics.csv
runs_10seed_compare_with_paper/mean_std_metrics.csv
runs_10seed_compare_with_paper/compare_with_paper.csv
runs_10seed_compare_with_paper/run_config.json
```

Do not run the training-only command and the combined command simultaneously into the same checkpoint directory. Choose either:

- the training-only runner followed by batch evaluation; or
- the recommended combined runner.

## 9. Evaluate locally generated checkpoints

For checkpoints generated by the training-only runner:

```bash
python scripts/run_seq2seq_eval.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/runs_seq2seq_eval" \
  --dataset_path ./data/dataset.csv \
  --eval_script ./src/eval_seq2seq_nacgvuln.py \
  --existing_ckpt_root "$PROJECT_ROOT/checkpoints_seq2seq_10seed/checkpoints" \
  --model_variation_seq2seq "$PROJECT_ROOT/hf_cache/codet5-base" \
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

For checkpoints produced by the combined runner, change `--existing_ckpt_root` to:

```text
$PROJECT_ROOT/runs_10seed_compare_with_paper/checkpoints
```

## 10. Reproduce the Self-Attention baseline and Wilcoxon comparison

After the function-level checkpoints and NacgVuln per-seed metrics exist:

```bash
python scripts/run_baseline_and_wilcoxon.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/runs_baseline_sigtest" \
  --baseline_script ./src/train_func_classifier.py \
  --improved_per_seed_csv ./runs_10seed_compare_with_paper/per_seed_metrics.csv \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --fine_tune no \
  --existing_ckpt_root ./checkpoints_func_10seed/checkpoints \
  --model_variation "$PROJECT_ROOT/hf_cache/codebert-base" \
  --sampling no \
  --remove_missing_line_labels yes \
  --explainer ATTENTION \
  --explain_only_tp no \
  --sort_by_lines yes \
  --resume_if_exists yes
```

## 11. Ablation and supplementary analyses

```bash
python ablation/run_seq2seq_ablation_suite.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/ablation_runs" \
  --dataset_path ./data/dataset.csv \
  --train_script ./src/train_seq2seq_nacgvuln.py \
  --eval_script ./src/eval_seq2seq_nacgvuln.py \
  --model_variation_seq2seq "$PROJECT_ROOT/hf_cache/codet5-base" \
  --run_groups all \
  --gpu_ids 0 \
  --max_parallel_seeds 1 \
  --batch_size 2 \
  --grad_accum_steps 4
```

Aggregate ablation and significance results:

```bash
python scripts/collect_ablation_perseed_and_wilcoxon.py \
  --ablation_root ./ablation_runs \
  --output_dir ./results/ablation
```

Long-function analysis:

```bash
python scripts/length_group_analysis.py \
  --ablation_root ./ablation_runs \
  --variants full_improved,chunk_infer_off,chunk_infer_on,chunk_train_off,chunk_train_on \
  --out_dir ./results/length_group
```

## 12. Verify the reproduction artifacts

```bash
find checkpoints_func_10seed/checkpoints -name best_weights.pt | sort
find checkpoints_seq2seq_10seed/checkpoints -name best_weights.pt | sort
find runs_10seed_compare_with_paper -maxdepth 2 -type f | sort
```

The expected number of `best_weights.pt` files is ten for each completed 10-seed training stage. A missing seed should be investigated through the corresponding file under the runner's `logs/` directory and `failed_seeds.csv`.
