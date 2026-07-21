# Standardized Reproduction Commands

All commands are executed from the repository root. Fine-tuned checkpoints are not distributed and must be generated locally. The complete, annotated procedure is in [`reproduction.md`](reproduction.md).

## 1. Environment and model cache

```bash
export PROJECT_ROOT="$(pwd)"
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT=https://hf-mirror.com
```

`HF_ENDPOINT` is optional. `HUGGINGFACE_HUB_CACHE` and `TRANSFORMERS_CACHE` are not required for the current workflow.

## 2. Install and download public models

```bash
conda env create -f environment.yml
conda activate vulnloc
python scripts/download_hf_models.py \
  --target-dir "$PROJECT_ROOT/hf_cache" \
  --hf-endpoint "$HF_ENDPOINT"
```

## 3. Construct the dataset

```bash
python data/data_mining.py
```

## 4. Function-level CodeBERT training, 10 seeds

```bash
python scripts/run_10seed_func_classifier_train.py \
  --project_root "$PROJECT_ROOT" \
  --output_root "$PROJECT_ROOT/checkpoints_func_10seed" \
  --train_script ./src/train_func_classifier.py \
  --model_variation "$PROJECT_ROOT/hf_cache/codebert-base" \
  --sampling no \
  --remove_missing_line_labels yes \
  --explainer ATTENTION \
  --explain_only_tp no \
  --sort_by_lines yes
```

## 5. NacgVuln CodeT5 training only, 10 seeds

```bash
python scripts/run_10seed_seq2seq_train.py \
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
  --quick_test no
```

## 6. NacgVuln training and evaluation, 10 seeds

This is the recommended PeerJ reproduction command. Use it instead of Step 5 when training and evaluation should be performed together.

```bash
python scripts/run_10seed_compare_with_paper.py \
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
  --sort_by_lines yes
```

## 7. Evaluate training-only checkpoints

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

## 8. Self-Attention baseline and paired Wilcoxon tests

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

## 9. Ablation and supplementary analysis

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

```bash
python scripts/collect_ablation_perseed_and_wilcoxon.py \
  --ablation_root ./ablation_runs \
  --output_dir ./results/ablation
```

```bash
python scripts/length_group_analysis.py \
  --ablation_root ./ablation_runs \
  --variants full_improved,chunk_infer_off,chunk_infer_on,chunk_train_off,chunk_train_on \
  --out_dir ./results/length_group
```
