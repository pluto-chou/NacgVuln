# Commands Used for Reproduction

This document records repository-relative commands for reproducing the experiments. Machine-specific paths from the original logs have been normalized to paths relative to the repository root.

## 1. Optional Hugging Face Cache Setup

```bash
export HF_HOME=./hf_cache
export HUGGINGFACE_HUB_CACHE=./hf_cache/hub
export TRANSFORMERS_CACHE=./hf_cache/transformers
```

For environments where the official Hugging Face endpoint is slow, an optional mirror can be set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 2. Dataset Construction

```bash
python data/data_mining.py
```

Expected output:

```text
data/dataset.csv
```

## 3. Single-Seed NacgVuln Training

```bash
python src/train_seq2seq_nacgvuln.py   --seed 1   --FINE_TUNE yes   --model_variation Salesforce/codet5-base   --checkpoint_dir ./checkpoints_seq2seq_10seed/checkpoints/seed1   --INCLUDE_NEGATIVES yes   --NEGATIVE_RATIO 0.5   --NO_VULN_TOKEN '<NO_VULN>'   --ROUGE_ON_VULN_ONLY yes   --MAX_INPUT_LEN 512   --MAX_TARGET_LEN_CAP 128   --BATCH_SIZE 2   --GRAD_ACCUM_STEPS 4   --NUM_EPOCHS 10   --PATIENCE 5   --LR 5e-5   --CHUNK_TRAINING yes   --CHUNK_MAX_TOKENS 512   --CHUNK_STRIDE_LINES 40   --CHUNK_PREFIX_LINES 8   --CHUNK_PREFIX_MAX_TOKENS 192   --NEG_CHUNKS_PER_FUNC 1   --QUICK_TEST no
```

## 4. Single-Seed Evaluation

```bash
python src/eval_seq2seq_nacgvuln.py   --seed 0   --checkpoint_dir_seq2seq ./checkpoints_seq2seq_10seed/checkpoints/seed0   --model_variation_seq2seq Salesforce/codet5-base   --output_dir ./runs_single_seed/eval_outputs   --run_name seed0_chunk_fast_balanced   --test_path ./runs_single_seed/preprocessed_tests/preprocessed_data_test_0.csv   --MAX_INPUT_LEN 512   --MAX_TARGET_LEN 128   --NUM_BEAMS 1   --NUM_RETURN_SEQS 1   --FALLBACK_SAMPLING no   --SAMPLE_RETURN_SEQS 1   --TOP_P 0.95   --TEMPERATURE 0.7   --CHUNK_LONG_FUNCS yes   --CHUNK_STRIDE_LINES 20   --SIMILARITY_REPLACEMENT yes   --SIM_WEIGHT 1.0   --SIM_THRESHOLD 0.0   --DEDUP yes   --RERANK no   --K 10   --EVAL_MODE locvul   --REMOVE_MISSING_LINE_LABELS yes   --EVAL_ONLY_VULN yes   --NO_VULN_TOKEN '<NO_VULN>'   --HANDLE_NO_VULN_TOKEN yes   --sort_by_lines yes
```

## 5. 10-Seed Batch Evaluation from Existing Checkpoints

```bash
python scripts/run_seq2seq_eval.py   --project_root ./   --output_root ./runs_seq2seq_eval_v3_batch_resilient   --dataset_path ./data/dataset.csv   --eval_script ./src/eval_seq2seq_nacgvuln.py   --existing_ckpt_root ./checkpoints_seq2seq_10seed/checkpoints   --model_variation_seq2seq Salesforce/codet5-base   --seeds 0,1,2,3,4,5,6,7,8,9   --resume_if_exists yes   --max_input_len 512   --max_target_len 256   --num_beams 1   --num_return_seqs 1   --fallback_sampling no   --sample_return_seqs 1   --chunk_long_funcs yes   --chunk_stride_lines_eval 20   --similarity_replacement yes   --dedup yes   --rerank no   --remove_missing_line_labels yes   --eval_only_vuln yes   --sort_by_lines yes   --continue_on_error yes   --max_retries_per_seed 2   --retry_disable_similarity yes   --retry_disable_rerank yes   --strict_failure no
```

## 6. 10-Seed Training + Evaluation

```bash
python scripts/run_10seed_compare_with_paper.py   --project_root ./   --output_root ./runs_10seed_compare_with_paper   --dataset_path ./data/dataset.csv   --train_script ./src/train_seq2seq_nacgvuln.py   --eval_script ./src/eval_seq2seq_nacgvuln.py   --model_variation Salesforce/codet5-base   --eval_model_variation_seq2seq Salesforce/codet5-base   --seeds 0,1,2,3,4,5,6,7,8,9   --resume_if_exists yes   --skip_train no   --skip_eval no   --include_negatives yes   --negative_ratio 0.5   --no_vuln_token '<NO_VULN>'   --rouge_on_vuln_only yes   --max_input_len 512   --max_target_len_cap 128   --max_target_len_eval 128   --batch_size 2   --grad_accum_steps 4   --num_epochs 10   --patience 5   --lr 5e-5   --chunk_training yes   --chunk_max_tokens 512   --chunk_stride_lines_train 40   --chunk_prefix_lines 8   --chunk_prefix_max_tokens 192   --neg_chunks_per_func 1   --num_beams 1   --num_return_seqs 1   --fallback_sampling no   --sample_return_seqs 1   --chunk_long_funcs yes   --chunk_stride_lines_eval 20   --similarity_replacement yes   --dedup yes   --rerank no   --eval_mode locvul   --remove_missing_line_labels yes   --sort_by_lines yes
```

## 7. Self-Attention Baseline and Wilcoxon Test

```bash
python scripts/run_baseline_and_wilcoxon.py   --project_root ./   --output_root ./runs_baseline_sigtest   --baseline_script ./baselines/vulnDet_pipeline.py   --improved_per_seed_csv ./results/main/Nacg_per_seed_metrics.csv   --seeds 0,1,2,3,4,5,6,7,8,9   --fine_tune no   --existing_ckpt_root ./checkpoints_func_10seed/checkpoints   --model_variation microsoft/codebert-base   --sampling no   --remove_missing_line_labels yes   --explainer ATTENTION   --explain_only_tp no   --sort_by_lines yes   --resume_if_exists yes
```

## 8. Ablation Suite

Examples are provided in:

```text
ablation/run_ablation_examples.sh
ablation/ablation_README.md
```

The main ablation runner is:

```bash
python ablation/run_seq2seq_ablation_suite.py   --project_root ./   --output_root ./ablation_runs   --dataset_path ./data/dataset.csv   --train_script ./src/train_seq2seq_nacgvuln.py   --eval_script ./src/eval_seq2seq_nacgvuln.py   --model_variation_seq2seq Salesforce/codet5-base   --run_groups all   --gpu_ids 0,1   --max_parallel_seeds 2   --batch_size 4   --grad_accum_steps 2
```

## 9. Ablation Wilcoxon Tests

```bash
python scripts/collect_ablation_perseed_and_wilcoxon.py   --ablation_root ./ablation_runs   --output_dir ./results/ablation
```

For pairwise tests, use:

```bash
python ablation/ablation_wilcoxon.py   --csv_a ./ablation_runs/neg_on/per_seed_metrics.csv   --csv_b ./ablation_runs/neg_off/per_seed_metrics.csv   --output_csv ./ablation_runs/wilcoxon_neg_on_vs_neg_off.csv
```

## 10. Long-Function Analysis

```bash
python scripts/length_group_analysis.py   --ablation_root ./ablation_runs   --variants full_improved,chunk_infer_off,chunk_infer_on,chunk_train_off,chunk_train_on   --out_dir ./results/length_group
```

## 11. Figure and Table Generation

```bash
python figures/make_chapter5_figures_with_compare.py
python figures/make_cwe_level_table.py
python figures/make_cwe_paper_tables.py
python figures/make_fig1_from_candidate.py
python figures/make_fig2.py
```
