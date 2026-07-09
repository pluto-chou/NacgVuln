#!/usr/bin/env bash
set -euo pipefail

# 修改成你自己的路径
PROJECT_ROOT="/path/to/LocVul-main"
PYTHON_BIN="/path/to/your/python"
DATASET_PATH="${PROJECT_ROOT}/data/dataset.csv"
BASE_MODEL="/path/to/hf_cache_clean/codet5-base"
OUT_ROOT="${PROJECT_ROOT}/ablation_runs"

# 1) 先跑完整改进版，给 eval-only 消融提供 reference checkpoints
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_seq2seq_ablation_suite.py" \
  --project_root "${PROJECT_ROOT}" \
  --output_root "${OUT_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --train_script "${PROJECT_ROOT}/Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py" \
  --eval_script "${PROJECT_ROOT}/seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py" \
  --python_bin "${PYTHON_BIN}" \
  --model_variation_seq2seq "${BASE_MODEL}" \
  --run_groups full

# 2) vulnerable-only vs include-negatives
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_seq2seq_ablation_suite.py" \
  --project_root "${PROJECT_ROOT}" \
  --output_root "${OUT_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --train_script "${PROJECT_ROOT}/Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py" \
  --eval_script "${PROJECT_ROOT}/seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py" \
  --python_bin "${PYTHON_BIN}" \
  --model_variation_seq2seq "${BASE_MODEL}" \
  --run_groups neg

# 3) no chunk training vs chunk training
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_seq2seq_ablation_suite.py" \
  --project_root "${PROJECT_ROOT}" \
  --output_root "${OUT_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --train_script "${PROJECT_ROOT}/Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py" \
  --eval_script "${PROJECT_ROOT}/seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py" \
  --python_bin "${PYTHON_BIN}" \
  --model_variation_seq2seq "${BASE_MODEL}" \
  --run_groups chunk_train

# 4) no chunk inference vs chunk inference（复用 full_improved）
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_seq2seq_ablation_suite.py" \
  --project_root "${PROJECT_ROOT}" \
  --output_root "${OUT_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --train_script "${PROJECT_ROOT}/Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py" \
  --eval_script "${PROJECT_ROOT}/seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py" \
  --python_bin "${PYTHON_BIN}" \
  --model_variation_seq2seq "${BASE_MODEL}" \
  --reference_ckpt_root "${OUT_ROOT}/full_improved/checkpoints" \
  --run_groups chunk_infer

# 5) without vs with similarity replacement（复用 full_improved）
"${PYTHON_BIN}" "${PROJECT_ROOT}/run_seq2seq_ablation_suite.py" \
  --project_root "${PROJECT_ROOT}" \
  --output_root "${OUT_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --train_script "${PROJECT_ROOT}/Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py" \
  --eval_script "${PROJECT_ROOT}/seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py" \
  --python_bin "${PYTHON_BIN}" \
  --model_variation_seq2seq "${BASE_MODEL}" \
  --reference_ckpt_root "${OUT_ROOT}/full_improved/checkpoints" \
  --run_groups sim

# 6) 可选：显著性检验示例
"${PYTHON_BIN}" "${PROJECT_ROOT}/ablation_wilcoxon.py" \
  --csv_a "${OUT_ROOT}/neg_on/per_seed_metrics.csv" \
  --csv_b "${OUT_ROOT}/neg_off/per_seed_metrics.csv" \
  --output_csv "${OUT_ROOT}/wilcoxon_neg_on_vs_neg_off.csv"
