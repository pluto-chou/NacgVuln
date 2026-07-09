# LocVul 消融实验运行说明

## 文件
- `run_seq2seq_ablation_suite.py`：统一批跑脚本
- `ablation_wilcoxon.py`：对两个 `per_seed_metrics.csv` 做 Wilcoxon signed-rank test
- `run_ablation_examples.sh`：示例命令

## 四组消融的严格控制方式

### 1. vulnerable-only vs include-negatives
- 训练侧唯一变化：`INCLUDE_NEGATIVES=no/yes`
- 其余保持不变：`CHUNK_TRAINING=yes`，`CHUNK_LONG_FUNCS=yes`，`SIMILARITY_REPLACEMENT=yes`

### 2. no chunk training vs chunk training
- 训练侧唯一变化：`CHUNK_TRAINING=no/yes`
- 其余保持不变：`INCLUDE_NEGATIVES=yes`，`CHUNK_LONG_FUNCS=yes`，`SIMILARITY_REPLACEMENT=yes`

### 3. no chunk inference vs chunk inference
- 不重新训练
- 复用同一套 reference checkpoints
- 评估侧唯一变化：`CHUNK_LONG_FUNCS=no/yes`

### 4. without similarity replacement vs with similarity replacement
- 不重新训练
- 复用同一套 reference checkpoints
- 评估侧唯一变化：`SIMILARITY_REPLACEMENT=no/yes`

## 推荐完整流程
1. 先准备 `data/dataset.csv`
2. 先跑 `full_improved`，得到 reference checkpoints
3. 跑 `neg`
4. 跑 `chunk_train`
5. 跑 `chunk_infer`（复用 full_improved）
6. 跑 `sim`（复用 full_improved）
7. 对每对消融结果做 Wilcoxon

## 每个变体输出
`<output_root>/<variant>/`
- `checkpoints/seed0..seed9/best_weights.pt`
- `eval_outputs/*_details.csv`
- `eval_outputs/*_summary.csv`
- `per_seed_metrics.csv`
- `mean_std_metrics.csv`
- `failed_seeds.csv`

## 总汇总
`<output_root>/ablation_suite_summary.csv`
