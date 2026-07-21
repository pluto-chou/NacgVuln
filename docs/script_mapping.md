# Original-to-Repository Script Mapping

The experiment was originally developed in a `LocVul-main` working directory. The public NacgVuln repository renames the principal implementation files and places them under `src/` and `scripts/`.

| Original experiment file | Public repository file | Role |
|---|---|---|
| `vulnDet_pipeline.py` | `src/train_func_classifier.py` | CodeBERT function-level classifier and Self-Attention localization baseline |
| `Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py` | `src/train_seq2seq_nacgvuln.py` | Negative-aware and chunk-enhanced CodeT5 training |
| `seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py` | `src/eval_seq2seq_nacgvuln.py` | NacgVuln line-level evaluation |
| `run_10seed_func_classifier_train.py` | `scripts/run_10seed_func_classifier_train.py` | Ten-seed function-level training runner |
| `run_10seed_seq2seq_train.py` | `scripts/run_10seed_seq2seq_train.py` | Ten-seed NacgVuln training-only runner |
| combined local train/evaluate command | `scripts/run_10seed_compare_with_paper.py` | Ten-seed NacgVuln training, evaluation, and aggregation |
| local batch evaluation runner | `scripts/run_seq2seq_eval.py` | Batch evaluation of locally generated checkpoints |
| local baseline/significance scripts | `scripts/run_baseline_and_wilcoxon.py` | Self-Attention baseline execution and paired Wilcoxon testing |

The files under `baselines/` retain the LocVul-style baseline implementations. For reproducing the NacgVuln implementation, use the renamed files under `src/`. The `src/` files contain the exact negative-aware, chunk-training, chunk-inference, and source-alignment options documented in the manuscript.

## Training stages

The complete system contains two separately trained model components:

1. **Function-level screening and Self-Attention baseline:** CodeBERT, trained with `src/train_func_classifier.py`.
2. **Line-level generative localization:** CodeT5-base, trained with `src/train_seq2seq_nacgvuln.py`.

The manuscript's main line-level experiments evaluate the second component on the vulnerable-function subset. The function-level model is still required for the complete two-stage deployment pipeline and for reproducing the Self-Attention baseline.
