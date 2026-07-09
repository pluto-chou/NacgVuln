#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一次性自动跑 10-seed 训练 + 评估，并把当前结果与 LocVul 论文公布数值做对比。

稳定版改动：
1) 评估默认参数与已验证可跑通的 offline/vuln-only 命令对齐：
   - eval_script 默认改为 seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py
   - NUM_BEAMS=1
   - NUM_RETURN_SEQS=1
   - FALLBACK_SAMPLING=no
   - SAMPLE_RETURN_SEQS=1
   - CHUNK_STRIDE_LINES(eval)=20
   - RERANK=no
   - MAX_TARGET_LEN(eval)=128
2) 支持独立指定评估侧基础模型目录（--eval_model_variation_seq2seq），避免训练模型源与评估 tokenizer/base model 源被强绑定。
3) 复用已有 checkpoint 根目录时，同时兼容以下结构：
   - seed0/best_weights.pt
   - checkpoints/seed0/best_weights.pt
   - seed0/ 目录本身
4) 如果评估子进程因 OOM/SIGKILL 退出（returncode=-9），可自动用更轻参数重试一次。
5) 自动给子进程注入：
   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   - TOKENIZERS_PARALLELISM=false
6) 保留 resume_if_exists / skip_train / skip_eval 逻辑。

默认脚本组合：
- 训练：Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py
- 评估：seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py

输出：
- output_root/logs/*.log
- output_root/checkpoints/seed*/best_weights.pt
- output_root/preprocessed_tests/*.csv
- output_root/eval_outputs/*_details.csv
- output_root/eval_outputs/*_summary.csv
- output_root/per_seed_metrics.csv
- output_root/mean_std_metrics.csv
- output_root/compare_with_paper.csv
- output_root/compare_with_paper.md
- output_root/run_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


PAPER_METRICS = {
    "A@10": 0.828,
    "P@10": 0.269,
    "R@10": 0.790,
    "MRR@10": 0.794,
    "MAP@10": 0.792,
    "Median IFA": 0.0,
    "Effort@20%Recall": 0.0057,
    "Recall@1%LOC": 0.298,
}

DEFAULT_SEEDS = list(range(10))


@dataclass
class Paths:
    root: Path
    logs: Path
    ckpts: Path
    tests: Path
    eval_outputs: Path


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_line(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return " ".join(s.split())


def split_locvul_lines(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []
    if "/~/" in s:
        return [x.strip() for x in s.split("/~/") if x.strip()]
    if "\n" in s:
        return [x.strip() for x in s.splitlines() if x.strip()]
    return [s]


def mean_safe(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def std_safe(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(statistics.stdev(vals))


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_args(d: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for k, v in d.items():
        out.extend([f"--{k}", str(v)])
    return out


def run_and_log(cmd: List[str], log_file: Path, cwd: Path, env: Dict[str, str]) -> int:
    ensure_dir(log_file.parent)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"[{now_str()}] RUN CMD\n")
        f.write(" ".join(cmd) + "\n")
        f.write("=" * 120 + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def build_paths(output_root: Path) -> Paths:
    paths = Paths(
        root=output_root,
        logs=output_root / "logs",
        ckpts=output_root / "checkpoints",
        tests=output_root / "preprocessed_tests",
        eval_outputs=output_root / "eval_outputs",
    )
    for p in [paths.root, paths.logs, paths.ckpts, paths.tests, paths.eval_outputs]:
        ensure_dir(p)
    return paths


def prepare_test_csvs(dataset_path: Path, out_dir: Path, seeds: List[int]) -> Dict[int, Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"缺少数据集：{dataset_path}。请先运行 python data_mining.py")

    df = pd.read_csv(dataset_path)
    required = ["processed_func", "target", "flaw_line", "flaw_line_index"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"dataset.csv 缺少列：{c}")

    val_ratio = 0.1
    n = int(val_ratio * len(df))
    if n <= 0:
        raise ValueError(f"dataset.csv 样本过少，无法按 10% 划分测试集：{len(df)}")

    test_df = df.iloc[-n:, :].copy()
    mapping: Dict[int, Path] = {}
    for seed in seeds:
        out_path = out_dir / f"preprocessed_data_test_{seed}.csv"
        test_df.to_csv(out_path, index=False)
        mapping[seed] = out_path
    return mapping


def compute_precision_recall_from_details(details_csv: Path, k: int) -> Dict[str, float]:
    df = pd.read_csv(details_csv)
    if df.empty:
        return {f"P@{k}": 0.0, f"R@{k}": 0.0}

    ranked_col = "ranked_topK" if "ranked_topK" in df.columns else "ranked_lines_topK"
    gt_col = "gt_lines" if "gt_lines" in df.columns else "Lines"

    precisions: List[float] = []
    recalls: List[float] = []

    for _, row in df.iterrows():
        gt = split_locvul_lines(row.get(gt_col, ""))
        ranked = split_locvul_lines(row.get(ranked_col, ""))[:k]

        gt_norm = [normalize_line(x) for x in gt if normalize_line(x)]
        gt_set = set(gt_norm)
        if not gt_set:
            continue

        count = 0
        for x in ranked:
            if normalize_line(x) in gt_set:
                count += 1

        precisions.append(count / k)
        recalls.append(count / len(gt_set))

    return {
        f"P@{k}": mean_safe(precisions),
        f"R@{k}": mean_safe(recalls),
    }


def fmt_metric(metric: str, value: float) -> str:
    if metric == "Median IFA":
        return f"{value:.4f}"
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value) * 100:.2f}%"


def make_markdown_compare(df: pd.DataFrame) -> str:
    lines = [
        "# 10-seed 与原论文对比",
        "",
        "| Metric | Your mean | Your std | Paper | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Metric']} | {r['Your mean']} | {r['Your std']} | {r['Paper']} | {r['Delta']} |"
        )
    return "\n".join(lines) + "\n"


def _candidate_existing_checkpoint_paths(seed: int, existing_ckpt_root: Path) -> List[Path]:
    seed_dir = existing_ckpt_root / f"seed{seed}"
    checkpoints_seed_dir = existing_ckpt_root / "checkpoints" / f"seed{seed}"
    return [
        seed_dir / "best_weights.pt",
        checkpoints_seed_dir / "best_weights.pt",
        existing_ckpt_root / f"seed{seed}" / "checkpoint" / "best_weights.pt",
    ]


def maybe_reuse_existing_checkpoint(seed: int, seed_ckpt_dir: Path, existing_ckpt_root: Path | None) -> bool:
    if existing_ckpt_root is None:
        return False
    dst = seed_ckpt_dir / "best_weights.pt"
    if dst.exists():
        return True

    for src in _candidate_existing_checkpoint_paths(seed, existing_ckpt_root):
        if src.exists():
            ensure_dir(seed_ckpt_dir)
            shutil.copy2(src, dst)
            log(f"seed={seed} 复用已有 checkpoint: {src} -> {dst}")
            return True

    log(
        f"seed={seed} 未找到可复用 checkpoint，已检查: " +
        ", ".join(str(p) for p in _candidate_existing_checkpoint_paths(seed, existing_ckpt_root))
    )
    return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", type=str)
    ap.add_argument("--output_root", default="./runs_10seed_compare_with_paper", type=str)
    ap.add_argument("--dataset_path", default="./data/dataset.csv", type=str)

    ap.add_argument(
        "--train_script",
        default="./Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py",
        type=str,
        help="当前改进版训练脚本",
    )
    ap.add_argument(
        "--eval_script",
        default="./seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py",
        type=str,
        help="当前改进版评估脚本（建议 offline_v3_vulnonly 版，兼容本地 HF cache/base model + best_weights.pt）",
    )

    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    ap.add_argument("--python_bin", default=sys.executable, type=str)
    ap.add_argument("--hf_endpoint", default="", type=str)
    ap.add_argument("--resume_if_exists", default="yes", choices=["yes", "no"])
    ap.add_argument("--skip_train", default="no", choices=["yes", "no"])
    ap.add_argument("--skip_eval", default="no", choices=["yes", "no"])
    ap.add_argument("--k", default=10, type=int)
    ap.add_argument("--existing_ckpt_root", default="", type=str,
                    help="可选：从该目录复用已有 checkpoints，目录结构要求为 seed0/seed1/.../best_weights.pt")

    # 训练参数
    ap.add_argument("--model_variation", default="Salesforce/codet5-base", type=str)
    ap.add_argument("--eval_model_variation_seq2seq", default="", type=str,
                    help="评估阶段加载 tokenizer/base model 的来源；为空时回退到 --model_variation")
    ap.add_argument("--include_negatives", default="yes", choices=["yes", "no"])
    ap.add_argument("--negative_ratio", default="0.5", type=str)
    ap.add_argument("--no_vuln_token", default="<NO_VULN>", type=str)
    ap.add_argument("--rouge_on_vuln_only", default="yes", choices=["yes", "no"])
    ap.add_argument("--max_input_len", default="512", type=str)
    ap.add_argument("--max_target_len_cap", default="128", type=str)
    ap.add_argument("--max_target_len_eval", default="128", type=str)
    ap.add_argument("--batch_size", default="2", type=str)
    ap.add_argument("--grad_accum_steps", default="4", type=str)
    ap.add_argument("--num_epochs", default="10", type=str)
    ap.add_argument("--patience", default="5", type=str)
    ap.add_argument("--lr", default="5e-5", type=str)
    ap.add_argument("--chunk_training", default="yes", choices=["yes", "no"])
    ap.add_argument("--chunk_max_tokens", default="512", type=str)
    ap.add_argument("--chunk_stride_lines_train", default="40", type=str)
    ap.add_argument("--chunk_prefix_lines", default="8", type=str)
    ap.add_argument("--chunk_prefix_max_tokens", default="192", type=str)
    ap.add_argument("--neg_chunks_per_func", default="1", type=str)

    # 评估参数：默认改成更稳的配置
    ap.add_argument("--num_beams", default="1", type=str)
    ap.add_argument("--num_return_seqs", default="1", type=str)
    ap.add_argument("--fallback_sampling", default="no", choices=["yes", "no"])
    ap.add_argument("--sample_return_seqs", default="1", type=str)
    ap.add_argument("--top_p", default="0.95", type=str)
    ap.add_argument("--temperature", default="0.7", type=str)
    ap.add_argument("--chunk_long_funcs", default="yes", choices=["yes", "no"])
    ap.add_argument("--chunk_stride_lines_eval", default="20", type=str)
    ap.add_argument("--similarity_replacement", default="yes", choices=["yes", "no"])
    ap.add_argument("--sim_weight", default="1.0", type=str)
    ap.add_argument("--sim_threshold", default="0.0", type=str)
    ap.add_argument("--dedup", default="yes", choices=["yes", "no"])
    ap.add_argument("--rerank", default="no", choices=["yes", "no"])
    ap.add_argument("--eval_mode", default="locvul", choices=["locvul", "gen_only"])
    ap.add_argument("--remove_missing_line_labels", default="yes", choices=["yes", "no"])
    ap.add_argument("--sort_by_lines", default="yes", choices=["yes", "no"])

    # OOM 自动重试
    ap.add_argument("--retry_eval_on_oom", default="yes", choices=["yes", "no"])
    ap.add_argument("--oom_retry_num_beams", default="2", type=str)
    ap.add_argument("--oom_retry_num_return_seqs", default="1", type=str)
    ap.add_argument("--oom_retry_sample_return_seqs", default="1", type=str)
    ap.add_argument("--oom_retry_fallback_sampling", default="no", choices=["yes", "no"])
    ap.add_argument("--oom_retry_chunk_stride_lines_eval", default="20", type=str)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    dataset_path = (project_root / args.dataset_path).resolve() if not os.path.isabs(args.dataset_path) else Path(args.dataset_path)
    train_script = (project_root / args.train_script).resolve() if not os.path.isabs(args.train_script) else Path(args.train_script)
    eval_script = (project_root / args.eval_script).resolve() if not os.path.isabs(args.eval_script) else Path(args.eval_script)
    existing_ckpt_root = None
    if args.existing_ckpt_root.strip():
        existing_ckpt_root = (project_root / args.existing_ckpt_root).resolve() if not os.path.isabs(args.existing_ckpt_root) else Path(args.existing_ckpt_root)
    eval_model_source = args.eval_model_variation_seq2seq.strip() or args.model_variation
    if eval_model_source and os.path.isabs(eval_model_source):
        eval_model_source = str(Path(eval_model_source).resolve())
    elif eval_model_source:
        eval_model_source = str((project_root / eval_model_source).resolve()) if os.path.exists(project_root / eval_model_source) else eval_model_source

    if not train_script.exists():
        raise FileNotFoundError(f"训练脚本不存在：{train_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"评估脚本不存在：{eval_script}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    if not seeds:
        seeds = DEFAULT_SEEDS

    paths = build_paths(output_root)
    write_json(
        output_root / "run_config.json",
        {
            "project_root": str(project_root),
            "output_root": str(output_root),
            "dataset_path": str(dataset_path),
            "train_script": str(train_script),
            "eval_script": str(eval_script),
            "seeds": seeds,
            "args": vars(args),
        },
    )

    log(f"project_root = {project_root}")
    log(f"train_script = {train_script}")
    log(f"eval_script = {eval_script}")
    log(f"output_root = {output_root}")
    log(f"seeds = {seeds}")
    if existing_ckpt_root is not None:
        log(f"existing_ckpt_root = {existing_ckpt_root}")
    log(f"eval_model_source = {eval_model_source}")

    test_csv_map = prepare_test_csvs(dataset_path, paths.tests, seeds)
    log(f"已生成测试集切分文件：{paths.tests}")

    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    per_seed_rows: List[dict] = []

    for seed in seeds:
        log(f"开始 seed={seed}")

        ckpt_dir = paths.ckpts / f"seed{seed}"
        ensure_dir(ckpt_dir)
        best_weights = ckpt_dir / "best_weights.pt"
        test_csv = test_csv_map[seed]
        run_name = f"seed{seed}_paper_compare"
        train_log = paths.logs / f"train_seed{seed}.log"
        eval_log = paths.logs / f"eval_seed{seed}.log"
        summary_csv = paths.eval_outputs / f"{run_name}_summary.csv"
        details_csv = paths.eval_outputs / f"{run_name}_details.csv"

        if args.resume_if_exists == "yes" and not best_weights.exists():
            maybe_reuse_existing_checkpoint(seed, ckpt_dir, existing_ckpt_root)

        if args.skip_train == "no":
            if args.resume_if_exists == "yes" and best_weights.exists():
                log(f"seed={seed} 检测到已有权重，跳过训练：{best_weights}")
            else:
                train_args = {
                    "seed": str(seed),
                    "FINE_TUNE": "yes",
                    "model_variation": args.model_variation,
                    "checkpoint_dir": str(ckpt_dir),
                    "INCLUDE_NEGATIVES": args.include_negatives,
                    "NEGATIVE_RATIO": args.negative_ratio,
                    "NO_VULN_TOKEN": args.no_vuln_token,
                    "ROUGE_ON_VULN_ONLY": args.rouge_on_vuln_only,
                    "MAX_INPUT_LEN": args.max_input_len,
                    "MAX_TARGET_LEN_CAP": args.max_target_len_cap,
                    "BATCH_SIZE": args.batch_size,
                    "GRAD_ACCUM_STEPS": args.grad_accum_steps,
                    "NUM_EPOCHS": args.num_epochs,
                    "PATIENCE": args.patience,
                    "LR": args.lr,
                    "CHUNK_TRAINING": args.chunk_training,
                    "CHUNK_MAX_TOKENS": args.chunk_max_tokens,
                    "CHUNK_STRIDE_LINES": args.chunk_stride_lines_train,
                    "CHUNK_PREFIX_LINES": args.chunk_prefix_lines,
                    "CHUNK_PREFIX_MAX_TOKENS": args.chunk_prefix_max_tokens,
                    "NEG_CHUNKS_PER_FUNC": args.neg_chunks_per_func,
                    "QUICK_TEST": "no",
                }
                cmd = [args.python_bin, str(train_script)] + flatten_args(train_args)
                rc = run_and_log(cmd, train_log, project_root, env)
                if rc != 0:
                    raise RuntimeError(f"训练失败（seed={seed}, 退出码={rc}），详见日志：{train_log}")

        if args.skip_eval == "no":
            if args.resume_if_exists == "yes" and summary_csv.exists() and details_csv.exists():
                log(f"seed={seed} 检测到已有评估结果，跳过评估：{summary_csv.name}")
            else:
                eval_args = {
                    "seed": str(seed),
                    "checkpoint_dir_seq2seq": str(ckpt_dir),
                    "model_variation_seq2seq": eval_model_source,
                    "output_dir": str(paths.eval_outputs),
                    "run_name": run_name,
                    "test_path": str(test_csv),
                    "MAX_INPUT_LEN": args.max_input_len,
                    "MAX_TARGET_LEN": args.max_target_len_eval,
                    "NUM_BEAMS": args.num_beams,
                    "NUM_RETURN_SEQS": args.num_return_seqs,
                    "FALLBACK_SAMPLING": args.fallback_sampling,
                    "SAMPLE_RETURN_SEQS": args.sample_return_seqs,
                    "TOP_P": args.top_p,
                    "TEMPERATURE": args.temperature,
                    "CHUNK_LONG_FUNCS": args.chunk_long_funcs,
                    "CHUNK_STRIDE_LINES": args.chunk_stride_lines_eval,
                    "SIMILARITY_REPLACEMENT": args.similarity_replacement,
                    "SIM_WEIGHT": args.sim_weight,
                    "SIM_THRESHOLD": args.sim_threshold,
                    "DEDUP": args.dedup,
                    "RERANK": args.rerank,
                    "K": str(args.k),
                    "EVAL_MODE": args.eval_mode,
                    "REMOVE_MISSING_LINE_LABELS": args.remove_missing_line_labels,
                    "NO_VULN_TOKEN": args.no_vuln_token,
                    "HANDLE_NO_VULN_TOKEN": "yes",
                    "sort_by_lines": args.sort_by_lines,
                }
                cmd = [args.python_bin, str(eval_script)] + flatten_args(eval_args)
                rc = run_and_log(cmd, eval_log, project_root, env)

                if rc == -9 and args.retry_eval_on_oom == "yes":
                    log(f"seed={seed} 评估疑似 OOM/SIGKILL，自动用更轻参数重试一次")
                    retry_args = dict(eval_args)
                    retry_args.update({
                        "NUM_BEAMS": args.oom_retry_num_beams,
                        "NUM_RETURN_SEQS": args.oom_retry_num_return_seqs,
                        "FALLBACK_SAMPLING": args.oom_retry_fallback_sampling,
                        "SAMPLE_RETURN_SEQS": args.oom_retry_sample_return_seqs,
                        "CHUNK_STRIDE_LINES": args.oom_retry_chunk_stride_lines_eval,
                    })
                    cmd_retry = [args.python_bin, str(eval_script)] + flatten_args(retry_args)
                    rc = run_and_log(cmd_retry, eval_log, project_root, env)

                if rc != 0:
                    raise RuntimeError(
                        f"评估失败（seed={seed}, 退出码={rc}），详见日志：{eval_log}。"
                        f"若是 OOM，可继续降低 --num_beams / --num_return_seqs / --chunk_stride_lines_eval。"
                    )

        if not summary_csv.exists():
            raise FileNotFoundError(f"缺少 summary.csv：{summary_csv}")
        if not details_csv.exists():
            raise FileNotFoundError(f"缺少 details.csv：{details_csv}")

        summary_df = pd.read_csv(summary_csv)
        if summary_df.empty:
            raise ValueError(f"空的 summary.csv：{summary_csv}")
        srow = summary_df.iloc[0].to_dict()
        pr = compute_precision_recall_from_details(details_csv, args.k)

        row = {
            "seed": seed,
            "A@10": float(srow.get("A@K", srow.get("accuracy", 0.0))),
            "P@10": float(pr[f"P@{args.k}"]),
            "R@10": float(pr[f"R@{args.k}"]),
            "MRR@10": float(srow.get("MRR@K", srow.get("MRR", 0.0))),
            "MAP@10": float(srow.get("MAP@K", srow.get("MAP", 0.0))),
            "Median IFA": float(srow.get("IFA_median", srow.get("Median_IFA", 0.0))),
            "IFA mean": float(srow.get("IFA_mean", 0.0)),
            "Effort@20%Recall": float(srow.get("Effort@20%Recall", srow.get("EffortRecall", 0.0))),
            "Recall@1%LOC": float(srow.get("Recall@1%LOC", srow.get("RecallLoc", 0.0))),
            "details_csv": str(details_csv),
            "summary_csv": str(summary_csv),
            "checkpoint_dir": str(ckpt_dir),
        }
        per_seed_rows.append(row)
        log(
            "seed={} 完成 | A@10={:.4f} P@10={:.4f} R@10={:.4f} MRR@10={:.4f} MAP@10={:.4f} Eff20={:.4f} R1LOC={:.4f}".format(
                seed,
                row["A@10"],
                row["P@10"],
                row["R@10"],
                row["MRR@10"],
                row["MAP@10"],
                row["Effort@20%Recall"],
                row["Recall@1%LOC"],
            )
        )

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)
    per_seed_path = output_root / "per_seed_metrics.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    metric_order = [
        "A@10",
        "P@10",
        "R@10",
        "MRR@10",
        "MAP@10",
        "Median IFA",
        "Effort@20%Recall",
        "Recall@1%LOC",
    ]

    mean_std_rows = []
    compare_rows = []
    for metric in metric_order:
        vals = per_seed_df[metric].astype(float).tolist()
        m = mean_safe(vals)
        s = std_safe(vals)
        mean_std_rows.append({"Metric": metric, "mean": m, "std": s})

        paper = PAPER_METRICS[metric]
        delta = m - paper
        compare_rows.append({
            "Metric": metric,
            "Your mean raw": m,
            "Your std raw": s,
            "Paper raw": paper,
            "Delta raw": delta,
            "Your mean": fmt_metric(metric, m),
            "Your std": fmt_metric(metric, s),
            "Paper": fmt_metric(metric, paper),
            "Delta": fmt_metric(metric, delta) if metric != "Median IFA" else f"{delta:.4f}",
        })

    mean_std_df = pd.DataFrame(mean_std_rows)
    compare_df = pd.DataFrame(compare_rows)

    mean_std_path = output_root / "mean_std_metrics.csv"
    compare_csv_path = output_root / "compare_with_paper.csv"
    compare_md_path = output_root / "compare_with_paper.md"

    mean_std_df.to_csv(mean_std_path, index=False)
    compare_df.to_csv(compare_csv_path, index=False)
    compare_md_path.write_text(make_markdown_compare(compare_df), encoding="utf-8")

    log("=" * 80)
    log("10-seed 平均结果：")
    for _, r in mean_std_df.iterrows():
        metric = r["Metric"]
        log(f"{metric}: mean={fmt_metric(metric, float(r['mean']))}, std={fmt_metric(metric, float(r['std']))}")
    log("=" * 80)
    log(f"per-seed 指标: {per_seed_path}")
    log(f"mean/std 汇总: {mean_std_path}")
    log(f"与论文对比: {compare_csv_path}")
    log(f"Markdown 对比表: {compare_md_path}")
    log("说明：这里只是与论文发表均值做比较；若要做显著性检验，请再准备一个同协议 baseline 的 10-seed 结果。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
