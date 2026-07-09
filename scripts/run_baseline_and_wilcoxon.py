#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行原始 LocVul baseline（Self-Attention）10-seed，并与改进版 per_seed_metrics.csv 做配对 Wilcoxon signed-rank test。

适配当前项目：
- baseline 脚本：vulnDet_pipeline.py
- 改进版 per-seed 指标：run_10seed_compare_with_paper_stable_fixed.py 产生的 per_seed_metrics.csv

输出：
- output_root/logs/seed*.log
- output_root/checkpoints/seed*/best_weights.pt
- output_root/baseline_per_seed_metrics.csv
- output_root/baseline_mean_std_metrics.csv
- output_root/aligned_baseline_vs_improved.csv
- output_root/wilcoxon_results.csv
- output_root/wilcoxon_results.md
- output_root/run_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_SEEDS = list(range(10))
METRIC_ORDER = [
    "A@10",
    "P@10",
    "R@10",
    "MRR@10",
    "MAP@10",
    "Median IFA",
    "Effort@20%Recall",
    "Recall@1%LOC",
]

# 与论文 / 原 statistical_test.py 口径一致：
# - A/P/R/MRR/MAP/Recall@1%LOC 越大越好 -> alternative='greater'（improved > baseline）
# - Median IFA / Effort@20%Recall 越小越好 -> alternative='less'（improved < baseline）
ALTERNATIVE_MAP = {
    "A@10": "greater",
    "P@10": "greater",
    "R@10": "greater",
    "MRR@10": "greater",
    "MAP@10": "greater",
    "Median IFA": "less",
    "Effort@20%Recall": "less",
    "Recall@1%LOC": "greater",
}


@dataclass
class Paths:
    root: Path
    logs: Path
    ckpts: Path


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def std_safe(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(statistics.stdev(vals))


def fmt_metric(metric: str, value: float) -> str:
    if metric == "Median IFA":
        return f"{value:.4f}"
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value) * 100:.2f}%"


def build_paths(output_root: Path) -> Paths:
    paths = Paths(
        root=output_root,
        logs=output_root / "logs",
        ckpts=output_root / "checkpoints",
    )
    for p in [paths.root, paths.logs, paths.ckpts]:
        ensure_dir(p)
    return paths


def candidate_existing_checkpoint_paths(seed: int, existing_ckpt_root: Path) -> List[Path]:
    return [
        existing_ckpt_root / f"seed{seed}" / "best_weights.pt",
        existing_ckpt_root / "checkpoints" / f"seed{seed}" / "best_weights.pt",
        existing_ckpt_root / f"seed{seed}" / "checkpoint" / "best_weights.pt",
        existing_ckpt_root / f"seed{seed}" / "pytorch_model.bin",
    ]


def maybe_reuse_existing_checkpoint(seed: int, dst_seed_ckpt_dir: Path, existing_ckpt_root: Optional[Path]) -> bool:
    if existing_ckpt_root is None:
        return False

    dst = dst_seed_ckpt_dir / "best_weights.pt"
    if dst.exists():
        return True

    for src in candidate_existing_checkpoint_paths(seed, existing_ckpt_root):
        if src.exists() and src.name == "best_weights.pt":
            ensure_dir(dst_seed_ckpt_dir)
            shutil.copy2(src, dst)
            log(f"seed={seed} 复用已有 checkpoint: {src} -> {dst}")
            return True

    return False


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"


def _find_last_float(text: str, patterns: List[str]) -> Optional[float]:
    for pat in patterns:
        ms = re.findall(pat, text, flags=re.MULTILINE)
        if ms:
            try:
                return float(ms[-1])
            except Exception:
                pass
    return None


def parse_baseline_metrics_from_log(log_path: Path) -> Dict[str, float]:
    text = read_text(log_path)

    # 优先解析百分比打印块；若没有，再退回原始小数打印块
    a10 = _find_last_float(text, [rf"Top-10 Accuracy:\s*{FLOAT}%", rf"Top-10 Accuracy:\s*{FLOAT}(?!%)"])
    p10 = _find_last_float(text, [rf"Top-10 Precision:\s*{FLOAT}%", rf"Top-10 Precision:\s*{FLOAT}(?!%)"])
    r10 = _find_last_float(text, [rf"Top-10 Recall:\s*{FLOAT}%", rf"Top-10 Recall:\s*{FLOAT}(?!%)"])
    mrr = _find_last_float(text, [rf"Top-10 MRR:\s*{FLOAT}%", rf"Top-10 Reciprocal Rank:\s*{FLOAT}(?!%)", rf"Top-10 MRR:\s*{FLOAT}(?!%)"])
    map10 = _find_last_float(text, [rf"Top-10 MAP:\s*{FLOAT}%", rf"Top-10 MAP:\s*{FLOAT}(?!%)"])
    ifa = _find_last_float(text, [rf"Median IFA:\s*{FLOAT}"])
    effort = _find_last_float(text, [rf"Effort@20%Recall:\s*{FLOAT}%", rf"Effort@20%Recall:\s*{FLOAT}(?!%)"])
    recall_loc = _find_last_float(text, [rf"Recall@1%LOC:\s*{FLOAT}%", rf"Recall@1%LOC:\s*{FLOAT}(?!%)"])

    if None in [a10, p10, r10, mrr, map10, ifa, effort, recall_loc]:
        missing = {
            "A@10": a10,
            "P@10": p10,
            "R@10": r10,
            "MRR@10": mrr,
            "MAP@10": map10,
            "Median IFA": ifa,
            "Effort@20%Recall": effort,
            "Recall@1%LOC": recall_loc,
        }
        raise ValueError(f"无法从日志解析完整指标: {log_path}\n{missing}")

    def pct_or_raw(v: float) -> float:
        # 71.4 -> 0.714; 0.714 -> 0.714
        return v / 100.0 if v > 1.0 else v

    return {
        "A@10": pct_or_raw(a10),
        "P@10": pct_or_raw(p10),
        "R@10": pct_or_raw(r10),
        "MRR@10": pct_or_raw(mrr),
        "MAP@10": pct_or_raw(map10),
        "Median IFA": ifa,
        "Effort@20%Recall": pct_or_raw(effort),
        "Recall@1%LOC": pct_or_raw(recall_loc),
    }


def compute_mean_std(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_ORDER:
        vals = per_seed_df[metric].astype(float).tolist()
        rows.append({
            "Metric": metric,
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "std": std_safe(vals),
            "n_success": len(vals),
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(improved_vals: List[float], baseline_vals: List[float], alternative: str) -> tuple[float, float]:
    if len(improved_vals) != len(baseline_vals):
        raise ValueError("Wilcoxon 需要等长配对样本")
    if len(improved_vals) == 0:
        return 0.0, 1.0

    diffs = [a - b for a, b in zip(improved_vals, baseline_vals)]
    if all(abs(x) < 1e-15 for x in diffs):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(improved_vals, baseline_vals, alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        # 常见于全零差值等特殊情形
        return 0.0, 1.0


def build_wilcoxon_table(aligned_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_ORDER:
        alt = ALTERNATIVE_MAP[metric]
        imp = aligned_df[f"improved::{metric}"].astype(float).tolist()
        base = aligned_df[f"baseline::{metric}"].astype(float).tolist()
        stat, p = paired_wilcoxon(imp, base, alt)
        imp_mean = sum(imp) / len(imp) if imp else 0.0
        base_mean = sum(base) / len(base) if base else 0.0
        delta = imp_mean - base_mean
        rows.append({
            "Metric": metric,
            "Alternative": alt,
            "N": len(imp),
            "Improved mean raw": imp_mean,
            "Baseline mean raw": base_mean,
            "Delta raw": delta,
            "Wilcoxon statistic": stat,
            "p-value": p,
            "Significant(p<0.05)": "yes" if p < 0.05 else "no",
            "Improved mean": fmt_metric(metric, imp_mean),
            "Baseline mean": fmt_metric(metric, base_mean),
            "Delta": fmt_metric(metric, delta) if metric != "Median IFA" else f"{delta:.4f}",
        })
    return pd.DataFrame(rows)


def make_markdown_summary(wilcox_df: pd.DataFrame, baseline_mean_std: pd.DataFrame, improved_mean_std: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# 原始 LocVul baseline vs 改进版：Wilcoxon 显著性检验")
    lines.append("")
    lines.append("## 1. baseline 10-seed 均值")
    lines.append("")
    lines.append("| Metric | mean | std | n_success |")
    lines.append("|---|---:|---:|---:|")
    for _, r in baseline_mean_std.iterrows():
        lines.append(f"| {r['Metric']} | {fmt_metric(r['Metric'], float(r['mean']))} | {fmt_metric(r['Metric'], float(r['std']))} | {int(r['n_success'])} |")

    lines.append("")
    lines.append("## 2. Wilcoxon 结果")
    lines.append("")
    lines.append("| Metric | Alternative | Improved mean | Baseline mean | Delta | p-value | Significant |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for _, r in wilcox_df.iterrows():
        lines.append(
            f"| {r['Metric']} | {r['Alternative']} | {r['Improved mean']} | {r['Baseline mean']} | {r['Delta']} | {float(r['p-value']):.6g} | {r['Significant(p<0.05)']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", type=str)
    ap.add_argument("--output_root", default="./runs_baseline_and_wilcoxon", type=str)
    ap.add_argument("--baseline_script", default="./vulnDet_pipeline.py", type=str)
    ap.add_argument("--improved_per_seed_csv", required=True, type=str,
                    help="改进版 run_10seed_compare_with_paper_stable_fixed.py 生成的 per_seed_metrics.csv")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)
    ap.add_argument("--python_bin", default=sys.executable, type=str)
    ap.add_argument("--resume_if_exists", default="yes", choices=["yes", "no"])
    ap.add_argument("--fine_tune", default="yes", choices=["yes", "no"],
                    help="yes=每个 seed 重新训练 baseline；no=仅加载 checkpoint 推理")
    ap.add_argument("--existing_ckpt_root", default="", type=str,
                    help="当 fine_tune=no 时，可指定已有 baseline checkpoint 根目录")
    ap.add_argument("--model_variation", default="microsoft/codebert-base", type=str)
    ap.add_argument("--sampling", default="no", choices=["yes", "no"])
    ap.add_argument("--remove_missing_line_labels", default="yes", choices=["yes", "no"])
    ap.add_argument("--explainer", default="ATTENTION", choices=["ATTENTION", "LIME"])
    ap.add_argument("--explain_only_tp", default="no", choices=["yes", "no"])
    ap.add_argument("--sort_by_lines", default="yes", choices=["yes", "no"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    baseline_script = (project_root / args.baseline_script).resolve() if not os.path.isabs(args.baseline_script) else Path(args.baseline_script)
    improved_csv = (project_root / args.improved_per_seed_csv).resolve() if not os.path.isabs(args.improved_per_seed_csv) else Path(args.improved_per_seed_csv)
    existing_ckpt_root = None
    if args.existing_ckpt_root.strip():
        existing_ckpt_root = (project_root / args.existing_ckpt_root).resolve() if not os.path.isabs(args.existing_ckpt_root) else Path(args.existing_ckpt_root)

    if not baseline_script.exists():
        raise FileNotFoundError(f"baseline 脚本不存在: {baseline_script}")
    if not improved_csv.exists():
        raise FileNotFoundError(f"改进版 per_seed_metrics.csv 不存在: {improved_csv}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    if not seeds:
        seeds = DEFAULT_SEEDS

    paths = build_paths(output_root)
    write_json(output_root / "run_config.json", {
        "project_root": str(project_root),
        "baseline_script": str(baseline_script),
        "improved_per_seed_csv": str(improved_csv),
        "seeds": seeds,
        "args": vars(args),
    })

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    baseline_rows: List[Dict[str, float]] = []

    for seed in seeds:
        seed_ckpt_dir = paths.ckpts / f"seed{seed}"
        seed_log = paths.logs / f"seed{seed}.log"
        seed_metrics_csv = output_root / f"seed{seed}_metrics.csv"

        if args.resume_if_exists == "yes" and seed_metrics_csv.exists():
            row = pd.read_csv(seed_metrics_csv).iloc[0].to_dict()
            baseline_rows.append(row)
            log(f"seed={seed} 已存在 metrics，跳过运行")
            continue

        if args.fine_tune == "no":
            maybe_reuse_existing_checkpoint(seed, seed_ckpt_dir, existing_ckpt_root)
            if not (seed_ckpt_dir / "best_weights.pt").exists():
                raise FileNotFoundError(
                    f"seed={seed} 缺少 baseline checkpoint。当前 fine_tune=no，且未找到 {seed_ckpt_dir / 'best_weights.pt'}"
                )

        cmd = [
            args.python_bin,
            str(baseline_script),
            "--seed", str(seed),
            "--FINE_TUNE", args.fine_tune,
            "--model_variation", args.model_variation,
            "--checkpoint_dir", str(seed_ckpt_dir),
            "--sampling", args.sampling,
            "--REMOVE_MISSING_LINE_LABELS", args.remove_missing_line_labels,
            "--EXPLAINER", args.explainer,
            "--EXPLAIN_ONLY_TP", args.explain_only_tp,
            "--sort_by_lines", args.sort_by_lines,
        ]

        log(f"seed={seed} 开始运行 baseline")
        rc = run_and_log(cmd, seed_log, project_root, env)
        if rc != 0:
            raise RuntimeError(f"seed={seed} baseline 运行失败，returncode={rc}，日志: {seed_log}")

        metrics = parse_baseline_metrics_from_log(seed_log)
        row = {"seed": seed, **metrics}
        pd.DataFrame([row]).to_csv(seed_metrics_csv, index=False)
        baseline_rows.append(row)
        log(f"seed={seed} 完成: {row}")

    baseline_df = pd.DataFrame(baseline_rows).sort_values("seed").reset_index(drop=True)
    baseline_per_seed_path = output_root / "baseline_per_seed_metrics.csv"
    baseline_df.to_csv(baseline_per_seed_path, index=False)

    baseline_mean_std = compute_mean_std(baseline_df)
    baseline_mean_std_path = output_root / "baseline_mean_std_metrics.csv"
    baseline_mean_std.to_csv(baseline_mean_std_path, index=False)

    improved_df = pd.read_csv(improved_csv)
    if "seed" not in improved_df.columns:
        raise ValueError(f"改进版 per_seed_metrics.csv 缺少 seed 列: {improved_csv}")

    # 只保留共同 seeds，保证配对 Wilcoxon 合法
    common_seeds = sorted(set(baseline_df["seed"].tolist()) & set(improved_df["seed"].tolist()))
    if not common_seeds:
        raise ValueError("baseline 与 improved 没有共同 seed，无法做配对检验")

    baseline_aligned = baseline_df[baseline_df["seed"].isin(common_seeds)].sort_values("seed").reset_index(drop=True)
    improved_aligned = improved_df[improved_df["seed"].isin(common_seeds)].sort_values("seed").reset_index(drop=True)

    if baseline_aligned["seed"].tolist() != improved_aligned["seed"].tolist():
        raise ValueError("共同 seeds 对齐失败")

    aligned_cols = {"seed": baseline_aligned["seed"]}
    for metric in METRIC_ORDER:
        aligned_cols[f"baseline::{metric}"] = baseline_aligned[metric].astype(float)
        aligned_cols[f"improved::{metric}"] = improved_aligned[metric].astype(float)
    aligned_df = pd.DataFrame(aligned_cols)
    aligned_path = output_root / "aligned_baseline_vs_improved.csv"
    aligned_df.to_csv(aligned_path, index=False)

    wilcox_df = build_wilcoxon_table(aligned_df)
    wilcox_path = output_root / "wilcoxon_results.csv"
    wilcox_df.to_csv(wilcox_path, index=False)

    improved_mean_std = compute_mean_std(improved_aligned)
    md = make_markdown_summary(wilcox_df, baseline_mean_std, improved_mean_std)
    md_path = output_root / "wilcoxon_results.md"
    md_path.write_text(md, encoding="utf-8")

    log("=" * 80)
    log(f"baseline per-seed: {baseline_per_seed_path}")
    log(f"baseline mean/std: {baseline_mean_std_path}")
    log(f"aligned paired data: {aligned_path}")
    log(f"Wilcoxon results: {wilcox_path}")
    log(f"Wilcoxon markdown: {md_path}")
    log("完成。")
    log("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
