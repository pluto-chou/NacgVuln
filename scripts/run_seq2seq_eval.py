#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版 Seq2Seq 评估批跑增强脚本（仅评估，不训练）。

目标：
1) 复用已有 seed0..seed9/best_weights.pt
2) 调用 seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py 逐 seed 评估
3) 失败不中断整批，自动记录失败并继续后续 seed
4) 支持 resume：已有 *_summary.csv + *_details.csv 直接跳过
5) 评估结束后汇总成功 seeds 的 per-seed / mean-std / compare-with-paper

说明：
- 兼容本地离线 CodeT5 目录（model_variation_seq2seq 指向完整本地目录）
- 默认只做评估；不涉及训练
- 如果某个 seed 首次评估失败，可按配置自动重试一次或两次，并可使用更轻量参数
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
from typing import Dict, List, Optional, Tuple

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


def normalize_line(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


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


def fmt_metric(metric: str, value: float) -> str:
    if metric == "Median IFA":
        return f"{value:.4f}"
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value) * 100:.2f}%"


def make_markdown_compare(df: pd.DataFrame) -> str:
    lines = [
        "# 改进版 10-seed 评估结果（仅成功 seeds）",
        "",
        "| Metric | Your mean | Your std | Paper | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Metric']} | {r['Your mean']} | {r['Your std']} | {r['Paper']} | {r['Delta']} |"
        )
    return "\n".join(lines) + "\n"


def prepare_test_csvs(dataset_path: Path, out_dir: Path, seeds: List[int]) -> Dict[int, Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"缺少数据集：{dataset_path}。请先准备 data/dataset.csv")

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
        cnt = 0
        for x in ranked:
            if normalize_line(x) in gt_set:
                cnt += 1
        precisions.append(cnt / k)
        recalls.append(cnt / len(gt_set))

    return {f"P@{k}": mean_safe(precisions), f"R@{k}": mean_safe(recalls)}


def maybe_reuse_existing_checkpoint(seed: int, seed_ckpt_dir: Path, existing_ckpt_root: Optional[Path]) -> bool:
    if existing_ckpt_root is None:
        return False
    src = existing_ckpt_root / f"seed{seed}" / "best_weights.pt"
    dst = seed_ckpt_dir / "best_weights.pt"
    if dst.exists():
        return True
    if src.exists():
        ensure_dir(seed_ckpt_dir)
        shutil.copy2(src, dst)
        log(f"seed={seed} 复用已有 checkpoint: {src} -> {dst}")
        return True
    return False


def parse_summary(summary_csv: Path, details_csv: Path, k: int) -> dict:
    sdf = pd.read_csv(summary_csv)
    if sdf.empty:
        raise ValueError(f"summary 为空：{summary_csv}")
    row = sdf.iloc[0].to_dict()
    pr = compute_precision_recall_from_details(details_csv, k)
    return {
        "seed": int(row.get("seed", -1)),
        f"A@{k}": float(row["A@K"]),
        f"P@{k}": float(pr[f"P@{k}"]),
        f"R@{k}": float(pr[f"R@{k}"]),
        f"MRR@{k}": float(row["MRR@K"]),
        f"MAP@{k}": float(row["MAP@K"]),
        "Median IFA": float(row["IFA_median"]),
        "Effort@20%Recall": float(row["Effort@20%Recall"]),
        "Recall@1%LOC": float(row["Recall@1%LOC"]),
        "n_rows": int(row.get("n_rows", 0)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_root", default="./runs_seq2seq_eval_v3_batch")
    ap.add_argument("--dataset_path", default="./data/dataset.csv")
    ap.add_argument("--eval_script", default="./seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--hf_endpoint", default="")
    ap.add_argument("--existing_ckpt_root", default="", help="seed0/seed1/.../best_weights.pt")
    ap.add_argument("--model_variation_seq2seq", default="Salesforce/codet5-base")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--resume_if_exists", default="yes", choices=["yes", "no"])
    ap.add_argument("--k", default=10, type=int)

    # 评估参数
    ap.add_argument("--max_input_len", default="512")
    ap.add_argument("--max_target_len", default="256")
    ap.add_argument("--num_beams", default="1")
    ap.add_argument("--num_return_seqs", default="1")
    ap.add_argument("--fallback_sampling", default="no", choices=["yes", "no"])
    ap.add_argument("--sample_return_seqs", default="1")
    ap.add_argument("--top_p", default="0.95")
    ap.add_argument("--temperature", default="0.7")
    ap.add_argument("--chunk_long_funcs", default="yes", choices=["yes", "no"])
    ap.add_argument("--chunk_stride_lines_eval", default="20")
    ap.add_argument("--similarity_replacement", default="yes", choices=["yes", "no"])
    ap.add_argument("--sim_weight", default="1.0")
    ap.add_argument("--sim_threshold", default="0.0")
    ap.add_argument("--dedup", default="yes", choices=["yes", "no"])
    ap.add_argument("--rerank", default="no", choices=["yes", "no"])
    ap.add_argument("--eval_mode", default="locvul", choices=["locvul", "gen_only"])
    ap.add_argument("--remove_missing_line_labels", default="yes", choices=["yes", "no"])
    ap.add_argument("--eval_only_vuln", default="yes", choices=["yes", "no"])
    ap.add_argument("--sort_by_lines", default="yes", choices=["yes", "no"])
    ap.add_argument("--no_vuln_token", default="<NO_VULN>")
    ap.add_argument("--handle_no_vuln_token", default="yes", choices=["yes", "no"])

    # 失败处理
    ap.add_argument("--continue_on_error", default="yes", choices=["yes", "no"])
    ap.add_argument("--strict_failure", default="no", choices=["yes", "no"], help="若有任意 seed 失败则退出码非 0")
    ap.add_argument("--max_retries_per_seed", default=2, type=int)
    ap.add_argument("--retry_disable_similarity", default="yes", choices=["yes", "no"])
    ap.add_argument("--retry_disable_rerank", default="yes", choices=["yes", "no"])
    ap.add_argument("--retry_num_beams", default="1")
    ap.add_argument("--retry_num_return_seqs", default="1")
    ap.add_argument("--retry_fallback_sampling", default="no", choices=["yes", "no"])
    ap.add_argument("--retry_sample_return_seqs", default="1")
    ap.add_argument("--retry_chunk_stride_lines_eval", default="10")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    dataset_path = (project_root / args.dataset_path).resolve() if not os.path.isabs(args.dataset_path) else Path(args.dataset_path)
    eval_script = (project_root / args.eval_script).resolve() if not os.path.isabs(args.eval_script) else Path(args.eval_script)
    existing_ckpt_root = None
    if args.existing_ckpt_root.strip():
        existing_ckpt_root = (project_root / args.existing_ckpt_root).resolve() if not os.path.isabs(args.existing_ckpt_root) else Path(args.existing_ckpt_root)

    if not eval_script.exists():
        raise FileNotFoundError(f"评估脚本不存在：{eval_script}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        seeds = DEFAULT_SEEDS

    paths = build_paths(output_root)
    write_json(output_root / "run_config.json", {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "dataset_path": str(dataset_path),
        "eval_script": str(eval_script),
        "seeds": seeds,
        "args": vars(args),
    })

    log(f"project_root = {project_root}")
    log(f"eval_script = {eval_script}")
    log(f"output_root = {output_root}")
    log(f"seeds = {seeds}")
    if existing_ckpt_root is not None:
        log(f"existing_ckpt_root = {existing_ckpt_root}")

    test_csv_map = prepare_test_csvs(dataset_path, paths.tests, seeds)
    log(f"已生成测试集切分文件：{paths.tests}")

    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    per_seed_rows: List[dict] = []
    fail_rows: List[dict] = []

    for seed in seeds:
        log(f"开始 seed={seed}")
        ckpt_dir = paths.ckpts / f"seed{seed}"
        ensure_dir(ckpt_dir)
        best_weights = ckpt_dir / "best_weights.pt"
        test_csv = test_csv_map[seed]
        run_name = f"seed{seed}_paper_compare"
        eval_log = paths.logs / f"eval_seed{seed}.log"
        summary_csv = paths.eval_outputs / f"{run_name}_summary.csv"
        details_csv = paths.eval_outputs / f"{run_name}_details.csv"

        if args.resume_if_exists == "yes" and not best_weights.exists():
            maybe_reuse_existing_checkpoint(seed, ckpt_dir, existing_ckpt_root)

        if not best_weights.exists():
            msg = f"缺少 best_weights.pt：{best_weights}"
            log(f"seed={seed} 失败 | {msg}")
            fail_rows.append({"seed": seed, "stage": "precheck", "returncode": None, "reason": msg, "log": str(eval_log)})
            if args.continue_on_error == "no":
                break
            continue

        if args.resume_if_exists == "yes" and summary_csv.exists() and details_csv.exists():
            log(f"seed={seed} 检测到已有评估结果，跳过评估：{summary_csv.name}")
            try:
                row = parse_summary(summary_csv, details_csv, args.k)
                per_seed_rows.append(row)
                log(
                    "seed={} 已复用 | A@{}={:.4f} P@{}={:.4f} R@{}={:.4f} MRR@{}={:.4f} MAP@{}={:.4f} Eff20={:.4f} R1LOC={:.4f}".format(
                        seed, args.k, row[f"A@{args.k}"], args.k, row[f"P@{args.k}"], args.k, row[f"R@{args.k}"], args.k, row[f"MRR@{args.k}"], args.k, row[f"MAP@{args.k}"], row["Effort@20%Recall"], row["Recall@1%LOC"],
                    )
                )
                continue
            except Exception as e:
                log(f"seed={seed} 已有 summary/details 解析失败，将重新评估：{repr(e)}")

        base_eval_args = {
            "seed": str(seed),
            "checkpoint_dir_seq2seq": str(ckpt_dir),
            "model_variation_seq2seq": args.model_variation_seq2seq,
            "output_dir": str(paths.eval_outputs),
            "run_name": run_name,
            "test_path": str(test_csv),
            "MAX_INPUT_LEN": args.max_input_len,
            "MAX_TARGET_LEN": args.max_target_len,
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
            "EVAL_ONLY_VULN": args.eval_only_vuln,
            "NO_VULN_TOKEN": args.no_vuln_token,
            "HANDLE_NO_VULN_TOKEN": args.handle_no_vuln_token,
            "sort_by_lines": args.sort_by_lines,
        }

        attempts: List[Tuple[str, Dict[str, str]]] = [("base", dict(base_eval_args))]
        if args.max_retries_per_seed >= 1:
            retry1 = dict(base_eval_args)
            retry1.update({
                "NUM_BEAMS": args.retry_num_beams,
                "NUM_RETURN_SEQS": args.retry_num_return_seqs,
                "FALLBACK_SAMPLING": args.retry_fallback_sampling,
                "SAMPLE_RETURN_SEQS": args.retry_sample_return_seqs,
                "CHUNK_STRIDE_LINES": args.retry_chunk_stride_lines_eval,
            })
            attempts.append(("retry1", retry1))
        if args.max_retries_per_seed >= 2:
            retry2 = dict(base_eval_args)
            retry2.update({
                "NUM_BEAMS": args.retry_num_beams,
                "NUM_RETURN_SEQS": args.retry_num_return_seqs,
                "FALLBACK_SAMPLING": args.retry_fallback_sampling,
                "SAMPLE_RETURN_SEQS": args.retry_sample_return_seqs,
                "CHUNK_STRIDE_LINES": args.retry_chunk_stride_lines_eval,
            })
            if args.retry_disable_similarity == "yes":
                retry2["SIMILARITY_REPLACEMENT"] = "no"
            if args.retry_disable_rerank == "yes":
                retry2["RERANK"] = "no"
            attempts.append(("retry2", retry2))

        success = False
        last_rc: Optional[int] = None
        last_reason = ""
        for tag, eval_args in attempts:
            cmd = [args.python_bin, str(eval_script)] + flatten_args(eval_args)
            rc = run_and_log(cmd, eval_log, project_root, env)
            last_rc = rc
            if rc == 0 and summary_csv.exists() and details_csv.exists():
                row = parse_summary(summary_csv, details_csv, args.k)
                per_seed_rows.append(row)
                log(
                    "seed={} 完成({}) | A@{}={:.4f} P@{}={:.4f} R@{}={:.4f} MRR@{}={:.4f} MAP@{}={:.4f} Eff20={:.4f} R1LOC={:.4f}".format(
                        seed, tag, args.k, row[f"A@{args.k}"], args.k, row[f"P@{args.k}"], args.k, row[f"R@{args.k}"], args.k, row[f"MRR@{args.k}"], args.k, row[f"MAP@{args.k}"], row["Effort@20%Recall"], row["Recall@1%LOC"],
                    )
                )
                success = True
                break
            last_reason = f"returncode={rc} 或缺少 summary/details"
            log(f"seed={seed} {tag} 失败 | {last_reason} | 日志: {eval_log}")

        if not success:
            fail_rows.append({
                "seed": seed,
                "stage": "eval",
                "returncode": last_rc,
                "reason": last_reason,
                "log": str(eval_log),
            })
            if args.continue_on_error == "no":
                break

    # 输出失败记录
    fail_df = pd.DataFrame(fail_rows)
    fail_csv = output_root / "failed_seeds.csv"
    fail_df.to_csv(fail_csv, index=False)

    # 成功结果汇总
    if per_seed_rows:
        per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)
        per_seed_path = output_root / "per_seed_metrics.csv"
        per_seed_df.to_csv(per_seed_path, index=False)

        metric_order = [
            "A@10", "P@10", "R@10", "MRR@10", "MAP@10",
            "Median IFA", "Effort@20%Recall", "Recall@1%LOC",
        ]
        mean_std_rows = []
        compare_rows = []
        for metric in metric_order:
            vals = per_seed_df[metric].astype(float).tolist()
            m = mean_safe(vals)
            s = std_safe(vals)
            mean_std_rows.append({"Metric": metric, "mean": m, "std": s, "n_success": len(vals)})
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
        log(f"成功 seeds: {sorted(per_seed_df['seed'].astype(int).tolist())}")
        if not fail_df.empty:
            log(f"失败 seeds: {sorted(fail_df['seed'].astype(int).tolist())}")
        log("平均结果（仅统计成功 seeds）：")
        for _, r in mean_std_df.iterrows():
            metric = r["Metric"]
            log(f"{metric}: mean={fmt_metric(metric, float(r['mean']))}, std={fmt_metric(metric, float(r['std']))}")
        log("=" * 80)
        log(f"per-seed 指标: {per_seed_path}")
        log(f"mean/std 汇总: {mean_std_path}")
        log(f"与论文对比: {compare_csv_path}")
        log(f"失败记录: {fail_csv}")
    else:
        log("没有任何 seed 评估成功。")
        log(f"失败记录: {fail_csv}")

    if args.strict_failure == "yes" and not fail_df.empty:
        return 2
    if not per_seed_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
