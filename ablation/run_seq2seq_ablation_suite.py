#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seq2Seq ablation suite runner for LocVul-style experiments.

修正版并行调度版（v2）
----------------------
相对上一版并行脚本，核心修复：
1) 不再一次性把所有 seed 全部 submit 给 ThreadPoolExecutor
2) 改成“动态补位”调度：哪个 GPU 槽位先空出来，就立刻在该槽位上提交下一个 seed
3) 因此当某个 seed 因为复用 checkpoint / 复用 summary 而秒结束时，GPU 不会空转
4) 仍保留：原目录结构、resume 逻辑、eval-only 复用 checkpoint、suite 汇总逻辑

推荐用法：
- 两张卡并发：--gpu_ids 0,1 --max_parallel_seeds 2
- 单卡串行：不传 --gpu_ids，或 --max_parallel_seeds 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_SEEDS = list(range(10))
METRIC_ORDER = [
    "A@10", "P@10", "R@10", "MRR@10", "MAP@10",
    "Median IFA", "Effort@20%Recall", "Recall@1%LOC",
]


@dataclass
class VariantConfig:
    name: str
    train_required: bool
    train_overrides: Dict[str, str]
    eval_overrides: Dict[str, str]
    description: str


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
        if "CUDA_VISIBLE_DEVICES" in env:
            f.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
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
        cnt = 0
        for x in ranked:
            if normalize_line(x) in gt_set:
                cnt += 1
        precisions.append(cnt / k)
        recalls.append(cnt / len(gt_set))

    return {f"P@{k}": mean_safe(precisions), f"R@{k}": mean_safe(recalls)}


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


def maybe_reuse_existing_checkpoint(seed: int, seed_ckpt_dir: Path, existing_ckpt_root: Optional[Path]) -> bool:
    if existing_ckpt_root is None:
        return False
    dst = seed_ckpt_dir / "best_weights.pt"
    if dst.exists():
        return True
    src = existing_ckpt_root / f"seed{seed}" / "best_weights.pt"
    if src.exists():
        ensure_dir(seed_ckpt_dir)
        shutil.copy2(src, dst)
        log(f"seed={seed} 复用已有 checkpoint: {src} -> {dst}")
        return True
    return False


def default_variants() -> Dict[str, VariantConfig]:
    return {
        "full_improved": VariantConfig(
            name="full_improved",
            train_required=True,
            train_overrides={
                "INCLUDE_NEGATIVES": "yes",
                "NEGATIVE_RATIO": "1.0",
                "CHUNK_TRAINING": "yes",
            },
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="改进完整版：include-negatives + chunk-train + chunk-infer + similarity-replacement",
        ),
        "neg_off": VariantConfig(
            name="neg_off",
            train_required=True,
            train_overrides={
                "INCLUDE_NEGATIVES": "no",
                "NEGATIVE_RATIO": "1.0",
                "CHUNK_TRAINING": "yes",
            },
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="vulnerable-only：关闭 include-negatives，其它保持改进版",
        ),
        "neg_on": VariantConfig(
            name="neg_on",
            train_required=True,
            train_overrides={
                "INCLUDE_NEGATIVES": "yes",
                "NEGATIVE_RATIO": "1.0",
                "CHUNK_TRAINING": "yes",
            },
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="include-negatives：开启 negatives + <NO_VULN>，其它保持改进版",
        ),
        "chunk_train_off": VariantConfig(
            name="chunk_train_off",
            train_required=True,
            train_overrides={
                "INCLUDE_NEGATIVES": "yes",
                "NEGATIVE_RATIO": "1.0",
                "CHUNK_TRAINING": "no",
            },
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="no chunk training：只关训练分块，评估仍开 chunk inference",
        ),
        "chunk_train_on": VariantConfig(
            name="chunk_train_on",
            train_required=True,
            train_overrides={
                "INCLUDE_NEGATIVES": "yes",
                "NEGATIVE_RATIO": "1.0",
                "CHUNK_TRAINING": "yes",
            },
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="chunk training：打开训练分块，评估仍开 chunk inference",
        ),
        "chunk_infer_off": VariantConfig(
            name="chunk_infer_off",
            train_required=False,
            train_overrides={},
            eval_overrides={
                "CHUNK_LONG_FUNCS": "no",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="no chunk inference：复用同一套 checkpoint，仅关闭评估分块",
        ),
        "chunk_infer_on": VariantConfig(
            name="chunk_infer_on",
            train_required=False,
            train_overrides={},
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="chunk inference：复用同一套 checkpoint，仅打开评估分块",
        ),
        "sim_off": VariantConfig(
            name="sim_off",
            train_required=False,
            train_overrides={},
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "no",
            },
            description="without similarity replacement：复用同一套 checkpoint，仅关闭相似行替换",
        ),
        "sim_on": VariantConfig(
            name="sim_on",
            train_required=False,
            train_overrides={},
            eval_overrides={
                "CHUNK_LONG_FUNCS": "yes",
                "SIMILARITY_REPLACEMENT": "yes",
            },
            description="with similarity replacement：复用同一套 checkpoint，仅打开相似行替换",
        ),
    }


def group_to_variants() -> Dict[str, List[str]]:
    return {
        "neg": ["neg_off", "neg_on"],
        "chunk_train": ["chunk_train_off", "chunk_train_on"],
        "chunk_infer": ["chunk_infer_off", "chunk_infer_on"],
        "sim": ["sim_off", "sim_on"],
        "full": ["full_improved"],
        "all": [
            "full_improved",
            "neg_off", "neg_on",
            "chunk_train_off", "chunk_train_on",
            "chunk_infer_off", "chunk_infer_on",
            "sim_off", "sim_on",
        ],
    }


def save_variant_aggregate(output_root: Path, per_seed_rows: List[dict], fail_rows: List[dict]) -> None:
    per_seed_df = pd.DataFrame(per_seed_rows)
    fail_df = pd.DataFrame(fail_rows)

    per_seed_path = output_root / "per_seed_metrics.csv"
    mean_std_path = output_root / "mean_std_metrics.csv"
    fail_csv = output_root / "failed_seeds.csv"

    if not per_seed_df.empty:
        per_seed_df = per_seed_df.sort_values(by="seed").reset_index(drop=True)
        per_seed_df.to_csv(per_seed_path, index=False)

        mean_std_rows = []
        for metric in METRIC_ORDER:
            vals = per_seed_df[metric].astype(float).tolist()
            mean_std_rows.append({
                "Metric": metric,
                "mean": mean_safe(vals),
                "std": std_safe(vals),
                "n_success": len(vals),
                "mean_fmt": fmt_metric(metric, mean_safe(vals)),
                "std_fmt": fmt_metric(metric, std_safe(vals)),
            })
        pd.DataFrame(mean_std_rows).to_csv(mean_std_path, index=False)
    else:
        pd.DataFrame(columns=["seed"] + METRIC_ORDER).to_csv(per_seed_path, index=False)
        pd.DataFrame(columns=["Metric", "mean", "std", "n_success", "mean_fmt", "std_fmt"]).to_csv(mean_std_path, index=False)

    if not fail_df.empty:
        fail_df.to_csv(fail_csv, index=False)
    else:
        pd.DataFrame(columns=["seed", "stage", "returncode", "reason", "log"]).to_csv(fail_csv, index=False)


def resolve_gpu_slots(args: argparse.Namespace) -> List[Optional[str]]:
    gpu_ids = [x.strip() for x in str(args.gpu_ids).split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        return [None]
    return gpu_ids


def run_one_seed(
    cfg: VariantConfig,
    seed: int,
    gpu_slot: Optional[str],
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    paths: Paths,
    test_csv: Path,
    reference_ckpt_root: Optional[Path],
) -> Tuple[Optional[dict], Optional[dict]]:
    train_script = Path(args.train_script)
    eval_script = Path(args.eval_script)

    ckpt_dir = paths.ckpts / f"seed{seed}"
    ensure_dir(ckpt_dir)
    best_weights = ckpt_dir / "best_weights.pt"

    run_name = f"{cfg.name}_seed{seed}"
    train_log = paths.logs / f"{run_name}_train.log"
    eval_log = paths.logs / f"{run_name}_eval.log"
    summary_csv = paths.eval_outputs / f"{run_name}_summary.csv"
    details_csv = paths.eval_outputs / f"{run_name}_details.csv"

    child_env = env.copy()
    if gpu_slot is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)

    log(f"[{cfg.name}] seed={seed} 开始 | GPU={gpu_slot if gpu_slot is not None else 'default'}")

    if cfg.train_required:
        if args.resume_if_exists == "yes" and best_weights.exists():
            log(f"[{cfg.name}] seed={seed} 检测到已有训练权重，跳过训练")
        else:
            train_args = {
                "seed": str(seed),
                "FINE_TUNE": "yes",
                "model_variation": args.model_variation_seq2seq,
                "checkpoint_dir": str(ckpt_dir),
                "INCLUDE_NEGATIVES": cfg.train_overrides.get("INCLUDE_NEGATIVES", args.include_negatives),
                "NEGATIVE_RATIO": cfg.train_overrides.get("NEGATIVE_RATIO", args.negative_ratio),
                "NO_VULN_TOKEN": args.no_vuln_token,
                "ROUGE_ON_VULN_ONLY": args.rouge_on_vuln_only,
                "MAX_INPUT_LEN": args.max_input_len,
                "MAX_TARGET_LEN_CAP": args.max_target_len_cap,
                "BATCH_SIZE": args.batch_size,
                "GRAD_ACCUM_STEPS": args.grad_accum_steps,
                "NUM_EPOCHS": args.num_epochs,
                "PATIENCE": args.patience,
                "LR": args.lr,
                "CHUNK_TRAINING": cfg.train_overrides.get("CHUNK_TRAINING", args.chunk_training),
                "CHUNK_STRIDE_LINES": args.chunk_stride_lines_train,
                "CHUNK_PREFIX_LINES": args.chunk_prefix_lines,
                "CHUNK_PREFIX_MAX_TOKENS": args.chunk_prefix_max_tokens,
                "CHUNK_MAX_TOKENS": args.chunk_max_tokens,
                "NEG_CHUNKS_PER_FUNC": args.neg_chunks_per_func,
                "QUICK_TEST": "no",
            }
            cmd = [args.python_bin, str(train_script)] + flatten_args(train_args)
            rc = run_and_log(cmd, train_log, project_root, child_env)
            if rc != 0 or not best_weights.exists():
                msg = f"训练失败或未生成 best_weights.pt: rc={rc}, ckpt={best_weights}"
                log(f"[{cfg.name}] seed={seed} 失败 | {msg}")
                return None, {"seed": seed, "stage": "train", "returncode": rc, "reason": msg, "log": str(train_log)}
    else:
        if not maybe_reuse_existing_checkpoint(seed, ckpt_dir, reference_ckpt_root):
            msg = (
                f"eval-only 变体缺少可复用 checkpoint。请提供 --reference_ckpt_root，或先运行 full_improved。期望路径: {reference_ckpt_root}/seed{seed}/best_weights.pt"
                if reference_ckpt_root is not None else
                "eval-only 变体缺少可复用 checkpoint。请提供 --reference_ckpt_root，或先运行 full_improved。"
            )
            log(f"[{cfg.name}] seed={seed} 失败 | {msg}")
            return None, {"seed": seed, "stage": "precheck", "returncode": None, "reason": msg, "log": str(eval_log)}

    if args.resume_if_exists == "yes" and summary_csv.exists() and details_csv.exists():
        try:
            row = parse_summary(summary_csv, details_csv, args.k)
            log(f"[{cfg.name}] seed={seed} 复用已有评估结果: {summary_csv.name}")
            return row, None
        except Exception:
            pass

    base_eval_args = {
        "seed": str(seed),
        "checkpoint_dir_seq2seq": str(ckpt_dir),
        "model_variation_seq2seq": args.model_variation_seq2seq,
        "output_dir": str(paths.eval_outputs),
        "run_name": run_name,
        "test_path": str(test_csv),
        "MAX_INPUT_LEN": args.max_input_len,
        "MAX_TARGET_LEN": args.eval_max_target_len,
        "NUM_BEAMS": args.num_beams,
        "NUM_RETURN_SEQS": args.num_return_seqs,
        "FALLBACK_SAMPLING": args.fallback_sampling,
        "SAMPLE_RETURN_SEQS": args.sample_return_seqs,
        "TOP_P": args.top_p,
        "TEMPERATURE": args.temperature,
        "CHUNK_LONG_FUNCS": cfg.eval_overrides.get("CHUNK_LONG_FUNCS", args.chunk_long_funcs),
        "CHUNK_STRIDE_LINES": args.chunk_stride_lines_eval,
        "SIMILARITY_REPLACEMENT": cfg.eval_overrides.get("SIMILARITY_REPLACEMENT", args.similarity_replacement),
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
    cmd = [args.python_bin, str(eval_script)] + flatten_args(base_eval_args)
    rc = run_and_log(cmd, eval_log, project_root, child_env)
    if rc != 0 or not summary_csv.exists() or not details_csv.exists():
        msg = f"评估失败或未生成 summary/details: rc={rc}, summary={summary_csv.exists()}, details={details_csv.exists()}"
        log(f"[{cfg.name}] seed={seed} 失败 | {msg}")
        return None, {"seed": seed, "stage": "eval", "returncode": rc, "reason": msg, "log": str(eval_log)}

    row = parse_summary(summary_csv, details_csv, args.k)
    log(
        "[{}] seed={} 完成 | A@{}={:.4f} P@{}={:.4f} R@{}={:.4f} MRR@{}={:.4f} MAP@{}={:.4f} Eff20={:.4f} R1LOC={:.4f}".format(
            cfg.name,
            seed,
            args.k,
            row[f"A@{args.k}"],
            args.k,
            row[f"P@{args.k}"],
            args.k,
            row[f"R@{args.k}"],
            args.k,
            row[f"MRR@{args.k}"],
            args.k,
            row[f"MAP@{args.k}"],
            row["Effort@20%Recall"],
            row["Recall@1%LOC"],
        )
    )
    return row, None


def _submit_seed(
    executor: ThreadPoolExecutor,
    cfg: VariantConfig,
    seed: int,
    gpu_slot: Optional[str],
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    paths: Paths,
    test_csv: Path,
    reference_ckpt_root: Optional[Path],
):
    return executor.submit(
        run_one_seed,
        cfg,
        seed,
        gpu_slot,
        args,
        project_root,
        env,
        paths,
        test_csv,
        reference_ckpt_root,
    )


def run_variant(
    cfg: VariantConfig,
    args: argparse.Namespace,
    project_root: Path,
    env: Dict[str, str],
    variants: Dict[str, VariantConfig],
) -> None:
    del variants

    output_root = Path(args.output_root) / cfg.name
    paths = build_paths(output_root)
    write_json(output_root / "variant_config.json", {
        "variant": cfg.name,
        "description": cfg.description,
        "train_required": cfg.train_required,
        "train_overrides": cfg.train_overrides,
        "eval_overrides": cfg.eval_overrides,
    })

    seeds = [int(x) for x in args.seeds.split(",") if str(x).strip() != ""]
    test_map = prepare_test_csvs(Path(args.dataset_path), paths.tests, seeds)

    reference_ckpt_root: Optional[Path] = None
    if args.reference_ckpt_root.strip():
        reference_ckpt_root = Path(args.reference_ckpt_root)
    else:
        auto_ref = Path(args.output_root) / "full_improved" / "checkpoints"
        if auto_ref.exists():
            reference_ckpt_root = auto_ref

    per_seed_rows: List[dict] = []
    fail_rows: List[dict] = []

    gpu_slots = resolve_gpu_slots(args)
    max_workers = max(1, min(int(args.max_parallel_seeds), len(seeds), len(gpu_slots)))

    if max_workers <= 1:
        for idx, seed in enumerate(seeds):
            gpu_slot = gpu_slots[idx % len(gpu_slots)]
            row, fail = run_one_seed(
                cfg=cfg,
                seed=seed,
                gpu_slot=gpu_slot,
                args=args,
                project_root=project_root,
                env=env,
                paths=paths,
                test_csv=test_map[seed],
                reference_ckpt_root=reference_ckpt_root,
            )
            if row is not None:
                per_seed_rows.append(row)
            if fail is not None:
                fail_rows.append(fail)
                if args.continue_on_error == "no":
                    break
    else:
        stop_on_fail = (args.continue_on_error == "no")
        next_seed_idx = 0
        saw_fail = False
        inflight = {}
        free_gpu_slots = gpu_slots[:max_workers]

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # 初始填满所有 GPU 槽位
            while free_gpu_slots and next_seed_idx < len(seeds):
                gpu_slot = free_gpu_slots.pop(0)
                seed = seeds[next_seed_idx]
                next_seed_idx += 1
                fu = _submit_seed(
                    ex, cfg, seed, gpu_slot, args, project_root, env, paths, test_map[seed], reference_ckpt_root
                )
                inflight[fu] = (seed, gpu_slot)
                log(f"[{cfg.name}] 已提交 seed={seed} 到 GPU={gpu_slot if gpu_slot is not None else 'default'}")

            # 动态补位：谁先完成，就在同一 GPU 槽位补下一个 seed
            while inflight:
                done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                for fu in done:
                    seed, gpu_slot = inflight.pop(fu)
                    row, fail = fu.result()
                    if row is not None:
                        per_seed_rows.append(row)
                    if fail is not None:
                        fail_rows.append(fail)
                        saw_fail = True

                    if stop_on_fail and saw_fail:
                        log(f"[{cfg.name}] continue_on_error=no，已检测到失败；停止提交新的 seed。")
                        continue

                    if next_seed_idx < len(seeds):
                        next_seed = seeds[next_seed_idx]
                        next_seed_idx += 1
                        new_fu = _submit_seed(
                            ex, cfg, next_seed, gpu_slot, args, project_root, env, paths, test_map[next_seed], reference_ckpt_root
                        )
                        inflight[new_fu] = (next_seed, gpu_slot)
                        log(f"[{cfg.name}] GPU={gpu_slot if gpu_slot is not None else 'default'} 空闲，补提 seed={next_seed}")

                if stop_on_fail and saw_fail:
                    # 不再提交新任务；已在跑的任务继续自然结束
                    pass

    save_variant_aggregate(output_root, per_seed_rows, fail_rows)


def aggregate_suite(output_root: Path, variant_names: List[str]) -> None:
    rows = []
    for name in variant_names:
        mean_std = output_root / name / "mean_std_metrics.csv"
        if not mean_std.exists():
            continue
        df = pd.read_csv(mean_std)
        if df.empty:
            continue
        row = {"variant": name}
        for _, r in df.iterrows():
            metric = r["Metric"]
            row[f"{metric}_mean"] = r["mean"]
            row[f"{metric}_std"] = r["std"]
        rows.append(row)
    out = pd.DataFrame(rows)
    ensure_dir(output_root)
    out.to_csv(output_root / "ablation_suite_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_root", default="./ablation_runs")
    ap.add_argument("--dataset_path", default="./data/dataset.csv")
    ap.add_argument("--train_script", default="./Seq2Seq_vulnDet_fixed_v2_chunktrain_v2_2_quickfix.py")
    ap.add_argument("--eval_script", default="./seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--hf_endpoint", default="")
    ap.add_argument("--reference_ckpt_root", default="", help="供 eval-only 变体复用的 reference checkpoints 根目录")
    ap.add_argument("--model_variation_seq2seq", default="Salesforce/codet5-base")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--resume_if_exists", default="yes", choices=["yes", "no"])
    ap.add_argument("--continue_on_error", default="yes", choices=["yes", "no"])
    ap.add_argument("--run_groups", default="all", help="可选: full,neg,chunk_train,chunk_infer,sim,all；多个用逗号分隔")

    ap.add_argument("--gpu_ids", default="", help="例如 0,1；为空表示不显式绑定 GPU")
    ap.add_argument("--max_parallel_seeds", default=1, type=int, help="同一变体内最大并发 seed 数")

    ap.add_argument("--include_negatives", default="yes", choices=["yes", "no"])
    ap.add_argument("--negative_ratio", default="1.0")
    ap.add_argument("--no_vuln_token", default="<NO_VULN>")
    ap.add_argument("--rouge_on_vuln_only", default="yes", choices=["yes", "no"])
    ap.add_argument("--max_input_len", default="512")
    ap.add_argument("--max_target_len_cap", default="128")
    ap.add_argument("--batch_size", default="2")
    ap.add_argument("--grad_accum_steps", default="4")
    ap.add_argument("--num_epochs", default="10")
    ap.add_argument("--patience", default="5")
    ap.add_argument("--lr", default="5e-5")
    ap.add_argument("--chunk_training", default="yes", choices=["yes", "no"])
    ap.add_argument("--chunk_stride_lines_train", default="80")
    ap.add_argument("--chunk_prefix_lines", default="8")
    ap.add_argument("--chunk_prefix_max_tokens", default="192")
    ap.add_argument("--chunk_max_tokens", default="512")
    ap.add_argument("--neg_chunks_per_func", default="1")

    ap.add_argument("--k", default=10, type=int)
    ap.add_argument("--eval_max_target_len", default="128")
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
    ap.add_argument("--handle_no_vuln_token", default="yes", choices=["yes", "no"])
    ap.add_argument("--sort_by_lines", default="yes", choices=["yes", "no"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.hf_endpoint.strip():
        env["HF_ENDPOINT"] = args.hf_endpoint.strip()

    variants = default_variants()
    group_map = group_to_variants()
    requested_groups = [x.strip() for x in args.run_groups.split(",") if x.strip()]
    selected_variants: List[str] = []
    for g in requested_groups:
        if g not in group_map:
            raise ValueError(f"未知 run_groups: {g}. 可选: {', '.join(sorted(group_map.keys()))}")
        for v in group_map[g]:
            if v not in selected_variants:
                selected_variants.append(v)

    write_json(output_root / "suite_config.json", {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "selected_variants": selected_variants,
        "args": vars(args),
    })

    for vname in selected_variants:
        cfg = variants[vname]
        log("=" * 90)
        log(f"开始变体: {cfg.name} | {cfg.description}")
        log("=" * 90)
        run_variant(cfg, args, project_root, env, variants)

    aggregate_suite(output_root, selected_variants)
    log(f"总汇总文件: {output_root / 'ablation_suite_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
