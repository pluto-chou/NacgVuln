#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate PeerJ-ready statistical reports for the Results section.

This script reads archived per-seed result files from the NacgVuln repository and
produces complete statistical outputs required by PeerJ, including:

- test name
- reason for choosing the test
- paired sample size
- Wilcoxon signed-rank statistic
- df field, marked as NA because Wilcoxon signed-rank test has no conventional df
- exact p-value when available from scipy
- Holm-adjusted p-value within each result family
- paired mean difference and bootstrap 95% CI
- effect sizes:
  1) rank-biserial correlation
  2) approximate z-based r

Expected input files:
- results/main/Nacg_per_seed_metrics.csv
- results/selfattention_baseline/baseline_per_seed_metrics.csv
- results/ablation/ablation_all_per_seed_metrics.csv
- results/clean_fpr/clean_fpr_per_seed.csv
- results/length_group/length_group_per_seed_metrics.csv

Optional:
- --locvul_per_seed_csv path/to/locvul_reproduced_per_seed_metrics.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


MAIN_METRICS = [
    "A@10",
    "P@10",
    "R@10",
    "MRR@10",
    "MAP@10",
    "Median IFA",
    "Effort@20%Recall",
    "Recall@1%LOC",
]

HIGHER_BETTER = {
    "A@10",
    "P@10",
    "R@10",
    "MRR@10",
    "MAP@10",
    "Recall@1%LOC",
    "clean_TNR",
    "NO_VULN_precision",
    "NO_VULN_recall",
}

LOWER_BETTER = {
    "Median IFA",
    "Effort@20%Recall",
    "clean_FPR",
    "false_positive_clean_functions",
    "false_negative_vuln_functions",
}

# Metrics where an increase is reported as a trade-off rather than as an improvement.
TRADEOFF_GREATER = {
    "vuln_empty_output_rate",
}

CLEAN_METRICS = [
    "clean_FPR",
    "clean_TNR",
    "NO_VULN_precision",
    "NO_VULN_recall",
    "vuln_empty_output_rate",
    "false_positive_clean_functions",
    "false_negative_vuln_functions",
]

LENGTH_METRICS = [
    "A@10",
    "P@10",
    "R@10",
    "MRR@10",
    "MAP@10",
    "Median IFA",
]

ABLATION_PAIRS = [
    ("neg_on", "neg_off", "negative-aware training"),
    ("chunk_train_on", "chunk_train_off", "training-time chunking"),
    ("chunk_infer_on", "chunk_infer_off", "inference-time chunking"),
    ("sim_on", "sim_off", "similar-line replacement"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".", help="Path to the NacgVuln repository root.")
    parser.add_argument("--out_dir", default="peerj_stat_results", help="Output directory.")
    parser.add_argument(
        "--locvul_per_seed_csv",
        default="",
        help="Optional per-seed CSV for locally reproduced LocVul-style Seq2Seq baseline.",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=10000,
        help="Bootstrap iterations for paired mean-difference confidence intervals.",
    )
    parser.add_argument("--bootstrap_seed", type=int, default=20260713)
    return parser.parse_args()


def read_csv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        print(f"[WARN] Missing optional file: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def normalize_metric(metric: str, value: float) -> float:
    """Normalize percentage-like metrics to fractions.

    Example:
    94.4 -> 0.944 for A@10
    0.944 -> 0.944
    Median IFA is not normalized.
    """
    if pd.isna(value):
        return np.nan
    v = float(value)
    if metric != "Median IFA" and abs(v) > 1.5:
        return v / 100.0
    return v


def normalize_columns(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        if metric in out.columns:
            out[metric] = out[metric].apply(lambda x: normalize_metric(metric, x))
    return out


def metric_alternative(metric: str, comparison_type: str = "improvement") -> str:
    """Return one-sided Wilcoxon alternative.

    For most comparisons, x is the proposed/better condition and y is the baseline/off condition.
    """
    if metric in HIGHER_BETTER:
        return "greater"
    if metric in LOWER_BETTER:
        return "less"
    if metric in TRADEOFF_GREATER:
        # Used for reporting the trade-off that negative-aware training may increase empty outputs.
        return "greater"
    return "two-sided"


def format_metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "NA"
    if metric in {"Median IFA", "false_positive_clean_functions", "false_negative_vuln_functions"}:
        return f"{value:.4f}"
    return f"{value * 100:.2f}%"


def mean_sd(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), sd


def bootstrap_ci_paired_delta(
    x: np.ndarray,
    y: np.ndarray,
    iters: int = 10000,
    seed: int = 20260713,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for paired mean difference x - y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = x - y
    boot = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def wilcoxon_effect_sizes(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute rank-biserial correlation and approximate z-based r.

    The effect-size calculation excludes zero differences, matching zero_method='wilcox'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diffs = x[mask] - y[mask]
    diffs = diffs[np.abs(diffs) > 1e-15]

    n_eff = len(diffs)
    if n_eff == 0:
        return {
            "n_nonzero": 0,
            "W_plus": 0.0,
            "W_minus": 0.0,
            "rank_biserial_r": 0.0,
            "z_approx": 0.0,
            "effect_r_z": 0.0,
        }

    ranks = rankdata(np.abs(diffs), method="average")
    W_plus = float(np.sum(ranks[diffs > 0]))
    W_minus = float(np.sum(ranks[diffs < 0]))
    total_rank = n_eff * (n_eff + 1) / 2.0

    rank_biserial = (W_plus - W_minus) / total_rank if total_rank > 0 else np.nan

    expected = n_eff * (n_eff + 1) / 4.0
    variance = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0
    if variance <= 0:
        z = 0.0
    else:
        z = (W_plus - expected) / math.sqrt(variance)

    effect_r_z = z / math.sqrt(n_eff) if n_eff > 0 else np.nan

    return {
        "n_nonzero": n_eff,
        "W_plus": W_plus,
        "W_minus": W_minus,
        "rank_biserial_r": float(rank_biserial),
        "z_approx": float(z),
        "effect_r_z": float(effect_r_z),
    }


def run_wilcoxon(x: np.ndarray, y: np.ndarray, alternative: str) -> Tuple[float, float]:
    """Run paired Wilcoxon signed-rank test.

    Tries exact calculation first. Falls back to scipy's default if exact mode is unavailable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan, np.nan

    diffs = x - y
    if np.all(np.abs(diffs) <= 1e-15):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative=alternative,
            method="exact",
        )
    except TypeError:
        stat, p = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative=alternative,
        )
    except ValueError:
        stat, p = np.nan, np.nan

    return float(stat), float(p)


def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    adjusted = [np.nan] * m

    valid = [(i, p) for i, p in enumerate(p_values) if pd.notna(p)]
    valid_sorted = sorted(valid, key=lambda t: t[1])

    running_max = 0.0
    k = len(valid_sorted)

    for rank, (original_i, p) in enumerate(valid_sorted):
        factor = k - rank
        adj = min(1.0, factor * float(p))
        running_max = max(running_max, adj)
        adjusted[original_i] = running_max

    return adjusted


def paired_test_row(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metric: str,
    x_label: str,
    y_label: str,
    comparison: str,
    family: str,
    alternative: Optional[str],
    bootstrap_iters: int,
    bootstrap_seed: int,
    merge_keys: List[str],
) -> Dict[str, object]:
    if alternative is None:
        alternative = metric_alternative(metric)

    required_cols = set(merge_keys + [metric])
    missing_x = required_cols - set(x_df.columns)
    missing_y = required_cols - set(y_df.columns)
    if missing_x:
        raise ValueError(f"{x_label} missing columns {missing_x}")
    if missing_y:
        raise ValueError(f"{y_label} missing columns {missing_y}")

    x_small = x_df[merge_keys + [metric]].copy()
    y_small = y_df[merge_keys + [metric]].copy()
    merged = x_small.merge(y_small, on=merge_keys, suffixes=("_x", "_y"))

    x = merged[f"{metric}_x"].astype(float).to_numpy()
    y = merged[f"{metric}_y"].astype(float).to_numpy()

    stat, p = run_wilcoxon(x, y, alternative)
    eff = wilcoxon_effect_sizes(x, y)
    ci_low, ci_high = bootstrap_ci_paired_delta(
        x,
        y,
        iters=bootstrap_iters,
        seed=bootstrap_seed,
    )

    x_mean, x_sd = mean_sd(x)
    y_mean, y_sd = mean_sd(y)
    delta = x_mean - y_mean

    reason = (
        "paired non-parametric comparison across the same random seeds; "
        "normality of seed-level metric differences is not assumed"
    )

    return {
        "family": family,
        "comparison": comparison,
        "x_label": x_label,
        "y_label": y_label,
        "metric": metric,
        "alternative": alternative,
        "test": "Wilcoxon signed-rank test",
        "reason_for_test": reason,
        "n_pairs": int(len(merged)),
        "n_nonzero_pairs": int(eff["n_nonzero"]),
        "df": "NA",
        "x_mean": x_mean,
        "x_sd": x_sd,
        "y_mean": y_mean,
        "y_sd": y_sd,
        "mean_delta_x_minus_y": delta,
        "mean_delta_95ci_low": ci_low,
        "mean_delta_95ci_high": ci_high,
        "wilcoxon_W": stat,
        "exact_p_value": p,
        "rank_biserial_r": eff["rank_biserial_r"],
        "z_approx": eff["z_approx"],
        "effect_r_z": eff["effect_r_z"],
        "x_mean_fmt": format_metric_value(metric, x_mean),
        "x_sd_fmt": format_metric_value(metric, x_sd),
        "y_mean_fmt": format_metric_value(metric, y_mean),
        "y_sd_fmt": format_metric_value(metric, y_sd),
        "delta_fmt": format_metric_value(metric, delta),
        "ci_low_fmt": format_metric_value(metric, ci_low),
        "ci_high_fmt": format_metric_value(metric, ci_high),
    }


def add_holm_by_family(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["holm_p_value"] = np.nan

    for family, group in out.groupby("family"):
        indices = list(group.index)
        adjusted = holm_adjust(group["exact_p_value"].tolist())
        for idx, adj in zip(indices, adjusted):
            out.loc[idx, "holm_p_value"] = adj

    out["significant_p_lt_0_05"] = out["exact_p_value"].apply(
        lambda p: bool(pd.notna(p) and p < 0.05)
    )
    out["significant_holm_p_lt_0_05"] = out["holm_p_value"].apply(
        lambda p: bool(pd.notna(p) and p < 0.05)
    )
    return out


def make_peerj_sentence(row: pd.Series) -> str:
    metric = row["metric"]

    return (
        f"{row['comparison']} for {metric}: "
        f"{row['x_label']} = {row['x_mean_fmt']} ± {row['x_sd_fmt']}, "
        f"{row['y_label']} = {row['y_mean_fmt']} ± {row['y_sd_fmt']}; "
        f"paired mean difference = {row['delta_fmt']} "
        f"(95% CI [{row['ci_low_fmt']}, {row['ci_high_fmt']}]); "
        f"Wilcoxon signed-rank test, W = {row['wilcoxon_W']:.6g}, "
        f"n = {int(row['n_pairs'])}, df = NA, "
        f"exact p = {row['exact_p_value']:.6g}, "
        f"Holm-adjusted p = {row['holm_p_value']:.6g}, "
        f"rank-biserial r = {row['rank_biserial_r']:.4f}, "
        f"z-based r = {row['effect_r_z']:.4f}."
    )


def write_table_and_markdown(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("No data available.")
    else:
        for _, row in df.iterrows():
            lines.append("- " + make_peerj_sentence(row))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_optional_locvul_csv(repo_root: Path, cli_path: str) -> Optional[Path]:
    if cli_path.strip():
        p = Path(cli_path)
        if not p.is_absolute():
            p = repo_root / p
        return p if p.exists() else None

    candidates = [
        repo_root / "results/baselines/locvul_reproduced_per_seed_metrics.csv",
        repo_root / "results/baselines/locvul_per_seed_metrics.csv",
        repo_root / "results/main/locvul_reproduced_per_seed_metrics.csv",
        repo_root / "results/main/locvul_per_seed_metrics.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def overall_baseline_tests(
    repo_root: Path,
    out_dir: Path,
    bootstrap_iters: int,
    bootstrap_seed: int,
    locvul_csv_path: Optional[Path],
) -> pd.DataFrame:
    nacg_path = repo_root / "results/main/Nacg_per_seed_metrics.csv"
    self_path = repo_root / "results/selfattention_baseline/baseline_per_seed_metrics.csv"

    nacg = normalize_columns(read_csv(nacg_path), MAIN_METRICS)
    selfattn = normalize_columns(read_csv(self_path), MAIN_METRICS)

    rows = []

    for metric in MAIN_METRICS:
        rows.append(
            paired_test_row(
                nacg,
                selfattn,
                metric=metric,
                x_label="NacgVuln",
                y_label="Self-Attention baseline",
                comparison="NacgVuln vs Self-Attention baseline",
                family="RQ1_RQ2_selfattention",
                alternative=metric_alternative(metric),
                bootstrap_iters=bootstrap_iters,
                bootstrap_seed=bootstrap_seed,
                merge_keys=["seed"],
            )
        )

    if locvul_csv_path is not None and locvul_csv_path.exists():
        locvul = normalize_columns(read_csv(locvul_csv_path), MAIN_METRICS)
        for metric in MAIN_METRICS:
            rows.append(
                paired_test_row(
                    nacg,
                    locvul,
                    metric=metric,
                    x_label="NacgVuln",
                    y_label="LocVul reproduced",
                    comparison="NacgVuln vs LocVul reproduced",
                    family="RQ1_RQ2_locvul_reproduced",
                    alternative=metric_alternative(metric),
                    bootstrap_iters=bootstrap_iters,
                    bootstrap_seed=bootstrap_seed,
                    merge_keys=["seed"],
                )
            )
    else:
        print(
            "[WARN] No LocVul reproduced per-seed CSV found. "
            "Cross-paper or single-mean comparison will be descriptive only; no paired Wilcoxon test is produced.",
            file=sys.stderr,
        )

    df = add_holm_by_family(pd.DataFrame(rows))
    write_table_and_markdown(
        df,
        out_dir / "peerj_overall_baseline_tests.csv",
        out_dir / "peerj_overall_baseline_statements.md",
        "PeerJ-ready overall baseline statistical tests",
    )
    return df


def ablation_tests(
    repo_root: Path,
    out_dir: Path,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    ablation_path = repo_root / "results/ablation/ablation_all_per_seed_metrics.csv"
    ablation = normalize_columns(read_csv(ablation_path), MAIN_METRICS)

    rows = []

    for on_variant, off_variant, module_name in ABLATION_PAIRS:
        on_df = ablation[ablation["variant"] == on_variant].copy()
        off_df = ablation[ablation["variant"] == off_variant].copy()

        if on_df.empty or off_df.empty:
            print(f"[WARN] Missing ablation pair: {on_variant} vs {off_variant}", file=sys.stderr)
            continue

        for metric in MAIN_METRICS:
            rows.append(
                paired_test_row(
                    on_df,
                    off_df,
                    metric=metric,
                    x_label=on_variant,
                    y_label=off_variant,
                    comparison=f"{module_name}: on vs off",
                    family=f"RQ3_ablation_{module_name}",
                    alternative=metric_alternative(metric),
                    bootstrap_iters=bootstrap_iters,
                    bootstrap_seed=bootstrap_seed,
                    merge_keys=["seed"],
                )
            )

    df = add_holm_by_family(pd.DataFrame(rows))
    write_table_and_markdown(
        df,
        out_dir / "peerj_ablation_tests.csv",
        out_dir / "peerj_ablation_statements.md",
        "PeerJ-ready ablation statistical tests",
    )
    return df


def clean_fpr_tests(
    repo_root: Path,
    out_dir: Path,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    clean_path = repo_root / "results/clean_fpr/clean_fpr_per_seed.csv"
    clean = read_csv(clean_path)

    rows = []

    neg_on = clean[clean["variant"] == "neg_on"].copy()
    neg_off = clean[clean["variant"] == "neg_off"].copy()

    if neg_on.empty or neg_off.empty:
        raise ValueError("clean_fpr_per_seed.csv must contain both neg_on and neg_off variants.")

    for metric in CLEAN_METRICS:
        if metric not in clean.columns:
            print(f"[WARN] Missing clean metric: {metric}", file=sys.stderr)
            continue

        if metric == "vuln_empty_output_rate":
            alt = "greater"
        else:
            alt = metric_alternative(metric)

        rows.append(
            paired_test_row(
                neg_on,
                neg_off,
                metric=metric,
                x_label="negative-aware on",
                y_label="negative-aware off",
                comparison="Clean-function rejection: negative-aware on vs off",
                family="RQ4_clean_function_rejection",
                alternative=alt,
                bootstrap_iters=bootstrap_iters,
                bootstrap_seed=bootstrap_seed,
                merge_keys=["seed"],
            )
        )

    df = add_holm_by_family(pd.DataFrame(rows))
    write_table_and_markdown(
        df,
        out_dir / "peerj_clean_fpr_tests.csv",
        out_dir / "peerj_clean_fpr_statements.md",
        "PeerJ-ready clean-function rejection statistical tests",
    )
    return df


def length_group_tests(
    repo_root: Path,
    out_dir: Path,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    length_path = repo_root / "results/length_group/length_group_per_seed_metrics.csv"
    length_df = normalize_columns(read_csv(length_path), LENGTH_METRICS)

    rows = []

    if "length_group" not in length_df.columns or "variant" not in length_df.columns:
        raise ValueError("length_group_per_seed_metrics.csv must contain length_group and variant columns.")

    for group in sorted(length_df["length_group"].dropna().unique().tolist()):
        on_df = length_df[
            (length_df["variant"] == "chunk_infer_on")
            & (length_df["length_group"] == group)
        ].copy()

        off_df = length_df[
            (length_df["variant"] == "chunk_infer_off")
            & (length_df["length_group"] == group)
        ].copy()

        if on_df.empty or off_df.empty:
            print(f"[WARN] Missing length group pair for {group}", file=sys.stderr)
            continue

        for metric in LENGTH_METRICS:
            if metric not in on_df.columns or metric not in off_df.columns:
                continue

            rows.append(
                paired_test_row(
                    on_df,
                    off_df,
                    metric=metric,
                    x_label="chunk inference on",
                    y_label="chunk inference off",
                    comparison=f"Long-function robustness, length group {group}",
                    family=f"Long_function_group_{group}",
                    alternative=metric_alternative(metric),
                    bootstrap_iters=bootstrap_iters,
                    bootstrap_seed=bootstrap_seed,
                    merge_keys=["seed"],
                )
            )

    df = add_holm_by_family(pd.DataFrame(rows))
    write_table_and_markdown(
        df,
        out_dir / "peerj_length_group_tests.csv",
        out_dir / "peerj_length_group_statements.md",
        "PeerJ-ready length-group statistical tests",
    )
    return df


def write_master_summary(out_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    lines = [
        "# PeerJ Results statistical report summary",
        "",
        "Use the generated Markdown files to revise the Results section.",
        "",
        "Important reporting rule:",
        "- Wilcoxon signed-rank test has no conventional degrees of freedom; report `df = NA`.",
        "- Prefer exact p-values from the CSV.",
        "- Report both the unadjusted exact p-value and the Holm-adjusted p-value when multiple tests are discussed together.",
        "- Use rank-biserial r as the main effect size for Wilcoxon tests.",
        "",
        "Generated files:",
    ]

    for name in tables:
        lines.append(f"- {name}")

    lines.append("")
    lines.append("Suggested Results wording pattern:")
    lines.append("")
    lines.append(
        "NacgVuln significantly improved A@10 compared with the Self-Attention baseline "
        "(mean ± SD: xx ± xx vs xx ± xx; paired mean difference = xx, 95% CI [xx, xx]; "
        "Wilcoxon signed-rank test, W = xx, n = 10, df = NA, exact p = xx, "
        "Holm-adjusted p = xx, rank-biserial r = xx)."
    )

    (out_dir / "README_peerj_stat_outputs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    locvul_csv_path = find_optional_locvul_csv(repo_root, args.locvul_per_seed_csv)

    print(f"[INFO] repo_root = {repo_root}")
    print(f"[INFO] out_dir = {out_dir}")
    if locvul_csv_path is not None:
        print(f"[INFO] LocVul reproduced per-seed CSV = {locvul_csv_path}")
    else:
        print("[INFO] LocVul reproduced per-seed CSV = not found")

    tables: Dict[str, pd.DataFrame] = {}

    tables["peerj_overall_baseline_tests.csv"] = overall_baseline_tests(
        repo_root=repo_root,
        out_dir=out_dir,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        locvul_csv_path=locvul_csv_path,
    )

    tables["peerj_ablation_tests.csv"] = ablation_tests(
        repo_root=repo_root,
        out_dir=out_dir,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    tables["peerj_clean_fpr_tests.csv"] = clean_fpr_tests(
        repo_root=repo_root,
        out_dir=out_dir,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    tables["peerj_length_group_tests.csv"] = length_group_tests(
        repo_root=repo_root,
        out_dir=out_dir,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    write_master_summary(out_dir, tables)

    print("\nSaved PeerJ-ready statistical files:")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())