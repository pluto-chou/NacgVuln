#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Chapter 5 figures for NacgVuln paper.

This version keeps the original Chapter 5 figure generation logic and adds
comparison-experiment figures placed after the previous figures:
  - fig5_8_reference_based_a10_comparison
  - fig5_9_local_reproduction_a10_comparison

Usage 1: generate all original figures + comparison figures
  python make_chapter5_figures_with_compare.py \
    --data_dir /path/to/csvs \
    --out_dir ./chapter5_figures \
    --external_a10_csv /path/to/chapter5_combined_a10.csv

Usage 2: generate only the comparison figures
  python make_chapter5_figures_with_compare.py \
    --only_compare \
    --external_a10_csv chapter5_combined_a10.csv \
    --out_dir ./chapter5_figures

Expected CSV files in --data_dir for the original figures:
  baseline_mean_std_metrics.csv
  Loc_mean_std_metrics.csv
  Nacg_mean_std_metrics.csv
  ablation_all_mean_std_metrics.csv
  clean_fpr_mean_std.csv
  length_group_mean_std_metrics.csv

Expected CSV for comparison figures:
  chapter5_combined_a10.csv

Required columns for comparison CSV:
  Approach,A@10,Source

Recommended Source values:
  Reference Table 4       -> literature/reference comparison
  Local reproduction      -> local reproduced baseline
  Local 10-seed mean      -> your final NacgVuln 10-seed mean

Notes:
  - The script normalizes metrics to paper-friendly units.
  - Accuracy/ranking/recall/effort/FPR values are shown as percentages.
  - Median IFA is shown as raw number of false alarms.
  - It uses matplotlib only and does not require seaborn.
  - Each figure is saved as PNG, PDF, and SVG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MAIN_METRICS = [
    "A@10", "P@10", "R@10", "MRR@10", "MAP@10",
    "Median IFA", "Effort@20%Recall", "Recall@1%LOC",
]

PERCENT_METRICS = {
    "A@10", "P@10", "R@10", "MRR@10", "MAP@10",
    "Effort@20%Recall", "Recall@1%LOC",
    "clean_FPR", "clean_TNR", "NO_VULN_precision", "NO_VULN_recall",
    "vuln_empty_output_rate",
}

METHOD_LABELS = {
    "baseline": " Self-Attention baseline",
    "LocVuln": "LocVul reproduced",
    "NacgVuln": "NacgVuln",
}

# Display names for comparison figures. Keep method names compact for SCI figures.
COMPARE_LABELS = {
    "CppCheck Analyzer": "CppCheck",
    "Cppcheck Analyzer": "Cppcheck",
    "Cppcheck Analyzer (local)": "Cppcheck\n(local)",
    "LineVul Attention (local)": "LineVul\nAttention",
    "LocVul reproduction": "LocVul\nreproduction",
    "Proposed method": "NacgVuln",
    "NacgVuln": "NacgVuln",
}


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_base: Path, dpi: int = 600) -> None:
    ensure_out_dir(out_base.parent)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def normalize_metric(metric: str, value: float) -> float:
    """Normalize metric to decimal for percentage metrics, raw for IFA."""
    if pd.isna(value):
        return np.nan
    v = float(value)
    if metric in PERCENT_METRICS:
        # Some files use 0-1 decimals, while other files use 0-100 percentages.
        if abs(v) > 1.5:
            v = v / 100.0
    return v


def pct_value(metric: str, value: float) -> float:
    """Return plotting value: percentage for percentage metrics; raw for IFA."""
    v = normalize_metric(metric, value)
    if metric in PERCENT_METRICS:
        return v * 100.0
    return v


def read_mean_std_metric_file(path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"Metric", "mean", "std"}.issubset(df.columns):
        out = df[["Metric", "mean", "std"]].copy()
        out = out.rename(columns={"Metric": "metric"})
    elif {"metric", "mean", "std"}.issubset(df.columns):
        out = df[["metric", "mean", "std"]].copy()
    else:
        raise ValueError(f"Unsupported mean/std file format: {path}")
    out["method"] = method_name
    out["mean_norm"] = out.apply(lambda r: normalize_metric(str(r["metric"]), r["mean"]), axis=1)
    out["std_norm"] = out.apply(lambda r: normalize_metric(str(r["metric"]), r["std"]), axis=1)
    return out


def load_main_results(data_dir: Path) -> pd.DataFrame:
    files = [
        ("baseline", data_dir / "baseline_mean_std_metrics.csv"),
        ("LocVuln", data_dir / "Loc_mean_std_metrics.csv"),
        ("NacgVuln", data_dir / "Nacg_mean_std_metrics.csv"),
    ]
    frames = []
    for method, path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        frames.append(read_mean_std_metric_file(path, method))
    return pd.concat(frames, ignore_index=True)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "%.2f", padding_ratio: float = 0.012) -> None:
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * padding_ratio
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            fmt % height,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def grouped_bar_from_main(
    df: pd.DataFrame,
    metrics: List[str],
    title: str,
    ylabel: str,
    out_base: Path,
    ylim: Tuple[float, float] | None = None,
) -> None:
    methods = ["baseline", "LocVuln", "NacgVuln"]
    labels = [METHOD_LABELS[m] for m in methods]
    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, method in enumerate(methods):
        vals = []
        errs = []
        for metric in metrics:
            row = df[(df["method"] == method) & (df["metric"] == metric)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0)
            else:
                vals.append(pct_value(metric, row.iloc[0]["mean"]))
                errs.append(pct_value(metric, row.iloc[0]["std"]))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=labels[i], yerr=errs, capsize=3)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    save_figure(fig, out_base)


def grouped_bar_cost(df: pd.DataFrame, out_base: Path) -> None:
    methods = ["baseline", "LocVuln", "NacgVuln"]
    labels = [METHOD_LABELS[m] for m in methods]
    metrics = ["Median IFA", "Effort@20%Recall", "Recall@1%LOC"]
    display_names = ["Median IFA", "Effort@20%Recall (%)", "Recall@1%LOC (%)"]
    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, method in enumerate(methods):
        vals = []
        errs = []
        for metric in metrics:
            row = df[(df["method"] == method) & (df["metric"] == metric)]
            vals.append(pct_value(metric, row.iloc[0]["mean"]))
            errs.append(pct_value(metric, row.iloc[0]["std"]))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=labels[i], yerr=errs, capsize=3)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title("Cost-effectiveness comparison")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_ablation_accuracy(data_dir: Path, out_base: Path) -> None:
    path = data_dir / "ablation_all_mean_std_metrics.csv"
    df = pd.read_csv(path)
    variants = [
        "sim_on",
        "chunk_infer_off",
        "chunk_train_off",
        "sim_off",
        "neg_off",
    ]
    labels = [
        "Full model",
        "w/o inference chunking",
        "w/o training chunking",
        "w/o similar-line replacement",
        "w/o negative-aware training",
    ]
    metrics = ["A@10", "R@10", "MRR@10", "MAP@10"]
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, (variant, label) in enumerate(zip(variants, labels)):
        row = df[df["variant"] == variant]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [pct_value(m, row[f"{m}_mean"]) for m in metrics]
        errs = [pct_value(m, row[f"{m}_std"]) for m in metrics]
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, yerr=errs, capsize=2)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

    ax.set_title("Ablation study on accuracy and ranking metrics")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_clean_fpr(data_dir: Path, out_base: Path) -> None:
    path = data_dir / "clean_fpr_mean_std.csv"
    df = pd.read_csv(path)
    variants = ["neg_off", "neg_on"]
    labels = ["w/o negative-aware", "with negative-aware"]
    metrics = ["clean_FPR", "NO_VULN_recall", "vuln_empty_output_rate"]
    metric_labels = ["Clean FPR", "NO_VULN Recall", "Vulnerable empty-output rate"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for i, (variant, label) in enumerate(zip(variants, labels)):
        row = df[df["variant"] == variant]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [pct_value(m, row[f"{m}_mean"]) for m in metrics]
        errs = [pct_value(m, row[f"{m}_std"]) for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, yerr=errs, capsize=3)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title("Impact of negative-aware training on clean-function outputs")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_length_group(data_dir: Path, out_base: Path) -> None:
    path = data_dir / "length_group_mean_std_metrics.csv"
    df = pd.read_csv(path)
    length_order = ["<=20", "21-50", "51-100", ">100"]
    variants = ["chunk_infer_off", "chunk_infer_on"]
    labels = ["w/o inference chunking", "with inference chunking"]
    metric = "A@10"
    x = np.arange(len(length_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for i, (variant, label) in enumerate(zip(variants, labels)):
        vals = []
        errs = []
        for lg in length_order:
            row = df[(df["variant"] == variant) & (df["length_group"] == lg)]
            if row.empty:
                vals.append(np.nan)
                errs.append(0)
            else:
                row = row.iloc[0]
                vals.append(pct_value(metric, row[f"{metric}_mean"]))
                errs.append(pct_value(metric, row[f"{metric}_std"]))
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, yerr=errs, capsize=3)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title("A@10 by function length group")
    ax.set_ylabel("A@10 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(length_order)
    ax.set_xlabel("Function length group (LOC)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_long_function_metrics(data_dir: Path, out_base: Path) -> None:
    path = data_dir / "length_group_mean_std_metrics.csv"
    df = pd.read_csv(path)
    variants = ["chunk_infer_off", "chunk_infer_on"]
    labels = ["w/o inference chunking", "with inference chunking"]
    metrics = ["A@10", "R@10", "MRR@10", "MAP@10"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, (variant, label) in enumerate(zip(variants, labels)):
        vals = []
        errs = []
        for metric in metrics:
            row = df[(df["variant"] == variant) & (df["length_group"] == ">100")]
            if row.empty:
                vals.append(np.nan)
                errs.append(0)
            else:
                row = row.iloc[0]
                vals.append(pct_value(metric, row[f"{metric}_mean"]))
                errs.append(pct_value(metric, row[f"{metric}_std"]))
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, yerr=errs, capsize=3)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title("Impact of inference-time chunking on long functions (>100 LOC)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_base)


def _pretty_compare_label(name: str) -> str:
    s = str(name).strip()
    return COMPARE_LABELS.get(s, s)


def _read_external_a10_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing external comparison CSV: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Approach", "A@10"}
    if not required.issubset(df.columns):
        raise ValueError(f"External A@10 CSV must contain columns {sorted(required)}; got {list(df.columns)}")
    if "Source" not in df.columns:
        df["Source"] = "Reference Table 4"
    out = df[["Approach", "A@10", "Source"]].copy()
    out["Approach"] = out["Approach"].astype(str).str.strip()
    out["Source"] = out["Source"].astype(str).str.strip()
    out["A@10"] = out["A@10"].astype(float).map(lambda x: pct_value("A@10", x))
    return out


def plot_reference_based_a10(external_csv: Path, out_base: Path) -> None:
    """Fig. 5-8: Reference-based A@10 comparison against existing methods."""
    df = _read_external_a10_csv(external_csv)
    source = df["Source"].str.lower()

    ref_mask = source.str.contains("reference")
    proposed_mask = source.str.contains("local 10-seed") & df["Approach"].str.contains(
        "proposed|nacgvuln", case=False, regex=True
    )
    plot_df = df.loc[ref_mask | proposed_mask, ["Approach", "A@10"]].copy()
    plot_df["Approach"] = plot_df["Approach"].replace({"Proposed method": "NacgVuln"})
    plot_df = plot_df.drop_duplicates(subset=["Approach"], keep="last")
    plot_df = plot_df.sort_values("A@10", ascending=False).reset_index(drop=True)
    plot_df["display"] = plot_df["Approach"].map(_pretty_compare_label)

    fig_width = max(9.6, len(plot_df) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["A@10"].values, width=0.68, edgecolor="black", linewidth=0.5)
    ax.set_title("Reference-based A@10 comparison", fontsize=12)
    ax.set_ylabel("A@10 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["display"].tolist(), rotation=35, ha="right")
    ax.set_ylim(0, max(100, float(plot_df["A@10"].max()) * 1.14))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    annotate_bars(ax, bars, fmt="%.1f")
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_local_reproduction_a10(external_csv: Path, out_base: Path) -> None:
    """Fig. 5-9: Local reproduction comparison under the local evaluation protocol."""
    df = _read_external_a10_csv(external_csv)
    source = df["Source"].str.lower()
    local_mask = source.str.contains("local reproduction") | source.str.contains("local 10-seed")
    plot_df = df.loc[local_mask, ["Approach", "A@10"]].copy()
    plot_df["Approach"] = plot_df["Approach"].replace({"Proposed method": "NacgVuln"})
    plot_df = plot_df.drop_duplicates(subset=["Approach"], keep="last")

    # Fixed order makes the figure easier to compare across drafts.
    order = [
        "Cppcheck Analyzer (local)",
        "LineVul Attention (local)",
        "LocVul reproduction",
        "NacgVuln",
    ]
    order_map = {name: i for i, name in enumerate(order)}
    plot_df["order"] = plot_df["Approach"].map(lambda x: order_map.get(x, len(order_map)))
    plot_df = plot_df.sort_values(["order", "A@10"], ascending=[True, False]).reset_index(drop=True)
    plot_df["display"] = plot_df["Approach"].map(_pretty_compare_label)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["A@10"].values, width=0.58, edgecolor="black", linewidth=0.5)
    ax.set_title("Local reproduction A@10 comparison", fontsize=12)
    ax.set_ylabel("A@10 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["display"].tolist())
    ax.set_ylim(0, max(100, float(plot_df["A@10"].max()) * 1.14))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    annotate_bars(ax, bars, fmt="%.1f")
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_comparison_experiments(external_csv: Path, out_dir: Path) -> None:
    plot_reference_based_a10(external_csv, out_dir / "fig5_8_reference_based_a10_comparison")
    plot_local_reproduction_a10(external_csv, out_dir / "fig5_9_local_reproduction_a10_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Directory containing CSV result files")
    parser.add_argument("--out_dir", default="chapter5_figures", help="Output directory for figures")
    parser.add_argument(
        "--external_a10_csv",
        default="chapter5_combined_a10.csv",
        help="CSV for later comparison figures. Required unless --no_compare is set.",
    )
    parser.add_argument("--no_compare", action="store_true", help="Do not generate fig5_8 and fig5_9")
    parser.add_argument("--only_compare", action="store_true", help="Only generate fig5_8 and fig5_9")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_out_dir(out_dir)

    if not args.only_compare:
        main_df = load_main_results(data_dir)
        grouped_bar_from_main(
            main_df,
            metrics=["A@10", "P@10", "R@10"],
            title="Top-10 accuracy, precision and recall comparison",
            ylabel="Percentage (%)",
            out_base=out_dir / "fig5_1_top10_accuracy_precision_recall",
            ylim=(0, 105),
        )
        grouped_bar_from_main(
            main_df,
            metrics=["MRR@10", "MAP@10"],
            title="Rank-aware line-level localization quality",
            ylabel="Percentage (%)",
            out_base=out_dir / "fig5_2_mrr_map",
            ylim=(0, 105),
        )
        grouped_bar_cost(main_df, out_dir / "fig5_3_cost_effectiveness")
        plot_ablation_accuracy(data_dir, out_dir / "fig5_4_ablation_accuracy")
        plot_clean_fpr(data_dir, out_dir / "fig5_5_clean_fpr_negative_aware")
        plot_length_group(data_dir, out_dir / "fig5_6_length_group_a10")
        plot_long_function_metrics(data_dir, out_dir / "fig5_7_long_function_chunking")

    if not args.no_compare:
        external_csv = Path(args.external_a10_csv)
        if not external_csv.is_absolute():
            data_candidate = data_dir / external_csv
            external_csv = data_candidate if data_candidate.exists() else external_csv.resolve()
        plot_comparison_experiments(external_csv, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
