#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

HIGHER_BETTER = {"A@10", "P@10", "R@10", "MRR@10", "MAP@10", "Recall@1%LOC"}
LOWER_BETTER = {"Median IFA", "Effort@20%Recall"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_a", required=True, help="variant A 的 per_seed_metrics.csv")
    ap.add_argument("--csv_b", required=True, help="variant B 的 per_seed_metrics.csv")
    ap.add_argument("--output_csv", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    a = pd.read_csv(args.csv_a)
    b = pd.read_csv(args.csv_b)
    merged = a.merge(b, on="seed", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("两个 per_seed_metrics.csv 没有共同 seed，无法做 Wilcoxon")

    rows = []
    metrics = [
        "A@10", "P@10", "R@10", "MRR@10", "MAP@10",
        "Median IFA", "Effort@20%Recall", "Recall@1%LOC",
    ]
    for metric in metrics:
        xa = merged[f"{metric}_a"].astype(float).tolist()
        xb = merged[f"{metric}_b"].astype(float).tolist()
        if metric in HIGHER_BETTER:
            alt = "greater"
        else:
            alt = "less"
        stat, p = wilcoxon(xa, xb, alternative=alt)
        rows.append({
            "metric": metric,
            "alternative": f"A {alt} than B",
            "n": len(xa),
            "A_mean": sum(xa) / len(xa),
            "B_mean": sum(xb) / len(xb),
            "statistic": stat,
            "p_value": p,
            "significant_p_lt_0_05": bool(p < 0.05),
        })

    out_df = pd.DataFrame(rows)
    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        out_path = Path("wilcoxon_results.csv")
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
