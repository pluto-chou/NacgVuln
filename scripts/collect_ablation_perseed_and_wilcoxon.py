from pathlib import Path
import argparse
import pandas as pd
from scipy.stats import wilcoxon

METRICS = [
    "A@10", "P@10", "R@10", "MRR@10", "MAP@10",
    "Median IFA", "Effort@20%Recall", "Recall@1%LOC"
]

HIGHER_BETTER = {"A@10", "P@10", "R@10", "MRR@10", "MAP@10", "Recall@1%LOC"}
LOWER_BETTER = {"Median IFA", "Effort@20%Recall"}

VARIANTS = [
    "full_improved",
    "neg_off", "neg_on",
    "chunk_train_off", "chunk_train_on",
    "chunk_infer_off", "chunk_infer_on",
    "sim_off", "sim_on",
]

PAIRS = [
    ("neg_on", "neg_off", "负样本感知"),
    ("chunk_train_on", "chunk_train_off", "训练期分块"),
    ("chunk_infer_on", "chunk_infer_off", "推理期分块"),
    ("sim_on", "sim_off", "相似行替换"),
]

def normalize_metric_value(metric: str, value: float) -> float:
    """把百分数和小数统一成小数。比如 94.4 -> 0.944。"""
    if pd.isna(value):
        return value
    v = float(value)
    if metric != "Median IFA" and abs(v) > 1.5:
        return v / 100.0
    return v

def read_variant(root: Path, variant: str) -> pd.DataFrame:
    path = root / variant / "per_seed_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    df = pd.read_csv(path)
    df["variant"] = variant

    for m in METRICS:
        if m not in df.columns:
            raise ValueError(f"{path} 缺少指标列: {m}")
        df[m] = df[m].apply(lambda x: normalize_metric_value(m, x))

    return df[["variant", "seed"] + METRICS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_root", required=True)
    ap.add_argument("--out_dir", default="paper_extra_results")
    args = ap.parse_args()

    root = Path(args.ablation_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for v in VARIANTS:
        p = root / v / "per_seed_metrics.csv"
        if p.exists():
            frames.append(read_variant(root, v))
        else:
            print(f"[WARN] 跳过不存在的变体: {v}")

    all_df = pd.concat(frames, ignore_index=True)
    all_path = out_dir / "ablation_all_per_seed_metrics.csv"
    all_df.to_csv(all_path, index=False)

    summary_rows = []
    for variant, g in all_df.groupby("variant"):
        row = {"variant": variant}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "ablation_all_mean_std_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    test_rows = []
    for on_variant, off_variant, label in PAIRS:
        if on_variant not in set(all_df["variant"]) or off_variant not in set(all_df["variant"]):
            print(f"[WARN] 跳过 Wilcoxon: {on_variant} vs {off_variant}")
            continue

        on_df = all_df[all_df["variant"] == on_variant].sort_values("seed")
        off_df = all_df[all_df["variant"] == off_variant].sort_values("seed")

        common = sorted(set(on_df["seed"]) & set(off_df["seed"]))
        on_df = on_df[on_df["seed"].isin(common)].sort_values("seed")
        off_df = off_df[off_df["seed"].isin(common)].sort_values("seed")

        for m in METRICS:
            x = on_df[m].astype(float).to_numpy()
            y = off_df[m].astype(float).to_numpy()

            try:
                stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), float("nan")

            delta = x.mean() - y.mean()
            if m in LOWER_BETTER:
                better = delta < 0
            else:
                better = delta > 0

            test_rows.append({
                "module": label,
                "on_variant": on_variant,
                "off_variant": off_variant,
                "metric": m,
                "on_mean": x.mean(),
                "off_mean": y.mean(),
                "delta": delta,
                "better": better,
                "wilcoxon_stat": stat,
                "p_value": p,
                "significant_0.05": bool(p < 0.05) if pd.notna(p) else False,
            })

    test_df = pd.DataFrame(test_rows)
    test_path = out_dir / "ablation_wilcoxon_tests.csv"
    test_df.to_csv(test_path, index=False)

    print("Saved:")
    print(all_path)
    print(summary_path)
    print(test_path)

if __name__ == "__main__":
    main()
