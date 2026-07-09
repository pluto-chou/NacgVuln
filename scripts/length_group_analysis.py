from pathlib import Path
import argparse
import pandas as pd
import numpy as np

K = 10

def normalize_line(s):
    return " ".join(str(s).strip().split())

def split_lines(s):
    if pd.isna(s):
        return []
    s = str(s)
    if not s.strip():
        return []
    if "/~/" in s:
        return [x.strip() for x in s.split("/~/") if x.strip()]
    return [x.strip() for x in s.splitlines() if x.strip()]

def length_group(loc):
    loc = int(loc)
    if loc <= 20:
        return "<=20"
    if loc <= 50:
        return "21-50"
    if loc <= 100:
        return "51-100"
    return ">100"

def compute_pr(row, k=10):
    ranked = split_lines(row.get("ranked_topK", ""))[:k]
    gt = split_lines(row.get("gt_lines", ""))
    gt_set = set(normalize_line(x) for x in gt if normalize_line(x))
    if not gt_set:
        return 0.0, 0.0

    hits = 0
    for line in ranked:
        if normalize_line(line) in gt_set:
            hits += 1

    precision = hits / k
    recall = hits / len(gt_set)
    return precision, recall

def infer_seed_from_name(path: Path):
    name = path.name
    # examples: chunk_infer_on_seed0_details.csv
    import re
    m = re.search(r"seed(\d+)", name)
    if m:
        return int(m.group(1))
    return -1

def read_variant_details(ablation_root: Path, variant: str):
    eval_dir = ablation_root / variant / "eval_outputs"
    if not eval_dir.exists():
        raise FileNotFoundError(f"缺少 eval_outputs 目录: {eval_dir}")

    detail_files = sorted(eval_dir.glob("*_details.csv"))
    if not detail_files:
        raise FileNotFoundError(f"没有找到 details.csv: {eval_dir}")

    frames = []
    for p in detail_files:
        df = pd.read_csv(p)
        if "target" in df.columns:
            df = df[df["target"].astype(int) == 1].copy()

        if df.empty:
            continue

        df["seed"] = infer_seed_from_name(p)
        df["variant"] = variant
        df["length_group"] = df["loc"].apply(length_group)

        pr = df.apply(lambda r: compute_pr(r, K), axis=1)
        df["P@10_row"] = [x[0] for x in pr]
        df["R@10_row"] = [x[1] for x in pr]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_root", required=True)
    ap.add_argument("--variants", default="full_improved,chunk_infer_off,chunk_infer_on,chunk_train_off,chunk_train_on")
    ap.add_argument("--out_dir", default="paper_extra_results/length_group")
    args = ap.parse_args()

    ablation_root = Path(args.ablation_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    frames = []

    for v in variants:
        try:
            df = read_variant_details(ablation_root, v)
            if not df.empty:
                frames.append(df)
                print(f"[OK] {v}: {len(df)} rows")
            else:
                print(f"[WARN] {v}: empty")
        except Exception as e:
            print(f"[WARN] skip {v}: {e}")

    if not frames:
        raise RuntimeError("没有可用 details 数据。请先运行消融评估，确保 eval_outputs/*_details.csv 存在。")

    all_df = pd.concat(frames, ignore_index=True)

    # 每个 seed、每个长度组先算一次，避免某个 seed 样本过多直接支配最终结果
    group_rows = []
    for (variant, seed, lg), g in all_df.groupby(["variant", "seed", "length_group"]):
        row = {
            "variant": variant,
            "seed": seed,
            "length_group": lg,
            "num_samples": len(g),
            "avg_loc": g["loc"].mean(),
            "A@10": g["A@K"].mean() if "A@K" in g.columns else np.nan,
            "P@10": g["P@10_row"].mean(),
            "R@10": g["R@10_row"].mean(),
            "MRR@10": g["MRR@K"].mean() if "MRR@K" in g.columns else np.nan,
            "MAP@10": g["MAP@K"].mean() if "MAP@K" in g.columns else np.nan,
            "Median IFA": g["IFA"].median() if "IFA" in g.columns else np.nan,
        }
        group_rows.append(row)

    per_seed_group = pd.DataFrame(group_rows)
    per_seed_path = out_dir / "length_group_per_seed_metrics.csv"
    per_seed_group.to_csv(per_seed_path, index=False)

    summary_rows = []
    metric_cols = ["num_samples", "avg_loc", "A@10", "P@10", "R@10", "MRR@10", "MAP@10", "Median IFA"]

    order = ["<=20", "21-50", "51-100", ">100"]

    for (variant, lg), g in per_seed_group.groupby(["variant", "length_group"]):
        row = {
            "variant": variant,
            "length_group": lg,
        }
        for c in metric_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["length_group"] = pd.Categorical(summary["length_group"], categories=order, ordered=True)
    summary = summary.sort_values(["variant", "length_group"])
    summary_path = out_dir / "length_group_mean_std_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:")
    print(per_seed_path)
    print(summary_path)

if __name__ == "__main__":
    main()
