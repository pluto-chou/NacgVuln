#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_cwe_paper_tables.py

Purpose:
1. Verify UNKNOWN CWE labels under the same evaluation filtering protocol.
2. Generate a filtered main-paper CWE-level table.
3. Generate a raw appendix CWE-level table.
4. Generate ready-to-use Chinese and English text for the paper.

Expected existing files:
- data/dataset.csv
- paper_extra_results/cwe_level/cwe_level_top10_table.csv
- paper_extra_results/cwe_level/cwe_level_joined_details.csv

Default output directory:
- paper_extra_results/cwe_level_paper/
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def canonicalize_cwe(x: Any) -> str:
    """Convert CWE labels into a stable format such as CWE-119."""
    if x is None:
        return "UNKNOWN"
    if isinstance(x, float) and math.isnan(x):
        return "UNKNOWN"

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return "UNKNOWN"

    m = re.search(r"CWE[-_\s]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"CWE-{int(m.group(1))}"

    m = re.search(r"\b(\d{1,5})\b", s)
    if m:
        return f"CWE-{int(m.group(1))}"

    return s


def write_markdown_table(df: pd.DataFrame, path: Path, title: str | None = None, note: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def load_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path)


def audit_unknown(
    dataset_path: Path,
    joined_details_path: Path,
    out_dir: Path,
) -> None:
    """Audit whether UNKNOWN labels come from raw missing CWE labels."""
    dataset = load_required_csv(dataset_path, "dataset.csv")
    joined = load_required_csv(joined_details_path, "cwe_level_joined_details.csv")

    required_dataset_cols = ["processed_func", "target", "flaw_line", "flaw_line_index"]
    for col in required_dataset_cols:
        if col not in dataset.columns:
            raise ValueError(f"dataset.csv missing required column: {col}")

    if "CWE ID" in dataset.columns:
        cwe_col = "CWE ID"
    elif "CWE" in dataset.columns:
        cwe_col = "CWE"
    else:
        raise ValueError("dataset.csv must contain either 'CWE ID' or 'CWE'.")

    if "row" not in joined.columns or "CWE" not in joined.columns:
        raise ValueError("joined_details must contain 'row' and 'CWE' columns.")

    # LocVul-style fixed-order split: last 10% as test set.
    n = int(0.1 * len(dataset))
    if n <= 0:
        raise ValueError(f"dataset.csv too small for 10% test split: {len(dataset)} rows")

    raw_test = dataset.iloc[-n:].copy().reset_index(drop=True)

    eval_test = pd.DataFrame({
        "raw_test_row": raw_test.index,
        "Text": raw_test["processed_func"],
        "target": raw_test["target"],
        "Lines": raw_test["flaw_line"],
        "Line_Index": raw_test["flaw_line_index"],
        "CWE_raw": raw_test[cwe_col],
        "CWE": raw_test[cwe_col].apply(canonicalize_cwe),
    })

    # REMOVE_MISSING_LINE_LABELS=yes:
    # remove vulnerable samples with missing line-level labels.
    missing_line_label = (
        (eval_test["target"].astype(int) == 1)
        & (eval_test["Lines"].isna() | eval_test["Line_Index"].isna())
    )
    eval_test = eval_test.loc[~missing_line_label].reset_index(drop=True)

    # EVAL_ONLY_VULN=yes:
    eval_test = eval_test.loc[eval_test["target"].astype(int) == 1].reset_index(drop=True)

    unknown_rows = sorted(
        joined[joined["CWE"].astype(str).str.upper().eq("UNKNOWN")]["row"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    valid_rows = [r for r in unknown_rows if 0 <= r < len(eval_test)]
    unknown_audit = eval_test.loc[valid_rows, ["raw_test_row", "target", "CWE_raw", "CWE", "Lines", "Line_Index"]].copy()
    cwe_counts = eval_test["CWE"].value_counts(dropna=False).reset_index()
    cwe_counts.columns = ["CWE", "#Functions"]

    out_dir.mkdir(parents=True, exist_ok=True)
    unknown_audit.to_csv(out_dir / "cwe_unknown_audit.csv", index=False)
    cwe_counts.to_csv(out_dir / "cwe_counts_in_filtered_eval_test.csv", index=False)

    write_markdown_table(
        unknown_audit,
        out_dir / "cwe_unknown_audit.md",
        title="UNKNOWN CWE audit",
        note="These rows are checked after applying REMOVE_MISSING_LINE_LABELS=yes and EVAL_ONLY_VULN=yes.",
    )

    write_markdown_table(
        cwe_counts,
        out_dir / "cwe_counts_in_filtered_eval_test.md",
        title="CWE counts in filtered vulnerable test set",
    )


def build_tables(
    raw_table_path: Path,
    out_dir: Path,
    min_funcs: int,
    exclude_unknown: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = load_required_csv(raw_table_path, "cwe_level_top10_table.csv")

    required_cols = ["CWE", "#Functions", "#VulnLines"]
    for col in required_cols:
        if col not in raw_df.columns:
            raise ValueError(f"raw table missing required column: {col}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw appendix table: keep everything, including UNKNOWN and low-frequency CWEs.
    appendix_df = raw_df.copy()
    appendix_df.to_csv(out_dir / "table_cwe_level_appendix_raw.csv", index=False)
    write_markdown_table(
        appendix_df,
        out_dir / "table_cwe_level_appendix_raw.md",
        title="Raw CWE-level localization results",
        note="This table keeps UNKNOWN labels and low-frequency CWE categories. It is intended for appendix or supplementary material.",
    )

    # Main-paper full metric table.
    main_df = raw_df.copy()

    if exclude_unknown:
        main_df = main_df[main_df["CWE"].astype(str).str.upper() != "UNKNOWN"].copy()

    main_df = main_df[main_df["#Functions"].astype(int) >= int(min_funcs)].copy()
    main_df = main_df.sort_values(["#Functions", "#VulnLines"], ascending=False).reset_index(drop=True)

    main_df.to_csv(out_dir / "table_cwe_level_main_full_metrics.csv", index=False)
    write_markdown_table(
        main_df,
        out_dir / "table_cwe_level_main_full_metrics.md",
        title="Main-paper CWE-level localization results",
        note=f"UNKNOWN labels and CWE groups with fewer than {min_funcs} vulnerable functions were excluded.",
    )

    # A compact main-paper table. This is recommended for the manuscript body.
    preferred_cols = [
        "CWE",
        "#Functions",
        "#VulnLines",
        "A@10",
        "P@10",
        "R@10",
        "MRR@10",
        "MAP@10",
        "Median IFA",
    ]
    compact_cols = [c for c in preferred_cols if c in main_df.columns]
    compact_df = main_df[compact_cols].copy()

    compact_df.to_csv(out_dir / "table_cwe_level_main_compact.csv", index=False)
    write_markdown_table(
        compact_df,
        out_dir / "table_cwe_level_main_compact.md",
        title="Compact CWE-level localization results for manuscript body",
        note=(
            f"UNKNOWN labels and CWE groups with fewer than {min_funcs} vulnerable functions were excluded. "
            "The complete metric table is provided separately."
        ),
    )

    return appendix_df, main_df, compact_df


def write_paper_text(out_dir: Path, min_funcs: int) -> None:
    cn_text = f"""# 可直接放入正文的中文内容

## Additional Analysis: CWE-level Performance

为进一步分析 NacgVuln 在不同漏洞类型上的表现，本文按照 CWE 标签对测试集中的漏洞函数进行分组，并统计高频 CWE 类型上的行级定位性能。CWE 是软件与硬件弱点的标准化分类体系，可用于描述不同漏洞弱点类型。对于缺失有效 CWE 标签的样本，本文将其标记为 UNKNOWN；由于 UNKNOWN 不对应具体 CWE 类型，主文表格中不纳入该类样本。同时，为避免低频类别造成不稳定估计，本文仅在主文中报告测试集中不少于 {min_funcs} 个漏洞函数的 CWE 类型，完整结果放入附录。

表 5-X 给出了高频 CWE 类型上的定位结果。可以看出，NacgVuln 在 CWE-119、CWE-264、CWE-20 和 CWE-125 等相对高频类型上能够保持较高的 Top-10 定位效果。然而，不同 CWE 类型之间的排序质量和审计成本仍存在差异。例如，部分 CWE 类型虽然 A@10 较高，但 MRR@10 或 MAP@10 相对较低，说明模型能够在 Top-10 范围内命中真实漏洞行，但候选行排序仍不稳定。该结果表明，生成式行级漏洞定位模型在不同漏洞模式上的学习难度并不一致，CWE-level 分析可作为整体指标之外的补充证据。

需要强调的是，CWE-level 结果主要用于补充说明模型在不同漏洞类型上的行为差异，而不应被解释为对跨 CWE 泛化能力的决定性证明。由于测试集中 CWE 标签分布不均衡，且部分 CWE 类型样本数较少，本文在 Threats to Validity 中进一步讨论该限制。
"""

    en_text = f"""# English text for manuscript

## Additional Analysis: CWE-level Performance

To further examine the behavior of NacgVuln across vulnerability types, we group vulnerable test functions by their CWE labels and report line-level localization results for frequent CWE categories. CWE provides a standardized taxonomy of software and hardware weaknesses. Samples without valid CWE labels were marked as UNKNOWN; since UNKNOWN does not correspond to a specific CWE category, these samples were excluded from the main table. To avoid unstable estimates caused by low-frequency categories, we only report CWE categories with at least {min_funcs} vulnerable functions in the main text, while the complete results are provided in the appendix.

Table 5-X presents the CWE-level localization results. The results show that NacgVuln achieves high Top-10 localization performance on frequent categories such as CWE-119, CWE-264, CWE-20, and CWE-125. However, rank-aware metrics and inspection-effort metrics still vary across CWE types. For some categories, A@10 remains high while MRR@10 or MAP@10 is relatively lower, indicating that the model can hit vulnerable lines within the Top-10 candidates but does not always rank them at the very top. This suggests that vulnerability-type-specific difficulty remains an important factor in generative line-level vulnerability localization.

It should be noted that the CWE-level results serve as complementary evidence on the behavior of the proposed method across vulnerability types, rather than a conclusive assessment of cross-CWE generalization. Since the CWE distribution in the test set is imbalanced and several CWE categories contain only a small number of samples, we further discuss this limitation in Threats to Validity.
"""

    threats_cn = """# Threats to Validity 可补充中文内容

CWE-level 分析受到测试集中 CWE 标签分布不均衡的影响。部分漏洞函数缺失有效 CWE 标签，且若干 CWE 类型仅包含少量样本。因此，本文仅将 CWE-level 结果作为整体指标之外的补充分析，而不将其解释为对跨 CWE 泛化能力的决定性证明。
"""

    threats_en = """# Threats to Validity English note

The CWE-level analysis is limited by the imbalanced distribution of CWE labels in the test set. Some vulnerable functions do not have valid CWE labels, and several CWE categories contain only a small number of samples. Therefore, the CWE-level results should be interpreted as complementary evidence rather than a conclusive assessment of cross-CWE generalization.
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "section_cwe_level_performance_cn.md").write_text(cn_text, encoding="utf-8")
    (out_dir / "section_cwe_level_performance_en.md").write_text(en_text, encoding="utf-8")
    (out_dir / "threats_cwe_note_cn.md").write_text(threats_cn, encoding="utf-8")
    (out_dir / "threats_cwe_note_en.md").write_text(threats_en, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/dataset.csv", type=str)
    parser.add_argument("--raw_table", default="paper_extra_results/cwe_level/cwe_level_top10_table.csv", type=str)
    parser.add_argument("--joined_details", default="paper_extra_results/cwe_level/cwe_level_joined_details.csv", type=str)
    parser.add_argument("--out_dir", default="paper_extra_results/cwe_level_paper", type=str)
    parser.add_argument("--min_funcs", default=3, type=int)
    parser.add_argument("--exclude_unknown", default="yes", choices=["yes", "no"])
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    raw_table_path = Path(args.raw_table)
    joined_details_path = Path(args.joined_details)
    out_dir = Path(args.out_dir)

    audit_unknown(
        dataset_path=dataset_path,
        joined_details_path=joined_details_path,
        out_dir=out_dir,
    )

    appendix_df, main_df, compact_df = build_tables(
        raw_table_path=raw_table_path,
        out_dir=out_dir,
        min_funcs=args.min_funcs,
        exclude_unknown=(args.exclude_unknown == "yes"),
    )

    write_paper_text(out_dir=out_dir, min_funcs=args.min_funcs)

    print("\n[OK] CWE paper tables generated.")
    print(f"Output directory: {out_dir}")
    print("\nMain files:")
    print(f"- {out_dir / 'table_cwe_level_main_compact.md'}")
    print(f"- {out_dir / 'table_cwe_level_main_full_metrics.md'}")
    print(f"- {out_dir / 'table_cwe_level_appendix_raw.md'}")
    print(f"- {out_dir / 'cwe_unknown_audit.md'}")
    print(f"- {out_dir / 'section_cwe_level_performance_cn.md'}")
    print(f"- {out_dir / 'section_cwe_level_performance_en.md'}")
    print(f"- {out_dir / 'threats_cwe_note_cn.md'}")
    print(f"- {out_dir / 'threats_cwe_note_en.md'}")

    print("\nMain compact table preview:")
    print(compact_df.to_string(index=False))


if __name__ == "__main__":
    main()
