#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_locvul_fig2_style_fixed.py

Generate a LocVul Fig.2-style comparison figure from a real dataset sample.

Panels:
  (A) Original code with ground-truth vulnerable lines highlighted in red.
  (B) XAI / Self-Attention ranked output using a real xai_row*.csv if provided.
      By default it shows Top-K rows and appends missed ground-truth rows, so that
      vulnerable lines ranked outside Top-K are still visible in the figure.
  (C) LocVul / NacgVuln output: predicted vulnerable lines are placed first.

Expected candidate CSV columns:
  processed_func, flaw_line_index, flaw_line, project, CWE ID

Expected XAI CSV columns, compatible with extract_attention_row.py:
  rank, line_index_0based, line_number, line_text, score, is_gt
Only line_number, line_text, and score are strictly required.

Example:
  python make_locvul_fig2_style_fixed.py \
    --csv picture_candidates.csv \
    --row-id 32 \
    --xai-csv xai_row32.csv \
    --pred-lines 6,9,11 \
    --out fig2_real_attention.png
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# -----------------------------
# Utilities
# -----------------------------

def norm_line(s: object) -> str:
    return " ".join(str(s).strip().split())


def shorten(s: object, max_chars: int) -> str:
    text = str(s).replace("\t", "    ").rstrip("\n")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def parse_int_list(x: object) -> List[int]:
    """Parse list-like values such as '[5,8,10]', '5,8,10', '5 /~/ 8'."""
    if x is None:
        return []
    try:
        if isinstance(x, float) and math.isnan(x):
            return []
    except Exception:
        pass

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []

    try:
        v = ast.literal_eval(s)
        if isinstance(v, int):
            return [int(v)]
        if isinstance(v, (list, tuple, set)):
            out: List[int] = []
            for item in v:
                try:
                    out.append(int(item))
                except Exception:
                    pass
            return out
    except Exception:
        pass

    return [int(n) for n in re.findall(r"-?\d+", s)]


def resolve_gt_indices(row: pd.Series, lines: Sequence[str]) -> List[int]:
    """Return zero-based ground-truth vulnerable line indices."""
    raw = parse_int_list(row.get("flaw_line_index", ""))
    flaw_text = str(row.get("flaw_line", ""))
    flaw_parts = [norm_line(p) for p in re.split(r"/~/?|\n", flaw_text) if norm_line(p)]

    # First use indices. In the LocVul/Big-Vul processing used here, indices are usually 0-based.
    out: List[int] = []
    for idx in raw:
        if 0 <= idx < len(lines):
            out.append(idx)
        elif 1 <= idx <= len(lines):
            out.append(idx - 1)

    if out:
        return sorted(set(out))

    # Fallback: exact/contained line text matching.
    for i, line in enumerate(lines):
        nline = norm_line(line)
        if any(part and (part == nline or part in nline or nline in part) for part in flaw_parts):
            out.append(i)
    return sorted(set(out))


def parse_pred_lines(pred_lines: str | None, default_idx: Sequence[int]) -> List[int]:
    """
    Parse predicted line numbers from CLI.
    Input is 1-based line numbers, e.g. '6,9,11'.
    If not provided, defaults to ground-truth indices for drawing a method illustration.
    """
    if pred_lines is None or str(pred_lines).strip() == "":
        return sorted(set(int(i) for i in default_idx))
    nums = parse_int_list(pred_lines)
    return sorted(set(n - 1 for n in nums if n > 0))


def crop_context(lines: Sequence[str], gt_idx: Sequence[int], pred_idx: Sequence[int], radius: int = 12) -> Tuple[int, int]:
    anchors = list(gt_idx) + list(pred_idx)
    if not anchors:
        return 0, min(len(lines), 35)
    lo = max(0, min(anchors) - radius)
    hi = min(len(lines), max(anchors) + radius + 1)
    if hi - lo < 18:
        hi = min(len(lines), lo + 18)
    return lo, hi


# -----------------------------
# XAI rows
# -----------------------------

def make_demo_xai(lines: Sequence[str], gt_idx: Sequence[int], display_range: Tuple[int, int]) -> List[Tuple[int, str, float, int | None]]:
    """Fallback demo ranking if real xai-csv is not provided."""
    lo, hi = display_range
    candidates = list(range(lo, hi))
    non_gt = [i for i in candidates if i not in set(gt_idx)]

    priority: List[int] = []
    for kw in ["if", "strstr", "return", "find", "log", "recvfrom", "memset", "sizeof"]:
        for i in non_gt:
            if kw in lines[i] and i not in priority:
                priority.append(i)

    order: List[int] = []
    order.extend(priority[:4])
    order.extend(list(gt_idx[:1]))
    order.extend(priority[4:7])
    order.extend(list(gt_idx[1:]))
    order.extend([i for i in candidates if i not in order])

    scores = [27.9, 23.9, 19.6, 18.2, 17.7, 13.2, 12.9, 11.6, 9.5, 5.7, 5.5, 3.9, 1.4, 0.9, 0.5, 0.0]
    rows: List[Tuple[int, str, float, int | None]] = []
    for k, i in enumerate(order[:16]):
        rows.append((i, lines[i], scores[k] if k < len(scores) else 0.0, k + 1))
    return rows


def load_xai_csv(path: str | Path) -> List[Tuple[int, str, float, int | None]]:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    line_no_col = cols.get("line_number") or cols.get("line") or cols.get("lineno")
    idx_col = cols.get("line_index_0based") or cols.get("line_index") or cols.get("idx")
    text_col = cols.get("line_text") or cols.get("text") or cols.get("code")
    score_col = cols.get("score") or cols.get("attention_score_pct") or cols.get("attention_score") or cols.get("attention")
    rank_col = cols.get("rank")

    if text_col is None or score_col is None or (line_no_col is None and idx_col is None):
        raise ValueError(
            "XAI CSV must contain line_number or line_index_0based, plus line_text and score columns."
        )

    rows: List[Tuple[int, str, float, int | None]] = []
    for _, r in df.iterrows():
        if idx_col is not None:
            idx0 = int(r[idx_col])
        else:
            idx0 = int(r[line_no_col]) - 1
        rank = int(r[rank_col]) if rank_col is not None and not pd.isna(r[rank_col]) else None
        rows.append((idx0, str(r[text_col]), float(r[score_col]), rank))

    # Preserve explicit rank if available; otherwise sort by score.
    if any(rank is not None for *_, rank in rows):
        rows.sort(key=lambda x: (x[3] if x[3] is not None else 10**9))
    else:
        rows.sort(key=lambda x: x[2], reverse=True)
        rows = [(idx, txt, score, k + 1) for k, (idx, txt, score, _) in enumerate(rows)]
    return rows


def select_xai_rows(
    xai_rows: Sequence[Tuple[int, str, float, int | None]],
    gt_idx: Sequence[int],
    top_n: int = 10,
    include_missed_gt: bool = True,
) -> List[Tuple[int, str, float, int | None]]:
    """Show Top-N and append missed GT rows ranked outside Top-N."""
    selected = list(xai_rows[:top_n])
    selected_idx = {r[0] for r in selected}
    if include_missed_gt:
        for r in xai_rows[top_n:]:
            if r[0] in set(gt_idx) and r[0] not in selected_idx:
                selected.append(r)
                selected_idx.add(r[0])
    return selected


# -----------------------------
# Drawing
# -----------------------------

def setup_panel(ax, title: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.025, title, fontsize=11, fontweight="bold", va="bottom")


def draw_code_panel(
    ax,
    title: str,
    lines: Sequence[str],
    line_numbers: Sequence[int],
    red_idx: Iterable[int] | None = None,
    gray_idx: Iterable[int] | None = None,
    max_chars: int = 72,
):
    red_idx = set(red_idx or [])
    gray_idx = set(gray_idx or [])
    setup_panel(ax, title)

    n = len(lines)
    top = 0.955
    bottom = 0.035
    line_h = (top - bottom) / max(n, 1)
    ax.add_patch(Rectangle((0, bottom - 0.012), 1, top - bottom + 0.024,
                           facecolor="#f7f7f7", edgecolor="#d0d0d0", linewidth=0.8))

    for j, (ln, txt) in enumerate(zip(line_numbers, lines)):
        y = top - j * line_h - line_h * 0.72
        idx0 = ln - 1
        if idx0 in gray_idx:
            ax.add_patch(Rectangle((0.035, y - line_h * 0.18), 0.94, line_h * 0.90,
                                   facecolor="#c9c9c9", alpha=0.82, edgecolor="none"))
        if idx0 in red_idx:
            ax.add_patch(Rectangle((0.035, y - line_h * 0.18), 0.94, line_h * 0.90,
                                   facecolor="#ff4b4b", alpha=0.80, edgecolor="none"))
        ax.text(0.015, y, f"{ln:>2}", fontsize=7.4, family="DejaVu Sans Mono", color="#666666")
        ax.text(0.072, y, shorten(txt, max_chars), fontsize=7.4, family="DejaVu Sans Mono", color="#202020")


def draw_xai_table(
    ax,
    title: str,
    xai_rows: Sequence[Tuple[int, str, float, int | None]],
    gt_idx: Sequence[int],
    top_n: int,
):
    """Draw panel B as a compact, publication-style ranking table.

    The vulnerable rows are now highlighted by a full-width red rectangle and an
    additional code-cell border, so the red frame completely encloses the code
    text instead of looking like a partial background band.
    """
    setup_panel(ax, title)

    gt_set = set(gt_idx)

    # Column anchors: keep enough width for code and avoid overlap with Score.
    x_rank = 0.018
    x_line = 0.092
    x_code = 0.168
    x_score = 0.972
    x_code_right = 0.905

    table_left = 0.006
    table_right = 0.992
    header_top = 0.962
    header_bottom = 0.925
    body_bottom = 0.062

    # Table background and header strip.
    ax.add_patch(Rectangle((table_left, body_bottom), table_right - table_left, header_top - body_bottom,
                           facecolor="#fbfbfb", edgecolor="#c9c9c9", linewidth=0.8, zorder=0))
    ax.add_patch(Rectangle((table_left, header_bottom), table_right - table_left, header_top - header_bottom,
                           facecolor="#f1f1f1", edgecolor="#c9c9c9", linewidth=0.6, zorder=1))

    # Header text.
    header_fs = 8.0
    ax.text(x_rank, 0.943, "Rank", fontsize=header_fs, fontweight="bold", va="center", zorder=3)
    ax.text(x_line, 0.943, "Line", fontsize=header_fs, fontweight="bold", va="center", zorder=3)
    ax.text(x_code, 0.943, "Code line", fontsize=header_fs, fontweight="bold", va="center", zorder=3)
    ax.text(x_score, 0.943, "Score", fontsize=header_fs, fontweight="bold", ha="right", va="center", zorder=3)

    # Subtle vertical separators.
    for x in [0.075, 0.148, x_code_right + 0.018]:
        ax.plot([x, x], [body_bottom, header_top], color="#dddddd", linewidth=0.45, zorder=1)

    n = len(xai_rows)
    row_h = (header_bottom - body_bottom) / max(n, 1)
    text_fs = 7.05 if n >= 12 else 7.35

    for k, (idx0, txt, score, rank) in enumerate(xai_rows):
        row_top = header_bottom - k * row_h
        row_bottom = row_top - row_h
        y = row_bottom + row_h * 0.50

        # Alternating background for readability.
        if k % 2 == 1:
            ax.add_patch(Rectangle((table_left, row_bottom), table_right - table_left, row_h,
                                   facecolor="#f8f8f8", edgecolor="none", zorder=0.5))

        # Ground-truth vulnerable rows: full-width red frame + code-cell red frame.
        if idx0 in gt_set:
            inset_y = row_h * 0.08
            ax.add_patch(Rectangle((table_left + 0.002, row_bottom + inset_y),
                                   table_right - table_left - 0.004, row_h - 2 * inset_y,
                                   facecolor="#ff6b6b", alpha=0.88,
                                   edgecolor="#c80000", linewidth=0.95, zorder=2))
            ax.add_patch(Rectangle((x_code - 0.010, row_bottom + row_h * 0.18),
                                   x_code_right - x_code + 0.014, row_h * 0.64,
                                   facecolor="none", edgecolor="#b00000",
                                   linewidth=0.85, zorder=3))

        # Row separator after highlight so the table keeps a neat grid.
        ax.plot([table_left, table_right], [row_bottom, row_bottom],
                color="#e1e1e1", linewidth=0.45, zorder=2.5)

        rank_str = str(rank if rank is not None else k + 1)
        if rank is not None and rank > top_n and idx0 in gt_set:
            rank_str = f"{rank}*"

        code_text = shorten(txt.strip(), 66)

        ax.text(x_rank, y, rank_str, fontsize=text_fs, family="DejaVu Sans Mono",
                va="center", color="#111111", zorder=4)
        ax.text(x_line, y, str(idx0 + 1), fontsize=text_fs, family="DejaVu Sans Mono",
                va="center", color="#111111", zorder=4)
        ax.text(x_code, y, code_text, fontsize=text_fs, family="DejaVu Sans Mono",
                va="center", color="#111111", zorder=4, clip_on=True)
        ax.text(x_score, y, f"{score:.2f}", fontsize=text_fs, family="DejaVu Sans Mono",
                ha="right", va="center", color="#111111", zorder=4)

    ax.text(table_left, 0.018, "*: ground-truth line ranked outside Top-K but appended for analysis.",
            fontsize=6.8, color="#555555", va="bottom")


def draw_xai_box(ax):
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#08345c", edgecolor="#08345c", linewidth=1.0))
    ax.text(0.5, 0.58, "XAI", fontsize=24, fontweight="bold", color="#ffb347", ha="center", va="center")
    ax.text(0.5, 0.28, "Self-Attention\nline ranking", fontsize=8.0, color="white", ha="center", va="center")


def add_arrow(fig, start: Tuple[float, float], end: Tuple[float, float]):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                   arrowstyle="->", mutation_scale=12,
                                   linewidth=1.2, color="black"))


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="picture_candidates.csv", help="candidate CSV")
    parser.add_argument("--row-id", type=int, default=32, help="row index in candidate CSV")
    parser.add_argument("--xai-csv", default=None, help="real XAI ranking CSV from extract_attention_row.py")
    parser.add_argument("--pred-lines", default=None, help="1-based predicted vulnerable line numbers, e.g. 6,9,11")
    parser.add_argument("--out", default="fig2_real_attention.png", help="output PNG path")
    parser.add_argument("--title", default="A motivating example of XAI ranking and generative line-level localization")
    parser.add_argument("--xai-top-n", type=int, default=10, help="number of top XAI rows shown before appending missed GT")
    parser.add_argument("--include-missed-gt", choices=["yes", "no"], default="yes")
    parser.add_argument("--context-radius", type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.row_id < 0 or args.row_id >= len(df):
        raise IndexError(f"--row-id {args.row_id} out of range; candidate CSV has {len(df)} rows")

    row = df.iloc[args.row_id]
    lines = str(row["processed_func"]).splitlines()
    gt_idx = resolve_gt_indices(row, lines)
    pred_idx = parse_pred_lines(args.pred_lines, gt_idx)

    lo, hi = crop_context(lines, gt_idx, pred_idx, radius=args.context_radius)
    disp_lines = lines[lo:hi]
    disp_nums = list(range(lo + 1, hi + 1))

    if args.xai_csv:
        all_xai_rows = load_xai_csv(args.xai_csv)
    else:
        all_xai_rows = make_demo_xai(lines, gt_idx, (lo, hi))

    xai_rows = select_xai_rows(
        all_xai_rows,
        gt_idx,
        top_n=args.xai_top_n,
        include_missed_gt=(args.include_missed_gt == "yes"),
    )

    # Panel C: predicted lines first, then remaining cropped context lines.
    valid_pred_idx = [i for i in pred_idx if 0 <= i < len(lines)]
    remaining_idx = [i for i in range(lo, hi) if i not in set(valid_pred_idx)]
    c_idx_order = valid_pred_idx + remaining_idx
    c_lines = [lines[i] for i in c_idx_order][:24]
    c_nums = [i + 1 for i in c_idx_order][:24]

    fig = plt.figure(figsize=(14.8, 8.4), dpi=220)
    fig.patch.set_facecolor("white")
    fig.suptitle(args.title, fontsize=14, fontweight="bold", y=0.985)

    ax_a = fig.add_axes([0.035, 0.18, 0.38, 0.58])
    ax_xai = fig.add_axes([0.245, 0.795, 0.16, 0.12])
    ax_b = fig.add_axes([0.50, 0.54, 0.46, 0.36])
    ax_c = fig.add_axes([0.50, 0.08, 0.46, 0.36])

    draw_code_panel(
        ax_a,
        "(A) Original code: ground-truth vulnerable lines in red",
        disp_lines,
        disp_nums,
        red_idx=gt_idx,
        max_chars=62,
    )
    draw_xai_box(ax_xai)
    draw_xai_table(
        ax_b,
        f"(B) XAI output: Top-{args.xai_top_n} ranking; true vulnerable lines in red",
        xai_rows,
        gt_idx,
        top_n=args.xai_top_n,
    )
    draw_code_panel(
        ax_c,
        "(C) LocVul/NacgVuln output: predicted vulnerable lines first in gray",
        c_lines,
        c_nums,
        gray_idx=valid_pred_idx,
        max_chars=72,
    )

    add_arrow(fig, (0.37, 0.715), (0.245, 0.855))
    add_arrow(fig, (0.405, 0.855), (0.50, 0.725))
    add_arrow(fig, (0.405, 0.255), (0.50, 0.255))

    fig.text(0.265, 0.105, "Seq2Seq localization\n(CodeT5 / NacgVuln)",
             ha="center", va="center", fontsize=9, fontweight="bold")

    project = row.get("project", "")
    cwe = row.get("CWE ID", row.get("CWE", ""))
    gt_lines = ", ".join(str(i + 1) for i in gt_idx)
    pred_lines = ", ".join(str(i + 1) for i in valid_pred_idx)
    fig.text(
        0.50,
        0.025,
        f"Dataset sample: {project}, {cwe}. Ground truth lines: {gt_lines}. Predicted-first lines: {pred_lines}.",
        ha="center",
        fontsize=8.8,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Selected row: {args.row_id}, project={project}, CWE={cwe}")
    print(f"Ground-truth vulnerable lines: {[i + 1 for i in gt_idx]}")
    print(f"Predicted-first lines: {[i + 1 for i in valid_pred_idx]}")
    if args.xai_csv:
        print(f"XAI rows shown: {len(xai_rows)} from {args.xai_csv}")


if __name__ == "__main__":
    main()
