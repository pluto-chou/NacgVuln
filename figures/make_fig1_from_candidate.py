#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_picture_from_candidate_compact.py

Compact, high-resolution version of the motivation figure.
Changes compared with the original script:
1) Larger code font.
2) Remove unnecessary blank lines while preserving original line numbers.
3) Smaller physical figure size.
4) Higher export resolution.
5) Optional CLI arguments for path / row / font / dpi.
"""

import argparse
import ast
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/tom/deeplearning/LocVul-main/picture_candidates.csv")
    parser.add_argument("--row-id", type=int, default=32)
    parser.add_argument("--out", default="/home/tom/deeplearning/LocVul-main/motivation_example_ssdp_compact.png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--code-font", type=float, default=8.8)
    parser.add_argument("--line-gap", type=float, default=0.78)
    parser.add_argument("--fig-width", type=float, default=9.6)
    parser.add_argument("--max-chars", type=int, default=118)
    parser.add_argument("--keep-blank-near-vuln", default="no", choices=["yes", "no"])
    parser.add_argument("--title", default="A motivating example of function-level and line-level vulnerability localization")
    parser.add_argument("--title-font", type=float, default=11.0)
    parser.add_argument("--label-font", type=float, default=7.2)
    parser.add_argument("--top-title-gap", type=float, default=1.85, help="Vertical gap between code block top and title")
    parser.add_argument("--top-label-gap", type=float, default=0.62, help="Vertical gap between code block top and function-level label")
    parser.add_argument("--gt-label-lift", type=float, default=0.76, help="Lift ground-truth label above the first vulnerable line, in units of line_gap")
    return parser.parse_args()


def parse_indices(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [int(i) for i in v]
        if isinstance(v, int):
            return [v]
    except Exception:
        pass
    return [int(n) for n in re.findall(r"\d+", s)]


def norm(s):
    return " ".join(str(s).strip().split())


def shorten(s, max_chars):
    s = str(s).replace("\t", "    ").rstrip("\n")
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def resolve_vulnerable_lines(row, func_lines):
    flaw_text = str(row["flaw_line"])
    flaw_parts = [p.strip() for p in re.split(r"/~/?|\n", flaw_text) if p.strip()]
    idxs = parse_indices(row["flaw_line_index"])

    vul_lines = set()
    for idx in idxs:
        if 0 <= idx < len(func_lines):
            code = norm(func_lines[idx])
            if any(code == norm(p) or norm(p) in code or code in norm(p) for p in flaw_parts):
                vul_lines.add(idx + 1)
        if 1 <= idx <= len(func_lines):
            code = norm(func_lines[idx - 1])
            if any(code == norm(p) or norm(p) in code or code in norm(p) for p in flaw_parts):
                vul_lines.add(idx)

    if not vul_lines:
        for line_no, line in enumerate(func_lines, start=1):
            code = norm(line)
            if any(code == norm(p) or norm(p) in code or code in norm(p) for p in flaw_parts):
                vul_lines.add(line_no)
    return sorted(vul_lines)


def make_display_rows(func_lines, vul_lines, keep_blank_near_vuln=False):
    """Return [(original_line_no, code_text, is_blank), ...].
    Blank lines are removed by default to reduce vertical whitespace.
    If keep_blank_near_vuln=True, keep blank lines adjacent to vulnerable lines.
    """
    vul_set = set(vul_lines)
    rows = []
    for line_no, line in enumerate(func_lines, start=1):
        is_blank = (line.strip() == "")
        if is_blank:
            if keep_blank_near_vuln and ((line_no - 1) in vul_set or (line_no + 1) in vul_set):
                rows.append((line_no, line, True))
            continue
        rows.append((line_no, line, False))
    return rows


def code_color(line):
    stripped = line.strip()
    if stripped.startswith(("if", "return", "static", "for", "while")):
        return "#c586c0"
    if "recvfrom" in line or "buf[" in line or "MAX_PKT_SIZE" in line:
        return "#dcdcaa"
    return "#d4d4d4"


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    row = df.iloc[args.row_id]
    func_lines = str(row["processed_func"]).splitlines()
    vul_lines = resolve_vulnerable_lines(row, func_lines)
    display_rows = make_display_rows(
        func_lines,
        vul_lines,
        keep_blank_near_vuln=(args.keep_blank_near_vuln == "yes"),
    )

    line_gap = float(args.line_gap)
    n_show = len(display_rows)

    # Smaller physical size, but high dpi.  Height follows displayed rows, not original rows.
    fig_w = float(args.fig_width)
    fig_h = max(5.2, n_show * 0.135 + 1.95)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    code_x = 3.0
    code_y = 0.95
    code_w = 94.0
    code_h = n_show * line_gap + 0.8
    top_y = code_y + code_h

    ax.set_xlim(0, 100)
    ax.set_ylim(0, top_y + max(2.35, args.top_title_gap + 0.55))

    # Title
    ax.text(
        2.0,
        top_y + args.top_title_gap,
        args.title,
        fontsize=args.title_font,
        fontweight="bold",
        color="black",
        va="center",
        clip_on=False,
        zorder=10,
    )

    # Code background
    ax.add_patch(
        Rectangle(
            (code_x, code_y),
            code_w,
            code_h,
            facecolor="#1e1e1e",
            edgecolor="#333333",
            linewidth=1.0,
            zorder=0,
        )
    )

    # Function-level outer box
    ax.add_patch(
        Rectangle(
            (code_x + 0.55, code_y + 0.35),
            code_w - 1.1,
            code_h - 0.7,
            fill=False,
            edgecolor="red",
            linewidth=1.7,
            zorder=2,
        )
    )

    y_for_orig_line = {}
    for pos, (orig_no, line, _is_blank) in enumerate(display_rows):
        y = top_y - 0.85 - pos * line_gap
        y_for_orig_line[orig_no] = y

        # Original line number is retained even after blank-line compression.
        ax.text(
            code_x + 1.15,
            y,
            f"{orig_no:>3}",
            fontfamily="DejaVu Sans Mono",
            fontsize=args.code_font - 1.1,
            color="#858585",
            va="center",
            zorder=4,
        )
        ax.text(
            code_x + 5.2,
            y,
            shorten(line, args.max_chars),
            fontfamily="DejaVu Sans Mono",
            fontsize=args.code_font,
            color=code_color(line),
            va="center",
            zorder=4,
        )

        if orig_no in vul_lines:
            ax.add_patch(
                Rectangle(
                    (code_x + 4.6, y - line_gap * 0.32),
                    code_w - 7.3,
                    line_gap * 0.64,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.25,
                    zorder=3,
                )
            )

    # Labels
    # Put the top red label in the gap between the title and the code block,
    # rather than on the outer red rectangle. The small white bbox prevents
    # visual collision with nearby strokes when the figure is scaled down.
    ax.text(
        code_x + 1.2,
        top_y + args.top_label_gap,
        "Function-level vulnerable region",
        fontsize=args.label_font,
        color="red",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        clip_on=False,
        zorder=11,
    )
    if vul_lines and vul_lines[0] in y_for_orig_line:
        # Lift the label above the first vulnerable-line rectangle and give it
        # the same dark background as the code area. This prevents the label
        # from being crossed by red highlight lines.
        ax.text(
            code_x + 50.0,
            y_for_orig_line[vul_lines[0]] + line_gap * args.gt_label_lift,
            "Ground-truth vulnerable lines",
            fontsize=args.label_font,
            color="red",
            va="bottom",
            ha="center",
            bbox=dict(facecolor="#1e1e1e", edgecolor="none", pad=1.2),
            clip_on=False,
            zorder=12,
        )

    caption = (
        f"Dataset sample: {row['project']}, {row['CWE ID']}. "
        "Outer red box denotes coarse-grained function-level prediction; "
        "inner red boxes denote line-level ground-truth vulnerable statements."
    )
    ax.text(2.0, 0.35, caption, fontsize=7.2, color="black", va="center")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # Also save a PDF for paper typesetting if the output is PNG/JPG.
    if out_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        pdf_path = out_path.with_suffix(".pdf")
        fig2, ax2 = plt.subplots(figsize=(0.1, 0.1))
        plt.close(fig2)
        # Re-render directly by calling this script with --out pdf is cleaner; avoid silent duplicate rendering here.

    print("Saved:", out_path)
    print("Selected row:", args.row_id)
    print("Project:", row["project"])
    print("CWE:", row["CWE ID"])
    print("Original lines:", len(func_lines))
    print("Displayed non-blank lines:", n_show)
    print("Removed blank lines:", len(func_lines) - n_show)
    print("Vulnerable display lines:", vul_lines)
    print("For paper, you can also run the same command with --out xxx.pdf to export vector PDF.")


if __name__ == "__main__":
    main()
