#!/usr/bin/env python
# coding: utf-8

"""seq2seq_eval_fixed_v3_compat_offline_v3_vulnonly.py

基于 offline_v2 的最终版修复：
1) 保留离线 Hugging Face cache / snapshot 解析能力
2) 保留 use_fast=False，减少 tokenizer 兼容问题
3) 使用 overflow-aware 长函数检测，避免长序列 warning
4) 新增 EVAL_ONLY_VULN（默认 yes）：仅在 target==1 样本上统计行级定位指标
   - 这样 A@K / MRR@K / MAP@K / IFA 与论文口径更一致
   - 避免把大量 target==0 样本纳入均值，导致指标被严重稀释
"""

import os
import re
import ast
import math
import random
import logging
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seq2seq_eval_fixed_v3")


# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=9, type=int)
parser.add_argument("--checkpoint_dir_seq2seq", default="./checkpoints_seq2seq", type=str)
parser.add_argument("--model_variation_seq2seq", default="Salesforce/codet5-base", type=str,
                    help="当 checkpoint_dir_seq2seq 仅包含 best_weights.pt 时，用该基础模型结构加载")
parser.add_argument("--output_dir", default="./ablation_outputs", type=str,
                    help="输出目录")
parser.add_argument("--run_name", default=None, type=str,
                    help="本次实验名称；默认自动生成")

parser.add_argument("--test_path", default=None, type=str,
                    help="默认 ./preprocessed_data_test_{seed}.csv")

# decoding
parser.add_argument("--MAX_INPUT_LEN", default=512, type=int)
parser.add_argument("--MAX_TARGET_LEN", default=128, type=int)
parser.add_argument("--NUM_BEAMS", default=4, type=int)
parser.add_argument("--NUM_RETURN_SEQS", default=1, type=int)
parser.add_argument("--FALLBACK_SAMPLING", default="yes", choices=["yes", "no"])
parser.add_argument("--SAMPLE_RETURN_SEQS", default=4, type=int)
parser.add_argument("--TOP_P", default=0.95, type=float)
parser.add_argument("--TEMPERATURE", default=0.7, type=float)

# long-function chunking
parser.add_argument(
    "--CHUNK_LONG_FUNCS",
    default="yes",
    choices=["yes", "no"],
    help="当函数超过 MAX_INPUT_LEN token 时，按行滑窗切块推理并合并候选（提升长函数覆盖率）",
)
parser.add_argument(
    "--CHUNK_STRIDE_LINES",
    default=40,
    type=int,
    help="滑窗重叠的行数（越大越稳，但更慢）",
)

# post-process
parser.add_argument("--SIMILARITY_REPLACEMENT", default="yes", choices=["yes", "no"])
parser.add_argument("--SIM_WEIGHT", default=1.0, type=float)
parser.add_argument("--SIM_THRESHOLD", default=0.0, type=float,
                    help="丢弃相似度低于阈值的候选行（只对 hallucination/不在函数内时有意义）")
parser.add_argument("--DEDUP", default="yes", choices=["yes", "no"])
parser.add_argument("--RERANK", default="yes", choices=["yes", "no"],
                    help="对多候选合并后的行按(序列得分+相似度)重排")

# evaluation
parser.add_argument("--K", default=10, type=int)
parser.add_argument("--EVAL_MODE", default="locvul", choices=["locvul", "gen_only"],
                    help="locvul: pred_lines + 其余原行补齐；gen_only: 仅评估模型生成的行")
parser.add_argument("--REMOVE_MISSING_LINE_LABELS", default="yes", choices=["yes", "no"])
parser.add_argument("--EVAL_ONLY_VULN", default="yes", choices=["yes", "no"],
                    help="If yes, evaluate localization metrics only on target==1 samples")

# UniLocVul
parser.add_argument("--NO_VULN_TOKEN", default="<NO_VULN>", type=str)
parser.add_argument("--HANDLE_NO_VULN_TOKEN", default="yes", choices=["yes", "no"])

# global ordering
parser.add_argument("--sort_by_lines", default="yes", choices=["yes", "no"],
                    help="与原 repo 一致：yes=优先检查被模型指出的行；no=按函数概率排序后逐函数检查")

args = parser.parse_args()
print(args)

seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]
seed = seeders[args.seed] if args.seed < len(seeders) else args.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


# -------------------------
# Helpers
# -------------------------

def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def _normalize_line(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _split_lines_from_text(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []
    if "/~/" in s:
        return [p.strip() for p in s.split("/~/") if p.strip()]
    if "\n" in s:
        return [p.strip() for p in s.splitlines() if p.strip()]
    return [s]


def _parse_line_index_field(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for i in x:
            try:
                out.append(int(i))
            except Exception:
                pass
        return out

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            out = []
            for i in v:
                try:
                    out.append(int(i))
                except Exception:
                    pass
            return out
    except Exception:
        pass

    for sep in ["\n", "/~/", ",", " ", ";", "|", "\t"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            out = []
            for p in parts:
                if p.isdigit():
                    out.append(int(p))
            if out:
                return out

    if s.isdigit():
        return [int(s)]
    return []


def _is_no_vuln_prediction(pred_text: str) -> bool:
    if pred_text is None:
        return True
    t = str(pred_text).strip()
    if t == "" or t.lower() == "nan":
        return True
    if args.HANDLE_NO_VULN_TOKEN == "yes" and t == args.NO_VULN_TOKEN:
        return True
    return False


# -------------------------
# Load data
# -------------------------

test_path = args.test_path or f"./preprocessed_data_test_{args.seed}.csv"
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Missing test csv: {test_path}")

test_raw = pd.read_csv(test_path)

required_cols = ["processed_func", "flaw_line", "flaw_line_index", "target"]
for c in required_cols:
    if c not in test_raw.columns:
        raise ValueError(f"Missing column '{c}' in test CSV: {test_path}")

test_data = pd.DataFrame({
    "Text": test_raw["processed_func"],
    "Lines": test_raw["flaw_line"],
    "Line_Index": test_raw["flaw_line_index"],
    "target": test_raw["target"],
})

if "proba" in test_raw.columns:
    test_data["proba"] = test_raw["proba"].astype(float)
else:
    test_data["proba"] = 1.0

test_data["Line_Index_parsed"] = test_data["Line_Index"].apply(_parse_line_index_field)

if args.REMOVE_MISSING_LINE_LABELS == "yes":
    before = len(test_data)
    miss = (
        (test_data["target"] == 1)
        & (test_data["Lines"].isna() | test_data["Line_Index"].isna())
    )
    test_data = test_data.loc[~miss].reset_index(drop=True)
    logger.info(f"REMOVE_MISSING_LINE_LABELS=yes: dropped {before-len(test_data)} samples")

if args.EVAL_ONLY_VULN == "yes":
    before = len(test_data)
    test_data = test_data.loc[test_data["target"] == 1].reset_index(drop=True)
    logger.info(f"EVAL_ONLY_VULN=yes: kept {len(test_data)} / {before} samples (target==1)")


# -------------------------
# Load model
# -------------------------

def _is_offline_mode() -> bool:
    vals = [
        os.environ.get("HF_HUB_OFFLINE", ""),
        os.environ.get("TRANSFORMERS_OFFLINE", ""),
    ]
    return any(str(v).strip().lower() in {"1", "true", "yes", "on"} for v in vals)


def _find_hf_snapshot_dir(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None

    direct_cfg = os.path.join(base_dir, "config.json")
    if os.path.exists(direct_cfg):
        return base_dir

    snapshots_dir = os.path.join(base_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    candidates = []
    for name in os.listdir(snapshots_dir):
        cand = os.path.join(snapshots_dir, name)
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "config.json")):
            candidates.append(cand)

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _resolve_model_source(model_name: str) -> str:
    if model_name is None:
        return model_name
    if os.path.isdir(model_name):
        snap = _find_hf_snapshot_dir(model_name)
        if snap is not None:
            if os.path.abspath(snap) != os.path.abspath(model_name):
                logger.info(f"Resolved local HF cache dir to snapshot: {snap}")
            return snap
    return model_name


def _load_state_dict_from_best_weights(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def _load_tokenizer(model_source: str):
    offline = _is_offline_mode()
    try:
        tok = AutoTokenizer.from_pretrained(
            model_source,
            use_fast=False,
            local_files_only=offline,
        )
        try:
            tok.model_max_length = int(1e9)
        except Exception:
            pass
        return tok
    except Exception as e:
        raise RuntimeError(
            "Failed to load tokenizer from model source: "
            f"{model_source}. "
            "Please ensure this directory contains a complete local CodeT5 model "
            "(config.json, tokenizer_config.json, and spiece.model/tokenizer files), "
            "or provide a valid Hugging Face snapshot path. Original error: "
            f"{repr(e)}"
        ) from e


def _load_model(model_source: str):
    offline = _is_offline_mode()
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
            local_files_only=offline,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load seq2seq base model from: "
            f"{model_source}. "
            "Please ensure the local model directory is complete and readable. "
            "Original error: "
            f"{repr(e)}"
        ) from e


def _maybe_load_hf_or_pt(checkpoint_dir: str, model_name: str):
    checkpoint_dir = _resolve_model_source(checkpoint_dir)
    model_source = _resolve_model_source(model_name)
    config_json = os.path.join(checkpoint_dir, "config.json")
    weight_pt = os.path.join(checkpoint_dir, "best_weights.pt")

    if os.path.exists(config_json):
        logger.info(f"Loading HF checkpoint dir: {checkpoint_dir}")
        tok = _load_tokenizer(checkpoint_dir)
        mdl = _load_model(checkpoint_dir)
        return tok, mdl

    logger.info(f"HF config not found under {checkpoint_dir}; fallback to base model + best_weights.pt")
    if not os.path.exists(weight_pt):
        raise FileNotFoundError(
            f"Neither HF checkpoint dir nor best_weights.pt found under: {checkpoint_dir}"
        )

    if model_source is None:
        raise ValueError("model_variation_seq2seq is empty; cannot load base model.")

    if os.path.isdir(model_source) and not os.path.exists(os.path.join(model_source, "config.json")):
        snapshots_dir = os.path.join(model_source, "snapshots")
        raise FileNotFoundError(
            "Local base model directory is incomplete: "
            f"{model_source}. Missing config.json. "
            f"If this is a Hugging Face cache root, it should contain snapshots/<hash>/config.json. "
            f"Current snapshots dir exists: {os.path.isdir(snapshots_dir)}"
        )

    tok = _load_tokenizer(model_source)
    mdl = _load_model(model_source)
    sd = _load_state_dict_from_best_weights(weight_pt)
    missing, unexpected = mdl.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"[Seq2Seq] Missing keys: {missing[:10]}")
    if unexpected:
        logger.warning(f"[Seq2Seq] Unexpected keys: {unexpected[:10]}")
    return tok, mdl


tokenizer, model = _maybe_load_hf_or_pt(args.checkpoint_dir_seq2seq, args.model_variation_seq2seq)
model = model.to(DEVICE)
model.eval()

ENCODER = model.get_encoder()


# -------------------------
# Similarity replacement (encoder cosine)
# -------------------------

def _encode_lines(lines: List[str], batch_size: int = 64) -> torch.Tensor:
    outs = []
    for i in range(0, len(lines), batch_size):
        chunk = lines[i:i + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=args.MAX_INPUT_LEN,
        ).to(DEVICE)
        with torch.no_grad():
            h = ENCODER(**inputs).last_hidden_state
            outs.append(h.mean(dim=1))
    return torch.cat(outs, dim=0) if outs else torch.empty((0, model.config.d_model), device=DEVICE)


def _best_match_with_sim(pred_line: str,
                         orig_lines: List[str],
                         orig_emb: torch.Tensor,
                         norm_to_first_idx: Dict[str, int]) -> Tuple[str, float]:
    pl = (pred_line or "").strip()
    if pl == "":
        return "", -1.0

    npl = _normalize_line(pl)
    if npl in norm_to_first_idx:
        idx = norm_to_first_idx[npl]
        return orig_lines[idx], 1.0

    if orig_emb.numel() == 0 or len(orig_lines) == 0:
        return pl, -1.0

    inp = tokenizer(
        pl,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=args.MAX_INPUT_LEN,
    ).to(DEVICE)
    with torch.no_grad():
        h = ENCODER(**inp).last_hidden_state.mean(dim=1)
        sims = F.cosine_similarity(h, orig_emb)
        best = int(torch.argmax(sims).item())
        return orig_lines[best], float(sims[best].item())


# -------------------------
# Decoding: multi-hypothesis
# -------------------------

def _generate_candidates(text: str) -> List[Tuple[str, float]]:
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=args.MAX_INPUT_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=args.MAX_TARGET_LEN,
            num_beams=max(1, args.NUM_BEAMS),
            num_return_sequences=max(1, args.NUM_RETURN_SEQS),
            return_dict_in_generate=True,
            output_scores=True,
        )

    seqs = out.sequences
    scores = getattr(out, "sequences_scores", None)
    if scores is None:
        scores = getattr(out, "sequence_scores", None)
    if scores is None:
        scores = torch.zeros((seqs.shape[0],), device=DEVICE)

    cands: List[Tuple[str, float]] = []
    for i in range(seqs.shape[0]):
        txt = tokenizer.decode(seqs[i], skip_special_tokens=True)
        if _is_no_vuln_prediction(txt):
            continue
        cands.append((txt, float(scores[i].item())))

    if (not cands) and args.FALLBACK_SAMPLING == "yes":
        with torch.no_grad():
            out2 = model.generate(
                **inputs,
                max_length=args.MAX_TARGET_LEN,
                do_sample=True,
                top_p=float(args.TOP_P),
                temperature=float(args.TEMPERATURE),
                num_return_sequences=max(1, args.SAMPLE_RETURN_SEQS),
                return_dict_in_generate=True,
            )
        for i in range(out2.sequences.shape[0]):
            txt = tokenizer.decode(out2.sequences[i], skip_special_tokens=True)
            if _is_no_vuln_prediction(txt):
                continue
            cands.append((txt, 0.0))

    return cands


def _build_line_chunks(full_text: str) -> List[str]:
    lines = str(full_text).splitlines()
    if not lines:
        return [""]

    max_tok = int(args.MAX_INPUT_LEN) - 8
    stride = max(0, int(args.CHUNK_STRIDE_LINES))
    line_tok = [len(tokenizer.encode(l, add_special_tokens=False)) for l in lines]

    chunks: List[str] = []
    start = 0
    n = len(lines)

    while start < n:
        tok_sum = 0
        end = start

        while end < n:
            add = line_tok[end]
            if end == start and add > max_tok:
                end += 1
                break
            if tok_sum + add > max_tok:
                break
            tok_sum += add
            end += 1

        # Safety guard: if no line was consumed for any unexpected reason,
        # force progress to avoid an infinite loop.
        if end <= start:
            end = start + 1

        chunk = "\n".join(lines[start:end])
        chunks.append(chunk)

        if end >= n:
            break

        # Critical fix:
        # The previous implementation used:
        #     start = max(0, end - stride)
        # If stride >= chunk_len, start could stay unchanged, causing an infinite loop.
        chunk_len = end - start
        effective_stride = min(stride, max(0, chunk_len - 1))
        next_start = end - effective_stride

        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


def _generate_candidates_for_function(full_text: str) -> List[Tuple[str, float]]:
    if args.CHUNK_LONG_FUNCS == "no":
        return _generate_candidates(full_text)

    try:
        enc = tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.MAX_INPUT_LEN,
            return_overflowing_tokens=True,
        )
        input_ids = enc.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            num_windows = len(input_ids)
            is_long = num_windows > 1
        else:
            overflow = enc.get("overflowing_tokens", [])
            is_long = bool(overflow)
    except Exception:
        is_long = False

    if not is_long:
        return _generate_candidates(full_text)

    cands: List[Tuple[str, float]] = []
    for chunk in _build_line_chunks(full_text):
        cands.extend(_generate_candidates(chunk))
    return cands


# -------------------------
# Build ranked list (LocVul style)
# -------------------------

def _rank_lines_for_function(text: str, candidates: List[Tuple[str, float]]) -> Tuple[List[str], List[str]]:
    orig_lines = str(text).splitlines()

    norm_to_first_idx: Dict[str, int] = {}
    for i, l in enumerate(orig_lines):
        nl = _normalize_line(l)
        if nl and nl not in norm_to_first_idx:
            norm_to_first_idx[nl] = i

    orig_emb = torch.empty((0, model.config.d_model), device=DEVICE)
    if args.SIMILARITY_REPLACEMENT == "yes" and orig_lines:
        orig_emb = _encode_lines(orig_lines)

    line2score: Dict[str, float] = {}
    line2first_seen: Dict[str, int] = {}
    seen_counter = 0

    for cand_text, base_score in candidates:
        for raw_line in _split_lines_from_text(cand_text):
            raw_line = raw_line.strip()
            if raw_line == "":
                continue

            if args.SIMILARITY_REPLACEMENT == "yes":
                best_line, sim = _best_match_with_sim(raw_line, orig_lines, orig_emb, norm_to_first_idx)
                if sim < args.SIM_THRESHOLD:
                    continue
                final_line = best_line
                final_score = base_score + args.SIM_WEIGHT * sim
            else:
                final_line = raw_line
                final_score = base_score

            key = _normalize_line(final_line)
            if key == "":
                continue

            if key not in line2score:
                line2score[key] = final_score
                line2first_seen[key] = seen_counter
            else:
                line2score[key] = max(line2score[key], final_score)
            seen_counter += 1

    pred_items = [(k, line2score[k], line2first_seen[k]) for k in line2score.keys()]

    if args.RERANK == "yes":
        pred_items.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    else:
        pred_items.sort(key=lambda x: x[2])

    pred_keys = [k for k, _, _ in pred_items]

    pred_lines: List[str] = []
    for k in pred_keys:
        if k in norm_to_first_idx:
            pred_lines.append(orig_lines[norm_to_first_idx[k]])
        else:
            pred_lines.append(k)

    if args.DEDUP == "yes":
        uniq = []
        seen = set()
        for l in pred_lines:
            nl = _normalize_line(l)
            if nl not in seen:
                seen.add(nl)
                uniq.append(l)
        pred_lines = uniq

    if args.EVAL_MODE == "gen_only":
        return pred_lines, pred_lines

    pred_set = set(_normalize_line(x) for x in pred_lines)
    ranked = list(pred_lines)
    for l in orig_lines:
        nl = _normalize_line(l)
        if nl not in pred_set:
            ranked.append(l)

    return pred_lines, ranked


# -------------------------
# Metrics (per function)
# -------------------------

def _a_at_k(ranked: List[str], gt: List[str], k: int) -> int:
    if not gt:
        return 0
    gt_set = set(_normalize_line(x) for x in gt)
    for l in ranked[:k]:
        if _normalize_line(l) in gt_set:
            return 1
    return 0


def _mrr_at_k(ranked: List[str], gt: List[str], k: int) -> float:
    if not gt:
        return 0.0
    gt_set = set(_normalize_line(x) for x in gt)
    for i, l in enumerate(ranked[:k], start=1):
        if _normalize_line(l) in gt_set:
            return 1.0 / i
    return 0.0


def _ap_at_k(ranked: List[str], gt: List[str], k: int) -> float:
    if not gt:
        return 0.0
    gt_set = set(_normalize_line(x) for x in gt)
    hit = 0
    s = 0.0
    for i, l in enumerate(ranked[:k], start=1):
        if _normalize_line(l) in gt_set:
            hit += 1
            s += hit / i
    if hit == 0:
        return 0.0
    return s / hit


def _ifa(ranked: List[str], gt: List[str]) -> int:
    if not gt:
        return 0
    gt_set = set(_normalize_line(x) for x in gt)
    cnt = 0
    for l in ranked:
        if _normalize_line(l) in gt_set:
            break
        cnt += 1
    return cnt


# -------------------------
# Ground-truth indices (for global metrics)
# -------------------------

def _derive_true_indices_from_text(gt_lines: List[str], orig_lines: List[str]) -> List[int]:
    if not gt_lines or not orig_lines:
        return []
    norm_to_idxs: Dict[str, List[int]] = {}
    for i, l in enumerate(orig_lines):
        nl = _normalize_line(l)
        norm_to_idxs.setdefault(nl, []).append(i)

    out = []
    for gl in gt_lines:
        nl = _normalize_line(gl)
        for i in norm_to_idxs.get(nl, []):
            out.append(i)
    return sorted(set(out))


# -------------------------
# Run evaluation
# -------------------------
rows = []
all_ranked_lines: List[List[str]] = []
all_pred_lines: List[List[str]] = []
all_gt_indices: List[List[int]] = []
all_locs: List[int] = []
all_func_proba: List[float] = []

logger.info("Generating + evaluating...")

for i in tqdm(range(len(test_data))):
    text = test_data.loc[i, "Text"]
    gt_text = test_data.loc[i, "Lines"]

    gt_lines = _split_lines_from_text(gt_text)
    cands = _generate_candidates_for_function(str(text))
    pred_lines, ranked_lines = _rank_lines_for_function(str(text), cands)

    orig_lines = str(text).splitlines()
    true_idx = test_data.loc[i, "Line_Index_parsed"]
    if not true_idx:
        true_idx = _derive_true_indices_from_text(gt_lines, orig_lines)

    a10 = _a_at_k(ranked_lines, gt_lines, args.K)
    mrr10 = _mrr_at_k(ranked_lines, gt_lines, args.K)
    map10 = _ap_at_k(ranked_lines, gt_lines, args.K)
    ifa = _ifa(ranked_lines, gt_lines)

    all_ranked_lines.append(ranked_lines)
    all_pred_lines.append(pred_lines)
    all_gt_indices.append(true_idx)
    all_locs.append(max(1, len(orig_lines)))
    all_func_proba.append(float(test_data.loc[i, "proba"]))

    rows.append({
        "row": i,
        "target": int(test_data.loc[i, "target"]),
        "loc": max(1, len(orig_lines)),
        "gt_lines": "/~/".join(gt_lines),
        "pred_lines": "/~/".join(pred_lines),
        "ranked_topK": "/~/".join(ranked_lines[:args.K]),
        "A@K": a10,
        "MRR@K": mrr10,
        "MAP@K": map10,
        "IFA": ifa,
        "n_pred_lines": len(pred_lines),
        "n_gt_lines": len(gt_lines),
    })

out_df = pd.DataFrame(rows)


# -------------------------
# Global cost-effectiveness
# -------------------------

def _global_line_stream(sort_by_lines: bool) -> List[int]:
    func_order = list(range(len(all_ranked_lines)))
    func_order.sort(key=lambda idx: all_func_proba[idx], reverse=True)

    pred_stream = []
    rest_stream = []

    for fi in func_order:
        ranked = all_ranked_lines[fi]
        pred_set = set(_normalize_line(x) for x in all_pred_lines[fi])
        true_set = set(all_gt_indices[fi])

        orig_lines = str(test_data.loc[fi, "Text"]).splitlines()
        norm_to_idxs = {}
        for ii, l in enumerate(orig_lines):
            norm_to_idxs.setdefault(_normalize_line(l), []).append(ii)

        def is_vuln(line_text: str) -> int:
            nl = _normalize_line(line_text)
            for ii in norm_to_idxs.get(nl, []):
                if ii in true_set:
                    return 1
            return 0

        for l in ranked:
            lab = is_vuln(l)
            if sort_by_lines:
                if _normalize_line(l) in pred_set:
                    pred_stream.append(lab)
                else:
                    rest_stream.append(lab)
            else:
                pred_stream.append(lab)

    return pred_stream + rest_stream if sort_by_lines else pred_stream


def _effort_at_20_recall() -> float:
    total_loc = sum(all_locs)
    total_vuln = sum(len(x) for x in all_gt_indices)
    if total_vuln == 0 or total_loc == 0:
        return 0.0

    needed = max(1, int(math.ceil(0.2 * total_vuln)))
    stream = _global_line_stream(sort_by_lines=(args.sort_by_lines == "yes"))

    inspected = 0
    found = 0
    for lab in stream:
        inspected += 1
        if lab == 1:
            found += 1
        if found >= needed:
            break

    inspected = min(inspected, total_loc)
    return inspected / total_loc


def _recall_at_1pct_loc() -> float:
    total_loc = sum(all_locs)
    total_vuln = sum(len(x) for x in all_gt_indices)
    if total_vuln == 0 or total_loc == 0:
        return 0.0

    budget = max(1, int(math.ceil(0.01 * total_loc)))
    stream = _global_line_stream(sort_by_lines=(args.sort_by_lines == "yes"))

    found = 0
    for lab in stream[:budget]:
        if lab == 1:
            found += 1

    return found / total_vuln


A_mean = float(out_df["A@K"].mean())
MRR_mean = float(out_df["MRR@K"].mean())
MAP_mean = float(out_df["MAP@K"].mean())

ifa_median = float(out_df["IFA"].median())
ifa_mean = float(out_df["IFA"].mean())

EFF20 = float(_effort_at_20_recall())
R1LOC = float(_recall_at_1pct_loc())

logger.info("=" * 80)
logger.info(f"A@{args.K}:   {A_mean:.4f}")
logger.info(f"MRR@{args.K}: {MRR_mean:.4f}")
logger.info(f"MAP@{args.K}: {MAP_mean:.4f}")
logger.info(f"IFA (median): {ifa_median:.4f} | IFA (mean): {ifa_mean:.4f}")
logger.info(f"Effort@20%Recall: {EFF20:.4f}")
logger.info(f"Recall@1%LOC:     {R1LOC:.4f}")
logger.info("=" * 80)

os.makedirs(args.output_dir, exist_ok=True)
run_name = args.run_name or (
    f"seed{args.seed}_k{args.K}_mode{args.EVAL_MODE}_sim{args.SIMILARITY_REPLACEMENT}_chunk{args.CHUNK_LONG_FUNCS}"
)
out_path = os.path.join(args.output_dir, f"{run_name}_details.csv")
out_df.to_csv(out_path, index=False)

summary_path = os.path.join(args.output_dir, f"{run_name}_summary.csv")
summary_df = pd.DataFrame([{
    "run_name": run_name,
    "seed": args.seed,
    "K": args.K,
    "EVAL_MODE": args.EVAL_MODE,
    "NUM_BEAMS": args.NUM_BEAMS,
    "NUM_RETURN_SEQS": args.NUM_RETURN_SEQS,
    "FALLBACK_SAMPLING": args.FALLBACK_SAMPLING,
    "SAMPLE_RETURN_SEQS": args.SAMPLE_RETURN_SEQS,
    "CHUNK_LONG_FUNCS": args.CHUNK_LONG_FUNCS,
    "CHUNK_STRIDE_LINES": args.CHUNK_STRIDE_LINES,
    "SIMILARITY_REPLACEMENT": args.SIMILARITY_REPLACEMENT,
    "SIM_WEIGHT": args.SIM_WEIGHT,
    "SIM_THRESHOLD": args.SIM_THRESHOLD,
    "DEDUP": args.DEDUP,
    "RERANK": args.RERANK,
    "sort_by_lines": args.sort_by_lines,
    "EVAL_ONLY_VULN": args.EVAL_ONLY_VULN,
    "A@K": A_mean,
    "MRR@K": MRR_mean,
    "MAP@K": MAP_mean,
    "IFA_median": ifa_median,
    "IFA_mean": ifa_mean,
    "Effort@20%Recall": EFF20,
    "Recall@1%LOC": R1LOC,
    "n_rows": len(out_df),
}])
summary_df.to_csv(summary_path, index=False)
logger.info(f"Saved detailed results to: {out_path}")
logger.info(f"Saved summary results to: {summary_path}")
