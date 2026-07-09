#!/usr/bin/env python
# coding: utf-8

"""
Seq2Seq_vulnDet_fixed_v2_chunktrain.py  (OOM-safe + UniLocVul-ready)

Key fixes:
1) Mask padding tokens in labels with -100 (HF seq2seq loss expects this)
2) Optional UniLocVul-style training: include non-vulnerable functions and map them to NO_VULN_TOKEN
3) ROUGE early-stopping computed only on vulnerable samples (optional)
4) OOM fixes:
   - cap target max length to 128
   - batch_size=2 + gradient accumulation=4 (effective batch=8)
   - AMP mixed precision
   - model.config.use_cache = False
5) Avoid evaluate.load('rouge') dependencies: use `rouge` python package directly (pip install rouge)

Dataset:
- read from: data/dataset.csv
- expects columns: processed_func, target, flaw_line, flaw_line_index
"""

import os
import time
import math
import random
import logging
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, get_scheduler
from transformers import logging as hf_logging

from rouge import Rouge


# --------------------------
# Args
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=9, type=int, required=False, choices=list(range(10)))
parser.add_argument("--FINE_TUNE", default="yes", type=str, required=False, choices=["yes", "no"])
parser.add_argument("--model_variation", default="Salesforce/codet5-base", type=str, required=False)
parser.add_argument("--checkpoint_dir", default="./checkpoints_seq2seq", type=str, required=False)
parser.add_argument("--hf_endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com/"), type=str, required=False,
                    help="HuggingFace Hub mirror endpoint, default https://hf-mirror.com/")

# UniLocVul
parser.add_argument("--INCLUDE_NEGATIVES", default="yes", type=str, required=False, choices=["yes", "no"])
parser.add_argument("--NEGATIVE_RATIO", default=1.0, type=float, required=False)  # neg : vuln = ratio : 1
parser.add_argument("--NO_VULN_TOKEN", default="<NO_VULN>", type=str, required=False)
parser.add_argument("--ROUGE_ON_VULN_ONLY", default="yes", type=str, required=False, choices=["yes", "no"])

# OOM controls (optional)
parser.add_argument("--MAX_INPUT_LEN", default=512, type=int, required=False)
parser.add_argument("--MAX_TARGET_LEN_CAP", default=128, type=int, required=False)  # IMPORTANT
parser.add_argument("--BATCH_SIZE", default=2, type=int, required=False)            # IMPORTANT
parser.add_argument("--GRAD_ACCUM_STEPS", default=4, type=int, required=False)      # IMPORTANT
parser.add_argument("--NUM_EPOCHS", default=10, type=int, required=False)
parser.add_argument("--PATIENCE", default=5, type=int, required=False)
parser.add_argument("--LR", default=5e-5, type=float, required=False)

# Long-function chunked training (recommended for better tail coverage)
parser.add_argument("--CHUNK_TRAINING", default="no", type=str, required=False, choices=["yes", "no"])
parser.add_argument("--CHUNK_STRIDE_LINES", default=80, type=int, required=False)  # overlap size
parser.add_argument("--CHUNK_PREFIX_LINES", default=8, type=int, required=False)  # always include function header context
parser.add_argument("--CHUNK_PREFIX_MAX_TOKENS", default=192, type=int, required=False)
parser.add_argument("--CHUNK_MAX_TOKENS", default=512, type=int, required=False)  # usually = MAX_INPUT_LEN
parser.add_argument("--NEG_CHUNKS_PER_FUNC", default=1, type=int, required=False)  # when CHUNK_TRAINING=yes, sample N chunks per negative func
# Post-training quick generation sanity check (can OOM if left too aggressive)
parser.add_argument("--QUICK_TEST", default="no", type=str, required=False, choices=["yes", "no"])
parser.add_argument("--QUICK_TEST_SAMPLES", default=3, type=int, required=False)
parser.add_argument("--QUICK_TEST_BEAMS", default=1, type=int, required=False)
parser.add_argument("--QUICK_TEST_MAX_LENGTH", default=64, type=int, required=False)

args = parser.parse_args()
print(args)

# --------------------------
# Logging & Seed
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]
seed = seeders[args.seed]

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def apply_hf_mirror(endpoint: str):
    endpoint = (endpoint or "https://hf-mirror.com/").strip().rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = endpoint
    os.environ["HUGGINGFACE_CO_URL_HOME"] = endpoint
    os.environ["HF_INFERENCE_ENDPOINT"] = endpoint
    logger.info(f"Using HuggingFace mirror endpoint: {endpoint}")
    return endpoint


HF_ENDPOINT = apply_hf_mirror(args.hf_endpoint)

# --------------------------
# Config
# --------------------------
FINE_TUNE = (args.FINE_TUNE.lower() == "yes")
INCLUDE_NEGATIVES = (args.INCLUDE_NEGATIVES.lower() == "yes")
NEGATIVE_RATIO = float(args.NEGATIVE_RATIO)
NO_VULN_TOKEN = args.NO_VULN_TOKEN
ROUGE_ON_VULN_ONLY = (args.ROUGE_ON_VULN_ONLY.lower() == "yes")

MAX_INPUT_LEN = int(args.MAX_INPUT_LEN)
MAX_TARGET_LEN_CAP = int(args.MAX_TARGET_LEN_CAP)  # default 128
BATCH_SIZE = int(args.BATCH_SIZE)                  # default 2
GRAD_ACCUM_STEPS = int(args.GRAD_ACCUM_STEPS)      # default 4

NUM_EPOCHS = int(args.NUM_EPOCHS)
PATIENCE = int(args.PATIENCE)
LR = float(args.LR)

os.makedirs(args.checkpoint_dir, exist_ok=True)
save_path = os.path.join(args.checkpoint_dir, "best_weights.pt")

# --------------------------
# Tokenizer (MUST be initialized before any optional chunked training)
# --------------------------
# NOTE: CHUNK_TRAINING builds chunk samples before we tokenize datasets;
# the chunk builder needs a tokenizer to measure/truncate tokens.
tokenizer = AutoTokenizer.from_pretrained(args.model_variation, use_fast=True)
# Silence HF warnings about long sequences; we always truncate before feeding the model.
hf_logging.set_verbosity_error()
try:
    tokenizer.model_max_length = int(1e9)
except Exception:
    pass

# --------------------------
# Load dataset
# --------------------------
root_path = os.getcwd()
dataset_path = os.path.join(root_path, "data", "dataset.csv")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Missing {dataset_path}. Please run: python data_mining.py")

dataset = pd.read_csv(dataset_path)
dataset = dataset.dropna(subset=["processed_func"]).reset_index(drop=True)

required_cols = ["processed_func", "target", "flaw_line", "flaw_line_index"]
for c in required_cols:
    if c not in dataset.columns:
        raise ValueError(f"dataset.csv missing column: {c}")

# Split (keep consistent with original repo style)
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_raw = dataset.iloc[-num_of_ratio:, :]
train_raw = data.iloc[0:-num_of_ratio, :]
val_raw = data.iloc[-num_of_ratio:, :]

del dataset

def replace_delimiter_with_newline(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("/~/", "\n")


def _split_lines_maybe(s: str):
    if s is None:
        return []
    s = str(s)
    if "/~/" in s:
        return s.split("/~/")
    # fallback: real newlines
    return s.splitlines()

def _parse_indices(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = s.split("/~/") if "/~/" in s else s.replace(",", " ").split()
    out = []
    for p in parts:
        p = p.strip()
        if p == "":
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out

def _join_lines(lines):
    return "\n".join([str(x) for x in lines if str(x).strip() != ""])

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)

def _make_chunks_for_function(func_text: str, max_tokens: int, overlap_lines: int, prefix_lines: int, prefix_max_tokens: int):
    """
    Return list of chunk texts (strings). Each chunk contains:
      - a (truncated) prefix header (first `prefix_lines`) to provide signature/ctx
      - a sliding window of lines with overlap.

    v2_2 changes vs v2_1:
      * NO tokenizer(truncation=False) on long sequences -> avoids 701>512 warning
      * O(n) chunking per function using per-line token lengths (no quadratic inner loop)
      * guarantee progress even if overlap_lines >= window size (prevents infinite loops / "卡住")
    """
    lines = _split_lines_maybe(func_text)
    if len(lines) == 0:
        return [""]

    # build (and optionally truncate) header
    header = _join_lines(lines[:max(0, prefix_lines)])
    if prefix_max_tokens is not None and prefix_max_tokens > 0:
        header = _truncate_to_tokens(header, prefix_max_tokens)

    whole = _join_lines(lines)

    # Pre-tokenize per-line to estimate window budgets efficiently
    try:
        line_enc = tokenizer(lines, add_special_tokens=False, truncation=False, padding=False)
        line_lens = [len(ids) for ids in line_enc["input_ids"]]
    except Exception:
        line_lens = [len(tokenizer(str(l), add_special_tokens=False, truncation=False)["input_ids"]) for l in lines]

    # quick fit check (approx): if whole likely fits, keep as one sample (no duplicated header)
    approx_whole = sum(line_lens) + max(0, len(lines) - 1) + 2  # +newlines + special tokens
    if approx_whole <= max_tokens:
        return [whole]

    # header token cost (approx)
    header_len = 0
    if header.strip():
        header_len = len(tokenizer(header, add_special_tokens=False, truncation=True, max_length=max(8, int(prefix_max_tokens or 2048)))["input_ids"])
    specials = 2
    window_budget = max(8, max_tokens - header_len - specials)

    chunks = []
    n = len(lines)
    overlap = max(0, int(overlap_lines))

    i = 0
    while i < n:
        cur = 0
        j = i
        while j < n:
            add = int(line_lens[j]) + 1  # +1 for newline-ish separator
            if cur + add <= window_budget:
                cur += add
                j += 1
            else:
                break

        if j == i:
            # single line too long -> hard truncate safely
            window = str(lines[i])
            candidate = (header + "\n" + window).strip() if header.strip() else window
            candidate = _truncate_to_tokens(candidate, max_tokens)
            chunks.append(candidate)
            i += 1
            continue

        window_lines = lines[i:j]
        window_txt = _join_lines(window_lines)
        candidate = (header + "\n" + window_txt).strip() if header.strip() else window_txt
        # final safety truncation (keeps <= max_tokens always)
        candidate = _truncate_to_tokens(candidate, max_tokens)
        chunks.append(candidate)

        if j >= n:
            break

        chunk_len = j - i
        # effective overlap cannot exceed chunk_len-1, otherwise i would not advance
        eff_overlap = min(overlap, max(0, chunk_len - 1))
        i_next = j - eff_overlap
        if i_next <= i:
            i_next = i + 1
        i = i_next

    return chunks


def build_seq2seq_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.dropna(subset=["processed_func"]).copy()

    # vulnerable samples must have flaw_line_index
    vuln = df[(df["target"] == 1) & (~df["flaw_line_index"].isna())].copy()
    vuln = vuln[["processed_func", "flaw_line", "flaw_line_index"]].reset_index(drop=True)

    # Optional: chunked training to reduce 512-token truncation mismatch on long functions.
    # We expand each vulnerable function into multiple chunks, keeping only chunks that contain at least
    # one vulnerable line index.
    if args.CHUNK_TRAINING == "yes" and len(vuln) > 0:
        expanded_rows = []
        for r in tqdm(vuln.itertuples(index=False), total=len(vuln), desc=f"[{split_name}] vuln chunking"):
            func_text = getattr(r, "processed_func")
            flaw_lines = _split_lines_maybe(getattr(r, "flaw_line"))
            flaw_idx   = _parse_indices(getattr(r, "flaw_line_index"))
            # map idx->line text (best-effort)
            idx2line = {}
            for k, idx in enumerate(flaw_idx):
                if k < len(flaw_lines):
                    idx2line[idx] = flaw_lines[k]
            chunks = _make_chunks_for_function(
                func_text,
                max_tokens=args.CHUNK_MAX_TOKENS,
                overlap_lines=args.CHUNK_STRIDE_LINES,
                prefix_lines=args.CHUNK_PREFIX_LINES,
                prefix_max_tokens=args.CHUNK_PREFIX_MAX_TOKENS,
            )
            # compute which chunk contains which line indices: approximate by line window boundaries
            # Since chunking is token-based, we do a simpler robust approach: keep ALL chunks, but build target
            # lines as those idx lines whose text appears in the chunk (string match). If none matched, drop.
            for ch in chunks:
                matched = []
                for idx, line in idx2line.items():
                    if str(line).strip() != "" and str(line) in ch:
                        matched.append(line)
                if len(matched) == 0:
                    continue
                expanded_rows.append({
                    "processed_func": ch,
                    "flaw_line": "/~/".join([str(x) for x in matched]),
                    "flaw_line_index": "",  # not used in training loss
                })
        if len(expanded_rows) > 0:
            vuln = pd.DataFrame(expanded_rows)
        logger.info(f"[{split_name}] CHUNK_TRAINING=yes: expanded vulnerable funcs to {len(vuln)} chunk-samples")

    if not INCLUDE_NEGATIVES:
        out = vuln
        logger.info(f"[{split_name}] vuln-only: {len(out)}")
    else:
        
        neg = df[df["target"] == 0].copy()
        neg = neg[["processed_func"]].reset_index(drop=True)

        # downsample negatives relative to vulnerable BEFORE chunk expansion (big speedup)
        target_neg = int(round(len(vuln) * NEGATIVE_RATIO))  # neg : vuln = ratio : 1
        if target_neg <= 0 or len(neg) == 0:
            neg = neg.iloc[0:0]
        else:
            if args.CHUNK_TRAINING == "yes" and args.NEG_CHUNKS_PER_FUNC > 0:
                # if each negative func yields `NEG_CHUNKS_PER_FUNC` chunks, sample fewer funcs first
                func_take = int(math.ceil(target_neg / max(1, int(args.NEG_CHUNKS_PER_FUNC))))
                if len(neg) > func_take:
                    neg = neg.sample(n=func_take, random_state=seed).reset_index(drop=True)

                expanded_neg = []
                for r in tqdm(neg.itertuples(index=False), total=len(neg), desc=f"[{split_name}] neg chunking"):
                    func_text = getattr(r, "processed_func")
                    chunks = _make_chunks_for_function(
                        func_text,
                        max_tokens=args.CHUNK_MAX_TOKENS,
                        overlap_lines=args.CHUNK_STRIDE_LINES,
                        prefix_lines=args.CHUNK_PREFIX_LINES,
                        prefix_max_tokens=args.CHUNK_PREFIX_MAX_TOKENS,
                    )
                    take = min(len(chunks), int(args.NEG_CHUNKS_PER_FUNC))
                    for ch in chunks[:take]:
                        expanded_neg.append({"processed_func": ch})

                neg = pd.DataFrame(expanded_neg) if len(expanded_neg) > 0 else neg.iloc[0:0]
                logger.info(f"[{split_name}] CHUNK_TRAINING=yes: expanded negatives to {len(neg)} chunk-samples")

                # final cap to target_neg
                if len(neg) > target_neg:
                    neg = neg.sample(n=target_neg, random_state=seed).reset_index(drop=True)
            else:
                # no chunk training: just sample negative functions directly
                if len(neg) > target_neg:
                    neg = neg.sample(n=target_neg, random_state=seed).reset_index(drop=True)

        neg["flaw_line"] = NO_VULN_TOKEN
        neg["flaw_line_index"] = ""  # empty

        out = pd.concat([vuln, neg], ignore_index=True)
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info(f"[{split_name}] vuln={len(vuln)} neg_used={len(out)-len(vuln)} total={len(out)}")
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info(f"[{split_name}] vuln={len(vuln)} neg_used={len(out)-len(vuln)} total={len(out)}")

    out_df = pd.DataFrame({
        "Text": out["processed_func"],
        "Lines": out["flaw_line"].apply(replace_delimiter_with_newline),
        "Line_Index": out["flaw_line_index"],
    })
    return out_df

train_df = build_seq2seq_dataframe(train_raw, "train")
val_df   = build_seq2seq_dataframe(val_raw, "val")
test_df  = build_seq2seq_dataframe(test_raw, "test")

# --------------------------
# Model
# --------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_variation)

# IMPORTANT OOM saver
model.config.use_cache = False

model = model.to(device)
if torch.cuda.device_count() > 1:
    logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# --------------------------
# Decide max target length (cap to 128)
# --------------------------
# Compute actual max tokenized length among vulnerable targets only (avoid NO_VULN_TOKEN)
vuln_lines = train_df[train_df["Lines"].astype(str).str.strip() != NO_VULN_TOKEN]["Lines"].tolist()
if len(vuln_lines) == 0:
    vuln_lines = train_df["Lines"].tolist()

enc = tokenizer(vuln_lines, add_special_tokens=True, truncation=False, padding=False)
true_max_target_len = max(len(x) for x in enc["input_ids"]) if len(enc["input_ids"]) > 0 else 16
max_len_lines = min(true_max_target_len, MAX_TARGET_LEN_CAP)  # <=128 by default
logger.info(f"True max target len (vuln-only): {true_max_target_len} | capped target len: {max_len_lines}")

# --------------------------
# Tokenize
# --------------------------
def tokenize_df(df: pd.DataFrame):
    inputs = tokenizer(
        df["Text"].tolist(),
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    targets = tokenizer(
        df["Lines"].tolist(),
        max_length=max_len_lines,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    labels = targets["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # IMPORTANT FIX

    return inputs["input_ids"], inputs["attention_mask"], labels

train_input_ids, train_attn, train_labels = tokenize_df(train_df)
val_input_ids,   val_attn,   val_labels   = tokenize_df(val_df)
test_input_ids,  test_attn,  test_labels  = tokenize_df(test_df)

train_dataset = TensorDataset(train_input_ids, train_attn, train_labels)
val_dataset   = TensorDataset(val_input_ids, val_attn, val_labels)
test_dataset  = TensorDataset(test_input_ids, test_attn, test_labels)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_loader   = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# --------------------------
# ROUGE util (no evaluate dependency)
# --------------------------
rouge = Rouge()

def rouge_l_f1(preds, refs) -> float:
    """
    Make `rouge` package robust:
    - it throws ValueError if any hypothesis is empty.
    Strategy:
      * drop pairs with empty reference
      * replace empty prediction with a placeholder token "<EMPTY>"
        (so it counts as mismatch but won't crash)
    """
    safe_preds = []
    safe_refs = []

    for p, r in zip(preds, refs):
        r = "" if r is None else str(r).strip()
        if r == "":
            continue  # no valid reference -> skip

        p = "" if p is None else str(p).strip()
        if p == "":
            p = "<EMPTY>"  # avoid "Hypothesis is empty." crash

        safe_preds.append(p)
        safe_refs.append(r)

    if len(safe_preds) == 0:
        return 0.0

    scores = rouge.get_scores(safe_preds, safe_refs, avg=True)
    return float(scores["rouge-l"]["f"])

# --------------------------
# Optimizer / Scheduler
# --------------------------
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)

max_steps = len(train_loader) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=max_steps // 5,
    num_training_steps=max_steps,
)

# AMP scaler
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# --------------------------
# Train
# --------------------------
if FINE_TUNE:
    logger.info("Starting training...")

    best_val_rouge = -1.0
    best_epoch = -1
    no_improve = 0

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    train_rouge_per_epoch = []
    val_rouge_per_epoch = []

    start_ms = int(round(time.time() * 1000))

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        total_preds = []
        total_refs = []

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids, attn_mask, labels = [x.to(device) for x in batch]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = outputs.loss
                loss = loss.mean() / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            # accumulate steps
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * GRAD_ACCUM_STEPS

            # collect preds for ROUGE (optional)
            # NOTE: generating every step is expensive; keep but you can comment out if too slow.
            with torch.no_grad():
                gen_model = model.module if hasattr(model, "module") else model
                gen_ids = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_length=max_len_lines,
                    num_beams=2,
                )
                decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
                decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

                if ROUGE_ON_VULN_ONLY:
                    for p, r in zip(decoded_preds, decoded_labels):
                        r_s = (r or "").strip()
                        if r_s and r_s != NO_VULN_TOKEN:
                            total_preds.append(p)
                            total_refs.append(r)
                else:
                    total_preds.extend(decoded_preds)
                    total_refs.extend(decoded_labels)

            # optional: free some cache
            if torch.cuda.is_available() and (step % 50 == 0):
                torch.cuda.empty_cache()

        avg_train_loss = running_loss / max(1, len(train_loader))
        train_loss_per_epoch.append(avg_train_loss)

        train_rouge = rouge_l_f1(total_preds, total_refs)
        train_rouge_per_epoch.append(train_rouge)

        # ---------------- validation ----------------
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_refs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, attn_mask, labels = [x.to(device) for x in batch]

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    vloss = outputs.loss.mean().item()
                val_running_loss += vloss

                gen_model = model.module if hasattr(model, "module") else model
                gen_ids = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_length=max_len_lines,
                    num_beams=4,
                )
                decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
                decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

                if ROUGE_ON_VULN_ONLY:
                    for p, r in zip(decoded_preds, decoded_labels):
                        r_s = (r or "").strip()
                        if r_s and r_s != NO_VULN_TOKEN:
                            val_preds.append(p)
                            val_refs.append(r)
                else:
                    val_preds.extend(decoded_preds)
                    val_refs.extend(decoded_labels)

        avg_val_loss = val_running_loss / max(1, len(val_loader))
        val_loss_per_epoch.append(avg_val_loss)

        val_rouge = rouge_l_f1(val_preds, val_refs)
        val_rouge_per_epoch.append(val_rouge)

        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Train ROUGE-L: {train_rouge:.4f} | Val ROUGE-L: {val_rouge:.4f}")

        # early stop
        if val_rouge > best_val_rouge:
            best_val_rouge = val_rouge
            best_epoch = epoch + 1
            no_improve = 0

            # save checkpoint
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "epoch": best_epoch,
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "train_loss_per_epoch": train_loss_per_epoch,
                    "val_loss_per_epoch": val_loss_per_epoch,
                    "train_rouge_per_epoch": train_rouge_per_epoch,
                    "val_rouge_per_epoch": val_rouge_per_epoch,
                },
                save_path,
            )
            logger.info(f"Saved best checkpoint at epoch {best_epoch} (ROUGE-L={best_val_rouge:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"Early stopping. Best epoch={best_epoch}, best ROUGE-L={best_val_rouge:.4f}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_ms = int(round(time.time() * 1000))
    logger.info(f"Training finished in {(end_ms - start_ms)//1000} seconds.")

# --------------------------
# Load best checkpoint
# --------------------------
if os.path.exists(save_path):
    logger.info(f"Loading best checkpoint: {save_path}")
    ckpt = torch.load(save_path, map_location=device)
    state_dict = ckpt["model"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
else:
    logger.warning("No checkpoint found. Using current model weights.")

# 额外导出为 HuggingFace save_pretrained 目录，方便评估阶段直接本地加载，避免再次依赖在线下载。
try:
    gen_model = model.module if hasattr(model, "module") else model
    gen_model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)
    logger.info(f"Exported HuggingFace model directory to: {args.checkpoint_dir}")
except Exception as e:
    logger.warning(f"Failed to export save_pretrained directory: {e}")

# --------------------------
# Quick test generation (optional)
# --------------------------
logger.info("Quick testing generation on a few samples...")
model.eval()
gen_model = model.module if hasattr(model, "module") else model

sample_n = min(3, len(test_df))
for i in range(sample_n):
    code = test_df["Text"].iloc[i]
    gt = str(test_df["Lines"].iloc[i])

    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_INPUT_LEN).to(device)
    with torch.no_grad():
        pred_ids = gen_model.generate(**inputs, max_length=max_len_lines, num_beams=4)
    pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    logger.info("-" * 60)
    logger.info(f"GT:   {gt[:200]}")
    logger.info(f"PRED: {pred[:200]}")

logger.info("Done.")