#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下载并校验两个本地模型目录：
1) Salesforce/codet5-base
2) microsoft/codebert-base

默认会在 --target_dir 下创建：
- codet5-base
- codebert-base

示例：
python download_code_both.py --target_dir ./hf_cache --force
python download_code_both.py --target_dir ./hf_cache --hf_endpoint https://hf-mirror.com --force
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_args():
    ap = argparse.ArgumentParser(description="下载并校验 CodeT5 + CodeBERT 本地目录")
    ap.add_argument("--target_dir", required=True, help="根目录；脚本会在其下创建 codet5-base 和 codebert-base")
    ap.add_argument("--cache_dir", default=None, help="可选：transformers cache_dir")
    ap.add_argument("--hf_endpoint", default=None, help="可选：例如 https://hf-mirror.com")
    ap.add_argument("--force", action="store_true", help="如果子目录已存在，先删除再重下")
    return ap.parse_args()


def remove_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def validate_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        json.load(f)


def _exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def validate_dir(target_dir: Path):
    required_common = [
        "config.json",
        "tokenizer_config.json",
    ]
    optional_json = [
        "special_tokens_map.json",
        "generation_config.json",
        "added_tokens.json",
    ]

    missing = []
    for name in required_common:
        p = target_dir / name
        if not _exists_and_nonempty(p):
            missing.append(name)

    spiece = target_dir / "spiece.model"
    vocab = target_dir / "vocab.json"
    merges = target_dir / "merges.txt"

    has_sentencepiece = _exists_and_nonempty(spiece)
    has_bpe = _exists_and_nonempty(vocab) and _exists_and_nonempty(merges)

    if not (has_sentencepiece or has_bpe):
        missing.append("spiece.model 或 vocab.json+merges.txt")

    safetensors_file = target_dir / "model.safetensors"
    bin_file = target_dir / "pytorch_model.bin"
    has_model = _exists_and_nonempty(safetensors_file) or _exists_and_nonempty(bin_file)
    if not has_model:
        missing.append("model.safetensors 或 pytorch_model.bin")

    if missing:
        raise RuntimeError(f"缺少关键文件或文件为空: {missing}")

    for name in required_common + [x for x in optional_json if _exists_and_nonempty(target_dir / x)]:
        validate_json(target_dir / name)

    return {
        "target_dir": str(target_dir.resolve()),
        "has_sentencepiece": has_sentencepiece,
        "has_bpe_tokenizer": has_bpe,
        "model_file": str(safetensors_file if _exists_and_nonempty(safetensors_file) else bin_file),
    }


def download_one(model_id: str, save_dir: Path, model_kind: str, cache_dir: Optional[str] = None):
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

    print(f"[INFO] model_id = {model_id}")
    print(f"[INFO] save_dir = {save_dir}")

    print("[INFO] 开始下载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        cache_dir=cache_dir,
    )
    tok.save_pretrained(str(save_dir))
    print("[INFO] tokenizer 已保存")

    print("[INFO] 开始下载 model ...")
    if model_kind == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
    elif model_kind == "encoder":
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"未知 model_kind: {model_kind}")

    model.save_pretrained(str(save_dir))
    print("[INFO] model 已保存")

    print("[INFO] 开始校验文件完整性 ...")
    info = validate_dir(save_dir)
    print("[INFO] 校验通过")
    for k, v in info.items():
        print(f"  - {k}: {v}")


def main():
    args = parse_args()
    target_root = Path(args.target_dir).expanduser().resolve()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    ensure_dir(target_root)

    try:
        import transformers  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "当前环境缺少 transformers。请先安装: pip install -U transformers sentencepiece safetensors huggingface_hub torch"
        ) from e

    jobs = [
        {
            "model_id": "Salesforce/codet5-base",
            "subdir": "codet5-base",
            "model_kind": "seq2seq",
        },
        {
            "model_id": "microsoft/codebert-base",
            "subdir": "codebert-base",
            "model_kind": "encoder",
        },
    ]

    for job in jobs:
        save_dir = target_root / job["subdir"]
        if args.force and save_dir.exists():
            print(f"[INFO] 删除已有目录: {save_dir}")
            remove_dir(save_dir)
        ensure_dir(save_dir)

        try:
            download_one(
                model_id=job["model_id"],
                save_dir=save_dir,
                model_kind=job["model_kind"],
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            eprint(f"[ERROR] 下载失败: {job['model_id']}")
            eprint(repr(e))
            eprint("\n可能原因：")
            eprint("1) 机器无法联网到 Hugging Face")
            eprint("2) 需要设置镜像，例如 --hf_endpoint https://hf-mirror.com")
            eprint("3) 磁盘空间不足")
            raise

    print("\n====== 后续参数示例 ======")
    print(f"--model_variation {target_root / 'codebert-base'}")
    print(f"--model_variation_seq2seq {target_root / 'codet5-base'}")
    print("\n[OK] CodeT5 和 CodeBERT 已下载完成。")


if __name__ == "__main__":
    main()
