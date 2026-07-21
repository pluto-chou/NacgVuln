#!/usr/bin/env python3
"""Download and validate the pretrained models used by NacgVuln.

By default, the script creates these repository-local directories:

- hf_cache/codet5-base for Salesforce/codet5-base
- hf_cache/codebert-base for microsoft/codebert-base

Examples:
    python scripts/download_hf_models.py --target-dir ./hf_cache
    python scripts/download_hf_models.py --target-dir ./hf_cache --force
    python scripts/download_hf_models.py --target-dir ./hf_cache \
        --hf-endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "codet5": {
        "model_id": "Salesforce/codet5-base",
        "subdir": "codet5-base",
        "kind": "seq2seq",
    },
    "codebert": {
        "model_id": "microsoft/codebert-base",
        "subdir": "codebert-base",
        "kind": "encoder",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CodeT5-base and CodeBERT-base into local directories."
    )
    parser.add_argument(
        "--target-dir",
        "--target_dir",
        dest="target_dir",
        type=Path,
        default=Path("./hf_cache"),
        help="Root directory for the local model folders (default: ./hf_cache).",
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        dest="cache_dir",
        type=Path,
        default=None,
        help="Optional Hugging Face download cache directory.",
    )
    parser.add_argument(
        "--hf-endpoint",
        "--hf_endpoint",
        dest="hf_endpoint",
        default=None,
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--models",
        default="codet5,codebert",
        help="Comma-separated models to download: codet5, codebert (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing model directory and download it again.",
    )
    return parser.parse_args()


def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def validate_json(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)


def model_weight_files(directory: Path) -> Iterable[Path]:
    yield from directory.glob("*.safetensors")
    yield from directory.glob("pytorch_model*.bin")


def validate_model_directory(directory: Path) -> None:
    required_json = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_json if not nonempty(directory / name)]

    has_sentencepiece = nonempty(directory / "spiece.model")
    has_bpe = nonempty(directory / "vocab.json") and nonempty(directory / "merges.txt")
    if not (has_sentencepiece or has_bpe):
        missing.append("spiece.model or vocab.json + merges.txt")

    weights = [path for path in model_weight_files(directory) if nonempty(path)]
    if not weights:
        missing.append("model.safetensors or pytorch_model*.bin")

    if missing:
        raise RuntimeError(
            f"Incomplete model directory {directory}: missing {', '.join(missing)}"
        )

    for name in required_json:
        validate_json(directory / name)
    for optional in ("generation_config.json", "special_tokens_map.json"):
        if nonempty(directory / optional):
            validate_json(directory / optional)


def download_model(model_id: str, destination: Path, kind: str, cache_dir: Optional[Path]) -> None:
    try:
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    print(f"[download] tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    tokenizer.save_pretrained(destination)

    print(f"[download] model: {model_id}")
    if kind == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    elif kind == "encoder":
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    model.save_pretrained(destination)
    validate_model_directory(destination)


def selected_models(raw_value: str) -> list[str]:
    values = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    invalid = sorted(set(values).difference(MODEL_SPECS))
    if invalid:
        raise ValueError(f"Unknown model aliases: {', '.join(invalid)}")
    if not values:
        raise ValueError("At least one model alias must be supplied")
    return values


def main() -> int:
    args = parse_args()
    target_root = args.target_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    target_root.mkdir(parents=True, exist_ok=True)
    for alias in selected_models(args.models):
        spec = MODEL_SPECS[alias]
        destination = target_root / spec["subdir"]

        if destination.exists() and args.force:
            print(f"[remove] {destination}")
            shutil.rmtree(destination)

        if destination.exists() and not args.force:
            try:
                validate_model_directory(destination)
                print(f"[reuse] {spec['model_id']} -> {destination}")
                continue
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}. Rerun with --force to replace it.") from exc

        destination.mkdir(parents=True, exist_ok=True)
        try:
            download_model(
                model_id=spec["model_id"],
                destination=destination,
                kind=spec["kind"],
                cache_dir=cache_dir,
            )
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        print(f"[ready] {spec['model_id']} -> {destination}")

    print("\nUse these repository-local model arguments:")
    print(f"  CodeT5:  {target_root / 'codet5-base'}")
    print(f"  CodeBERT:{target_root / 'codebert-base'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
