#!/usr/bin/env python3
"""Train the function-level CodeBERT classifier for multiple seeds.

This runner replaces the original root-level
``run_10seed_func_classifier_train.py`` command while targeting the renamed
implementation at ``src/train_func_classifier.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_SEEDS = list(range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_root", default="./checkpoints_func_10seed")
    parser.add_argument("--train_script", default="./src/train_func_classifier.py")
    parser.add_argument("--model_variation", default="./hf_cache/codebert-base")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--sampling", default="no", choices=["yes", "no"])
    parser.add_argument("--remove_missing_line_labels", default="yes", choices=["yes", "no"])
    parser.add_argument("--explainer", default="ATTENTION", choices=["ATTENTION", "LIME"])
    parser.add_argument("--explain_only_tp", default="no", choices=["yes", "no"])
    parser.add_argument("--sort_by_lines", default="yes", choices=["yes", "no"])
    parser.add_argument("--resume_if_exists", default="yes", choices=["yes", "no"])
    parser.add_argument("--continue_on_error", default="no", choices=["yes", "no"])
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = resolve_path(project_root, args.output_root)
    train_script = resolve_path(project_root, args.train_script)
    model_source = args.model_variation
    if Path(model_source).expanduser().is_absolute():
        model_source = str(Path(model_source).expanduser().resolve())
    elif (project_root / model_source).exists():
        model_source = str((project_root / model_source).resolve())

    dataset_path = project_root / "data" / "dataset.csv"
    if not project_root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {project_root}")
    if not train_script.is_file():
        raise FileNotFoundError(f"Training script does not exist: {train_script}")
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_path}. Run: python data/data_mining.py"
        )

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        seeds = DEFAULT_SEEDS
    invalid = [seed for seed in seeds if seed not in DEFAULT_SEEDS]
    if invalid:
        raise ValueError(f"Seeds must be in 0..9: {invalid}")

    logs_dir = output_root / "logs"
    checkpoints_dir = output_root / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "train_script": str(train_script),
        "model_variation": model_source,
        "seeds": seeds,
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    failures: list[dict[str, object]] = []
    for seed in seeds:
        seed_dir = checkpoints_dir / f"seed{seed}"
        best_weights = seed_dir / "best_weights.pt"
        log_path = logs_dir / f"seed{seed}.log"
        seed_dir.mkdir(parents=True, exist_ok=True)

        if args.resume_if_exists == "yes" and best_weights.is_file():
            print(f"[SKIP] seed={seed}: {best_weights}", flush=True)
            continue

        cmd = [
            args.python_bin,
            "-u",
            str(train_script),
            "--seed", str(seed),
            "--FINE_TUNE", "yes",
            "--model_variation", model_source,
            "--checkpoint_dir", str(seed_dir),
            "--sampling", args.sampling,
            "--REMOVE_MISSING_LINE_LABELS", args.remove_missing_line_labels,
            "--EXPLAINER", args.explainer,
            "--EXPLAIN_ONLY_TP", args.explain_only_tp,
            "--sort_by_lines", args.sort_by_lines,
        ]
        print(f"[RUN] seed={seed}: {' '.join(cmd)}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n" + "=" * 100 + "\n")
            log_file.write(" ".join(cmd) + "\n")
            log_file.flush()
            completed = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        if completed.returncode != 0 or not best_weights.is_file():
            failure = {
                "seed": seed,
                "returncode": completed.returncode,
                "checkpoint": str(best_weights),
                "log": str(log_path),
            }
            failures.append(failure)
            print(f"[FAIL] {failure}", flush=True)
            if args.continue_on_error == "no":
                break
        else:
            print(f"[OK] seed={seed}: {best_weights}", flush=True)

    with (output_root / "failed_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "returncode", "checkpoint", "log"])
        writer.writeheader()
        writer.writerows(failures)

    if failures:
        print(f"Completed with {len(failures)} failed seed(s).", flush=True)
        return 1
    print(f"All function-level checkpoints are under: {checkpoints_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
