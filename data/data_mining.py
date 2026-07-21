#!/usr/bin/env python3
"""Download and assemble the third-party Big-Vul files used by NacgVuln.

The three CSV files are the preprocessed Big-Vul splits distributed by the
LineVul replication package. They are concatenated in the fixed order
train.csv, val.csv, test.csv to create data/dataset.csv.

The script is idempotent: an existing, valid dataset.csv is reused unless
--force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

BIG_VUL_REPOSITORY = (
    "https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset"
)
LINEVUL_REPOSITORY = "https://github.com/awsm-research/LineVul"

SOURCES = (
    ("train.csv", "1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw"),
    ("val.csv", "1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ"),
    ("test.csv", "1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V"),
)

REQUIRED_COLUMNS = {
    "processed_func",
    "target",
    "flaw_line",
    "flaw_line_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the LineVul-preprocessed Big-Vul splits and concatenate "
            "them into data/dataset.csv."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for downloaded files and dataset.csv (default: repository data/).",
    )
    parser.add_argument(
        "--output-name",
        default="dataset.csv",
        help="Combined CSV file name (default: dataset.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload source files and overwrite the combined output.",
    )
    parser.add_argument(
        "--keep-components",
        action="store_true",
        help="Keep train.csv, val.csv, and test.csv after successful assembly.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_gdown():
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required. Install the project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    return gdown


def download_source(file_id: str, destination: Path, force: bool) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[reuse] {destination}")
        return

    gdown = require_gdown()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    print(f"[download] Google Drive file {file_id} -> {destination}")
    result = gdown.download(id=file_id, output=str(temporary), quiet=False)
    if result is None or not temporary.exists() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed or produced an empty file: {destination.name}")

    temporary.replace(destination)


def load_and_validate(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read CSV file {path}: {exc}") from exc

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise RuntimeError(f"{path.name} contains no rows")

    print(f"[validated] {path.name}: {len(frame):,} rows, {len(frame.columns)} columns")
    return frame


def remove_files(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = data_dir / args.output_name
    component_paths = [data_dir / name for name, _ in SOURCES]
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Third-party data provenance:")
    print(f"  Original Big-Vul repository: {BIG_VUL_REPOSITORY}")
    print(f"  Preprocessed split source:   {LINEVUL_REPOSITORY}")

    if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
        existing = load_and_validate(output_path)
        print(f"[reuse] {output_path}: {len(existing):,} rows")
        print(f"[sha256] {sha256(output_path)}")
        print("Use --force to redownload and rebuild the dataset.")
        return 0

    for (file_name, file_id), destination in zip(SOURCES, component_paths):
        assert destination.name == file_name
        download_source(file_id, destination, force=args.force)

    frames = [load_and_validate(path) for path in component_paths]
    combined = pd.concat(frames, ignore_index=True)

    temporary_output = output_path.with_suffix(output_path.suffix + ".part")
    temporary_output.unlink(missing_ok=True)
    combined.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)

    validated = load_and_validate(output_path)
    print(f"[created] {output_path}: {len(validated):,} rows")
    print(f"[sha256] {sha256(output_path)}")

    if not args.keep_components:
        remove_files(component_paths)
        print("[cleanup] Removed train.csv, val.csv, and test.csv.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
