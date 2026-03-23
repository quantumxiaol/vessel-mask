#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


DEFAULT_RECORD_ID = "14191592"
DEFAULT_ARCHIVE_KEY = "topcow_claim_models.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TopCoW model archive from Zenodo and prepare checkpoints/",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: load .env in current directory if present)",
    )
    parser.add_argument(
        "--record-id",
        default=DEFAULT_RECORD_ID,
        help=f"Zenodo record id (default: {DEFAULT_RECORD_ID})",
    )
    parser.add_argument(
        "--archive-key",
        default=DEFAULT_ARCHIVE_KEY,
        help=f"Archive filename in Zenodo record (default: {DEFAULT_ARCHIVE_KEY})",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination checkpoints directory (default: VESSEL_MASK_CHECKPOINTS_DIR or ./checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload archive and overwrite existing unpacked model files",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep downloaded zip archive after extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve URLs and print plan without downloading",
    )
    return parser.parse_args()


def fetch_zenodo_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as response:  # noqa: S310
        return json.load(response)


def select_archive_file(record: dict, archive_key: str) -> dict:
    files = record.get("files", [])
    for f in files:
        if f.get("key") == archive_key:
            return f

    available = ", ".join(sorted(f.get("key", "") for f in files))
    raise RuntimeError(
        f"Archive '{archive_key}' not found in record {record.get('id')}. Available: {available}"
    )


def verify_file_size(file_path: Path, expected_size: int | None) -> None:
    if expected_size is None:
        print(f"[verify] skipped size check (no expected size from server): {file_path}")
        return

    actual_size = file_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Downloaded file size mismatch for {file_path}: expected {expected_size}, got {actual_size}"
        )
    print(f"[verify] file size ok: {actual_size} bytes")


def download_file(url: str, dest_file: Path, expected_size: int | None, force: bool) -> None:
    if dest_file.exists() and not force:
        if expected_size is None or dest_file.stat().st_size == expected_size:
            print(f"[skip] Archive already exists: {dest_file}")
            verify_file_size(dest_file, expected_size)
            return

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if dest_file.exists() and force:
        dest_file.unlink()

    print(f"[download] {url}")
    print(f"[to] {dest_file}")

    with urllib.request.urlopen(url) as response, dest_file.open("wb") as out_file:  # noqa: S310
        total = response.headers.get("Content-Length")
        total_size = int(total) if total is not None else expected_size
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
            mininterval=0.1,
            dynamic_ncols=True,
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

    print("[done] download complete")
    verify_file_size(dest_file, expected_size)


def _find_yolo_file(extract_root: Path) -> Path:
    candidates = sorted(extract_root.rglob("yolo-cow-detection.pt"))
    if not candidates:
        raise FileNotFoundError("Could not find yolo-cow-detection.pt inside extracted archive.")
    return candidates[0]


def _find_segmodel_dir(extract_root: Path) -> Path:
    preferred_names = ("topcow-claim-models", "topcow_claim_models")
    for name in preferred_names:
        for path in sorted(extract_root.rglob(name)):
            if path.is_dir():
                return path

    for fold_0 in sorted(extract_root.rglob("fold_0")):
        if fold_0.is_dir():
            return fold_0.parent

    raise FileNotFoundError("Could not find topcow-claim-models directory inside extracted archive.")


def extract_and_place_models(
    archive_path: Path,
    checkpoints_dir: Path,
    force: bool,
) -> tuple[Path, Path]:
    yolo_target = checkpoints_dir / "yolo-cow-detection.pt"
    seg_target = checkpoints_dir / "topcow-claim-models"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if yolo_target.exists() and seg_target.is_dir() and not force:
        print(f"[skip] Existing model files found in {checkpoints_dir}, use --force to overwrite.")
        return yolo_target, seg_target

    with tempfile.TemporaryDirectory(prefix="topcow_extract_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        print(f"[extract] {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_dir)

        yolo_source = _find_yolo_file(tmp_dir)
        seg_source = _find_segmodel_dir(tmp_dir)

        if force and yolo_target.exists():
            yolo_target.unlink()
        shutil.copy2(yolo_source, yolo_target)

        if force and seg_target.exists():
            shutil.rmtree(seg_target)
        shutil.copytree(seg_source, seg_target, dirs_exist_ok=True)

    return yolo_target, seg_target


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=args.env_file, override=False)

    checkpoints_dir = Path(
        args.dest
        or Path.cwd() / Path(os.environ.get("VESSEL_MASK_CHECKPOINTS_DIR", "checkpoints"))
    ).expanduser()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    try:
        record = fetch_zenodo_record(args.record_id)
        archive_meta = select_archive_file(record, args.archive_key)
        download_url = archive_meta["links"]["self"]
        archive_size = archive_meta.get("size")
        archive_path = checkpoints_dir / args.archive_key

        print(f"[record] {record.get('id')} - {record.get('title')}")
        print(f"[archive] {args.archive_key} ({archive_size} bytes)")
        print(f"[checkpoints] {checkpoints_dir.resolve()}")

        if args.dry_run:
            print("[dry-run] done")
            return 0

        download_file(
            url=download_url,
            dest_file=archive_path,
            expected_size=archive_size,
            force=args.force,
        )

        yolo_path, segmodel_dir = extract_and_place_models(
            archive_path=archive_path,
            checkpoints_dir=checkpoints_dir,
            force=args.force,
        )

        if not args.keep_archive and archive_path.exists():
            archive_path.unlink()
            print(f"[cleanup] removed archive {archive_path}")

        print("[ready]")
        print(f"VESSEL_MASK_YOLO_PATH={yolo_path}")
        print(f"VESSEL_MASK_SEGMODEL_DIR={segmodel_dir}")
        return 0
    except (RuntimeError, FileNotFoundError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
