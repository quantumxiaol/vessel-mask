from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RuntimeConfig:
    repo_dir: Path
    checkpoints_dir: Path
    yolo_path: Path
    segmodel_dir: Path
    device: str


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def load_runtime_config(env_file: str | Path | None = None) -> RuntimeConfig:
    load_dotenv(dotenv_path=env_file, override=False)

    repo_dir = _path_from_env(
        "VESSEL_MASK_REPO_DIR",
        "third_party/TopCoW_2024_MRA_winning_solution",
    )
    checkpoints_dir = _path_from_env("VESSEL_MASK_CHECKPOINTS_DIR", "checkpoints")
    yolo_path = _path_from_env(
        "VESSEL_MASK_YOLO_PATH",
        str(checkpoints_dir / "yolo-cow-detection.pt"),
    )
    segmodel_dir = _path_from_env(
        "VESSEL_MASK_SEGMODEL_DIR",
        str(checkpoints_dir / "topcow-claim-models"),
    )
    device = os.getenv("VESSEL_MASK_DEVICE", "auto").strip()
    if not device:
        device = "auto"

    return RuntimeConfig(
        repo_dir=repo_dir,
        checkpoints_dir=checkpoints_dir,
        yolo_path=yolo_path,
        segmodel_dir=segmodel_dir,
        device=device,
    )


def resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized and normalized != "auto":
        return device

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"

