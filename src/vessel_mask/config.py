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
    cta_window_mode: str
    cta_window_level: float
    cta_window_width: float
    yolo_conf: float
    roi_pad_xy: int
    roi_pad_z: int


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw!r}") from exc


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc


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

    cta_window_mode = os.getenv("VESSEL_MASK_CTA_WINDOW_MODE", "auto").strip().lower() or "auto"
    if cta_window_mode not in {"auto", "on", "off"}:
        raise ValueError(
            "VESSEL_MASK_CTA_WINDOW_MODE must be one of: auto, on, off; "
            f"got: {cta_window_mode!r}"
        )

    cta_window_level = _float_from_env("VESSEL_MASK_CTA_WINDOW_LEVEL", 300.0)
    cta_window_width = _float_from_env("VESSEL_MASK_CTA_WINDOW_WIDTH", 1000.0)
    if cta_window_width <= 0:
        raise ValueError(
            "VESSEL_MASK_CTA_WINDOW_WIDTH must be > 0; "
            f"got: {cta_window_width}"
        )

    yolo_conf = _float_from_env("VESSEL_MASK_YOLO_CONF", 0.25)
    if not 0.0 <= yolo_conf <= 1.0:
        raise ValueError(
            "VESSEL_MASK_YOLO_CONF must be within [0, 1]; "
            f"got: {yolo_conf}"
        )

    roi_pad_xy = _int_from_env("VESSEL_MASK_ROI_PAD_XY", 10)
    roi_pad_z = _int_from_env("VESSEL_MASK_ROI_PAD_Z", 5)
    if roi_pad_xy < 0 or roi_pad_z < 0:
        raise ValueError(
            "VESSEL_MASK_ROI_PAD_XY and VESSEL_MASK_ROI_PAD_Z must be >= 0; "
            f"got: xy={roi_pad_xy}, z={roi_pad_z}"
        )

    return RuntimeConfig(
        repo_dir=repo_dir,
        checkpoints_dir=checkpoints_dir,
        yolo_path=yolo_path,
        segmodel_dir=segmodel_dir,
        device=device,
        cta_window_mode=cta_window_mode,
        cta_window_level=cta_window_level,
        cta_window_width=cta_window_width,
        yolo_conf=yolo_conf,
        roi_pad_xy=roi_pad_xy,
        roi_pad_z=roi_pad_z,
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
