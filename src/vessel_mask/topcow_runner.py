from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vessel_mask.io import multiclass_to_binary_vessel_mask


def _replace_assignment(script: str, name: str, value: str) -> str:
    """Replace one module-level assignment robustly, including multiline RHS."""
    lines = script.splitlines(keepends=True)
    module = ast.parse(script)
    target_node: ast.Assign | ast.AnnAssign | None = None

    for stmt in module.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    target_node = stmt
                    break
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                target_node = stmt
        if target_node is not None:
            break

    if target_node is None or target_node.end_lineno is None:
        raise RuntimeError(f"Failed to patch assignment for {name}.")

    replacement = f"{name} = {value!r}\n"
    start_idx = target_node.lineno - 1
    end_idx = target_node.end_lineno
    return "".join(lines[:start_idx] + [replacement] + lines[end_idx:])


def _patch_topcow_script(
    script_path: Path,
    yolo_path: Path,
    segmodel_dir: Path,
    img_dir: Path,
    out_dir: Path,
    device: str,
) -> Path:
    script = script_path.read_text(encoding="utf-8")
    script = _replace_assignment(script, "YOLO_FILE_PATH", str(yolo_path))
    script = _replace_assignment(script, "SEGMODEL_DIR_PATH", str(segmodel_dir))
    script = _replace_assignment(script, "IMG_FOLDER", str(img_dir))
    script = _replace_assignment(script, "SEGMENTATION_FOLDER", str(out_dir))

    script, device_replacements = re.subn(
        r"torch\.device\(\s*['\"]cuda['\"]\s*,\s*0\s*\)",
        f"torch.device({device!r})",
        script,
    )
    if device_replacements < 1:
        raise RuntimeError("Failed to patch device in upstream TopCoW script.")

    run_fully_on_device = device.lower().startswith("cuda")
    script = re.sub(
        r"perform_everything_on_device\s*=\s*True",
        f"perform_everything_on_device={run_fully_on_device}",
        script,
    )

    patched_path = script_path.parent / "_segment_circle_of_willis_patched.py"
    patched_path.write_text(script, encoding="utf-8")
    return patched_path


def _assert_file_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _assert_dir_exists(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _validate_input_name(path: Path) -> None:
    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"Input must be a .nii.gz file, got: {path}")


def run_topcow_single(
    input_nifti: str | Path,
    output_binary_mask: str | Path,
    repo_dir: str | Path,
    yolo_path: str | Path,
    segmodel_dir: str | Path,
    device: str = "cuda:0",
    cta_window_mode: str = "auto",
    cta_window_level: float = 300.0,
    cta_window_width: float = 1000.0,
    yolo_conf: float = 0.25,
    roi_pad_xy: int = 10,
    roi_pad_z: int = 5,
) -> None:
    repo_dir = Path(repo_dir).resolve()
    input_nifti = Path(input_nifti).resolve()
    output_binary_mask = Path(output_binary_mask).resolve()
    yolo_path = Path(yolo_path).resolve()
    segmodel_dir = Path(segmodel_dir).resolve()

    _validate_input_name(input_nifti)
    _assert_file_exists(input_nifti, "Input NIfTI")
    _assert_file_exists(yolo_path, "YOLO model")
    _assert_dir_exists(segmodel_dir, "Segmentation model directory")

    script_path = repo_dir / "segment_circle_of_willis.py"
    _assert_file_exists(script_path, "TopCoW script")

    patched_script: Path | None = None

    try:
        with tempfile.TemporaryDirectory(prefix="topcow_run_") as tmp_dir:
            tmp_dir = Path(tmp_dir)
            img_dir = tmp_dir / "inputs"
            out_dir = tmp_dir / "outputs"
            img_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(input_nifti, img_dir / input_nifti.name)

            patched_script = _patch_topcow_script(
                script_path=script_path,
                yolo_path=yolo_path,
                segmodel_dir=segmodel_dir,
                img_dir=img_dir,
                out_dir=out_dir,
                device=device,
            )

            child_env = os.environ.copy()
            child_env["TOPCOW_CTA_WINDOW_MODE"] = cta_window_mode
            child_env["TOPCOW_CTA_WINDOW_LEVEL"] = str(cta_window_level)
            child_env["TOPCOW_CTA_WINDOW_WIDTH"] = str(cta_window_width)
            child_env["TOPCOW_YOLO_CONF"] = str(yolo_conf)
            child_env["TOPCOW_ROI_PAD_XY"] = str(roi_pad_xy)
            child_env["TOPCOW_ROI_PAD_Z"] = str(roi_pad_z)

            subprocess.run(
                [sys.executable, str(patched_script)],
                cwd=str(repo_dir),
                check=True,
                env=child_env,
            )

            output_candidates = sorted(out_dir.glob("*_seg.nii.gz"))
            if len(output_candidates) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one TopCoW output in {out_dir}, found {len(output_candidates)}."
                )

            multiclass_to_binary_vessel_mask(output_candidates[0], output_binary_mask)
    finally:
        if patched_script is not None and patched_script.exists():
            patched_script.unlink()
