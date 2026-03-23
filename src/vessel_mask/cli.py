from __future__ import annotations

import argparse
from pathlib import Path

from vessel_mask.config import RuntimeConfig, load_runtime_config, resolve_device
from vessel_mask.topcow_runner import run_topcow_single


def build_parser(defaults: RuntimeConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vessel-mask-topcow",
        description="Run TopCoW inference for one .nii.gz file and output a binary vessel mask.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: load .env in current directory if present)",
    )
    parser.add_argument("--input", required=True, help="Input CTA/TOF-MRA .nii.gz file")
    parser.add_argument("--output", required=True, help="Output binary vessel-mask .nii.gz file")
    parser.add_argument(
        "--repo-dir",
        default=str(defaults.repo_dir),
        help="Path to vendored TopCoW repository root",
    )
    parser.add_argument(
        "--yolo-path",
        default=str(defaults.yolo_path),
        help="Path to YOLO detector checkpoint",
    )
    parser.add_argument(
        "--segmodel-dir",
        default=str(defaults.segmodel_dir),
        help="Path to nnUNet TopCoW model directory",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="Torch device string: auto, cuda:0, mps, or cpu",
    )
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_runtime_config(pre_args.env_file)

    parser = build_parser(defaults)
    args = parser.parse_args()
    resolved_device = resolve_device(args.device)

    run_topcow_single(
        input_nifti=Path(args.input),
        output_binary_mask=Path(args.output),
        repo_dir=Path(args.repo_dir),
        yolo_path=Path(args.yolo_path),
        segmodel_dir=Path(args.segmodel_dir),
        device=resolved_device,
    )


if __name__ == "__main__":
    main()
