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
    parser.add_argument(
        "--cta-window-mode",
        default=defaults.cta_window_mode,
        choices=("auto", "on", "off"),
        help="CTA windowing mode: auto (filename heuristic), on, or off",
    )
    parser.add_argument(
        "--cta-window-level",
        type=float,
        default=defaults.cta_window_level,
        help="CTA window level (WL), used when CTA windowing is enabled",
    )
    parser.add_argument(
        "--cta-window-width",
        type=float,
        default=defaults.cta_window_width,
        help="CTA window width (WW), used when CTA windowing is enabled",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=defaults.yolo_conf,
        help="YOLO confidence threshold in [0, 1]",
    )
    parser.add_argument(
        "--roi-pad-xy",
        type=int,
        default=defaults.roi_pad_xy,
        help="ROI crop padding (voxels) in x/y direction",
    )
    parser.add_argument(
        "--roi-pad-z",
        type=int,
        default=defaults.roi_pad_z,
        help="ROI crop padding (voxels) in z direction",
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

    if args.cta_window_width <= 0:
        parser.error("--cta-window-width must be > 0")
    if not 0.0 <= args.yolo_conf <= 1.0:
        parser.error("--yolo-conf must be within [0, 1]")
    if args.roi_pad_xy < 0 or args.roi_pad_z < 0:
        parser.error("--roi-pad-xy and --roi-pad-z must be >= 0")

    run_topcow_single(
        input_nifti=Path(args.input),
        output_binary_mask=Path(args.output),
        repo_dir=Path(args.repo_dir),
        yolo_path=Path(args.yolo_path),
        segmodel_dir=Path(args.segmodel_dir),
        device=resolved_device,
        cta_window_mode=args.cta_window_mode,
        cta_window_level=args.cta_window_level,
        cta_window_width=args.cta_window_width,
        yolo_conf=args.yolo_conf,
        roi_pad_xy=args.roi_pad_xy,
        roi_pad_z=args.roi_pad_z,
    )


if __name__ == "__main__":
    main()
