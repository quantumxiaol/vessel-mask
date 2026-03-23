from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def multiclass_to_binary_vessel_mask(input_path: str | Path, output_path: str | Path) -> None:
    """Convert TopCoW multiclass segmentation to a binary vessel mask."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    img = nib.load(str(input_path))
    data = np.asarray(img.get_fdata())

    # TopCoW uses 0 as background and non-zero values for vessel classes.
    binary = (data > 0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(binary, img.affine, img.header)
    nib.save(out, str(output_path))

