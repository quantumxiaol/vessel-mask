import shutil
import numpy as np
import SimpleITK as sitk
import os
import torch
import nibabel as nib
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.ensembling.ensemble import ensemble_folders
import glob
from dotenv import load_dotenv

from ultralytics import YOLO
import cv2

#######################################################################################
# Load .env from current working directory if present
load_dotenv()


def _env_raw(primary: str, secondary: str | None = None) -> str | None:
    raw = os.getenv(primary)
    if raw is None and secondary is not None:
        raw = os.getenv(secondary)
    if raw is None:
        return None
    return raw.strip()


def _env_float(primary: str, secondary: str | None, default: float) -> float:
    raw = _env_raw(primary, secondary)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{primary} must be a float, got: {raw!r}") from exc


def _env_int(primary: str, secondary: str | None, default: int) -> int:
    raw = _env_raw(primary, secondary)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{primary} must be an int, got: {raw!r}") from exc


def _env_choice(primary: str, secondary: str | None, default: str, choices: set[str]) -> str:
    raw = _env_raw(primary, secondary)
    value = (raw or default).lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{primary} must be one of [{allowed}], got: {value!r}")
    return value


YOLO_FILE_PATH = os.getenv(
    "TOPCOW_YOLO_FILE_PATH",
    os.getenv("VESSEL_MASK_YOLO_PATH", "./models/yolo-cow-detection.pt"),
)
SEGMODEL_DIR_PATH = os.getenv(
    "TOPCOW_SEGMODEL_DIR_PATH",
    os.getenv("VESSEL_MASK_SEGMODEL_DIR", "./models/topcow-claim-models"),
)
IMG_FOLDER = os.getenv("TOPCOW_IMG_FOLDER", "./folder_to_predict")
SEGMENTATION_FOLDER = os.getenv("TOPCOW_SEGMENTATION_FOLDER", "./outputs_segmentations")

DEVICE_STR = os.getenv("TOPCOW_DEVICE", os.getenv("VESSEL_MASK_DEVICE", "cuda:0"))
TORCH_LOAD_WEIGHTS_ONLY = os.getenv("TOPCOW_TORCH_LOAD_WEIGHTS_ONLY", "false").strip().lower()
CTA_WINDOW_MODE = _env_choice("TOPCOW_CTA_WINDOW_MODE", "VESSEL_MASK_CTA_WINDOW_MODE", "auto", {"auto", "on", "off"})
CTA_WINDOW_LEVEL = _env_float("TOPCOW_CTA_WINDOW_LEVEL", "VESSEL_MASK_CTA_WINDOW_LEVEL", 300.0)
CTA_WINDOW_WIDTH = _env_float("TOPCOW_CTA_WINDOW_WIDTH", "VESSEL_MASK_CTA_WINDOW_WIDTH", 1000.0)
YOLO_CONF = _env_float("TOPCOW_YOLO_CONF", "VESSEL_MASK_YOLO_CONF", 0.25)
ROI_PAD_XY = _env_int("TOPCOW_ROI_PAD_XY", "VESSEL_MASK_ROI_PAD_XY", 10)
ROI_PAD_Z = _env_int("TOPCOW_ROI_PAD_Z", "VESSEL_MASK_ROI_PAD_Z", 5)

if CTA_WINDOW_WIDTH <= 0:
    raise ValueError(f"CTA window width must be > 0, got: {CTA_WINDOW_WIDTH}")
if not 0.0 <= YOLO_CONF <= 1.0:
    raise ValueError(f"YOLO confidence must be within [0, 1], got: {YOLO_CONF}")
if ROI_PAD_XY < 0 or ROI_PAD_Z < 0:
    raise ValueError(f"ROI paddings must be >= 0, got: xy={ROI_PAD_XY}, z={ROI_PAD_Z}")
#######################################################################################


def _patch_torch_load_default_weights_only():
    """
    Compat for PyTorch >= 2.6 where torch.load defaults to weights_only=True.
    TopCoW checkpoints contain python objects and require weights_only=False.
    """
    if getattr(torch.load, "__name__", "") == "_topcow_torch_load_compat":
        return

    original_torch_load = torch.load

    def _topcow_torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = TORCH_LOAD_WEIGHTS_ONLY == "true"
        return original_torch_load(*args, **kwargs)

    torch.load = _topcow_torch_load_compat


_patch_torch_load_default_weights_only()

def main():
    print("Starting segmentation...")
    print("Model:", SEGMODEL_DIR_PATH)
    print("Input:", IMG_FOLDER)
    print("Output:", SEGMENTATION_FOLDER)
    print("Device:", DEVICE_STR)
    print("Resolved device:", resolve_device(DEVICE_STR))
    print(
        "CTA window:",
        f"mode={CTA_WINDOW_MODE}, WL={CTA_WINDOW_LEVEL}, WW={CTA_WINDOW_WIDTH}",
    )
    print("YOLO conf:", YOLO_CONF)
    print("ROI pad:", f"xy={ROI_PAD_XY}, z={ROI_PAD_Z}")

    if not os.path.exists(SEGMENTATION_FOLDER):
        os.makedirs(SEGMENTATION_FOLDER)

    img_files = resolve_input_paths(IMG_FOLDER)
    if not img_files:
        raise FileNotFoundError(f"No .nii.gz files found in input: {IMG_FOLDER}")

    for img_file in img_files:
        # Load the NIfTI image
        print("------------------------------------")
        print(os.path.basename(img_file))

        try:
            # Perform segmentation
            seg_data = segment_nifti(img_file)

            # Save the segmentation as a NIfTI file
            seg_file = os.path.basename(img_file).split(".")[0] + "_seg.nii.gz"
            seg_file = os.path.join(SEGMENTATION_FOLDER, seg_file)
            save_nifti_file(seg_data, seg_file, img_file)

            # Load and check saved segmentation
            load_seg, _ = load_nifti_file(seg_file)
            print("LOADED SAVED seg_data unique = ", np.unique(load_seg))
    
        except Exception as e:
            # Print the error and continue to the next file
            print(f"Error encountered while processing {img_file}: {str(e)}")

    print("Segmentation done!")


def resolve_input_paths(input_path: str) -> list[str]:
    input_obj = Path(input_path).expanduser()
    if input_obj.is_file():
        if not input_obj.name.endswith(".nii.gz"):
            raise ValueError(f"Input file must end with .nii.gz: {input_obj}")
        return [str(input_obj.resolve())]

    if input_obj.is_dir():
        return sorted(
            str(p.resolve())
            for p in input_obj.glob("*.nii.gz")
            if p.is_file()
        )

    raise FileNotFoundError(f"Input path does not exist: {input_obj}")


def load_nifti_file(nifti_path):
    nifti_img = nib.load(nifti_path)
    try:
        nifti_data = nifti_img.get_fdata()
    except Exception as e:
        print(f"Warning: nib.get_fdata failed for {nifti_path}: {e}")
        print("Trying fallback conversion for vector/structured NIfTI input.")
        nifti_data = _coerce_to_scalar_volume(nifti_img)

    nifti_data = np.asarray(nifti_data)
    if nifti_data.ndim != 3:
        raise ValueError(
            f"Expected 3D scalar volume after loading, but got shape {nifti_data.shape} for {nifti_path}."
        )
    return nifti_data, nifti_img


def save_nifti_file(data, output_path, reference_nifti_path):
    """
    Save the 3D segmentation volume as a NIfTI file.
    Use the reference NIfTI file to copy the affine and header information.
    """
    reference_img = nib.load(reference_nifti_path)
    affine = reference_img.affine
    header = reference_img.header.copy()
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        header.set_data_dtype(np.asarray(data).dtype)
    else:
        header.set_data_dtype(np.float32)

    nifti_img = nib.Nifti1Image(data, affine, header)
    nib.save(nifti_img, output_path)


def _coerce_to_scalar_volume(nifti_img):
    """
    Convert vector/structured NIfTI payloads to a 3D scalar volume.
    """
    raw = np.asanyarray(nifti_img.dataobj)

    # Case 1: structured dtype, e.g. RGB24 as fields (R, G, B)
    if raw.dtype.fields is not None:
        field_names = raw.dtype.names or ()
        if {"R", "G", "B"}.issubset(set(field_names)):
            rgb = np.stack(
                [raw["R"].astype(np.float32), raw["G"].astype(np.float32), raw["B"].astype(np.float32)],
                axis=-1,
            )
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            print("Detected structured RGB NIfTI. Converted to grayscale volume.")
            return gray

        # Generic structured fallback: use first field
        first = field_names[0]
        print(f"Detected structured dtype with fields {field_names}. Using first field '{first}'.")
        return raw[first].astype(np.float32)

    # Case 2: explicit channel axis in last dimension
    if raw.ndim == 4 and raw.shape[-1] in (3, 4):
        rgb = raw[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        print("Detected multi-channel 4D NIfTI. Converted first 3 channels to grayscale volume.")
        return gray

    return raw.astype(np.float32)


def window_CTA(CTA_array, WL=300, WW=1000):
    # Calculate the lower and upper bounds based on WL and WW
    lower_bound = WL - (WW / 2)
    upper_bound = WL + (WW / 2)

    # Clip the CTA array to the calculated bounds
    windowed_CTA_array = np.clip(CTA_array, lower_bound, upper_bound)

    return windowed_CTA_array


def _is_ct_case(nifti_path: str) -> bool:
    filename = os.path.basename(nifti_path).lower()
    return "cta" in filename or "_ct" in filename or filename.startswith("ct")


def _should_apply_ct_window(nifti_path: str) -> bool:
    if CTA_WINDOW_MODE == "on":
        return True
    if CTA_WINDOW_MODE == "off":
        return False
    return _is_ct_case(nifti_path)


def process_nifti_with_yolo(model, nifti_path, output_crop_path, yolo_device):
    """
    Perform YOLO inference on each 2D slice of a NIfTI volume and output the cropped 3D volume.
    The 3D crop is centered around the median bounding box center, and the crop size is based on the median bounding box size.
    """
    # Load the NIfTI image
    nifti_data, nifti_img = load_nifti_file(nifti_path)

    # Squeeze the nifti_data to remove any singleton dimensions
    nifti_data = np.squeeze(nifti_data)

    if _should_apply_ct_window(nifti_path):
        print(
            f"Applying CTA window to input volume (WL={CTA_WINDOW_LEVEL}, WW={CTA_WINDOW_WIDTH})."
        )
        nifti_data = window_CTA(nifti_data, WL=CTA_WINDOW_LEVEL, WW=CTA_WINDOW_WIDTH)

    # Initialize lists to store bounding box centers and sizes for each slice
    bbox_centers = []
    bbox_widths = []
    bbox_heights = []
    slices_with_roi = []

    # Perform inference on each slice of the NIfTI volume
    for slice_idx in range(nifti_data.shape[2]):
        # Get the 2D slice from the NIfTI volume
        slice_data = nifti_data[:, :, slice_idx]

        # Normalize and convert the slice to uint8 format for YOLO
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max > slice_min:
            normalized_slice = (slice_data - slice_min) / (slice_max - slice_min) * 255.0
        else:
            normalized_slice = np.zeros_like(slice_data)

        normalized_slice = normalized_slice.astype(np.uint8)

        # YOLO requires 3 channels, so we need to convert the 2D slice to a 3-channel image
        slice_rgb = cv2.cvtColor(normalized_slice, cv2.COLOR_GRAY2RGB)
        # slice_rgb = np.stack((normalized_slice, normalized_slice, normalized_slice), axis=-1)

        # Perform YOLO inference on the 2D slice
        results = model.predict(
            source=slice_rgb,
            verbose=False,
            device=yolo_device,
            conf=YOLO_CONF,
        )

        # Loop through detected bounding boxes
        for result in results:
            if len(result.boxes.xyxy) > 0:
                # Mark the slice as containing a region of interest (ROI)
                slices_with_roi.append(slice_idx)

            for bbox in result.boxes.xyxy:
                # Extract bounding box coordinates (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Calculate the center of the bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Calculate the width and height of the bounding box
                width = x_max - x_min
                height = y_max - y_min

                # Store the center, width, and height
                bbox_centers.append((center_x, center_y, slice_idx))
                bbox_widths.append(width)
                bbox_heights.append(height)

    # If no bounding boxes were detected, return the original NIfTI data
    if not bbox_centers:
        print("No bounding boxes detected. Using full volume!")
        cropped_volume = nifti_data
        save_nifti_file(cropped_volume, output_crop_path, nifti_path)
        print(f"Cropped volume saved with shape: {cropped_volume.shape}")

        crop_dict = {"size": [cropped_volume.shape[0], cropped_volume.shape[1], cropped_volume.shape[2]],
                     "location": [0, 0, 0]
                     }
        print("Cropped dict:", crop_dict)
        print("Original size:", nifti_data.shape)

        return cropped_volume, crop_dict, nifti_data.shape

    print("Bounding box detected. Cropping volume!")

    # Compute the median center of all bounding boxes
    median_center_x = int(np.median([center[0] for center in bbox_centers]))
    median_center_y = int(np.median([center[1] for center in bbox_centers]))
    median_center_z = int(np.median([center[2] for center in bbox_centers]))

    # Compute the median width and height of all bounding boxes
    median_width = int(np.median(bbox_widths))
    median_height = int(np.median(bbox_heights))

    # Determine the crop size in x and y directions based on the median width and height
    crop_x_size = median_width
    crop_y_size = median_height

    # Determine the crop size in the z direction based on the number of slices with ROIs
    crop_z_size = len(set(slices_with_roi))

    # Define the crop region around the median center
    crop_x_min = max(0, median_center_x - crop_x_size // 2)
    crop_x_max = min(nifti_data.shape[0], median_center_x + crop_x_size // 2)

    crop_y_min = max(0, median_center_y - crop_y_size // 2)
    crop_y_max = min(nifti_data.shape[1], median_center_y + crop_y_size // 2)

    crop_z_min = max(0, median_center_z - crop_z_size // 2)
    crop_z_max = min(nifti_data.shape[2], median_center_z + crop_z_size // 2)

    pad_xy = ROI_PAD_XY
    pad_z = ROI_PAD_Z

    y0 = max(0, crop_y_min - pad_xy)
    y1 = min(nifti_data.shape[1], crop_y_max + pad_xy)
    x0 = max(0, crop_x_min - pad_xy)
    x1 = min(nifti_data.shape[0], crop_x_max + pad_xy)
    z0 = max(0, crop_z_min - pad_z)
    z1 = min(nifti_data.shape[2], crop_z_max + pad_z)

    cropped_volume = nifti_data[y0:y1, x0:x1, z0:z1]

    # Save the cropped 3D volume as a NIfTI file
    save_nifti_file(cropped_volume, output_crop_path, nifti_path)
    print(f"Cropped volume saved with shape: {cropped_volume.shape}")

    crop_dict = {
        "size": [y1 - y0, x1 - x0, z1 - z0],
        "location": [y0, x0, z0],
    }
    print("Cropped dict:", crop_dict)
    print("Original size:", nifti_data.shape)

    return cropped_volume, crop_dict, nifti_data.shape


def segment_nifti(img_file) -> np.array:
    """
    args:
    returns:
        np.array - prediction
    """

    resolved_device = resolve_device(DEVICE_STR)
    yolo_device = resolve_yolo_device_arg(resolved_device)

    ### YOLO DETECTION ###
    yolo_model = YOLO(YOLO_FILE_PATH)  # Load the trained YOLO model

    # Create a temporary folders to store the cropped NIfTI volume
    temp_save_path = "./temp_save"
    shutil.rmtree(temp_save_path, ignore_errors=True)
    os.makedirs(temp_save_path)
    ROI_path = os.path.join(temp_save_path, "ROI_0000.nii.gz")

    _, ROI_dict, original_size = process_nifti_with_yolo(
        yolo_model,
        img_file,
        ROI_path,
        yolo_device=yolo_device,
    )

    print(f"list temp_save_path = {os.listdir(temp_save_path)}")

    #### SEGMENTATION ###
    # Create a temporary folder to store the segmentation
    temp_save_path_seg_best = "./temp_save_seg_best"
    shutil.rmtree(temp_save_path_seg_best, ignore_errors=True)
    os.makedirs(temp_save_path_seg_best)
    temp_save_path_seg_final = "./temp_save_seg_final"
    shutil.rmtree(temp_save_path_seg_final, ignore_errors=True)
    os.makedirs(temp_save_path_seg_final)
    temp_save_path_seg_ensemble = "./temp_save_seg_ensemble"
    shutil.rmtree(temp_save_path_seg_ensemble, ignore_errors=True)
    os.makedirs(temp_save_path_seg_ensemble)

    perform_on_device = str(resolved_device).startswith("cuda")

    predictor_best = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=perform_on_device,
        device=resolved_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor_final = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=perform_on_device,
        device=resolved_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # initializes the network architecture, loads the checkpoint
    predictor_best.initialize_from_trained_model_folder(SEGMODEL_DIR_PATH,
                                                   use_folds=(0, 1, 2, 3, 4),
                                                   checkpoint_name='checkpoint_best.pth',
                                                   # select whether to use the best validation checkpoint or the model from the final epoch.
                                                   )
    predictor_final.initialize_from_trained_model_folder(SEGMODEL_DIR_PATH,
                                                        use_folds=(0, 1, 2, 3, 4),
                                                        checkpoint_name='checkpoint_final.pth',
                                                        # select whether to use the best validation checkpoint or the model from the final epoch.
                                                        )

    predictor_best.predict_from_files(temp_save_path,
                                 temp_save_path_seg_best,
                                 save_probabilities=True, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None)
    predictor_final.predict_from_files(temp_save_path,
                                      temp_save_path_seg_final,
                                      save_probabilities=True, overwrite=False,
                                      num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                      folder_with_segs_from_prev_stage=None)

    print(f"list temp_save_path = {os.listdir(temp_save_path)}")
    print(f"list temp_save_path_seg_best = {os.listdir(temp_save_path_seg_best)}")
    print(f"list temp_save_path_seg_final = {os.listdir(temp_save_path_seg_final)}")

    # Ensamble the segmentations from checkpoint best and final
    ensemble_folders([temp_save_path_seg_best, temp_save_path_seg_final], temp_save_path_seg_ensemble, num_processes=4)

    print(f"list temp_save_path_seg_final = {os.listdir(temp_save_path_seg_ensemble)}")

    # Load the ROI segmentation prediction
    pred_array, _ = load_nifti_file(os.path.join(temp_save_path_seg_ensemble, "ROI.nii.gz"))
    # Make sure the classes are integers
    pred_array = np.round(pred_array).astype(np.uint8)

    # Resize the cropped prediction to the original size
    pred_array_resized = np.zeros(original_size, dtype=np.uint8)
    pred_array_resized[ROI_dict["location"][0]:ROI_dict["location"][0] + ROI_dict["size"][0],
                       ROI_dict["location"][1]:ROI_dict["location"][1] + ROI_dict["size"][1],
                       ROI_dict["location"][2]:ROI_dict["location"][2] + ROI_dict["size"][2]] = pred_array
    pred_array = pred_array_resized

    # Translate the class 13 to class 15
    pred_array[pred_array == 13] = 15

    # The output np.array needs to have the same shape as track modality input
    print("pred_array.shape = ", pred_array.shape)
    print("original shape = ", original_size)

    # Remove the temporary folders
    shutil.rmtree(temp_save_path)
    shutil.rmtree(temp_save_path_seg_best)
    shutil.rmtree(temp_save_path_seg_final)
    shutil.rmtree(temp_save_path_seg_ensemble)

    return pred_array


def resolve_device(device_str: str):
    normalized = device_str.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized.startswith("cuda:"):
        idx = int(normalized.split(":", 1)[1])
        return torch.device("cuda", idx)
    if normalized == "cuda":
        return torch.device("cuda", 0)
    if normalized == "mps":
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")

    # Let torch validate any other custom device strings
    return torch.device(device_str)


def resolve_yolo_device_arg(device: torch.device) -> str:
    if device.type == "cuda":
        return str(0 if device.index is None else device.index)
    if device.type == "mps":
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
