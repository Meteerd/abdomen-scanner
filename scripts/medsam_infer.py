"""
Phase 2 - MedSAM Inference for 2D Mask Generation

Purpose:
- For each bbox annotation in CSV files, run MedSAM on the 2D DICOM/NIfTI slice to create a binary mask.
- Merges TRAININGDATA.csv and COMPETITIONDATA.csv to process all 42,450 annotations.
- Supports parallel execution across multiple GPUs.

Usage:
    # Single GPU
    python scripts/medsam_infer.py --csv_dir data_raw/annotations --dicom_root data_raw/dicom_files --out_root data_processed/medsam_2d_masks --medsam_ckpt models/medsam_vit_b.pth
    
    # Multi-GPU (split by GPU in SLURM script)
    CUDA_VISIBLE_DEVICES=0 python scripts/medsam_infer.py --csv_dir data_raw/annotations --gpu_idx 0 --num_gpus 2 ...
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import pydicom
import nibabel as nib
from PIL import Image

# MedSAM imports (requires medsam package installed)
try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    print("Warning: segment_anything not installed. Install with:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
    sam_model_registry = None


def load_medsam_model(checkpoint_path: Path, device: str = 'cuda') -> Tuple[object, object]:
    """
    Load MedSAM model from checkpoint.
    
    Args:
        checkpoint_path: Path to medsam_vit_b.pth
        device: Device to load model on
        
    Returns:
        Tuple of (model, transform)
    """
    if sam_model_registry is None:
        raise ImportError("segment_anything package not installed")
    
    # Load model (MedSAM uses vit_b architecture)
    model_type = "vit_b"
    sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam_model = sam_model.to(device)
    sam_model.eval()
    
    # Create transform
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    
    return sam_model, transform


def preprocess_image(image: np.ndarray, transform: object, device: str) -> torch.Tensor:
    """
    Preprocess image for MedSAM inference.
    
    Args:
        image: 2D numpy array (H, W) or (H, W, 3)
        transform: MedSAM transform
        device: Device to put tensor on
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to 3-channel if grayscale
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Ensure uint8
    if image.dtype != np.uint8:
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply transform
    image_transformed = transform.apply_image(image)
    image_tensor = torch.as_tensor(image_transformed, device=device)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]  # (1, 3, H, W)
    
    return image_tensor


def get_bbox_prompt(bbox: Tuple[int, int, int, int], original_size: Tuple[int, int], transform: object) -> np.ndarray:
    """
    Convert bounding box to MedSAM prompt format.
    
    Args:
        bbox: (x_min, y_min, x_max, y_max) in original image coordinates
        original_size: (H, W) of original image
        transform: MedSAM transform
        
    Returns:
        Bbox prompt array in transformed coordinates
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Transform bbox coordinates
    bbox_array = np.array([[x_min, y_min], [x_max, y_max]])
    bbox_transformed = transform.apply_boxes(bbox_array, original_size)
    
    return bbox_transformed


def run_medsam_inference(model: object, transform: object, image: np.ndarray, bbox: Tuple[int, int, int, int], device: str) -> np.ndarray:
    """
    Run MedSAM inference on a single image with bbox prompt.
    
    Args:
        model: MedSAM model
        transform: MedSAM transform
        image: 2D image array (H, W)
        bbox: (x_min, y_min, x_max, y_max) bounding box
        device: Device
        
    Returns:
        Binary mask (H, W) as uint8 (0 or 255)
    """
    original_size = image.shape[:2]  # (H, W)
    
    # Preprocess image
    image_tensor = preprocess_image(image, transform, device)
    
    # Get bbox prompt
    bbox_prompt = get_bbox_prompt(bbox, original_size, transform)
    bbox_tensor = torch.as_tensor(bbox_prompt, dtype=torch.float, device=device)[None, :]  # (1, 2, 2)
    
    # Run inference
    with torch.no_grad():
        # Encode image
        image_embedding = model.image_encoder(image_tensor)
        
        # Predict mask
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=bbox_tensor,
            masks=None,
        )
        
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upscale mask to original size
        masks = model.postprocess_masks(
            low_res_masks,
            input_size=image_tensor.shape[-2:],
            original_size=original_size,
        )
        
        # Convert to binary mask
        mask = masks[0, 0].cpu().numpy()  # (H, W)
        mask_binary = (mask > 0).astype(np.uint8) * 255
    
    return mask_binary


def parse_bbox_data(data_str: str) -> Tuple[int, int, int, int]:
    """
    Parse bounding box coordinates from Data column.
    
    Format: "xmin,ymin-xmax,ymax"
    Returns: (xmin, ymin, xmax, ymax)
    """
    min_str, max_str = data_str.split('-')
    xmin, ymin = map(int, min_str.split(','))
    xmax, ymax = map(int, max_str.split(','))
    return xmin, ymin, xmax, ymax


def build_dicom_manifest(dicom_root: Path) -> Dict[Tuple[int, int], Path]:
    """
    Build a manifest mapping (Case Number, Image Id) to DICOM file path.
    
    The Image Id corresponds to the DICOM InstanceNumber tag,
    NOT the filename (e.g., filename "100007.dcm" may have InstanceNumber=77).
    
    Returns:
        Dictionary mapping (case_number, image_id) to file path
    """
    print(f"Building DICOM manifest from {dicom_root}...")
    manifest = {}
    
    # Find all case directories
    case_dirs = [d for d in dicom_root.iterdir() if d.is_dir()]
    
    for case_dir in tqdm(case_dirs, desc="Scanning DICOM directories"):
        # Extract case number from directory name
        try:
            case_number = int(''.join(filter(str.isdigit, case_dir.name)))
        except ValueError:
            continue
        
        # Scan all DICOM files in this case
        for dcm_file in case_dir.glob("*.dcm"):
            try:
                # Read DICOM header only (fast - doesn't load pixel data)
                dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                
                # Use InstanceNumber tag as Image Id (what CSV references)
                image_id = int(dcm.InstanceNumber)
                
                # Store mapping
                manifest[(case_number, image_id)] = dcm_file
                
            except Exception as e:
                continue
    
    print(f"Found {len(manifest)} DICOM files across {len(case_dirs)} cases")
    return manifest


def apply_ct_window(pixel_array: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    """
    Apply CT windowing (soft tissue window) and convert to 8-bit image.
    
    Args:
        pixel_array: Raw DICOM pixel array (Hounsfield units)
        window_center: Center of the window (default: 40 HU for soft tissue)
        window_width: Width of the window (default: 400 HU)
        
    Returns:
        8-bit grayscale image (0-255) as 2D array
    """
    # Ensure 2D array (squeeze any extra dimensions)
    if pixel_array.ndim > 2:
        pixel_array = pixel_array.squeeze()
    
    # If still not 2D, take first slice
    if pixel_array.ndim > 2:
        pixel_array = pixel_array[0]
    
    # Convert to float to avoid overflow
    pixel_array = pixel_array.astype(np.float32)
    
    img_min = float(window_center - window_width // 2)
    img_max = float(window_center + window_width // 2)
    
    # Clip and normalize to 0-255
    windowed = np.clip(pixel_array, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min) * 255.0)
    windowed = windowed.astype(np.uint8)
    
    return windowed


def process_annotations(
    csv_dir: Path,
    dicom_root: Path,
    out_root: Path,
    medsam_ckpt: Path,
    gpu_idx: int,
    num_gpus: int,
    device: str
):
    """
    Process all annotations and generate MedSAM masks.
    
    Args:
        csv_dir: Directory containing TRAININGDATA.csv and COMPETITIONDATA.csv
        dicom_root: Root directory containing DICOM files
        out_root: Output root directory
        medsam_ckpt: Path to MedSAM checkpoint
        gpu_idx: Index of current GPU
        num_gpus: Total number of GPUs
        device: Device string
    """
    # Create output directory
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Load and merge annotations
    train_csv = csv_dir / "TRAININGDATA.csv"
    comp_csv = csv_dir / "COMPETITIONDATA.csv"
    
    print(f"Loading annotations from:")
    print(f"  - {train_csv}")
    print(f"  - {comp_csv}")
    
    train_df = pd.read_csv(train_csv)
    comp_df = pd.read_csv(comp_csv)
    df = pd.concat([train_df, comp_df], ignore_index=True)
    
    print(f"Merged {len(train_df)} training + {len(comp_df)} competition = {len(df)} total annotations")
    
    # Filter to bounding boxes only
    df = df[df['Type'] == 'Bounding Box'].copy()
    print(f"Filtered to {len(df)} bounding box annotations")
    
    # Show class distribution
    print("\nClass distribution:")
    print(df['Class'].value_counts())
    print()
    
    # Split annotations across GPUs
    if num_gpus > 1:
        annotations_per_gpu = len(df) // num_gpus
        start_idx = gpu_idx * annotations_per_gpu
        end_idx = start_idx + annotations_per_gpu if gpu_idx < num_gpus - 1 else len(df)
        df = df.iloc[start_idx:end_idx]
        print(f"GPU {gpu_idx}: Processing annotations {start_idx} to {end_idx} ({len(df)} total)")
    
    # Build DICOM manifest (FIX BUG #2)
    print("\nBuilding DICOM manifest...")
    dicom_manifest = build_dicom_manifest(dicom_root)
    
    # Load MedSAM model
    print(f"\nLoading MedSAM model from {medsam_ckpt}...")
    model, transform = load_medsam_model(medsam_ckpt, device)
    print(f"Model loaded on {device}")
    
    # Process each annotation
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu_idx} - MedSAM inference"):
        try:
            # FIX BUG #1: Use correct CSV column names
            case_number = int(row['Case Number'])
            image_id = int(row['Image Id'])
            
            # Parse bounding box from Data column
            data_str = row['Data']
            x_min, y_min, x_max, y_max = parse_bbox_data(data_str)
            
            # Check if mask already exists
            case_out_dir = out_root / f"case_{case_number}"
            mask_path = case_out_dir / f"image_{image_id}_mask.npy"
            
            if mask_path.exists():
                skipped_count += 1
                continue
            
            # FIX BUG #2: Load DICOM using manifest
            dicom_key = (case_number, image_id)
            if dicom_key not in dicom_manifest:
                failed_count += 1
                continue
            
            dicom_path = dicom_manifest[dicom_key]
            
            # Read DICOM file
            dcm = pydicom.dcmread(str(dicom_path))
            pixel_array = dcm.pixel_array
            
            # Convert to Hounsfield Units if needed
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Apply CT windowing to convert to 8-bit
            image = apply_ct_window(pixel_array)
            
            # Run MedSAM inference
            mask = run_medsam_inference(model, transform, image, (x_min, y_min, x_max, y_max), device)
            
            # Save mask as numpy array
            case_out_dir.mkdir(parents=True, exist_ok=True)
            np.save(mask_path, mask)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            failed_count += 1
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GPU {gpu_idx} - MedSAM inference complete!")
    print(f"Successfully processed: {success_count} new masks")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count} annotations")
    print(f"Total annotations: {len(df)}")
    print(f"Output directory: {out_root}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='MedSAM inference for 2D mask generation')
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory containing TRAININGDATA.csv and COMPETITIONDATA.csv"
    )
    parser.add_argument(
        "--dicom_root",
        type=str,
        required=True,
        help="Root directory containing DICOM files"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data_processed/medsam_2d_masks",
        help="Output root directory for masks"
    )
    parser.add_argument(
        "--medsam_ckpt",
        type=str,
        required=True,
        help="Path to MedSAM checkpoint (medsam_vit_b.pth)"
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help="GPU index for multi-GPU processing"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Total number of GPUs"
    )
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    dicom_root = Path(args.dicom_root)
    out_root = Path(args.out_root)
    medsam_ckpt = Path(args.medsam_ckpt)
    
    # Check for CSV files
    train_csv = csv_dir / "TRAININGDATA.csv"
    comp_csv = csv_dir / "COMPETITIONDATA.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(f"TRAININGDATA.csv not found: {train_csv}")
    if not comp_csv.exists():
        raise FileNotFoundError(f"COMPETITIONDATA.csv not found: {comp_csv}")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    if not medsam_ckpt.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {medsam_ckpt}")
    
    # FIX BUG #3: Set device to cuda:0 (CUDA_VISIBLE_DEVICES handles GPU selection)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Starting MedSAM inference...")
    print(f"Merging annotations from:")
    print(f"  - {train_csv}")
    print(f"  - {comp_csv}")
    print(f"DICOM root: {dicom_root}")
    print(f"MedSAM checkpoint: {medsam_ckpt}")
    print(f"Output: {out_root}")
    print(f"Device: {device}\n")
    
    process_annotations(csv_dir, dicom_root, out_root, medsam_ckpt, args.gpu_idx, args.num_gpus, device)


if __name__ == "__main__":
    main()
