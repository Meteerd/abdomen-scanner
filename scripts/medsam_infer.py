"""
Phase 2 - MedSAM Inference for 2D Mask Generation

Purpose:
- For each bbox annotation in CSV, run MedSAM on the 2D DICOM/NIfTI slice to create a binary mask.
- Supports parallel execution across multiple GPUs.

Usage:
    # Single GPU
    python scripts/medsam_infer.py --master_csv data_raw/annotations/TRAININGDATA.csv --dicom_root data_raw/dicom_files --out_root data_processed/medsam_2d_masks --medsam_ckpt models/medsam_vit_b.pth
    
    # Multi-GPU (split by GPU in SLURM script)
    CUDA_VISIBLE_DEVICES=0 python scripts/medsam_infer.py --master_csv data_raw/annotations/TRAININGDATA.csv --gpu_idx 0 --num_gpus 2 ...
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional
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


def load_slice_from_dicom(dicom_root: Path, case_id: str, slice_idx: int) -> Optional[np.ndarray]:
    """
    Load a specific slice from DICOM files.
    
    Args:
        dicom_root: Root directory containing DICOM files
        case_id: Case identifier
        slice_idx: Slice index
        
    Returns:
        2D numpy array or None if not found
    """
    # This is a placeholder - actual implementation depends on your DICOM structure
    # You may need to adjust based on how DICOMs are organized
    
    case_dir = dicom_root / case_id
    if not case_dir.exists():
        return None
    
    # Find the DICOM file for this slice
    dicom_files = sorted(case_dir.glob("*.dcm"))
    
    if slice_idx >= len(dicom_files):
        return None
    
    # Load DICOM
    dcm = pydicom.dcmread(str(dicom_files[slice_idx]))
    image = dcm.pixel_array.astype(np.float32)
    
    return image


def process_annotations(master_csv: Path, dicom_root: Path, out_root: Path, medsam_ckpt: Path, gpu_idx: int = 0, num_gpus: int = 1, device: str = 'cuda'):
    """
    Process all annotations and generate MedSAM masks.
    
    Args:
        master_csv: Path to master CSV
        dicom_root: Root directory with DICOM files
        out_root: Output root for masks
        medsam_ckpt: Path to MedSAM checkpoint
        gpu_idx: GPU index for multi-GPU processing
        num_gpus: Total number of GPUs
        device: Device string
    """
    # Create output directory
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {master_csv}...")
    df = pd.read_csv(master_csv)
    print(f"Found {len(df)} annotations")
    
    # Split annotations across GPUs
    if num_gpus > 1:
        annotations_per_gpu = len(df) // num_gpus
        start_idx = gpu_idx * annotations_per_gpu
        end_idx = start_idx + annotations_per_gpu if gpu_idx < num_gpus - 1 else len(df)
        df = df.iloc[start_idx:end_idx]
        print(f"GPU {gpu_idx}: Processing annotations {start_idx} to {end_idx} ({len(df)} total)")
    
    # Load MedSAM model
    print(f"Loading MedSAM model from {medsam_ckpt}...")
    model, transform = load_medsam_model(medsam_ckpt, device)
    print(f"Model loaded on {device}")
    
    # Process each annotation
    success_count = 0
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu_idx} - MedSAM inference"):
        try:
            case_id = row['case_id']
            slice_idx = int(row['slice_idx'])
            x_min = int(row['x_min'])
            y_min = int(row['y_min'])
            x_max = int(row['x_max'])
            y_max = int(row['y_max'])
            
            # Load slice
            image = load_slice_from_dicom(dicom_root, case_id, slice_idx)
            
            if image is None:
                failed_count += 1
                continue
            
            # Run MedSAM inference
            mask = run_medsam_inference(model, transform, image, (x_min, y_min, x_max, y_max), device)
            
            # Save mask
            case_out_dir = out_root / case_id
            case_out_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path = case_out_dir / f"slice_{slice_idx:04d}.png"
            cv2.imwrite(str(mask_path), mask)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            failed_count += 1
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GPU {gpu_idx} - MedSAM inference complete!")
    print(f"Successfully processed: {success_count}/{len(df)} annotations")
    print(f"Failed: {failed_count} annotations")
    print(f"Output directory: {out_root}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="MedSAM inference for 2D mask generation")
    parser.add_argument(
        "--master_csv",
        type=str,
        required=True,
        help="Path to master CSV annotation file"
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
    
    master_csv = Path(args.master_csv)
    dicom_root = Path(args.dicom_root)
    out_root = Path(args.out_root)
    medsam_ckpt = Path(args.medsam_ckpt)
    
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    if not medsam_ckpt.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {medsam_ckpt}")
    
    # Set device
    device = f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu'
    
    print(f"Starting MedSAM inference...")
    print(f"Annotations: {master_csv}")
    print(f"DICOM root: {dicom_root}")
    print(f"MedSAM checkpoint: {medsam_ckpt}")
    print(f"Output: {out_root}")
    print(f"Device: {device}\n")
    
    process_annotations(master_csv, dicom_root, out_root, medsam_ckpt, args.gpu_idx, args.num_gpus, device)


if __name__ == "__main__":
    main()
