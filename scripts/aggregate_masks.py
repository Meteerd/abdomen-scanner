"""
Phase 2 - Aggregate 2D MedSAM Masks into 3D Labels

Purpose:
- Aggregate 2D MedSAM masks into 3D NIfTI label volumes, aligned with the 3D images.
- Resolves overlaps for multi-class segmentation.

Usage:
    python scripts/aggregate_masks.py --masks2d_root data_processed/medsam_2d_masks --nifti_dir data_processed/nifti_images --out_dir data_processed/nifti_labels_medsam
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import pandas as pd


def load_2d_masks(masks_dir: Path, case_id: str) -> Dict[int, np.ndarray]:
    """
    Load all 2D masks for a case.
    
    Args:
        masks_dir: Root directory containing 2D masks
        case_id: Case identifier
        
    Returns:
        Dictionary mapping slice index to 2D mask array
    """
    case_masks_dir = masks_dir / case_id
    
    if not case_masks_dir.exists():
        return {}
    
    masks = {}
    
    for mask_file in sorted(case_masks_dir.glob("slice_*.png")):
        # Extract slice index from filename (e.g., slice_0042.png)
        slice_idx = int(mask_file.stem.split('_')[1])
        
        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        masks[slice_idx] = mask
    
    return masks


def aggregate_into_3d(nifti_img: nib.Nifti1Image, masks_2d: Dict[int, np.ndarray], case_id: str) -> nib.Nifti1Image:
    """
    Aggregate 2D masks into a 3D label volume.
    
    Args:
        nifti_img: Original NIfTI image (for shape and header)
        masks_2d: Dictionary mapping slice index to 2D mask
        case_id: Case identifier
        
    Returns:
        3D NIfTI label volume
    """
    # Get image shape
    img_data = nifti_img.get_fdata()
    shape = img_data.shape  # (H, W, D)
    
    # Initialize empty label volume
    label_volume = np.zeros(shape, dtype=np.uint8)
    
    if len(masks_2d) == 0:
        print(f"Warning: No masks found for case {case_id}")
        return nib.Nifti1Image(label_volume, nifti_img.affine, nifti_img.header)
    
    # Place each 2D mask in the 3D volume
    for slice_idx, mask_2d in masks_2d.items():
        if slice_idx < 0 or slice_idx >= shape[2]:
            print(f"Warning: Slice index {slice_idx} out of bounds for case {case_id}")
            continue
        
        # Resize mask if needed to match image size
        if mask_2d.shape != (shape[0], shape[1]):
            mask_2d = cv2.resize(mask_2d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize (0 or 1)
        mask_binary = (mask_2d > 127).astype(np.uint8)
        
        # Place in volume (simple assignment - could be improved with class IDs)
        label_volume[:, :, slice_idx] = np.maximum(label_volume[:, :, slice_idx], mask_binary)
    
    # Create NIfTI image with same header as original
    label_img = nib.Nifti1Image(label_volume, nifti_img.affine, nifti_img.header)
    
    return label_img


def aggregate_all_masks(masks2d_root: Path, nifti_dir: Path, out_dir: Path):
    """
    Aggregate all 2D masks into 3D label volumes.
    
    Args:
        masks2d_root: Root directory containing 2D MedSAM masks
        nifti_dir: Directory containing NIfTI image volumes
        out_dir: Output directory for 3D label volumes
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of NIfTI files
    nifti_files = sorted(nifti_dir.glob("*.nii.gz"))
    print(f"Found {len(nifti_files)} NIfTI volumes in {nifti_dir}")
    
    # Process each case
    success_count = 0
    skipped_count = 0
    
    for nifti_path in tqdm(nifti_files, desc="Aggregating masks"):
        case_id = nifti_path.stem.replace('.nii', '')  # Remove .nii.gz
        
        try:
            # Load NIfTI image
            nifti_img = nib.load(str(nifti_path))
            
            # Load 2D masks for this case
            masks_2d = load_2d_masks(masks2d_root, case_id)
            
            if len(masks_2d) == 0:
                skipped_count += 1
                continue
            
            # Aggregate into 3D
            label_img = aggregate_into_3d(nifti_img, masks_2d, case_id)
            
            # Save label volume
            out_path = out_dir / f"{case_id}.nii.gz"
            nib.save(label_img, str(out_path))
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Mask aggregation complete!")
    print(f"Successfully processed: {success_count} cases")
    print(f"Skipped (no masks): {skipped_count} cases")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate 2D MedSAM masks into 3D NIfTI labels")
    parser.add_argument(
        "--masks2d_root",
        type=str,
        default="data_processed/medsam_2d_masks",
        help="Root directory containing 2D MedSAM masks"
    )
    parser.add_argument(
        "--nifti_dir",
        type=str,
        default="data_processed/nifti_images",
        help="Directory containing NIfTI image volumes"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_processed/nifti_labels_medsam",
        help="Output directory for 3D label volumes"
    )
    
    args = parser.parse_args()
    
    masks2d_root = Path(args.masks2d_root)
    nifti_dir = Path(args.nifti_dir)
    out_dir = Path(args.out_dir)
    
    if not masks2d_root.exists():
        raise FileNotFoundError(f"2D masks root not found: {masks2d_root}")
    if not nifti_dir.exists():
        raise FileNotFoundError(f"NIfTI directory not found: {nifti_dir}")
    
    print(f"Starting mask aggregation...")
    print(f"2D masks: {masks2d_root}")
    print(f"Images: {nifti_dir}")
    print(f"Output: {out_dir}\n")
    
    aggregate_all_masks(masks2d_root, nifti_dir, out_dir)


if __name__ == "__main__":
    main()
