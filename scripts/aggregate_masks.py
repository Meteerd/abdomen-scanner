"""
Phase 2 - Aggregate 2D MedSAM Masks into 3D Labels

Purpose:
- Aggregate 2D MedSAM masks (NPY format) into 3D NIfTI label volumes.
- Handles multiple pathologies per slice (multi-class segmentation).
- Maps image_id (filename) → InstanceNumber → slice position in 3D volume.

Input Format:
- 2D masks: data_processed/medsam_2d_masks/case_{case_number}/image_{image_id}_class_{class_id}_mask.npy
- Images: data_processed/nifti_images/*.nii.gz

Output:
- 3D labels: data_processed/nifti_labels_medsam/*.nii.gz (multi-class volumes)

Usage:
    python scripts/aggregate_masks.py \
        --masks2d_root data_processed/medsam_2d_masks \
        --nifti_dir data_processed/nifti_images \
        --dicom_root data/AbdomenDataSet \
        --out_dir data_processed/nifti_labels_medsam
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import pydicom


def build_dicom_manifest(dicom_root: Path) -> Dict[Tuple[int, int, str], Path]:
    """
    Build a manifest mapping (Case Number, Image Id, Dataset) to DICOM file path.
    
    CRITICAL FIX: Training and Competition datasets contain DIFFERENT scans for the same case numbers.
    This function now properly tracks which dataset each Image ID belongs to.
    
    The Image Id in the CSV corresponds to the FILENAME (e.g., "100007"),
    NOT the DICOM InstanceNumber tag!
    
    Returns:
        Dictionary mapping (case_number, image_id_from_filename, dataset) to file path
    """
    print(f"Building DICOM manifest from {dicom_root}...")
    manifest = {}
    
    # Process Training-DataSets
    train_root = dicom_root / 'Training-DataSets'
    if train_root.exists():
        train_cases = [d for d in train_root.iterdir() if d.is_dir()]
        print(f"  Found {len(train_cases)} cases in Training-DataSets")
        
        for case_dir in tqdm(train_cases, desc="Scanning Training-DataSets"):
            try:
                case_number = int(''.join(filter(str.isdigit, case_dir.name)))
            except ValueError:
                continue
            
            for dcm_file in case_dir.glob("*.dcm"):
                try:
                    image_id = int(dcm_file.stem)
                    manifest[(case_number, image_id, 'Training-DataSets')] = dcm_file
                except Exception:
                    continue
    
    # Process Competition-DataSets
    comp_root = dicom_root / 'Competition-DataSets'
    if comp_root.exists():
        comp_cases = [d for d in comp_root.iterdir() if d.is_dir()]
        print(f"  Found {len(comp_cases)} cases in Competition-DataSets")
        
        for case_dir in tqdm(comp_cases, desc="Scanning Competition-DataSets"):
            try:
                case_number = int(''.join(filter(str.isdigit, case_dir.name)))
            except ValueError:
                continue
            
            for dcm_file in case_dir.glob("*.dcm"):
                try:
                    image_id = int(dcm_file.stem)
                    manifest[(case_number, image_id, 'Competition-DataSets')] = dcm_file
                except Exception:
                    continue
    
    print(f"Found {len(manifest)} DICOM files total")
    print(f"  Training: {len([k for k in manifest.keys() if k[2] == 'Training-DataSets'])} files")
    print(f"  Competition: {len([k for k in manifest.keys() if k[2] == 'Competition-DataSets'])} files")
    
    return manifest


def build_instance_to_slice_map(dicom_root: Path, case_number: int, dataset: str) -> Dict[int, int]:
    """
    Build a mapping from image_id (filename) to slice index in 3D volume.
    
    Args:
        dicom_root: Root directory containing DICOM files
        case_number: Case number (e.g., 20001)
        dataset: 'Training-DataSets' or 'Competition-DataSets'
        
    Returns:
        Dictionary mapping image_id (filename) to slice_index (position in sorted volume)
    """
    case_dir = dicom_root / dataset / str(case_number)
    
    if not case_dir.exists():
        return {}
    
    # Collect all DICOM files with their InstanceNumber
    dicom_files = []
    for dcm_file in case_dir.glob("*.dcm"):
        try:
            dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            image_id = int(dcm_file.stem)  # Filename (e.g., 100007)
            instance_number = int(dcm.InstanceNumber)  # DICOM tag
            dicom_files.append((instance_number, image_id))
        except Exception:
            continue
    
    # Sort by InstanceNumber to get correct slice order
    dicom_files.sort(key=lambda x: x[0])
    
    # Create mapping: image_id → slice_index
    instance_to_slice = {}
    for slice_idx, (instance_num, image_id) in enumerate(dicom_files):
        instance_to_slice[image_id] = slice_idx
    
    return instance_to_slice


def aggregate_case_masks(
    case_number: int,
    masks2d_dir: Path,
    nifti_path: Path,
    dicom_root: Path,
    dataset: str
) -> np.ndarray:
    """
    Aggregate all 2D masks for a single case into a 3D label volume.
    
    Args:
        case_number: Case number
        masks2d_dir: Directory containing 2D masks for this case
        nifti_path: Path to NIfTI image (for shape)
        dicom_root: Root directory containing DICOM files
        dataset: 'Training-DataSets' or 'Competition-DataSets'
        
    Returns:
        3D label volume with class IDs
    """
    # Load NIfTI image to get shape
    nifti_img = nib.load(str(nifti_path))
    img_data = nifti_img.get_fdata()
    shape = img_data.shape  # (H, W, D)
    
    # Initialize empty label volume
    label_volume = np.zeros(shape, dtype=np.uint8)
    
    # Build image_id → slice_index mapping
    instance_to_slice = build_instance_to_slice_map(dicom_root, case_number, dataset)
    
    if len(instance_to_slice) == 0:
        print(f"  Warning: No DICOM mapping found for case {case_number}, dataset {dataset}")
        return label_volume
    
    # Process all mask files from the dataset-specific directory
    # masks2d_dir format: TRAIN_case_20050 or COMP_case_20050
    if not masks2d_dir.exists():
        return label_volume
    
    mask_files = list(masks2d_dir.glob("image_*_class_*_mask.npy"))
    
    for mask_file in mask_files:
        try:
            # Parse filename: image_{image_id}_class_{class_id}_mask.npy
            parts = mask_file.stem.split('_')
            image_id = int(parts[1])  # Extract 100007 from image_100007_class_1_mask
            class_id = int(parts[3])  # Extract 1 from image_100007_class_1_mask
            
            # Get slice index
            if image_id not in instance_to_slice:
                continue
            
            slice_idx = instance_to_slice[image_id]
            
            if slice_idx < 0 or slice_idx >= shape[2]:
                continue
            
            # Load 2D mask
            mask_2d = np.load(str(mask_file))
            
            # Resize if needed
            if mask_2d.shape != (shape[0], shape[1]):
                mask_2d = cv2.resize(mask_2d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Binarize mask
            mask_binary = (mask_2d > 127).astype(np.uint8)
            
            # Paint class_id onto the volume (use maximum to handle overlaps)
            current_slice = label_volume[:, :, slice_idx]
            label_volume[:, :, slice_idx] = np.maximum(current_slice, mask_binary * class_id)
            
        except Exception as e:
            continue
    
    return label_volume


def aggregate_all_masks(masks2d_root: Path, nifti_dir: Path, dicom_root: Path, out_dir: Path):
    """
    Aggregate all 2D masks into 3D label volumes.
    
    CRITICAL FIX: Training and Competition are separate datasets, process them independently.
    
    Args:
        masks2d_root: Root directory containing 2D MedSAM masks
        nifti_dir: Directory containing NIfTI image volumes
        dicom_root: Root directory containing DICOM files
        out_dir: Output directory for 3D label volumes
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of cases with masks (now includes TRAIN_ and COMP_ prefixes)
    case_dirs = [d for d in masks2d_root.iterdir() if d.is_dir() and ('case_' in d.name)]
    print(f"Found {len(case_dirs)} cases with 2D masks")
    
    train_cases = len([d for d in case_dirs if d.name.startswith('TRAIN_')])
    comp_cases = len([d for d in case_dirs if d.name.startswith('COMP_')])
    print(f"  Training cases: {train_cases}")
    print(f"  Competition cases: {comp_cases}")
    
    # Build DICOM manifest for dataset lookup
    print("\nBuilding DICOM manifest...")
    dicom_manifest = build_dicom_manifest(dicom_root)
    
    # Process each case
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for case_dir in tqdm(case_dirs, desc="Aggregating masks"):
        try:
            # Extract dataset and case number from directory name
            # Format: TRAIN_case_20050 or COMP_case_20050
            dir_name = case_dir.name
            
            if dir_name.startswith('TRAIN_case_'):
                dataset = 'Training-DataSets'
                case_number = int(dir_name.replace('TRAIN_case_', ''))
                nifti_prefix = 'TRAIN'
            elif dir_name.startswith('COMP_case_'):
                dataset = 'Competition-DataSets'
                case_number = int(dir_name.replace('COMP_case_', ''))
                nifti_prefix = 'COMP'
            else:
                # Old format without dataset prefix - skip or handle gracefully
                failed_count += 1
                continue
            
            # Find corresponding NIfTI file with dataset prefix
            nifti_files = list(nifti_dir.glob(f"{nifti_prefix}_{case_number}.nii.gz"))
            
            if len(nifti_files) == 0:
                # No matching NIfTI file
                skipped_count += 1
                continue
            
            nifti_path = nifti_files[0]
            
            # Check if output already exists
            out_path = out_dir / nifti_path.name
            if out_path.exists():
                skipped_count += 1
                continue
            
            # Aggregate masks for this case (pass the case_dir which has dataset prefix)
            label_volume = aggregate_case_masks(
                case_number,
                case_dir,  # This is already the dataset-specific directory (TRAIN_case_X or COMP_case_X)
                nifti_path,
                dicom_root,
                dataset
            )
            
            # Load NIfTI image to get header
            nifti_img = nib.load(str(nifti_path))
            
            # Create label NIfTI with same header
            label_img = nib.Nifti1Image(label_volume, nifti_img.affine, nifti_img.header)
            
            # Save
            nib.save(label_img, str(out_path))
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing case {case_dir.name}: {e}")
            failed_count += 1
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Mask aggregation complete!")
    print(f"Successfully processed: {success_count} cases")
    print(f"Skipped (already exist or no NIfTI): {skipped_count} cases")
    print(f"Failed: {failed_count} cases")
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
        "--dicom_root",
        type=str,
        required=True,
        help="Root directory containing DICOM files (for InstanceNumber mapping)"
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
    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)
    
    if not masks2d_root.exists():
        raise FileNotFoundError(f"2D masks root not found: {masks2d_root}")
    if not nifti_dir.exists():
        raise FileNotFoundError(f"NIfTI directory not found: {nifti_dir}")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    
    print(f"Starting mask aggregation...")
    print(f"2D masks: {masks2d_root}")
    print(f"Images: {nifti_dir}")
    print(f"DICOM root: {dicom_root}")
    print(f"Output: {out_dir}\n")
    
    aggregate_all_masks(masks2d_root, nifti_dir, dicom_root, out_dir)


if __name__ == "__main__":
    main()
