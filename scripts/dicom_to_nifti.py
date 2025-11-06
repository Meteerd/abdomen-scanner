"""
Phase 1 - Step 1: DICOM to NIfTI Conversion

CRITICAL FIX: Training and Competition datasets contain COMPLETELY DIFFERENT CT scans
for the same case numbers (20001-20359). This script now processes BOTH datasets
separately and prefixes output files with TRAIN_ or COMP_ to avoid confusion.

Purpose:
- Process Training-DataSets/ and Competition-DataSets/ independently
- Create unique NIfTI files: TRAIN_20001.nii.gz, COMP_20001.nii.gz, etc.
- Sort DICOM slices by InstanceNumber, stack into 3D volumes
- Preserve DICOM spacing/origin/direction in NIfTI headers

Expected Output:
- 736 files from Training-DataSets (TRAIN_20001.nii.gz through TRAIN_20736.nii.gz)
- 359 files from Competition-DataSets (COMP_20001.nii.gz through COMP_20359.nii.gz)
- Total: 1,095 unique CT scans (not 736!)

Usage:
    python scripts/dicom_to_nifti.py --dicom_root data/AbdomenDataSet --out_dir data_processed/nifti_images
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pydicom
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def load_dicom_series(dicom_files: List[Path]) -> sitk.Image:
    """
    Load a series of DICOM files and return a 3D SimpleITK Image.
    
    Args:
        dicom_files: List of DICOM file paths (already sorted)
        
    Returns:
        3D SimpleITK Image with proper spacing/origin/direction
    """
    # Read series using SimpleITK's robust reader
    reader = sitk.ImageSeriesReader()
    dicom_names = [str(f) for f in dicom_files]
    reader.SetFileNames(dicom_names)
    
    # Load image
    image = reader.Execute()
    
    return image


def sort_dicom_slices(dicom_files: List[Path]) -> List[Path]:
    """
    Sort DICOM slices by InstanceNumber or ImagePositionPatient Z-coordinate.
    
    Args:
        dicom_files: List of DICOM file paths
        
    Returns:
        Sorted list of DICOM file paths
    """
    slice_info = []
    
    for dcm_path in dicom_files:
        try:
            dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            
            # Try InstanceNumber first (most reliable)
            if hasattr(dcm, 'InstanceNumber'):
                sort_key = int(dcm.InstanceNumber)
            # Fallback to ImagePositionPatient Z-coordinate
            elif hasattr(dcm, 'ImagePositionPatient'):
                sort_key = float(dcm.ImagePositionPatient[2])
            else:
                # Last resort: slice location
                sort_key = float(dcm.get('SliceLocation', 0))
            
            slice_info.append((sort_key, dcm_path))
        except Exception as e:
            print(f"Warning: Could not read {dcm_path}: {e}")
            continue
    
    # Sort by the extracted key
    slice_info.sort(key=lambda x: x[0])
    
    return [path for _, path in slice_info]


def group_dicom_by_case(dicom_root: Path) -> Dict[str, List[Path]]:
    """
    Group DICOM files by case, processing Training and Competition datasets separately.
    
    CRITICAL FIX: Training-DataSets/20050 and Competition-DataSets/20050 contain
    COMPLETELY DIFFERENT CT scans (different Image IDs, different slices, different anatomy).
    They share the same SeriesInstanceUID but are NOT the same patient scan!
    
    Solution: Create unique case IDs with dataset prefix:
    - Training-DataSets/20050 → "TRAIN_20050" → TRAIN_20050.nii.gz
    - Competition-DataSets/20050 → "COMP_20050" → COMP_20050.nii.gz
    
    Args:
        dicom_root: Root directory (should contain Training-DataSets/ and Competition-DataSets/)
        
    Returns:
        Dictionary mapping unique case IDs (e.g., "TRAIN_20050", "COMP_20050") to DICOM file lists
        
    Expected Output:
        - 736 TRAIN_ entries (cases 20001-20736, though not all numbers exist)
        - 359 COMP_ entries (cases 20001-20359, overlapping with Training case numbers)
        - Total: 1,095 unique case IDs
    """
    cases = defaultdict(list)
    
    # Process Training-DataSets (736 cases)
    train_root = dicom_root / 'Training-DataSets'
    if train_root.exists():
        print(f"Processing Training-DataSets...")
        for case_dir in train_root.iterdir():
            if not case_dir.is_dir():
                continue
            case_number = case_dir.name  # e.g., "20050"
            # Create unique case ID with TRAIN_ prefix
            case_id = f"TRAIN_{case_number}"
            
            # Collect all DICOM files in this case directory
            for dcm_file in case_dir.glob("*.dcm"):
                cases[case_id].append(dcm_file)
        
        train_count = len([k for k in cases.keys() if k.startswith('TRAIN_')])
        print(f"  Found {train_count} cases in Training-DataSets")
    
    # Process Competition-DataSets (359 cases with SAME case numbers as Training)
    comp_root = dicom_root / 'Competition-DataSets'
    if comp_root.exists():
        print(f"Processing Competition-DataSets...")
        for case_dir in comp_root.iterdir():
            if not case_dir.is_dir():
                continue
            case_number = case_dir.name  # e.g., "20050" (same as Training!)
            # Create unique case ID with COMP_ prefix to distinguish from Training
            case_id = f"COMP_{case_number}"
            
            # Collect all DICOM files in this case directory
            for dcm_file in case_dir.glob("*.dcm"):
                cases[case_id].append(dcm_file)
        
        comp_count = len([k for k in cases.keys() if k.startswith('COMP_')])
        print(f"  Found {comp_count} cases in Competition-DataSets")
        print(f"  NOTE: These are DIFFERENT scans than Training, despite same case numbers!")
    
    # Fallback: If neither subdirectory exists, try old behavior (shouldn't happen)
    if not train_root.exists() and not comp_root.exists():
        print("WARNING: Neither Training-DataSets nor Competition-DataSets found!")
        print("Falling back to scanning all .dcm files...")
        for dcm_file in dicom_root.rglob("*.dcm"):
            try:
                dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                
                # Use SeriesInstanceUID as case ID (old behavior)
                if hasattr(dcm, 'SeriesInstanceUID'):
                    case_id = dcm.SeriesInstanceUID
                else:
                    case_id = dcm_file.parent.name
                
                cases[case_id].append(dcm_file)
            except Exception as e:
                print(f"Warning: Could not read {dcm_file}: {e}")
                continue
    
    return dict(cases)


def convert_dicom_to_nifti(dicom_root: Path, out_dir: Path):
    """
    Convert all DICOM series to NIfTI format, processing Training and Competition separately.
    
    This function will create 1,095 NIfTI files:
    - 736 from Training-DataSets (TRAIN_20001.nii.gz, TRAIN_20002.nii.gz, ...)
    - 359 from Competition-DataSets (COMP_20001.nii.gz, COMP_20002.nii.gz, ...)
    
    Args:
        dicom_root: Root directory (should contain Training-DataSets/ and Competition-DataSets/)
        out_dir: Output directory for NIfTI files (e.g., data_processed/nifti_images/)
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group DICOM files by case (with TRAIN_/COMP_ prefixes)
    print(f"Scanning DICOM files in {dicom_root}...")
    cases = group_dicom_by_case(dicom_root)
    
    train_cases = [k for k in cases.keys() if k.startswith('TRAIN_')]
    comp_cases = [k for k in cases.keys() if k.startswith('COMP_')]
    
    print(f"\nFound {len(cases)} total unique case IDs:")
    print(f"  Training-DataSets: {len(train_cases)} cases")
    print(f"  Competition-DataSets: {len(comp_cases)} cases")
    print(f"  Expected total: ~1,095 cases (736 + 359)")
    
    if len(cases) < 1000:
        print(f"  WARNING: Expected ~1,095 cases, but found only {len(cases)}")
        print(f"  Check that both Training-DataSets/ and Competition-DataSets/ exist!")
    
    # Process each case
    success_count = 0
    skipped_count = 0
    failed_cases = []
    
    for case_id, dicom_files in tqdm(cases.items(), desc="Converting to NIfTI"):
        try:
            # Filename already has TRAIN_ or COMP_ prefix from group_dicom_by_case()
            # e.g., case_id = "TRAIN_20050" → filename = "TRAIN_20050.nii.gz"
            safe_case_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in case_id)
            out_path = out_dir / f"{safe_case_id}.nii.gz"
            
            # Skip if already exists (allows resuming interrupted jobs)
            if out_path.exists():
                skipped_count += 1
                continue
            
            # Sort slices by InstanceNumber (most reliable) or ImagePosition Z-coordinate
            sorted_files = sort_dicom_slices(dicom_files)
            
            # Sanity check: need at least 5 slices for a valid 3D volume
            if len(sorted_files) < 5:
                print(f"\nWarning: Case {case_id} has only {len(sorted_files)} slices, skipping")
                failed_cases.append(case_id)
                continue
            
            # Load DICOM series as 3D SimpleITK image (preserves spacing/orientation)
            image = load_dicom_series(sorted_files)
            
            # Save as compressed NIfTI
            sitk.WriteImage(image, str(out_path), useCompression=True)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            failed_cases.append(case_id)
            continue
    
    # Summary report
    print(f"\n{'='*70}")
    print(f"DICOM to NIfTI Conversion Complete!")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count} new cases")
    print(f"Skipped (already exist): {skipped_count} cases")
    print(f"Failed: {len(failed_cases)} cases")
    print(f"Total processed: {len(cases)} cases")
    print(f"\nOutput directory: {out_dir}")
    print(f"\nExpected output:")
    print(f"  - {len([k for k in cases.keys() if k.startswith('TRAIN_')])} TRAIN_*.nii.gz files")
    print(f"  - {len([k for k in cases.keys() if k.startswith('COMP_')])} COMP_*.nii.gz files")
    
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for case_id in failed_cases[:10]:  # Show first 10
            print(f"  - {case_id}")
        if len(failed_cases) > 10:
            print(f"  ... and {len(failed_cases) - 10} more")
    
    # Critical validation
    if success_count + skipped_count < 1000:
        print(f"\n⚠️  WARNING: Expected ~1,095 total cases, but only have {success_count + skipped_count}")
        print(f"    Check that both Training-DataSets/ and Competition-DataSets/ were processed!")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to NIfTI format (Training and Competition datasets separately)"
    )
    parser.add_argument(
        "--dicom_root",
        type=str,
        default="data/AbdomenDataSet",
        help="Root directory containing Training-DataSets/ and Competition-DataSets/ subdirectories"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_processed/nifti_images",
        help="Output directory for NIfTI files (will create TRAIN_*.nii.gz and COMP_*.nii.gz)"
    )
    
    args = parser.parse_args()
    
    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)
    
    # Validate input directory exists
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root directory not found: {dicom_root}")
    
    # Check for expected subdirectories
    train_root = dicom_root / 'Training-DataSets'
    comp_root = dicom_root / 'Competition-DataSets'
    
    if not train_root.exists():
        print(f"WARNING: {train_root} not found!")
    if not comp_root.exists():
        print(f"WARNING: {comp_root} not found!")
    
    if not train_root.exists() and not comp_root.exists():
        raise FileNotFoundError(
            f"Neither Training-DataSets nor Competition-DataSets found in {dicom_root}!\n"
            f"Expected structure:\n"
            f"  {dicom_root}/Training-DataSets/20001/*.dcm\n"
            f"  {dicom_root}/Competition-DataSets/20001/*.dcm"
        )
    
    print(f"="*70)
    print(f"DICOM to NIfTI Conversion - Phase 1")
    print(f"="*70)
    print(f"Input:  {dicom_root}")
    print(f"Output: {out_dir}")
    print(f"\nCRITICAL: Training and Competition datasets contain DIFFERENT scans!")
    print(f"Both will be processed separately with TRAIN_/COMP_ prefixes.")
    print(f"="*70)
    print()
    
    convert_dicom_to_nifti(dicom_root, out_dir)


if __name__ == "__main__":
    main()
