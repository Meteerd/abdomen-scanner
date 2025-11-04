"""
Phase 1 - Step 1: DICOM to NIfTI Conversion

Purpose:
- Group DICOM series by case, sort slices, stack into 3D arrays.
- Save volumes as NIfTI (.nii.gz), preserving spacing/origin/direction.
- Designed for HPC cluster execution with parallel processing.

Usage:
    python scripts/dicom_to_nifti.py --dicom_root data_raw/dicom_files --out_dir data_processed/nifti_images
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
    Group DICOM files by case/series.
    
    Args:
        dicom_root: Root directory containing DICOM files
        
    Returns:
        Dictionary mapping case IDs to lists of DICOM file paths
    """
    cases = defaultdict(list)
    
    # Walk through directory structure
    for dcm_file in dicom_root.rglob("*.dcm"):
        try:
            dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            
            # Use SeriesInstanceUID or PatientID + StudyInstanceUID as case ID
            if hasattr(dcm, 'SeriesInstanceUID'):
                case_id = dcm.SeriesInstanceUID
            elif hasattr(dcm, 'PatientID') and hasattr(dcm, 'StudyInstanceUID'):
                case_id = f"{dcm.PatientID}_{dcm.StudyInstanceUID}"
            else:
                # Fallback: use parent directory name
                case_id = dcm_file.parent.name
            
            cases[case_id].append(dcm_file)
        except Exception as e:
            print(f"Warning: Could not read {dcm_file}: {e}")
            continue
    
    return dict(cases)


def convert_dicom_to_nifti(dicom_root: Path, out_dir: Path):
    """
    Convert all DICOM series to NIfTI format.
    
    Args:
        dicom_root: Root directory containing DICOM files
        out_dir: Output directory for NIfTI files
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group DICOM files by case
    print(f"Scanning DICOM files in {dicom_root}...")
    cases = group_dicom_by_case(dicom_root)
    print(f"Found {len(cases)} unique cases/series")
    
    # Process each case
    success_count = 0
    skipped_count = 0
    failed_cases = []
    
    for case_id, dicom_files in tqdm(cases.items(), desc="Converting to NIfTI"):
        try:
            # Create safe filename
            safe_case_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in case_id)
            out_path = out_dir / f"{safe_case_id}.nii.gz"
            
            # Skip if already exists
            if out_path.exists():
                skipped_count += 1
                continue
            
            # Sort slices
            sorted_files = sort_dicom_slices(dicom_files)
            
            if len(sorted_files) < 5:
                print(f"Warning: Case {case_id} has only {len(sorted_files)} slices, skipping")
                continue
            
            # Load as 3D volume
            image = load_dicom_series(sorted_files)
            
            # Save as NIfTI
            sitk.WriteImage(image, str(out_path), useCompression=True)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            failed_cases.append(case_id)
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {success_count} new cases")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Total cases: {len(cases)}")
    print(f"Output directory: {out_dir}")
    
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for case_id in failed_cases[:10]:  # Show first 10
            print(f"  - {case_id}")
        if len(failed_cases) > 10:
            print(f"  ... and {len(failed_cases) - 10} more")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM series to NIfTI format")
    parser.add_argument(
        "--dicom_root",
        type=str,
        default="data_raw/dicom_files",
        help="Root directory containing DICOM files"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_processed/nifti_images",
        help="Output directory for NIfTI files"
    )
    
    args = parser.parse_args()
    
    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)
    
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root directory not found: {dicom_root}")
    
    print(f"Starting DICOM to NIfTI conversion...")
    print(f"Input: {dicom_root}")
    print(f"Output: {out_dir}\n")
    
    convert_dicom_to_nifti(dicom_root, out_dir)


if __name__ == "__main__":
    main()
