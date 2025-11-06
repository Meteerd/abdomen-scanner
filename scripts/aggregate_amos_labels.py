#!/usr/bin/env python3
"""
Aggregate AMOS 2022 Individual Organ Segmentations into Multi-Class Labels

Purpose:
- AMOS dataset has 117 separate organ segmentation files per case
- Training requires a single multi-class label volume
- This script combines the 15 main organs into a single NIfTI file

Input:
    data/AbdomenDataSet/AMOS-DataSet/s0000/segmentations/*.nii.gz (117 files)

Output:
    data/AbdomenDataSet/AMOS-DataSet/s0000/label.nii.gz (1 file, 16 classes)

Usage:
    python scripts/aggregate_amos_labels.py --amos_dir data/AbdomenDataSet/AMOS-DataSet
"""

import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# AMOS 2022 Organ Class Mapping
# Based on AMOS paper (arXiv:2206.08023) and config_pretrain.yaml
AMOS_ORGAN_MAP = {
    1: 'spleen.nii.gz',
    2: 'kidney_right.nii.gz',
    3: 'kidney_left.nii.gz',
    4: 'gallbladder.nii.gz',
    5: 'esophagus.nii.gz',
    6: 'liver.nii.gz',
    7: 'stomach.nii.gz',
    8: 'aorta.nii.gz',
    9: 'inferior_vena_cava.nii.gz',
    10: 'portal_vein_and_splenic_vein.nii.gz',
    11: 'pancreas.nii.gz',
    12: 'adrenal_gland_right.nii.gz',
    13: 'adrenal_gland_left.nii.gz',
    14: 'duodenum.nii.gz',
    15: 'urinary_bladder.nii.gz',
}


def aggregate_case_labels(case_dir: Path) -> Dict[str, any]:
    """
    Aggregate individual organ segmentations into a single multi-class label volume.
    
    Args:
        case_dir: Path to case directory (e.g., s0000/)
        
    Returns:
        Dict with status and statistics
    """
    case_id = case_dir.name
    seg_dir = case_dir / 'segmentations'
    output_path = case_dir / 'label.nii.gz'
    
    # Skip if already processed
    if output_path.exists():
        return {
            'case_id': case_id,
            'status': 'skipped',
            'reason': 'label.nii.gz already exists'
        }
    
    # Check if segmentations directory exists
    if not seg_dir.exists():
        return {
            'case_id': case_id,
            'status': 'error',
            'reason': 'segmentations/ directory not found'
        }
    
    # Load CT image to get reference geometry
    ct_path = case_dir / 'ct.nii.gz'
    if not ct_path.exists():
        return {
            'case_id': case_id,
            'status': 'error',
            'reason': 'ct.nii.gz not found'
        }
    
    try:
        # Load CT image as reference
        ct_img = sitk.ReadImage(str(ct_path))
        
        # Create empty label volume (same size as CT)
        label_array = np.zeros(sitk.GetArrayFromImage(ct_img).shape, dtype=np.uint8)
        
        # Track which organs were found
        found_organs = []
        missing_organs = []
        
        # Aggregate each organ into the label volume
        for class_id, organ_file in AMOS_ORGAN_MAP.items():
            organ_path = seg_dir / organ_file
            
            if organ_path.exists():
                # Load organ segmentation
                organ_img = sitk.ReadImage(str(organ_path))
                organ_array = sitk.GetArrayFromImage(organ_img)
                
                # Add to label volume (overwrite with higher class ID if overlap)
                # This handles edge cases where organs overlap
                mask = organ_array > 0
                label_array[mask] = class_id
                
                found_organs.append(organ_file)
            else:
                missing_organs.append(organ_file)
        
        # Create output image with same geometry as CT
        label_img = sitk.GetImageFromArray(label_array)
        label_img.CopyInformation(ct_img)
        
        # Save aggregated label
        sitk.WriteImage(label_img, str(output_path), useCompression=True)
        
        return {
            'case_id': case_id,
            'status': 'success',
            'found_organs': len(found_organs),
            'missing_organs': len(missing_organs),
            'output_size_mb': output_path.stat().st_size / (1024**2)
        }
        
    except Exception as e:
        return {
            'case_id': case_id,
            'status': 'error',
            'reason': str(e)
        }


def process_cases_parallel(amos_dir: Path, num_workers: int = 16):
    """
    Process all AMOS cases in parallel.
    """
    # Find all case directories
    case_dirs = sorted([d for d in amos_dir.iterdir() if d.is_dir() and d.name.startswith('s')])
    
    print(f"Found {len(case_dirs)} AMOS cases to process")
    print(f"Using {num_workers} worker processes")
    print("")
    
    results = {
        'success': [],
        'skipped': [],
        'error': []
    }
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_case = {
            executor.submit(aggregate_case_labels, case_dir): case_dir 
            for case_dir in case_dirs
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_case), total=len(case_dirs), desc="Aggregating labels"):
            result = future.result()
            status = result['status']
            results[status].append(result)
    
    return results


def print_summary(results: Dict):
    """Print aggregation summary."""
    print("\n" + "="*70)
    print("AMOS Label Aggregation Summary")
    print("="*70)
    
    print(f"\nTotal cases processed: {sum(len(v) for v in results.values())}")
    print(f"  ✓ Successfully aggregated: {len(results['success'])}")
    print(f"  ⊘ Skipped (already exists): {len(results['skipped'])}")
    print(f"  ✗ Errors: {len(results['error'])}")
    
    if results['success']:
        # Calculate average statistics
        avg_found = np.mean([r['found_organs'] for r in results['success']])
        avg_missing = np.mean([r['missing_organs'] for r in results['success']])
        avg_size = np.mean([r['output_size_mb'] for r in results['success']])
        
        print(f"\nLabel Statistics (averaged):")
        print(f"  Organs found per case: {avg_found:.1f} / 15")
        print(f"  Missing organs per case: {avg_missing:.1f} / 15")
        print(f"  Output file size: {avg_size:.1f} MB")
    
    if results['error']:
        print(f"\nErrors encountered:")
        for result in results['error'][:5]:  # Show first 5 errors
            print(f"  {result['case_id']}: {result['reason']}")
        if len(results['error']) > 5:
            print(f"  ... and {len(results['error']) - 5} more errors")
    
    print("\n" + "="*70)
    print("Aggregation complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Aggregate AMOS organ segmentations into multi-class labels')
    parser.add_argument(
        '--amos_dir',
        type=str,
        default='data/AbdomenDataSet/AMOS-DataSet',
        help='Path to AMOS dataset directory (contains s0000/, s0001/, ...)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    amos_dir = Path(args.amos_dir)
    
    # Validate input
    if not amos_dir.exists():
        print(f"ERROR: AMOS directory not found: {amos_dir}")
        return 1
    
    print("="*70)
    print("AMOS 2022 Label Aggregation")
    print("="*70)
    print(f"AMOS directory: {amos_dir}")
    print(f"Workers: {args.num_workers}")
    print("")
    print("Organ mapping (15 classes):")
    for class_id, organ_file in AMOS_ORGAN_MAP.items():
        organ_name = organ_file.replace('_', ' ').replace('.nii.gz', '').title()
        print(f"  Class {class_id:2d}: {organ_name}")
    print("")
    
    # Process all cases
    results = process_cases_parallel(amos_dir, args.num_workers)
    
    # Print summary
    print_summary(results)
    
    return 0


if __name__ == '__main__':
    exit(main())
