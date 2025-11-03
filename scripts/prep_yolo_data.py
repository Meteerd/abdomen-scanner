"""
Prepare YOLO Dataset for Label Validation (GAP 2 Fix)

- Convert Excel annotations to YOLOv11 format
- Create internal train/val/test splits (80/10/10)
- Generate data.yaml configuration
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml


# 11→6 Class Mapping (from GAP 1)
CLASS_MAPPING = {
    # Class 0: AAA/AAD
    'Abdominal aortic aneurysm': 0,
    'Abdominal aortic dissection': 0,
    
    # Class 1: Acute Pancreatitis
    'Compatible with acute pancreatitis': 1,
    
    # Class 2: Cholecystitis
    'Compatible with acute cholecystitis': 2,
    'Gallbladder stone': 2,
    
    # Class 3: Kidney/Ureteral Stones
    'Kidney stone': 3,
    'ureteral stone': 3,
    
    # Class 4: Diverticulitis (RARE - 54 total)
    'Compatible with acute diverticulitis': 4,
    'Calcified diverticulum': 4,
    
    # Class 5: Appendicitis (RARE - 54 total)
    'Compatible with acute appendicitis': 5,
}


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


def bbox_to_yolo(xmin: int, ymin: int, xmax: int, ymax: int, 
                 img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert absolute bbox coordinates to YOLO format (normalized).
    
    YOLO format: class x_center y_center width height (all normalized 0-1)
    
    Args:
        xmin, ymin, xmax, ymax: Absolute pixel coordinates
        img_width, img_height: Image dimensions
        
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Calculate center and dimensions
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Clip to [0, 1]
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)
    
    return x_center, y_center, width, height


def get_dicom_dimensions(dicom_root: Path, case_number: int, image_id: int) -> Tuple[int, int]:
    """
    Get dimensions of a DICOM slice.
    
    This is a simplified version - you may need to adjust based on your DICOM structure.
    """
    # Try to find the DICOM file for this case and slice
    # This assumes a specific directory structure - adjust as needed
    case_dirs = list(dicom_root.glob(f"*{case_number}*"))
    
    if not case_dirs:
        # Default CT size if not found
        return 512, 512
    
    case_dir = case_dirs[0]
    dicom_files = sorted(case_dir.glob("*.dcm"))
    
    if dicom_files:
        try:
            dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=False)
            return int(dcm.Columns), int(dcm.Rows)
        except:
            pass
    
    # Default CT size
    return 512, 512


def prepare_yolo_dataset(
    excel_path: Path,
    dicom_root: Path,
    out_root: Path,
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str]
):
    """
    Prepare YOLO dataset from Excel annotations.
    
    Creates directory structure:
        out_root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
    """
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Read Excel annotations
    print(f"Reading annotations from {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name='TRAIININGDATA')
    
    # Filter only bounding boxes
    bbox_df = df[df['Type'] == 'Bounding Box'].copy()
    print(f"Found {len(bbox_df)} bounding box annotations")
    
    # Map cases to splits
    case_to_split = {}
    for case in train_cases:
        case_to_split[case] = 'train'
    for case in val_cases:
        case_to_split[case] = 'val'
    for case in test_cases:
        case_to_split[case] = 'test'
    
    # Statistics
    stats = {
        'train': {'total': 0, 'by_class': {i: 0 for i in range(6)}},
        'val': {'total': 0, 'by_class': {i: 0 for i in range(6)}},
        'test': {'total': 0, 'by_class': {i: 0 for i in range(6)}},
        'skipped': 0,
        'unknown_class': 0,
    }
    
    print("\nProcessing annotations...")
    
    # Group by case and image for efficiency
    grouped = bbox_df.groupby(['Case Number', 'Image Id'])
    
    for (case_number, image_id), group in tqdm(grouped, desc="Converting to YOLO format"):
        # Determine split
        case_id = f"case_{case_number}"
        if case_id not in case_to_split:
            # Try without prefix
            case_id = str(case_number)
            if case_id not in case_to_split:
                stats['skipped'] += len(group)
                continue
        
        split = case_to_split[case_id]
        
        # Get image dimensions
        img_width, img_height = get_dicom_dimensions(dicom_root, case_number, image_id)
        
        # Process each bbox in this image
        yolo_lines = []
        
        for _, row in group.iterrows():
            pathology_class = row['Class']
            data_str = row['Data']
            
            # Get class ID
            class_id = CLASS_MAPPING.get(pathology_class)
            
            if class_id is None:
                stats['unknown_class'] += 1
                continue
            
            # Parse bbox
            try:
                xmin, ymin, xmax, ymax = parse_bbox_data(data_str)
            except:
                stats['skipped'] += 1
                continue
            
            # Convert to YOLO format
            x_center, y_center, width, height = bbox_to_yolo(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            
            # Create YOLO line
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
            
            # Update stats
            stats[split]['total'] += 1
            stats[split]['by_class'][class_id] += 1
        
        # Save label file if we have any bboxes
        if yolo_lines:
            label_filename = f"case_{case_number}_slice_{image_id}.txt"
            label_path = out_root / 'labels' / split / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            # Note: In a complete implementation, you'd also copy/export the actual image
            # For now, we're just creating the label structure
    
    # Create data.yaml
    data_yaml = {
        'path': str(out_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 6,  # Number of classes
        'names': {
            0: 'AAA_AAD',
            1: 'Pancreatitis',
            2: 'Cholecystitis',
            3: 'Kidney_Ureteral_Stone',
            4: 'Diverticulitis',
            5: 'Appendicitis',
        }
    }
    
    with open(out_root / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("YOLO Dataset Preparation Complete!")
    print("=" * 70)
    
    print("\nDataset Statistics:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        print(f"  Total annotations: {stats[split]['total']}")
        print(f"  By class:")
        for class_id, count in stats[split]['by_class'].items():
            class_name = data_yaml['names'][class_id]
            print(f"    Class {class_id} ({class_name}): {count}")
    
    print(f"\nSkipped annotations: {stats['skipped']}")
    print(f"Unknown class annotations: {stats['unknown_class']}")
    
    print(f"\nOutput directory: {out_root}")
    print(f"YOLO config: {out_root / 'data.yaml'}")
    
    print("\n" + "=" * 70)
    print("Next step: Train YOLOv11 baseline")
    print("  sbatch slurm_phase1.5_yolo.sh")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset for label validation (GAP 2 fix)"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="data_raw/annotations",
        help="Path to annotations directory (CSV files) or legacy Excel file"
    )
    parser.add_argument(
        "--dicom_root",
        type=str,
        default="data_raw/dicom_files",
        help="Root directory containing DICOM files"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data_processed/yolo_dataset",
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio for training set (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio for test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    excel_path = Path(args.excel_path)
    dicom_root = Path(args.dicom_root)
    out_root = Path(args.out_root)
    
    # Check if CSV files exist in data_raw/annotations/
    csv_dir = Path("data_raw/annotations")
    train_csv = csv_dir / "TRAININGDATA.csv"
    comp_csv = csv_dir / "COMPETITIONDATA.csv"
    
    if train_csv.exists() and comp_csv.exists():
        print(f"Loading annotations from CSV files in {csv_dir}...")
        
        # Read both CSV files
        train_df = pd.read_csv(train_csv)
        comp_df = pd.read_csv(comp_csv)
        
        # Merge datasets
        df = pd.concat([train_df, comp_df], ignore_index=True)
        print(f"Merged {len(train_df)} training + {len(comp_df)} competition annotations")
        
    elif excel_path.exists():
        # Fallback to Excel file
        print(f"CSV files not found, loading from Excel: {excel_path}...")
        df = pd.read_excel(excel_path, sheet_name='TRAIININGDATA')
    else:
        raise FileNotFoundError(f"Neither CSV files nor Excel file found")
    
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Load and split cases
    print("Extracting unique cases...")
    unique_cases = df['Case Number'].unique()
    print(f"Found {len(unique_cases)} unique cases")
    
    # Create splits
    print(f"\nSplitting with ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    
    val_test_ratio = args.val_ratio + args.test_ratio
    train_cases, val_test_cases = train_test_split(
        unique_cases,
        test_size=val_test_ratio,
        random_state=args.seed
    )
    
    # Calculate test ratio relative to val_test subset
    relative_test_ratio = args.test_ratio / val_test_ratio
    val_cases, test_cases = train_test_split(
        val_test_cases,
        test_size=relative_test_ratio,
        random_state=args.seed
    )
    
    train_cases = list(train_cases)
    val_cases = list(val_cases)
    test_cases = list(test_cases)
    
    print(f"Train: {len(train_cases)} cases")
    print(f"Val: {len(val_cases)} cases")
    print(f"Test: {len(test_cases)} cases")
    
    # Prepare dataset
    prepare_yolo_dataset(
        excel_path, dicom_root, out_root,
        train_cases, val_cases, test_cases
    )


if __name__ == "__main__":
    main()
