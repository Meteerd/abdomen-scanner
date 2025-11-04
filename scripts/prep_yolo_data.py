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
import cv2


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


def prepare_yolo_dataset(
    dicom_manifest: Dict[Tuple[int, int], Path],
    df: pd.DataFrame,
    out_root: Path,
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str]
):
    """
    Prepare YOLO dataset from annotations and DICOM files.
    
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
    
    # Filter only bounding boxes
    bbox_df = df[df['Type'] == 'Bounding Box'].copy()
    print(f"Found {len(bbox_df)} bounding box annotations")
    
    # Map cases to splits
    case_to_split = {}
    for case in train_cases:
        case_to_split[str(case)] = 'train'
    for case in val_cases:
        case_to_split[str(case)] = 'val'
    for case in test_cases:
        case_to_split[str(case)] = 'test'
    
    # Statistics
    stats = {
        'train': {'total': 0, 'by_class': {i: 0 for i in range(6)}},
        'val': {'total': 0, 'by_class': {i: 0 for i in range(6)}},
        'test': {'total': 0, 'by_class': {i: 0 for i in range(6)}},
        'skipped_no_dicom': 0,
        'skipped_no_split': 0,
        'unknown_class': 0,
        'images_exported': 0,
    }
    
    print("\nProcessing annotations and exporting images...")
    
    # Group by case and image for efficiency
    grouped = bbox_df.groupby(['Case Number', 'Image Id'])
    
    for (case_number, image_id), group in tqdm(grouped, desc="Exporting DICOM slices"):
        # Determine split
        case_id = str(case_number)
        if case_id not in case_to_split:
            stats['skipped_no_split'] += len(group)
            continue
        
        split = case_to_split[case_id]
        
        # Find DICOM file in manifest
        dicom_key = (case_number, image_id)
        if dicom_key not in dicom_manifest:
            stats['skipped_no_dicom'] += len(group)
            continue
        
        dicom_path = dicom_manifest[dicom_key]
        
        # Read DICOM file
        try:
            dcm = pydicom.dcmread(str(dicom_path))
            pixel_array = dcm.pixel_array
            
            # Convert to Hounsfield Units if needed
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            img_height = int(dcm.Rows)
            img_width = int(dcm.Columns)
        except Exception as e:
            stats['skipped_no_dicom'] += len(group)
            continue
        
        # Apply CT windowing to convert to 8-bit
        img_8bit = apply_ct_window(pixel_array)
        
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
                continue
            
            # Convert to YOLO format using actual DICOM dimensions
            x_center, y_center, width, height = bbox_to_yolo(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            
            # Create YOLO line
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
            
            # Update stats
            stats[split]['total'] += 1
            stats[split]['by_class'][class_id] += 1
        
        # Save image and label files if we have any bboxes
        if yolo_lines:
            base_filename = f"case_{case_number}_slice_{image_id}"
            
            # Skip if both files already exist
            img_path = out_root / 'images' / split / f"{base_filename}.png"
            label_path = out_root / 'labels' / split / f"{base_filename}.txt"
            
            if img_path.exists() and label_path.exists():
                stats[split]['total'] += len(yolo_lines)
                for line in yolo_lines:
                    class_id = int(line.split()[0])
                    stats[split]['by_class'][class_id] += 1
                continue
            
            # Save PNG image
            cv2.imwrite(str(img_path), img_8bit)
            stats['images_exported'] += 1
            
            # Save label file
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
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
    
    print(f"\nImages exported: {stats['images_exported']}")
    
    print("\nDataset Statistics:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        print(f"  Total annotations: {stats[split]['total']}")
        print(f"  By class:")
        for class_id, count in stats[split]['by_class'].items():
            class_name = data_yaml['names'][class_id]
            print(f"    Class {class_id} ({class_name}): {count}")
    
    print(f"\nSkipped (no DICOM file): {stats['skipped_no_dicom']}")
    print(f"Skipped (no split assignment): {stats['skipped_no_split']}")
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
    
    # Build DICOM manifest
    dicom_manifest = build_dicom_manifest(dicom_root)
    
    if not dicom_manifest:
        raise RuntimeError("No DICOM files found in manifest!")
    
    # Prepare dataset
    prepare_yolo_dataset(
        dicom_manifest, df, out_root,
        train_cases, val_cases, test_cases
    )


if __name__ == "__main__":
    main()
