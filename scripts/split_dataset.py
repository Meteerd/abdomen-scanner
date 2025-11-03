"""
Phase 1 - Step 3: Split Dataset into Train/Val/Test

Purpose:
- Split unique case IDs into train/val/test sets.
- Creates JSON format files compatible with train_monai.py

Usage:
    python scripts/split_dataset.py --nifti_dir data_processed/nifti_images --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import random
import json


def get_case_ids(nifti_dir: Path) -> List[str]:
    """
    Get list of unique case IDs from NIfTI directory.
    
    Args:
        nifti_dir: Directory containing NIfTI files
        
    Returns:
        List of case IDs (filenames without .nii.gz extension)
    """
    nifti_files = sorted(nifti_dir.glob("*.nii.gz"))
    case_ids = [f.stem.replace('.nii', '') for f in nifti_files]
    return case_ids


def split_cases(case_ids: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Split case IDs into train/val/test sets.
    
    Args:
        case_ids: List of case IDs
        train_ratio: Fraction for training (e.g., 0.8)
        val_ratio: Fraction for validation (e.g., 0.1)
        test_ratio: Fraction for testing (e.g., 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_cases, val_cases, test_cases)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle case IDs
    shuffled_cases = case_ids.copy()
    random.shuffle(shuffled_cases)
    
    # Calculate split indices
    n_total = len(shuffled_cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_cases = shuffled_cases[:n_train]
    val_cases = shuffled_cases[n_train:n_train + n_val]
    test_cases = shuffled_cases[n_train + n_val:]
    
    return train_cases, val_cases, test_cases


def save_split(case_ids: List[str], out_path: Path, image_dir: Path, label_dir: Path):
    """
    Save split to file in JSON format (one dict per line).
    
    Args:
        case_ids: List of case IDs
        out_path: Output file path
        image_dir: Directory containing NIfTI images
        label_dir: Directory containing NIfTI labels
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        for case_id in case_ids:
            data_dict = {
                "image": str(image_dir / f"{case_id}.nii.gz"),
                "label": str(label_dir / f"{case_id}.nii.gz"),
                "case_id": case_id
            }
            f.write(json.dumps(data_dict) + "\n")


def create_dataset_splits(nifti_dir: Path, label_dir: Path, train_out: Path, val_out: Path, test_out: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    Create train/val/test splits and save to JSON format files.
    
    Args:
        nifti_dir: Directory containing NIfTI images
        label_dir: Directory containing NIfTI labels
        train_out: Output path for train split
        val_out: Output path for val split
        test_out: Output path for test split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    """
    # Get case IDs
    print(f"Scanning NIfTI files in {nifti_dir}...")
    case_ids = get_case_ids(nifti_dir)
    print(f"Found {len(case_ids)} cases")
    
    if len(case_ids) == 0:
        raise ValueError(f"No NIfTI files found in {nifti_dir}")
    
    # Split cases
    print(f"\nSplitting with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    
    train_cases, val_cases, test_cases = split_cases(case_ids, train_ratio, val_ratio, test_ratio, seed)
    
    # Save splits
    print(f"\nSaving splits...")
    save_split(train_cases, train_out, nifti_dir, label_dir)
    save_split(val_cases, val_out, nifti_dir, label_dir)
    save_split(test_cases, test_out, nifti_dir, label_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset split complete!")
    print(f"  Train: {len(train_cases)} cases → {train_out}")
    print(f"  Val:   {len(val_cases)} cases → {val_out}")
    print(f"  Test:  {len(test_cases)} cases → {test_out}")
    print(f"  Total: {len(case_ids)} cases")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--nifti_dir",
        type=str,
        default="data_processed/nifti_images",
        help="Directory containing NIfTI images"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="data_processed/nifti_labels_medsam",
        help="Directory containing NIfTI labels"
    )
    parser.add_argument(
        "--train_out",
        type=str,
        default="splits/train_cases.txt",
        help="Output path for train split"
    )
    parser.add_argument(
        "--val_out",
        type=str,
        default="splits/val_cases.txt",
        help="Output path for val split"
    )
    parser.add_argument(
        "--test_out",
        type=str,
        default="splits/test_cases.txt",
        help="Output path for test split"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    label_dir = Path(args.label_dir)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    test_out = Path(args.test_out)
    
    if not nifti_dir.exists():
        raise FileNotFoundError(f"NIfTI directory not found: {nifti_dir}")
    
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    create_dataset_splits(
        nifti_dir=nifti_dir,
        label_dir=label_dir,
        train_out=train_out,
        val_out=val_out,
        test_out=test_out,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
    
    print(f"Starting dataset split...")
    print(f"Input: {nifti_dir}\n")
    
    create_dataset_splits(nifti_dir, train_out, val_out, test_out, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
