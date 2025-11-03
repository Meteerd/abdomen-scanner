"""
Phase 2.5 - Quality Control for Rare Classes (GAP 4)

Purpose:
- Sample cases from rare pathology classes (Class 5 & 6)
- Generate file paths for manual inspection in ITK-SNAP or 3D Slicer
- Catch systematic errors before expensive Phase 3 training

Dataset:
- Class 5 (Diverticulitis): 54 examples
- Class 6 (Appendicitis): 54 examples

Critical Issue (GAP 4):
- These classes have 181:1 imbalance ratio
- MedSAM may fail on subtle pathology
- Manual QC is MANDATORY before Phase 3

Usage:
    python scripts/sample_rare_classes.py \
        --excel_path Temp/Information.xlsx \
        --medsam_labels data_processed/nifti_labels_medsam \
        --nifti_images data_processed/nifti_images \
        --out_file phase2_qc_checklist.txt \
        --samples_per_class 10
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
import random


# Class mapping (11 → 6 classes, same as throughout project)
CLASS_MAPPING = {
    1: 1,   # Organ → Organ
    2: 2,   # Bowel → Bowel
    3: 1,   # Organ → Organ
    4: 3,   # Vessel → Vessel
    5: 4,   # Free Fluid → Free Fluid
    6: 5,   # Diverticulitis → Diverticulitis
    7: 6,   # Appendicitis → Appendicitis
    8: 1,   # Organ → Organ
    9: 3,   # Vessel → Vessel
    10: 3,  # Vessel → Vessel
    11: 3,  # Vessel → Vessel
}


def load_annotations(excel_path: Path) -> pd.DataFrame:
    """Load annotations from Excel file."""
    df = pd.read_excel(excel_path, sheet_name='TRAIININGDATA')
    return df


def get_rare_class_cases(df: pd.DataFrame, target_class: int) -> List[str]:
    """
    Get all case IDs for a specific class.
    
    Args:
        df: DataFrame with annotations
        target_class: Class to filter (5 or 6 in new mapping)
        
    Returns:
        List of unique case IDs
    """
    # Apply class mapping
    df['new_class'] = df['class'].map(CLASS_MAPPING)
    
    # Filter by target class
    class_df = df[df['new_class'] == target_class]
    
    # Get unique case IDs
    case_ids = class_df['case'].unique().tolist()
    
    return case_ids


def sample_cases(case_ids: List[str], num_samples: int, seed: int = 42) -> List[str]:
    """
    Randomly sample cases from list.
    
    Args:
        case_ids: List of case IDs
        num_samples: Number of cases to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled case IDs
    """
    random.seed(seed)
    
    # If fewer cases than requested, return all
    if len(case_ids) <= num_samples:
        return sorted(case_ids)
    
    # Random sample
    sampled = random.sample(case_ids, num_samples)
    return sorted(sampled)


def generate_qc_checklist(
    sampled_cases: Dict[int, List[str]],
    nifti_images: Path,
    medsam_labels: Path,
    out_file: Path
):
    """
    Generate QC checklist with file paths for manual inspection.
    
    Args:
        sampled_cases: Dict mapping class_id → list of case_ids
        nifti_images: Directory with NIfTI images
        medsam_labels: Directory with MedSAM pseudo-labels
        out_file: Output text file path
    """
    
    class_names = {
        5: "Diverticulitis",
        6: "Appendicitis",
    }
    
    with open(out_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 2.5: Quality Control Checklist (GAP 4)\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        f.write("PURPOSE:\n")
        f.write("  Manually inspect MedSAM pseudo-labels for rare pathology classes\n")
        f.write("  before expensive Phase 3 training.\n")
        f.write("\n")
        f.write("CRITICAL ISSUE (GAP 4):\n")
        f.write("  - Class 5 (Diverticulitis): 54 examples (181:1 imbalance)\n")
        f.write("  - Class 6 (Appendicitis):   54 examples (181:1 imbalance)\n")
        f.write("  - MedSAM may fail on subtle pathology\n")
        f.write("  - Manual QC prevents wasted compute on bad labels\n")
        f.write("\n")
        f.write("INSPECTION TOOLS:\n")
        f.write("  - ITK-SNAP: http://www.itksnap.org/\n")
        f.write("  - 3D Slicer: https://www.slicer.org/\n")
        f.write("\n")
        f.write("WHAT TO CHECK:\n")
        f.write("  1. Segmentation coverage (does mask cover full pathology?)\n")
        f.write("  2. Boundary accuracy (is mask boundary clean or noisy?)\n")
        f.write("  3. False positives (any background wrongly segmented?)\n")
        f.write("  4. Missing slices (z-axis gaps in annotations?)\n")
        f.write("\n")
        f.write("DECISION CRITERIA:\n")
        f.write("  PASS (>70% good):   Proceed to Phase 3\n")
        f.write("  MARGINAL (50-70%):  Consider re-running MedSAM with tuned prompts\n")
        f.write("  FAIL (<50%):        Stop, fix MedSAM prompts before Phase 3\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        for class_id, cases in sampled_cases.items():
            class_name = class_names[class_id]
            
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write(f"Class {class_id}: {class_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sampled: {len(cases)} cases (out of 54 total)\n")
            f.write("\n")
            
            for idx, case_id in enumerate(cases, 1):
                image_path = nifti_images / f"{case_id}.nii.gz"
                label_path = medsam_labels / f"{case_id}.nii.gz"
                
                f.write(f"\n[{idx}] Case: {case_id}\n")
                f.write(f"    Image: {image_path}\n")
                f.write(f"    Label: {label_path}\n")
                
                # Check if files exist
                if not image_path.exists():
                    f.write(f"    ⚠ WARNING: Image file not found!\n")
                if not label_path.exists():
                    f.write(f"    ⚠ WARNING: Label file not found!\n")
                
                f.write(f"\n    ITK-SNAP command:\n")
                f.write(f"      itksnap -g {image_path} -s {label_path}\n")
                f.write(f"\n    Quality Check:\n")
                f.write(f"      [ ] Coverage: Full / Partial / Missing\n")
                f.write(f"      [ ] Boundary: Clean / Noisy / Incorrect\n")
                f.write(f"      [ ] False Positives: None / Few / Many\n")
                f.write(f"      [ ] Overall: PASS / MARGINAL / FAIL\n")
                f.write(f"      Notes: ___________________________________________\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        for class_id, cases in sampled_cases.items():
            class_name = class_names[class_id]
            f.write(f"Class {class_id} ({class_name}):\n")
            f.write(f"  Total checked: {len(cases)}\n")
            f.write(f"  PASS:     ___ / {len(cases)}\n")
            f.write(f"  MARGINAL: ___ / {len(cases)}\n")
            f.write(f"  FAIL:     ___ / {len(cases)}\n")
            f.write(f"  Pass rate: ____%\n")
            f.write("\n")
        
        f.write("OVERALL DECISION:\n")
        f.write("  [ ] PASS: Proceed to Phase 3 (sbatch slurm_phase3_training.sh)\n")
        f.write("  [ ] MARGINAL: Re-run MedSAM with adjusted prompts\n")
        f.write("  [ ] FAIL: Fix MedSAM inference before Phase 3\n")
        f.write("\n")
        f.write("Reviewer: _________________    Date: __________\n")
        f.write("\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample rare class cases for manual QC (GAP 4)"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="Temp/Information.xlsx",
        help="Path to Excel file with annotations"
    )
    parser.add_argument(
        "--medsam_labels",
        type=str,
        default="data_processed/nifti_labels_medsam",
        help="Directory with MedSAM pseudo-labels"
    )
    parser.add_argument(
        "--nifti_images",
        type=str,
        default="data_processed/nifti_images",
        help="Directory with NIfTI images"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="phase2_qc_checklist.txt",
        help="Output checklist file"
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=10,
        help="Number of cases to sample per class"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    excel_path = Path(args.excel_path)
    medsam_labels = Path(args.medsam_labels)
    nifti_images = Path(args.nifti_images)
    out_file = Path(args.out_file)
    
    # Check paths
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not medsam_labels.exists():
        raise FileNotFoundError(f"MedSAM labels directory not found: {medsam_labels}")
    if not nifti_images.exists():
        raise FileNotFoundError(f"NIfTI images directory not found: {nifti_images}")
    
    print("=" * 80)
    print("Phase 2.5: Quality Control Sampling (GAP 4)")
    print("=" * 80)
    print()
    
    # Load annotations
    print(f"Loading annotations from: {excel_path}")
    df = load_annotations(excel_path)
    print(f"Total annotations: {len(df)}")
    print()
    
    # Sample rare classes
    sampled_cases = {}
    
    for class_id, class_name in [(5, "Diverticulitis"), (6, "Appendicitis")]:
        print(f"Class {class_id} ({class_name}):")
        
        # Get all cases for this class
        case_ids = get_rare_class_cases(df, class_id)
        print(f"  Total cases: {len(case_ids)}")
        
        # Sample
        sampled = sample_cases(case_ids, args.samples_per_class, args.seed)
        sampled_cases[class_id] = sampled
        print(f"  Sampled: {len(sampled)} cases")
        print()
    
    # Generate QC checklist
    print(f"Generating QC checklist: {out_file}")
    generate_qc_checklist(sampled_cases, nifti_images, medsam_labels, out_file)
    
    print()
    print("=" * 80)
    print("QC Checklist Generated!")
    print("=" * 80)
    print()
    print(f"File: {out_file}")
    print()
    print("Next steps:")
    print("  1. Open checklist file:")
    print(f"     cat {out_file}")
    print()
    print("  2. Manually inspect each case in ITK-SNAP or 3D Slicer")
    print()
    print("  3. Record quality assessments in the checklist")
    print()
    print("  4. Calculate pass rate:")
    print("     - >70% PASS: Proceed to Phase 3")
    print("     - 50-70% PASS: Consider re-running MedSAM")
    print("     - <50% PASS: Fix MedSAM before Phase 3")
    print()
    print("  5. If PASS, start Phase 3 training:")
    print("     sbatch slurm_phase3_training.sh")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
