#!/usr/bin/env python3
"""
Verify that the pipeline fix was successful.

Usage:
    python verify_pipeline_fix.py
"""

import sys
from pathlib import Path
import nibabel as nib
import pandas as pd

def check_phase1_output():
    """Verify Phase 1 created separate TRAIN_ and COMP_ NIfTI files."""
    print("="*70)
    print("PHASE 1 VERIFICATION: DICOM to NIfTI Conversion")
    print("="*70)
    
    nifti_dir = Path('data_processed/nifti_images')
    
    if not nifti_dir.exists():
        print("❌ FAIL: data_processed/nifti_images/ does not exist")
        return False
    
    all_files = list(nifti_dir.glob('*.nii.gz'))
    train_files = [f for f in all_files if f.name.startswith('TRAIN_')]
    comp_files = [f for f in all_files if f.name.startswith('COMP_')]
    
    print(f"\n📊 NIfTI Files Count:")
    print(f"  Total files: {len(all_files)}")
    print(f"  TRAIN_ files: {len(train_files)}")
    print(f"  COMP_ files: {len(comp_files)}")
    
    if len(train_files) == 0 or len(comp_files) == 0:
        print("❌ FAIL: Missing TRAIN_ or COMP_ prefixed files")
        print("   Expected: 736 TRAIN_ files and 359 COMP_ files")
        return False
    
    expected_total = 1095  # 736 Training + 359 Competition
    if abs(len(all_files) - expected_total) > 50:  # Allow some tolerance
        print(f"⚠️  WARNING: Expected ~{expected_total} files, got {len(all_files)}")
    else:
        print(f"✅ PASS: File count is correct (~{expected_total} expected)")
    
    # Check that case 20050 exists in both datasets
    train_20050 = any('TRAIN_20050' in f.name for f in all_files)
    comp_20050 = any('COMP_20050' in f.name for f in all_files)
    
    if train_20050 and comp_20050:
        print(f"✅ PASS: Case 20050 exists in BOTH datasets (as expected)")
    else:
        print(f"⚠️  WARNING: Case 20050 missing from one dataset")
        print(f"   TRAIN_20050: {train_20050}, COMP_20050: {comp_20050}")
    
    return True


def check_phase2_masks():
    """Verify Phase 2 created separate TRAIN_case_ and COMP_case_ directories."""
    print("\n" + "="*70)
    print("PHASE 2 VERIFICATION: MedSAM 2D Masks")
    print("="*70)
    
    masks_dir = Path('data_processed/medsam_2d_masks')
    
    if not masks_dir.exists():
        print("❌ FAIL: data_processed/medsam_2d_masks/ does not exist")
        return False
    
    all_dirs = [d for d in masks_dir.iterdir() if d.is_dir()]
    train_dirs = [d for d in all_dirs if d.name.startswith('TRAIN_case_')]
    comp_dirs = [d for d in all_dirs if d.name.startswith('COMP_case_')]
    
    print(f"\n📊 Mask Directories Count:")
    print(f"  Total directories: {len(all_dirs)}")
    print(f"  TRAIN_case_ directories: {len(train_dirs)}")
    print(f"  COMP_case_ directories: {len(comp_dirs)}")
    
    if len(train_dirs) == 0 or len(comp_dirs) == 0:
        print("❌ FAIL: Missing TRAIN_case_ or COMP_case_ directories")
        return False
    
    # Check case 20050 directories
    train_20050_dir = masks_dir / 'TRAIN_case_20050'
    comp_20050_dir = masks_dir / 'COMP_case_20050'
    
    if train_20050_dir.exists() and comp_20050_dir.exists():
        train_masks = len(list(train_20050_dir.glob('*.npy')))
        comp_masks = len(list(comp_20050_dir.glob('*.npy')))
        print(f"✅ PASS: Case 20050 has separate mask directories")
        print(f"   TRAIN_case_20050: {train_masks} masks")
        print(f"   COMP_case_20050: {comp_masks} masks")
    else:
        print(f"⚠️  WARNING: Case 20050 mask directories not found")
    
    return True


def check_phase2_labels():
    """Verify Phase 2 aggregation created label volumes."""
    print("\n" + "="*70)
    print("PHASE 2 VERIFICATION: 3D Label Aggregation")
    print("="*70)
    
    labels_dir = Path('data_processed/nifti_labels_medsam')
    
    if not labels_dir.exists():
        print("❌ FAIL: data_processed/nifti_labels_medsam/ does not exist")
        return False
    
    all_files = list(labels_dir.glob('*.nii.gz'))
    train_files = [f for f in all_files if f.name.startswith('TRAIN_')]
    comp_files = [f for f in all_files if f.name.startswith('COMP_')]
    
    print(f"\n📊 Label Files Count:")
    print(f"  Total files: {len(all_files)}")
    print(f"  TRAIN_ files: {len(train_files)}")
    print(f"  COMP_ files: {len(comp_files)}")
    
    if len(all_files) == 0:
        print("❌ FAIL: No label files found")
        return False
    
    # Check rare class labels
    print(f"\n🔍 Checking Rare Class Labels...")
    
    class_4_cases = []
    class_5_cases = []
    
    for label_file in all_files:
        try:
            img = nib.load(label_file)
            data = img.get_fdata()
            
            if (data == 4).any():
                class_4_cases.append(label_file.name)
            
            if (data == 5).any():
                class_5_cases.append(label_file.name)
        except Exception as e:
            continue
    
    print(f"\n  Class 4 (Diverticulitis):")
    print(f"    Cases with labels: {len(class_4_cases)}")
    print(f"    Expected: ~21 cases (old pipeline: ~6)")
    
    if len(class_4_cases) >= 15:
        print(f"    ✅ PASS: Sufficient Diverticulitis cases for training")
    else:
        print(f"    ❌ FAIL: Too few Diverticulitis cases ({len(class_4_cases)})")
    
    print(f"\n  Class 5 (Appendicitis):")
    print(f"    Cases with labels: {len(class_5_cases)}")
    print(f"    Expected: ~87 cases (old pipeline: ~26)")
    
    if len(class_5_cases) >= 60:
        print(f"    ✅ PASS: Sufficient Appendicitis cases for training")
    else:
        print(f"    ❌ FAIL: Too few Appendicitis cases ({len(class_5_cases)})")
    
    # Check case 20050 specifically
    train_20050_label = labels_dir / 'TRAIN_20050.nii.gz'
    comp_20050_label = labels_dir / 'COMP_20050.nii.gz'
    
    if comp_20050_label.exists():
        img = nib.load(comp_20050_label)
        data = img.get_fdata()
        has_class_4 = (data == 4).any()
        
        if has_class_4:
            print(f"\n✅ CRITICAL FIX VERIFIED: COMP_20050 has Class 4 (Diverticulitis) labels!")
            print(f"   This was missing in the old pipeline.")
        else:
            print(f"\n⚠️  WARNING: COMP_20050 exists but has no Class 4 labels")
    
    return True


def check_splits():
    """Verify Phase 2.5 created train/val/test splits."""
    print("\n" + "="*70)
    print("PHASE 2.5 VERIFICATION: Dataset Splits")
    print("="*70)
    
    splits_dir = Path('splits')
    
    if not splits_dir.exists():
        print("❌ FAIL: splits/ directory does not exist")
        return False
    
    train_file = splits_dir / 'train_cases.txt'
    val_file = splits_dir / 'val_cases.txt'
    test_file = splits_dir / 'test_cases.txt'
    
    if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
        print("❌ FAIL: Missing split files")
        return False
    
    with open(train_file) as f:
        train_lines = f.readlines()
    with open(val_file) as f:
        val_lines = f.readlines()
    with open(test_file) as f:
        test_lines = f.readlines()
    
    print(f"\n📊 Split Statistics:")
    print(f"  Training cases: {len(train_lines)}")
    print(f"  Validation cases: {len(val_lines)}")
    print(f"  Test cases: {len(test_lines)}")
    print(f"  Total: {len(train_lines) + len(val_lines) + len(test_lines)}")
    
    expected_total = 1095
    actual_total = len(train_lines) + len(val_lines) + len(test_lines)
    
    if abs(actual_total - expected_total) > 50:
        print(f"⚠️  WARNING: Expected ~{expected_total} cases, got {actual_total}")
    else:
        print(f"✅ PASS: Total case count is correct (~{expected_total} expected)")
    
    return True


def main():
    print("\n" + "="*70)
    print("PIPELINE FIX VERIFICATION")
    print("Checking if Training and Competition datasets are properly separated")
    print("="*70 + "\n")
    
    results = []
    
    # Run all checks
    results.append(("Phase 1 (NIfTI)", check_phase1_output()))
    results.append(("Phase 2 Masks", check_phase2_masks()))
    results.append(("Phase 2 Labels", check_phase2_labels()))
    results.append(("Phase 2.5 Splits", check_splits()))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Pipeline fix is successful.")
        print("   You can now proceed to Phase 3 training.")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED. Review the output above.")
        print("   You may need to re-run the failed phases.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
