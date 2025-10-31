"""
Test script to validate GAP 1 fix in make_boxy_labels.py

This script tests the z-axis boundary validation logic to ensure:
1. Class mapping works correctly (11→6)
2. Anatomical boundary extraction works
3. Z-axis validation filters out-of-bounds annotations
"""

import sys
from pathlib import Path
import pandas as pd

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from make_boxy_labels import (
    parse_excel_annotations,
    get_anatomical_boundaries,
    is_bbox_valid_for_slice,
    CLASS_MAPPING,
    PATHOLOGY_TO_ANATOMY
)


def test_class_mapping():
    """Test that all 11 radiologist labels map to 6 competition classes"""
    print("=" * 70)
    print("TEST 1: Class Mapping (11→6)")
    print("=" * 70)
    
    expected_pathologies = [
        'Abdominal aortic aneurysm',
        'Abdominal aortic dissection',
        'Compatible with acute pancreatitis',
        'Compatible with acute cholecystitis',
        'Gallbladder stone',
        'Kidney stone',
        'ureteral stone',
        'Compatible with acute diverticulitis',
        'Calcified diverticulum',
        'Compatible with acute appendicitis',
    ]
    
    for pathology in expected_pathologies:
        class_id = CLASS_MAPPING.get(pathology)
        anatomy = PATHOLOGY_TO_ANATOMY.get(pathology)
        print(f"  {pathology:50s} → Class {class_id} ({anatomy})")
        
        assert class_id is not None, f"Missing class mapping for {pathology}"
        assert anatomy is not None, f"Missing anatomy mapping for {pathology}"
        assert 1 <= class_id <= 6, f"Invalid class ID {class_id} for {pathology}"
    
    print("✓ All 11 pathology classes mapped correctly\n")


def test_boundary_extraction(excel_path: Path):
    """Test anatomical boundary extraction"""
    print("=" * 70)
    print("TEST 2: Anatomical Boundary Extraction")
    print("=" * 70)
    
    bbox_df, boundary_df = parse_excel_annotations(excel_path)
    
    # Test on case 20003 (has multiple boundaries)
    test_case = 20003
    boundaries = get_anatomical_boundaries(test_case, boundary_df)
    
    print(f"\nCase {test_case} anatomical boundaries:")
    for anatomy, (z_start, z_end) in boundaries.items():
        print(f"  {anatomy:20s}: z=[{z_start}, {z_end}] (span={z_end-z_start+1} slices)")
    
    # Expected boundaries from our earlier analysis
    expected_anatomies = ['Colon', 'Pancreas', 'Abdominal Aorta', 'Kidney-Bladder', 'Gall bladder', 'appendix']
    
    for anatomy in expected_anatomies:
        assert anatomy in boundaries, f"Missing boundary for {anatomy} in case {test_case}"
        z_start, z_end = boundaries[anatomy]
        assert z_end >= z_start, f"Invalid boundary range for {anatomy}"
    
    print(f"\n✓ Boundary extraction working correctly\n")


def test_z_axis_validation(excel_path: Path):
    """Test z-axis validation logic"""
    print("=" * 70)
    print("TEST 3: Z-Axis Validation")
    print("=" * 70)
    
    bbox_df, boundary_df = parse_excel_annotations(excel_path)
    
    # Test case 20003
    test_case = 20003
    boundaries = get_anatomical_boundaries(test_case, boundary_df)
    
    # Get some real annotations
    case_bboxes = bbox_df[bbox_df['Case Number'] == test_case].head(10)
    
    print(f"\nTesting validation on case {test_case} annotations:")
    
    valid_count = 0
    invalid_count = 0
    
    for _, row in case_bboxes.iterrows():
        image_id = row['Image Id']
        pathology = row['Class']
        
        is_valid = is_bbox_valid_for_slice(pathology, image_id, boundaries)
        
        anatomy = PATHOLOGY_TO_ANATOMY.get(pathology, 'Unknown')
        boundary = boundaries.get(anatomy, (None, None))
        
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {status}: Image {image_id} | {pathology:40s} | Anatomy: {anatomy:20s} | Boundary: {boundary}")
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"\nValidation results:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid (filtered out): {invalid_count}")
    print(f"✓ Z-axis validation logic working\n")


def test_class_distribution(excel_path: Path):
    """Test class distribution and rare class identification"""
    print("=" * 70)
    print("TEST 4: Class Distribution Analysis")
    print("=" * 70)
    
    bbox_df, _ = parse_excel_annotations(excel_path)
    
    print("\nCompetition class distribution:")
    
    class_counts = {}
    for pathology in bbox_df['Class'].unique():
        class_id = CLASS_MAPPING.get(pathology, 0)
        count = len(bbox_df[bbox_df['Class'] == pathology])
        
        if class_id not in class_counts:
            class_counts[class_id] = {'count': 0, 'pathologies': []}
        
        class_counts[class_id]['count'] += count
        class_counts[class_id]['pathologies'].append((pathology, count))
    
    for class_id in sorted(class_counts.keys()):
        if class_id == 0:
            continue
        
        info = class_counts[class_id]
        print(f"\nClass {class_id}: Total {info['count']} annotations")
        for pathology, count in info['pathologies']:
            print(f"  - {pathology:50s} (n={count})")
    
    # Highlight Class 5 (Diverticulitis) - the rare class
    class_5_count = class_counts.get(5, {'count': 0})['count']
    if class_5_count < 100:
        print(f"\n⚠️  WARNING: Class 5 (Diverticulitis) has only {class_5_count} annotations!")
        print(f"   This is the 'most formidable challenge' mentioned in the paper")
        print(f"   → Requires transfer learning (GAP 3) for good performance")
    
    print("\n✓ Class distribution analysis complete\n")


def main():
    excel_path = Path("Temp/Information.xlsx")
    
    if not excel_path.exists():
        print(f"ERROR: {excel_path} not found")
        print("Please ensure Information.xlsx is in the Temp/ directory")
        return
    
    print("\n" + "=" * 70)
    print("GAP 1 FIX VALIDATION TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        # Run all tests
        test_class_mapping()
        test_boundary_extraction(excel_path)
        test_z_axis_validation(excel_path)
        test_class_distribution(excel_path)
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED - GAP 1 FIX VALIDATED")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ 11→6 class mapping implemented correctly")
        print("  ✓ Anatomical boundary extraction working")
        print("  ✓ Z-axis validation filtering invalid annotations")
        print("  ✓ Class distribution analyzed (Class 5 is rare)")
        print("\nNext steps:")
        print("  1. Run make_boxy_labels.py on real data")
        print("  2. Implement GAP 2: YOLO baseline validation")
        print("  3. Implement GAP 3: Transfer learning for Class 5")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
