"""
Phase 1 - Step 2: Generate 3D Boxy Labels from CSV Annotations with Z-Axis Validation

Purpose:
- Build 3D integer label volumes by drawing 2D CSV bboxes per slice
- Uses Boundary Slice annotations to validate 3D z-axis extent per anatomical region
- Maps 11 radiologist labels to 6 competition classes (per Koç et al. 2024 Table 2)
- Creates "weak" bounding-box labels for initial training or MedSAM prompting

Critical Changes (GAP 1 Fix):
- Added 11→6 class mapping from paper
- Added z-axis boundary validation using "Boundary Slice" annotations
- Bounding boxes are only drawn within their valid anatomical z-range

Usage:
    python scripts/make_boxy_labels.py --excel_path Temp/Information.xlsx --nifti_dir data_processed/nifti_images --out_dir data_processed/nifti_labels_boxy
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm


# ============================================================================
# CLASS MAPPING: 11 Radiologist Labels → 6 Competition Classes
# Based on Koç et al. (2024) Table 2
# ============================================================================
CLASS_MAPPING = {
    'background': 0,
    
    # Competition Class 1: AAA/AAD (Abdominal Aortic Aneurysm/Dissection)
    'Abdominal aortic aneurysm': 1,
    'Abdominal aortic dissection': 1,
    
    # Competition Class 2: Acute Pancreatitis
    'Compatible with acute pancreatitis': 2,
    
    # Competition Class 3: Cholecystitis
    'Compatible with acute cholecystitis': 3,
    'Gallbladder stone': 3,
    
    # Competition Class 4: Kidney/Ureteral Stones
    'Kidney stone': 4,
    'ureteral stone': 4,
    
    # Competition Class 5: Diverticulitis (RARE - only 54 total annotations)
    'Compatible with acute diverticulitis': 5,
    'Calcified diverticulum': 5,
    
    # Competition Class 6: Appendicitis
    'Compatible with acute appendicitis': 6,
}

# ============================================================================
# ANATOMICAL BOUNDARY MAPPING
# Maps pathology classes to their anatomical organ boundaries for z-axis validation
# ============================================================================
PATHOLOGY_TO_ANATOMY = {
    # Class 1: AAA/AAD → Abdominal Aorta
    'Abdominal aortic aneurysm': 'Abdominal Aorta',
    'Abdominal aortic dissection': 'Abdominal Aorta',
    
    # Class 2: Acute Pancreatitis → Pancreas
    'Compatible with acute pancreatitis': 'Pancreas',
    
    # Class 3: Cholecystitis → Gall bladder
    'Compatible with acute cholecystitis': 'Gall bladder',
    'Gallbladder stone': 'Gall bladder',
    
    # Class 4: Kidney/Ureteral Stones → Kidney-Bladder
    'Kidney stone': 'Kidney-Bladder',
    'ureteral stone': 'Kidney-Bladder',
    
    # Class 5: Diverticulitis → Colon
    'Compatible with acute diverticulitis': 'Colon',
    'Calcified diverticulum': 'Colon',
    
    # Class 6: Appendicitis → appendix
    'Compatible with acute appendicitis': 'appendix',
}


def parse_excel_annotations(excel_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and parse CSV annotation files (TRAININGDATA.csv + COMPETITIONDATA.csv).
    
    Args:
        excel_path: Path to annotations directory or legacy Excel file (for backward compatibility)
        
    Returns:
        Tuple of (bounding_boxes_df, boundary_slices_df)
    """
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
        combined_df = pd.concat([train_df, comp_df], ignore_index=True)
        print(f"Merged {len(train_df)} training + {len(comp_df)} competition annotations")
        
    else:
        # Fallback to Excel file
        print(f"CSV files not found, loading from Excel: {excel_path}...")
        train_df = pd.read_excel(excel_path, sheet_name='TRAIININGDATA')
        combined_df = train_df
    
    # Split into bounding boxes and boundary slices
    bbox_df = combined_df[combined_df['Type'] == 'Bounding Box'].copy()
    boundary_df = combined_df[combined_df['Type'] == 'Boundary Slice'].copy()
    
    print(f"Found {len(bbox_df)} bounding boxes")
    print(f"Found {len(boundary_df)} boundary slices")
    print(f"Total cases: {combined_df['Case Number'].nunique()}")
    
    # Show class distribution
    print("\nBounding Box class distribution:")
    for cls_name, count in bbox_df['Class'].value_counts().items():
        comp_class = CLASS_MAPPING.get(cls_name, 0)
        print(f"  {cls_name:50s} → Class {comp_class} (n={count})")
    
    return bbox_df, boundary_df


def extract_bbox_coords(data_str: str) -> Tuple[int, int, int, int]:
    """
    Parse bounding box coordinates from Data column format: "xmin,ymin-xmax,ymax"
    
    Args:
        data_str: String like "251,290-262,302"
        
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    # Split by '-' to get min and max points
    min_str, max_str = data_str.split('-')
    x_min, y_min = map(int, min_str.split(','))
    x_max, y_max = map(int, max_str.split(','))
    
    return x_min, y_min, x_max, y_max


def get_anatomical_boundaries(case_number: int, boundary_df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """
    Extract anatomical z-axis boundaries for a case from Boundary Slice annotations.
    
    Each anatomical structure (e.g., 'Pancreas', 'Kidney-Bladder') has 2 boundary slices
    marking the start and end of the 3D region.
    
    Args:
        case_number: Case ID
        boundary_df: DataFrame with Boundary Slice annotations
        
    Returns:
        Dict mapping anatomical class name to (z_start, z_end) slice range
    """
    case_boundaries = boundary_df[boundary_df['Case Number'] == case_number]
    
    boundaries = {}
    
    for anatomy_class in case_boundaries['Class'].unique():
        # Get both boundary slices for this anatomy
        class_slices = case_boundaries[case_boundaries['Class'] == anatomy_class]['Image Id'].values
        
        if len(class_slices) == 2:
            z_start = min(class_slices)
            z_end = max(class_slices)
            boundaries[anatomy_class] = (z_start, z_end)
        elif len(class_slices) == 1:
            # Only one boundary - assume it's valid for that slice only
            boundaries[anatomy_class] = (class_slices[0], class_slices[0])
        else:
            print(f"Warning: Case {case_number} has {len(class_slices)} boundaries for {anatomy_class}")
    
    return boundaries


def is_bbox_valid_for_slice(pathology_class: str, image_id: int, boundaries: Dict[str, Tuple[int, int]]) -> bool:
    """
    Check if a bounding box annotation is within the valid anatomical z-range.
    
    This implements the critical GAP 1 fix: validates that each bounding box is only
    drawn within the anatomically valid z-axis extent defined by Boundary Slice annotations.
    
    Args:
        pathology_class: Pathology class name (e.g., 'Kidney stone')
        image_id: Slice Image ID from annotation
        boundaries: Dict of anatomical boundaries from get_anatomical_boundaries()
        
    Returns:
        True if bbox should be drawn on this slice, False otherwise
    """
    # Get the anatomical structure for this pathology
    anatomy = PATHOLOGY_TO_ANATOMY.get(pathology_class)
    
    if anatomy is None:
        print(f"Warning: Unknown pathology class '{pathology_class}' - no anatomy mapping")
        return True  # Draw anyway if we don't know the anatomy
    
    # Check if this anatomy has boundary annotations
    if anatomy not in boundaries:
        print(f"Warning: No boundary annotations for anatomy '{anatomy}' (pathology: {pathology_class})")
        return True  # Draw anyway if no boundaries defined
    
    # Check if image_id is within the anatomical z-range
    z_start, z_end = boundaries[anatomy]
    
    if image_id < z_start or image_id > z_end:
        return False  # Outside valid z-range - skip this bbox
    
    return True


def draw_bbox_on_slice(label_slice: np.ndarray, bbox: Tuple[int, int, int, int], class_id: int) -> np.ndarray:
    """
    Draw a bounding box on a 2D label slice.
    
    Args:
        label_slice: 2D numpy array (H, W)
        bbox: (x_min, y_min, x_max, y_max) in pixel coordinates
        class_id: Integer class ID to fill the box with
        
    Returns:
        Modified label_slice with bbox drawn
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Clip to image bounds
    h, w = label_slice.shape
    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w - 1))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h - 1))
    
    # Draw filled rectangle
    label_slice[y_min:y_max+1, x_min:x_max+1] = class_id
    
    return label_slice


def create_boxy_label_volume(
    nifti_img: nib.Nifti1Image,
    bbox_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    case_number: int,
    image_id_to_slice_idx: Dict[int, int]
) -> Tuple[nib.Nifti1Image, Dict[str, int]]:
    """
    Create a 3D label volume with boxy annotations + z-axis boundary validation.
    
    This implements GAP 1 fix: validates each bounding box against anatomical boundaries
    before drawing, ensuring labels are only applied within valid z-ranges.
    
    Args:
        nifti_img: Original NIfTI image (for shape and header)
        bbox_df: DataFrame with Bounding Box annotations
        boundary_df: DataFrame with Boundary Slice annotations
        case_number: Case Number (e.g., 20001)
        image_id_to_slice_idx: Mapping from Image ID to slice index in NIfTI volume
        
    Returns:
        Tuple of (label_volume, validation_stats)
    """
    # Get image shape
    img_data = nifti_img.get_fdata()
    shape = img_data.shape  # (H, W, D)
    
    # Initialize empty label volume
    label_volume = np.zeros(shape, dtype=np.uint8)
    
    # Get anatomical boundaries for this case
    boundaries = get_anatomical_boundaries(case_number, boundary_df)
    
    if not boundaries:
        print(f"Warning: Case {case_number} has no boundary slice annotations")
    
    # Filter annotations for this case
    case_annotations = bbox_df[bbox_df['Case Number'] == case_number]
    
    # Validation statistics
    stats = {
        'total_boxes': len(case_annotations),
        'drawn_boxes': 0,
        'skipped_out_of_bounds': 0,
        'skipped_unknown_class': 0,
        'skipped_outside_anatomy': 0,
    }
    
    if len(case_annotations) == 0:
        print(f"Warning: No bounding box annotations found for case {case_number}")
        return nib.Nifti1Image(label_volume, nifti_img.affine, nifti_img.header), stats
    
    # Draw each bounding box (with z-axis validation)
    for _, row in case_annotations.iterrows():
        image_id = int(row['Image Id'])
        pathology_class = row['Class']
        data_str = row['Data']
        
        # Get class ID from mapping
        class_id = CLASS_MAPPING.get(pathology_class)
        
        if class_id is None:
            stats['skipped_unknown_class'] += 1
            continue
        
        # Z-AXIS VALIDATION: Check if bbox is within anatomical boundaries
        if not is_bbox_valid_for_slice(pathology_class, image_id, boundaries):
            stats['skipped_outside_anatomy'] += 1
            continue
        
        # Map Image ID to slice index in NIfTI volume
        if image_id not in image_id_to_slice_idx:
            print(f"Warning: Image ID {image_id} not found in slice mapping for case {case_number}")
            stats['skipped_out_of_bounds'] += 1
            continue
        
        slice_idx = image_id_to_slice_idx[image_id]
        
        # Check slice index is valid
        if slice_idx < 0 or slice_idx >= shape[2]:
            stats['skipped_out_of_bounds'] += 1
            continue
        
        # Parse bbox coordinates
        try:
            x_min, y_min, x_max, y_max = extract_bbox_coords(data_str)
        except Exception as e:
            print(f"Warning: Failed to parse bbox data '{data_str}': {e}")
            continue
        
        # Draw on the appropriate slice
        label_volume[:, :, slice_idx] = draw_bbox_on_slice(
            label_volume[:, :, slice_idx],
            (x_min, y_min, x_max, y_max),
            class_id
        )
        
        stats['drawn_boxes'] += 1
    
    # Create NIfTI image with same header as original
    label_img = nib.Nifti1Image(label_volume, nifti_img.affine, nifti_img.header)
    
    return label_img, stats


def build_image_id_mapping(nifti_dir: Path) -> Dict[int, Dict[int, int]]:
    """
    Build a mapping from Case Number → (Image ID → Slice Index).
    
    This assumes Image IDs correspond to slice order in the NIfTI volume.
    For proper mapping, we need the DICOM metadata or sequence info.
    
    Args:
        nifti_dir: Directory with NIfTI files
        
    Returns:
        Nested dict: {case_number: {image_id: slice_idx}}
    """
    # For now, we make a simple assumption:
    # Image IDs are sequential and correspond to slice indices
    # This should be improved with actual DICOM metadata
    
    print("\nWarning: Using simple Image ID → Slice Index mapping")
    print("For production, integrate with DICOM metadata from dicom_to_nifti.py")
    
    return {}


def generate_boxy_labels(excel_path: Path, nifti_dir: Path, out_dir: Path):
    """
    Generate boxy label volumes for all cases with z-axis validation.
    
    Args:
        excel_path: Path to Information.xlsx with TRAININGDATA/COMPETITIONDATA
        nifti_dir: Directory containing NIfTI image volumes
        out_dir: Output directory for label volumes
    """
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    bbox_df, boundary_df = parse_excel_annotations(excel_path)
    
    # Get list of NIfTI files
    nifti_files = sorted(nifti_dir.glob("*.nii.gz"))
    print(f"\nFound {len(nifti_files)} NIfTI volumes in {nifti_dir}")
    
    # Get unique case numbers from annotations
    annotated_cases = set(bbox_df['Case Number'].unique())
    print(f"Found {len(annotated_cases)} cases with annotations")
    
    # Process each case
    success_count = 0
    skipped_count = 0
    total_stats = {
        'total_boxes': 0,
        'drawn_boxes': 0,
        'skipped_out_of_bounds': 0,
        'skipped_unknown_class': 0,
        'skipped_outside_anatomy': 0,
    }
    
    for nifti_path in tqdm(nifti_files, desc="Generating boxy labels"):
        # Extract case number from filename
        # Format: "1_2_840_10009_1_2_3_10001_20001.nii.gz" → 20001
        # Or: "case_20001.nii.gz" → 20001
        case_str = nifti_path.stem.replace('.nii', '')
        
        # Try to extract numeric case number
        try:
            if 'case_' in case_str:
                # Format: case_20001
                case_number = int(case_str.split('case_')[1])
            elif '_' in case_str:
                # Format: DICOM_UID_20001 - take last part
                case_number = int(case_str.split('_')[-1])
            else:
                # Plain number
                case_number = int(case_str)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse case number from filename: {nifti_path.name}")
            skipped_count += 1
            continue
        
        # Check if this case has annotations
        if case_number not in annotated_cases:
            skipped_count += 1
            continue
        
        # Check if output already exists
        out_path = out_dir / f"case_{case_number}.nii.gz"
        if out_path.exists():
            skipped_count += 1
            continue
        
        try:
            # Load NIfTI image
            nifti_img = nib.load(str(nifti_path))
            
            # Build simple Image ID → Slice Index mapping
            # Assumes Image IDs are sequential starting from some base
            shape = nifti_img.shape
            case_image_ids = bbox_df[bbox_df['Case Number'] == case_number]['Image Id'].unique()
            
            if len(case_image_ids) > 0:
                min_image_id = case_image_ids.min()
                # Map: image_id → (image_id - min_image_id)
                image_id_to_slice_idx = {
                    img_id: img_id - min_image_id
                    for img_id in case_image_ids
                }
            else:
                image_id_to_slice_idx = {}
            
            # Create label volume with z-axis validation
            label_img, stats = create_boxy_label_volume(
                nifti_img, bbox_df, boundary_df, case_number, image_id_to_slice_idx
            )
            
            # Save label volume
            nib.save(label_img, str(out_path))
            
            # Update statistics
            for key in total_stats:
                total_stats[key] += stats[key]
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing case {case_number}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Boxy label generation complete!")
    print(f"Successfully processed: {success_count} cases")
    print(f"Skipped (no annotations): {skipped_count} cases")
    print(f"\nZ-Axis Validation Statistics:")
    print(f"  Total bounding boxes in annotations: {total_stats['total_boxes']}")
    print(f"  ✓ Drawn in label volumes: {total_stats['drawn_boxes']}")
    print(f"  ✗ Skipped (outside anatomy z-range): {total_stats['skipped_outside_anatomy']}")
    print(f"  ✗ Skipped (out of volume bounds): {total_stats['skipped_out_of_bounds']}")
    print(f"  ✗ Skipped (unknown class): {total_stats['skipped_unknown_class']}")
    
    if total_stats['skipped_outside_anatomy'] > 0:
        pct_filtered = 100.0 * total_stats['skipped_outside_anatomy'] / total_stats['total_boxes']
        print(f"\n  → {pct_filtered:.1f}% of bboxes filtered by anatomical z-validation")
        print(f"  → This prevents training on anatomically invalid annotations!")
    
    print(f"\nOutput directory: {out_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D boxy labels from Excel annotations with z-axis validation"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="Temp/Information.xlsx",
        help="Path to Information.xlsx with TRAININGDATA/COMPETITIONDATA sheets"
    )
    parser.add_argument(
        "--nifti_dir",
        type=str,
        default="data_processed/nifti_images",
        help="Directory containing NIfTI image volumes"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data_processed/nifti_labels_boxy",
        help="Output directory for boxy label volumes"
    )
    
    args = parser.parse_args()
    
    excel_path = Path(args.excel_path)
    nifti_dir = Path(args.nifti_dir)
    out_dir = Path(args.out_dir)
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not nifti_dir.exists():
        raise FileNotFoundError(f"NIfTI directory not found: {nifti_dir}")
    
    print(f"{'='*70}")
    print(f"GAP 1 FIX: Boxy Label Generation with Z-Axis Validation")
    print(f"{'='*70}")
    print(f"Annotations: {excel_path}")
    print(f"Images: {nifti_dir}")
    print(f"Output: {out_dir}")
    print(f"\nCritical improvements:")
    print(f"  ✓ Using 11→6 radiologist label mapping from Koç et al. (2024)")
    print(f"  ✓ Validating bboxes against anatomical boundary slices")
    print(f"  ✓ Preventing anatomically invalid labels in 3D volumes")
    print(f"{'='*70}\n")
    
    generate_boxy_labels(excel_path, nifti_dir, out_dir)


if __name__ == "__main__":
    main()
