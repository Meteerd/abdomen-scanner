#!/usr/bin/env python3
"""
Phase 4: Clinical Inference Pipeline

Takes a trained model and a new DICOM series, outputs:
1. 3D segmentation mask (NIfTI format)
2. JSON report with classification and volume calculation

Usage:
    python scripts/inference.py \
        --dicom_in data_raw/new_patient/series_001 \
        --out_dir results/patient_001 \
        --model_ckpt models/phase3b_finetune/best.ckpt \
        --config configs/config.yaml
"""

import argparse
import os
import json
import SimpleITK as sitk
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, ToTensord
)
from train_monai import SegmentationModel


def load_dicom_series_as_nifti(dicom_dir: Path, out_nifti_path: Path) -> bool:
    """
    Convert DICOM series to single NIfTI file.
    
    Args:
        dicom_dir: Directory containing DICOM files
        out_nifti_path: Output path for NIfTI file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        
        if not dicom_names:
            print(f"Error: No DICOM series found in {dicom_dir}")
            return False
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        sitk.WriteImage(image, str(out_nifti_path))
        print(f"DICOM series converted to {out_nifti_path}")
        return True
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        return False


def get_inference_transforms(config: Dict) -> Compose:
    """
    Get MONAI transforms for inference (no augmentation).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MONAI Compose transform
    """
    preproc = config.get('preproc', {})
    target_spacing = preproc.get('target_spacing', [1.5, 1.5, 2.0])
    hu_min = preproc.get('hu_window_min', -175)
    hu_max = preproc.get('hu_window_max', 250)
    
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ])


def calculate_volumes(mask: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Calculate volume of each class in mm^3.
    
    Args:
        mask: 3D segmentation mask (H, W, D)
        voxel_spacing: Voxel spacing (x, y, z) in mm
        
    Returns:
        Dictionary mapping class names to volumes in mm^3
    """
    # Class mapping from docs/DATA_FORMAT.md
    class_map = {
        1: "AAA/AAD",
        2: "Pancreatitis",
        3: "Cholecystitis",
        4: "Kidney/Ureteral Stones",
        5: "Diverticulitis",
        6: "Appendicitis"
    }
    
    voxel_volume = np.prod(voxel_spacing)  # e.g., 1.5 * 1.5 * 2.0 = 4.5 mm^3
    volumes = {}
    
    for class_idx, class_name in class_map.items():
        voxel_count = np.sum(mask == class_idx)
        volume_mm3 = float(voxel_count * voxel_volume)
        volumes[class_name] = volume_mm3
        
    return volumes


def main():
    parser = argparse.ArgumentParser(description="Run 3D Segmentation Inference")
    parser.add_argument(
        "--dicom_in",
        type=str,
        required=True,
        help="Path to directory with new DICOM series"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save output mask and JSON"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to trained .ckpt model from Phase 3.B"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to Phase 3.B config file"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Phase 4: Clinical Inference Pipeline")
    print("=" * 60)
    print(f"Input DICOM: {args.dicom_in}")
    print(f"Output directory: {out_dir}")
    print(f"Model checkpoint: {args.model_ckpt}")
    print(f"Device: {device}")
    print()
    
    # Load config
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading model from {args.model_ckpt}...")
    model = SegmentationModel.load_from_checkpoint(args.model_ckpt, config=config)
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    print()
    
    # Prepare input image
    print(f"Processing input DICOM from {args.dicom_in}...")
    temp_nifti_path = out_dir / "input_image.nii.gz"
    
    if not load_dicom_series_as_nifti(Path(args.dicom_in), temp_nifti_path):
        return 1
    
    # Create data dict for MONAI loader
    data_dict = {"image": str(temp_nifti_path)}
    
    # Apply transforms
    print("Applying preprocessing transforms...")
    transforms = get_inference_transforms(config)
    transformed_data = transforms(data_dict)
    
    # Get voxel spacing from transformed image
    voxel_spacing = transformed_data['image_meta_dict']['pixdim'][1:4].numpy()
    print(f"Voxel spacing: {voxel_spacing} mm")
    print()
    
    # Add batch dimension and send to device
    input_tensor = transformed_data["image"].unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run inference
    print("Running sliding window inference...")
    patch_size = config.get('inference', {}).get('roi_size', [192, 192, 160])
    sw_batch_size = config.get('inference', {}).get('sw_batch_size', 2)
    overlap = config.get('inference', {}).get('overlap', 0.5)
    
    print(f"  Patch size: {patch_size}")
    print(f"  Batch size: {sw_batch_size}")
    print(f"  Overlap: {overlap}")
    
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    
    # Post-process: argmax to get segmentation mask
    mask_tensor = torch.argmax(output, dim=1).detach().cpu().numpy().squeeze()
    print(f"Output mask shape: {mask_tensor.shape}")
    print()
    
    # Save segmentation mask
    print("Saving segmentation mask...")
    mask_img = sitk.GetImageFromArray(mask_tensor.astype(np.uint8))
    
    # Load original nifti to get spacing/origin/direction
    original_nifti = sitk.ReadImage(str(temp_nifti_path))
    mask_img.CopyInformation(original_nifti)
    
    out_mask_path = out_dir / "segmentation_mask.nii.gz"
    sitk.WriteImage(mask_img, str(out_mask_path))
    print(f"✓ Segmentation mask saved to: {out_mask_path}")
    print()
    
    # Post-processing: Classification & Volume
    print("Performing post-processing...")
    volumes = calculate_volumes(mask_tensor, voxel_spacing)
    
    # Simple classification (any voxels present = positive)
    classification = {
        "AAA/AAD": bool(volumes["AAA/AAD"] > 0),
        "Pancreatitis": bool(volumes["Pancreatitis"] > 0),
        "Cholecystitis": bool(volumes["Cholecystitis"] > 0),
        "Kidney/Ureteral Stones": bool(volumes["Kidney/Ureteral Stones"] > 0),
        "Diverticulitis": bool(volumes["Diverticulitis"] > 0),
        "Appendicitis": bool(volumes["Appendicitis"] > 0),
    }
    
    # Create results report
    results = {
        "case_id": str(args.dicom_in),
        "model_checkpoint": str(args.model_ckpt),
        "voxel_spacing_mm": voxel_spacing.tolist(),
        "classification_report": classification,
        "volume_report_mm3": volumes,
        "output_mask": str(out_mask_path)
    }
    
    # Save JSON report
    out_json_path = out_dir / "report.json"
    with open(out_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Inference report saved to: {out_json_path}")
    print()
    
    # Display results
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print()
    print("Classification Results:")
    for pathology, is_positive in classification.items():
        status = "POSITIVE" if is_positive else "NEGATIVE"
        print(f"  {pathology:25s}: {status}")
    
    print()
    print("Volume Measurements (mm³):")
    for pathology, volume in volumes.items():
        if volume > 0:
            print(f"  {pathology:25s}: {volume:,.1f} mm³")
    
    print()
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
