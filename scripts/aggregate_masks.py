"""
PLACEHOLDER ONLY — NO CODE.

Purpose:
- Aggregate 2D MedSAM masks into 3D NIfTI label volumes, aligned with the 3D images.

Inputs (intended):
- --masks2d_root data_processed/medsam_2d_masks
- --nifti_dir    data_processed/nifti_images
- --out_dir      data_processed/nifti_labels_medsam

Implementation notes:
- Resolve overlaps if multi-class (priority rules or argmax).
- Preserve spacing/origin/direction.
"""
