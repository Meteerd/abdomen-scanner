"""
PLACEHOLDER ONLY — NO CODE.

Purpose:
- Build 3D integer label volumes by drawing 2D CSV bboxes per slice.

Inputs (intended):
- --master_csv data_raw/annotations/MASTER.csv
- --nifti_dir  data_processed/nifti_images
- --out_dir    data_processed/nifti_labels_boxy

Implementation notes:
- Match slice indices; draw rectangles with class IDs.
- Copy NIfTI header/affine from image volumes.
"""
