"""
PLACEHOLDER ONLY — NO CODE.

Purpose:
- For each bbox row in MASTER.csv, run MedSAM on the 2D DICOM slice to create a binary mask.

Inputs (intended):
- --master_csv data_raw/annotations/MASTER.csv
- --dicom_root data_raw/dicom_files
- --out_root   data_processed/medsam_2d_masks
- --medsam_ckpt /path/to/medsam_vit_b.pth

Implementation notes:
- Ensure consistent orientation/resolution across saved masks.
- Save masks under out_root/<CASE_ID>/<SLICE>.png
"""
