"""
PLACEHOLDER ONLY — NO CODE.

Purpose:
- Group DICOM series by case, sort slices, stack into 3D arrays.
- Save volumes as NIfTI (.nii.gz), preserving spacing/origin/direction.

Inputs (intended):
- --dicom_root data_raw/dicom_files
- --out_dir    data_processed/nifti_images

Implementation notes:
- pydicom for reading metadata.
- Sort by InstanceNumber or ImagePositionPatient.
- Save with SimpleITK or nibabel.
"""
