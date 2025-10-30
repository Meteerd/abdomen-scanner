#!/bin/bash
#SBATCH --job-name=dicom_to_nifti
#SBATCH --output=logs/dicom_%j.out
#SBATCH --error=logs/dicom_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================================
# Phase 1: DICOM to NIfTI Conversion
# ============================================================================
# This script converts raw DICOM slices into 3D NIfTI volumes.
# Does NOT require GPU - runs on CPU nodes.
#
# Usage: sbatch slurm_dicom_to_nifti.sh
# ============================================================================

echo "=========================================="
echo "DICOM to NIfTI Conversion - Phase 1"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo ""

# Load conda environment
source ~/.bashrc
conda activate abdomen_scanner

# Verify environment
echo "Python version:"
python --version
echo ""

# Run conversion script
echo "Starting DICOM to NIfTI conversion..."
python scripts/dicom_to_nifti.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Conversion completed successfully!"
    echo "Output: data_processed/nifti_images/"
else
    echo ""
    echo "✗ Conversion failed! Check error log."
    exit 1
fi

echo ""
echo "Finished at: $(date)"
echo "=========================================="
