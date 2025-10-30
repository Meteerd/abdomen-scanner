#!/bin/bash
#SBATCH --job-name=boxy_labels
#SBATCH --output=logs/boxy_%j.out
#SBATCH --error=logs/boxy_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================================
# Phase 1: Create Boxy 3D Labels
# ============================================================================
# This script creates 3D label volumes by drawing bounding boxes from CSV
# annotations onto the correct slices.
# Does NOT require GPU - runs on CPU nodes.
#
# Usage: sbatch slurm_make_boxy_labels.sh
# ============================================================================

echo "=========================================="
echo "Boxy Label Generation - Phase 1"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo ""

# Load conda environment
source ~/.bashrc
conda activate abdomen_scanner

# Run boxy label generation
echo "Creating 3D boxy labels from CSV annotations..."
python scripts/make_boxy_labels.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Boxy labels created successfully!"
    echo "Output: data_processed/nifti_labels_boxy/"
else
    echo ""
    echo "✗ Label generation failed! Check error log."
    exit 1
fi

echo ""
echo "Finished at: $(date)"
echo "=========================================="
