#!/bin/bash
#SBATCH --job-name=Phase1_DataPrep
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --partition=mesh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64        # Use 64 CPU cores for parallel processing
#SBATCH --mem=100G                # Request 100GB RAM
#SBATCH --time=02:00:00           # 2 hours should be sufficient

echo "=========================================="
echo "Phase 1: Full Data Processing Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /home/mete/abdomen-scanner || exit 1

# Activate Python virtual environment
echo "Activating virtual environment..."
source /home/mete/abdomen-scanner/venv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Step 1: Convert DICOM to NIfTI
echo "=========================================="
echo "Step 1: Converting DICOM to NIfTI"
echo "=========================================="
python scripts/dicom_to_nifti.py \
    --dicom_root data/AbdomenDataSet/Training-DataSets \
    --out_dir data_processed/nifti_images

if [ $? -ne 0 ]; then
    echo "ERROR: DICOM to NIfTI conversion failed!"
    exit 1
fi
echo ""

# Step 2: Generate 3D Boxy Labels with Z-Axis Validation (GAP 1 Fix)
echo "=========================================="
echo "Step 2: Generating 3D Boxy Labels (GAP 1)"
echo "=========================================="
python scripts/make_boxy_labels.py \
    --nifti_dir data_processed/nifti_images \
    --out_dir data_processed/nifti_labels_boxy

if [ $? -ne 0 ]; then
    echo "ERROR: Boxy label generation failed!"
    exit 1
fi
echo ""

# Phase 1 Complete
echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo "Finished at: $(date)"
echo ""
echo "Data processed:"
echo "  - NIfTI images: data_processed/nifti_images/"
echo "  - Boxy labels: data_processed/nifti_labels_boxy/"
echo ""
echo "Next steps:"
echo "  1. Verify outputs:"
echo "     ls -lh data_processed/nifti_images/ | head"
echo ""
echo "  2. Proceed to Phase 1.5 (YOLO label validation):"
echo "     sbatch slurm_phase1.5_yolo.sh"
echo ""
echo "Note: Dataset splits for 3D training will be created"
echo "      after Phase 2 (MedSAM) completes."
echo "=========================================="
