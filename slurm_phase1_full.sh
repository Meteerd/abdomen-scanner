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

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate abdomen_scanner

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Step 1: Convert DICOM to NIfTI
echo "=========================================="
echo "Step 1: Converting DICOM to NIfTI"
echo "=========================================="
python scripts/dicom_to_nifti.py \
    --dicom_root data_raw/dicom_files \
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
    --excel_path Temp/Information.xlsx \
    --nifti_dir data_processed/nifti_images \
    --out_dir data_processed/nifti_labels_boxy

if [ $? -ne 0 ]; then
    echo "ERROR: Boxy label generation failed!"
    exit 1
fi
echo ""

# Step 3: Create Dataset Splits
echo "=========================================="
echo "Step 3: Creating Dataset Splits"
echo "=========================================="
python scripts/split_dataset.py \
    --nifti_dir data_processed/nifti_images \
    --train_out splits/train_cases.txt \
    --val_out splits/val_cases.txt \
    --test_out splits/test_cases.txt \
    --train 0.8 \
    --val 0.1 \
    --test 0.1 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset splitting failed!"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo "Finished at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Version data with DVC:"
echo "     dvc add data_processed/nifti_images"
echo "     dvc add data_processed/nifti_labels_boxy"
echo "     git add data_processed/*.dvc splits/*.txt"
echo "     git commit -m 'feat: Complete Phase 1 data processing'"
echo ""
echo "  2. Proceed to Phase 2 (MedSAM inference):"
echo "     sbatch slurm_phase2_medsam.sh"
echo "=========================================="
