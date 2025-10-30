#!/bin/bash
#SBATCH --job-name=medsam_infer
#SBATCH --output=logs/medsam_%j.out
#SBATCH --error=logs/medsam_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=mesh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --oversubscribe

# ============================================================================
# Phase 2: MedSAM Inference for Pseudo-Mask Generation
# ============================================================================
# This script uses MedSAM to generate high-quality segmentation masks
# from bounding box prompts. This is the core innovation of the project!
# REQUIRES GPU.
#
# Usage: sbatch slurm_medsam_infer.sh
# ============================================================================

echo "=========================================="
echo "MedSAM Pseudo-Mask Generation - Phase 2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo ""

# Load conda environment
source ~/.bashrc
conda activate abdomen_scanner

# Check GPU availability
echo "Checking GPU access..."
nvidia-smi
echo ""

# Verify PyTorch can see GPU
echo "PyTorch CUDA check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run MedSAM inference
echo "Starting MedSAM inference..."
echo "This will process all cases and generate precise segmentation masks."
echo ""

python scripts/medsam_infer.py \
    --input-dir data_processed/nifti_images \
    --label-dir data_processed/nifti_labels_boxy \
    --output-dir data_processed/nifti_labels_medsam \
    --batch-size 4

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ MedSAM inference completed successfully!"
    echo "Output: data_processed/nifti_labels_medsam/"
    echo ""
    echo "Next step: Run slurm_train_unet.sh for Phase 3 training"
else
    echo ""
    echo "✗ MedSAM inference failed! Check error log."
    exit 1
fi

echo ""
echo "Finished at: $(date)"
echo "=========================================="
