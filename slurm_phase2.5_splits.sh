#!/bin/bash
#SBATCH --job-name=Phase2.5_Splits
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=mesh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:30:00           # 2.5 hours (+2h buffer)
#SBATCH --oversubscribe

echo "=========================================="
echo "Phase 2.5: Creating 3D Training Splits"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo ""

# Create logs directory
mkdir -p logs

# Navigate to project directory
cd /home/mete/abdomen-scanner || exit 1

# Activate environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# This script creates JSON-format split files for 3D training
# It reads MedSAM labels which were generated in Phase 2
# Split files contain: {"image": "/path/img.nii.gz", "label": "/path/lbl.nii.gz"}

echo "=========================================="
echo "Creating JSON split files for 3D training"
echo "=========================================="
echo "Input:"
echo "  - Images: data_processed/nifti_images/"
echo "  - Labels: data_processed/nifti_labels_medsam/"
echo ""
echo "Output:"
echo "  - splits/train_cases.txt"
echo "  - splits/val_cases.txt"
echo "  - splits/test_cases.txt"
echo ""
echo "Ratios: 80% train / 10% val / 10% test"
echo "Random seed: 42 (same as YOLO split for consistency)"
echo ""

python scripts/split_dataset.py \
    --nifti_dir data_processed/nifti_images \
    --label_dir data_processed/nifti_labels_medsam \
    --train_out splits/train_cases.txt \
    --val_out splits/val_cases.txt \
    --test_out splits/test_cases.txt \
    --train 0.8 \
    --val 0.1 \
    --test 0.1 \
    --seed 42

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase 2.5 Complete!"
    echo "=========================================="
    echo "Finished at: $(date)"
    echo ""
    echo "Split files created:"
    echo "  - splits/train_cases.txt"
    echo "  - splits/val_cases.txt"
    echo "  - splits/test_cases.txt"
    echo ""
    echo "Verify splits:"
    echo "  head -3 splits/train_cases.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Manual QC for rare classes:"
    echo "     python scripts/sample_rare_classes.py"
    echo ""
    echo "  2. Upload AMOS 2022 dataset:"
    echo "     ./verify_amos_upload.sh"
    echo ""
    echo "  3. Prepare AMOS splits:"
    echo "     python scripts/prepare_amos_dataset.py"
    echo ""
    echo "  4. Phase 3.A (AMOS Pre-training):"
    echo "     sbatch slurm_phase3a_pretrain.sh"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Phase 2.5 Failed!"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check logs:"
    echo "  logs/phase2.5_splits_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "Job completed at $(date)"
