#!/bin/bash
#SBATCH --job-name=Phase2_MedSAM
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=mesh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2              # Request BOTH GPUs
#SBATCH --cpus-per-task=32        # 32 CPU cores
#SBATCH --mem=90G                 # 90GB RAM
#SBATCH --time=60:00:00           # 60 hours (safe buffer)
#SBATCH --oversubscribe

echo "=========================================="
echo "Phase 2: MedSAM Inference (Dual-GPU)"
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
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Check for MedSAM checkpoint
MEDSAM_CKPT="models/medsam_vit_b.pth"
if [ ! -f "$MEDSAM_CKPT" ]; then
    echo "ERROR: MedSAM checkpoint not found at $MEDSAM_CKPT"
    echo "Please download it from: https://github.com/bowang-lab/MedSAM"
    echo "Or use: wget -P models/ <medsam_checkpoint_url>"
    exit 1
fi

echo "MedSAM checkpoint found: $MEDSAM_CKPT"
echo ""

# Display GPU status
echo "GPU Status:"
nvidia-smi
echo ""

# CSV files location - medsam_infer.py will merge both TRAININGDATA.csv and COMPETITIONDATA.csv
CSV_DIR="data_raw/annotations"
echo "Using annotation directory: $CSV_DIR"
echo "  - TRAININGDATA.csv (28,135 annotations)"
echo "  - COMPETITIONDATA.csv (14,315 annotations)"
echo "  - Total: 42,450 annotations"
echo ""

# Step 1: Run MedSAM inference on GPU 0 and GPU 1 in parallel
echo "=========================================="
echo "Step 1: Running MedSAM Inference (Parallel)"
echo "=========================================="
echo "Launching 2 parallel processes (one per GPU)..."

# Launch process for GPU 0 on the first half of the data
echo "Starting GPU 0 process..."
CUDA_VISIBLE_DEVICES=0 python scripts/medsam_infer.py \
    --csv_dir "$CSV_DIR" \
    --dicom_root data_raw/dicom_files \
    --out_root data_processed/medsam_2d_masks \
    --medsam_ckpt "$MEDSAM_CKPT" \
    --gpu_idx 0 \
    --num_gpus 2 \
    > logs/medsam_gpu0_$SLURM_JOB_ID.log 2>&1 &

PID_GPU0=$!
echo "GPU 0 process started with PID: $PID_GPU0"

# Launch process for GPU 1 on the second half of the data
echo "Starting GPU 1 process..."
CUDA_VISIBLE_DEVICES=1 python scripts/medsam_infer.py \
    --csv_dir "$CSV_DIR" \
    --dicom_root data_raw/dicom_files \
    --out_root data_processed/medsam_2d_masks \
    --medsam_ckpt "$MEDSAM_CKPT" \
    --gpu_idx 1 \
    --num_gpus 2 \
    > logs/medsam_gpu1_$SLURM_JOB_ID.log 2>&1 &

PID_GPU1=$!
echo "GPU 1 process started with PID: $PID_GPU1"

echo ""
echo "Both processes launched. Waiting for completion..."
echo "You can monitor progress with:"
echo "  tail -f logs/medsam_gpu0_$SLURM_JOB_ID.log"
echo "  tail -f logs/medsam_gpu1_$SLURM_JOB_ID.log"
echo ""

# Wait for both background processes to finish
wait $PID_GPU0
EXIT_CODE_GPU0=$?
echo "GPU 0 process finished with exit code: $EXIT_CODE_GPU0"

wait $PID_GPU1
EXIT_CODE_GPU1=$?
echo "GPU 1 process finished with exit code: $EXIT_CODE_GPU1"

if [ $EXIT_CODE_GPU0 -ne 0 ] || [ $EXIT_CODE_GPU1 -ne 0 ]; then
    echo "ERROR: One or more MedSAM processes failed!"
    echo "Check logs:"
    echo "  logs/medsam_gpu0_$SLURM_JOB_ID.log"
    echo "  logs/medsam_gpu1_$SLURM_JOB_ID.log"
    exit 1
fi

echo ""
echo "Both MedSAM inference processes completed successfully!"
echo ""

# Step 2: Aggregate 2D masks into 3D labels
echo "=========================================="
echo "Step 2: Aggregating 2D Masks into 3D Labels"
echo "=========================================="
python scripts/aggregate_masks.py \
    --masks2d_root data_processed/medsam_2d_masks \
    --nifti_dir data_processed/nifti_images \
    --out_dir data_processed/nifti_labels_medsam

if [ $? -ne 0 ]; then
    echo "ERROR: Mask aggregation failed!"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo "Phase 2 Complete!"
echo "=========================================="
echo "Finished at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Version data with DVC:"
echo "     dvc add data_processed/nifti_labels_medsam"
echo "     git add data_processed/nifti_labels_medsam.dvc"
echo "     git commit -m 'feat: Complete Phase 2 MedSAM mask generation'"
echo ""
echo "  2. Proceed to Phase 2.5 (Create 3D Training Splits):"
echo "     sbatch slurm_phase2.5_splits.sh"
echo "=========================================="
