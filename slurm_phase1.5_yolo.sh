#!/bin/bash
#SBATCH --job-name=Phase1.5_YOLO
#SBATCH --output=logs/phase1.5_yolo_%j.out
#SBATCH --error=logs/phase1.5_yolo_%j.err
#SBATCH --partition=mesh
#SBATCH --gres=gpu:1              # Single GPU sufficient for YOLO
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00           # 8 hours max

echo "=========================================="
echo "Phase 1.5: YOLO Baseline Validation (GAP 2)"
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

# Check GPU
echo "GPU Status:"
nvidia-smi
echo ""

# Step 1: Install YOLO if not already installed
echo "=========================================="
echo "Step 1: Installing YOLOv11 (ultralytics)"
echo "=========================================="
pip install -U ultralytics
echo "YOLOv11 installed"
echo ""

# Step 2: Prepare YOLO dataset (if not already done)
echo "=========================================="
echo "Step 2: Preparing YOLO Dataset"
echo "=========================================="

if [ ! -f "data_processed/yolo_dataset/data.yaml" ]; then
    python scripts/prep_yolo_data.py \
        --dicom_root data/AbdomenDataSet/Training-DataSets \
        --out_root data_processed/yolo_dataset \
        --seed 42
    
    if [ $? -ne 0 ]; then
        echo "ERROR: YOLO data preparation failed!"
        exit 1
    fi
else
    echo "✓ YOLO dataset already prepared"
fi

echo ""

# Step 3: Train YOLOv11
echo "=========================================="
echo "Step 3: Training YOLOv11 Baseline"
echo "=========================================="
echo "Model: yolo11x.pt (largest, most accurate)"
echo "Epochs: 100"
echo "Image size: 640"
echo "Batch size: 16"
echo ""

yolo detect train \
    data=data_processed/yolo_dataset/data.yaml \
    model=yolo11x.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=models/yolo \
    name=baseline_validation \
    patience=20 \
    save=True \
    plots=True \
    val=True

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "YOLO Training Complete!"
    echo "=========================================="
    echo "Finished at: $(date)"
    echo ""
    echo "Results saved to: models/yolo/baseline_validation/"
    echo ""
    echo "Check validation metrics:"
    echo "  mAP@0.5 should be > 0.70 for label validation"
    echo "  If mAP < 0.50, labels may be problematic"
    echo ""
    echo "View results:"
    echo "  cat models/yolo/baseline_validation/results.csv"
    echo "  Open models/yolo/baseline_validation/results.png"
    echo ""
    echo "✅ SUCCESS CRITERIA:"
    echo "   mAP@0.5 > 0.70: Labels validated, proceed to Phase 2"
    echo "   mAP@0.5 = 0.50-0.70: Labels acceptable, consider cleaning"
    echo "   mAP@0.5 < 0.50: Labels problematic, investigate before Phase 2"
    echo ""
    echo "Next step (if mAP > 0.70):"
    echo "  sbatch slurm_phase2_medsam.sh"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "YOLO Training Failed!"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check logs:"
    echo "  logs/phase1.5_yolo_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "Job completed at $(date)"
