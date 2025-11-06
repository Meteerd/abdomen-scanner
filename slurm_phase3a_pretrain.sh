#!/bin/bash
#SBATCH --job-name=Phase3A_Pretrain
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=mesh
#SBATCH --gres=gpu:2              # 2x RTX 6000 GPUs
#SBATCH --cpus-per-task=32        # 16 workers per GPU
#SBATCH --mem=128G                # Full node memory
#SBATCH --time=90:00:00           # 90 hours (+2h buffer)
#SBATCH --oversubscribe

echo "=========================================="
echo "Phase 3.A: Pre-training on AMOS 2022 (GAP 3)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo ""

# Create logs directory
mkdir -p logs

# Activate virtual environment
echo "Activating virtual environment..."
source /home/mete/abdomen-scanner/venv/bin/activate

# Verify Python environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Check GPU status
echo "GPU Status:"
nvidia-smi
echo ""

# Set CUDA devices (for dual GPU setup)
export CUDA_VISIBLE_DEVICES=0,1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Set environment variables for distributed training
export MASTER_PORT=12355
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Verifying AMOS 2022 Dataset"
echo "=========================================="

AMOS_DIR="data/AbdomenDataSet/AMOS-DataSet"
META_CSV="data/meta.csv"

# Verify AMOS 2022 dataset exists
if [ ! -d "$AMOS_DIR" ]; then
    echo "ERROR: AMOS dataset directory not found!"
    echo ""
    echo "Expected location: $AMOS_DIR"
    echo ""
    echo "Please upload the AMOS dataset using WinSCP or scp to:"
    echo "  /home/mete/abdomen-scanner/data/AbdomenDataSet/AMOS-DataSet/"
    echo ""
    echo "The directory should contain folders like:"
    echo "  s0000/, s0001/, s0002/, ..."
    echo ""
    exit 1
fi

if [ ! -f "$META_CSV" ]; then
    echo "ERROR: meta.csv not found!"
    echo ""
    echo "Expected location: $META_CSV"
    echo ""
    exit 1
fi

# Count cases in AMOS directory
NUM_CASES=$(find "$AMOS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "AMOS dataset found: $NUM_CASES case folders"
echo "Meta CSV found: $META_CSV"
echo ""

# Aggregate organ segmentations into multi-class labels
echo "=========================================="
echo "Aggregating AMOS Organ Segmentations"
echo "=========================================="
echo "Combining 117 individual organ files into single multi-class labels..."
echo "This creates label.nii.gz (16 classes) for each case."
echo ""

python scripts/aggregate_amos_labels.py \
    --amos_dir "$AMOS_DIR" \
    --num_workers 16

if [ $? -ne 0 ]; then
    echo "ERROR: AMOS label aggregation failed!"
    exit 1
fi
echo ""

# Prepare dataset (create split files from meta.csv)
echo "=========================================="
echo "Preparing AMOS Dataset"
echo "=========================================="

if [ ! -f "splits/amos_train_cases.txt" ]; then
    echo "Split files not found. Generating from meta.csv..."
    python scripts/prepare_amos_dataset.py \
        --amos_dir "$AMOS_DIR" \
        --meta_csv "$META_CSV" \
        --output_dir splits \
        --inventory_output data/amos_inventory.csv
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to prepare AMOS dataset"
        exit 1
    fi
    echo ""
else
    echo "Split files already exist. Skipping preparation."
    echo "  - splits/amos_train_cases.txt"
    echo "  - splits/amos_val_cases.txt"
    echo "  - splits/amos_test_cases.txt"
    echo ""
fi

# Display split statistics
echo "Split Statistics:"
echo "  Train cases: $(wc -l < splits/amos_train_cases.txt)"
echo "  Val cases:   $(wc -l < splits/amos_val_cases.txt)"
echo "  Test cases:  $(wc -l < splits/amos_test_cases.txt)"
echo ""

# Verify config file
CONFIG_FILE="configs/config_pretrain.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting Pre-training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Duration: 3 days (72 hours)"
echo ""

# Run distributed training
python scripts/train_monai.py \
    --config "$CONFIG_FILE" \
    --experiment_name phase3a_amos_pretrain

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Phase 3.A Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Pre-training successful!"
    echo ""
    echo "Next steps:"
    echo "1. Find best checkpoint:"
    echo "   BEST_CKPT=\$(ls -t models/phase3a_amos_pretrain/best_model-*.ckpt | head -1)"
    echo ""
    echo "2. Run Phase 3.B (fine-tuning on pathology):"
    echo "   sbatch slurm_phase3b_finetune.sh \$BEST_CKPT"
    echo ""
else
    echo ""
    echo "Pre-training failed. Check logs for errors."
    echo ""
fi

exit $EXIT_CODE
