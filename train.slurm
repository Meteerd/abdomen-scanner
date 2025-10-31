#!/bin/bash
#SBATCH --job-name=Phase3_Training
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=mesh
#SBATCH --gres=gpu:2              # Request both GPUs for DDP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        # 32 CPU cores for data loading
#SBATCH --mem=90G                 # 90GB RAM
#SBATCH --oversubscribe

echo "=========================================="
echo "Phase 3: 3D U-Net Training (DDP)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Navigate to project directory
cd /home/mete/abdomen-scanner || exit 1

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate abdomen_scanner

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Display GPU status
echo "GPU Status:"
nvidia-smi
echo ""

# Set environment variables for training
export PYTHONPATH="${PYTHONPATH}:/home/mete/abdomen-scanner"
export WANDB_PROJECT="${WANDB_PROJECT:-abdomen-segmentation}"
export WANDB_NAME="${WANDB_RUN_NAME:-slurm-job-$SLURM_JOB_ID}"
export WANDB_TAGS="gpu_dual,${WANDB_RUN_NAME}"

# Print summary
echo "--- Training Configuration ---"
echo "Experiment Name: ${WANDB_RUN_NAME}"
echo "W&B Project: $WANDB_PROJECT"
echo "Config: configs/config.yaml"
echo "GPUs: 2 (DDP strategy)"
echo "----------------------------"
echo ""

# Run the training script
echo "Starting training at $(date)..."
python scripts/train_monai.py \
    --config configs/config.yaml \
    --experiment_name "${WANDB_RUN_NAME:-phase3_unet_baseline}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Finished at: $(date)"
    echo ""
    echo "Model saved to: models/${WANDB_RUN_NAME}"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate on test set"
    echo "  2. Version model with DVC:"
    echo "     dvc add models/${WANDB_RUN_NAME}"
    echo "  3. Deploy for inference"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code: $EXIT_CODE"
    echo "=========================================="
    echo "Check logs:"
    echo "  logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "Job completed at $(date)"