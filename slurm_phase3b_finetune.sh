#!/bin/bash
#SBATCH --job-name=Phase3B_Finetune
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=mesh
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=170:00:00          # 170 hours (+2h buffer)
#SBATCH --oversubscribe

echo "=========================================="
echo "Phase 3.B: Fine-tuning on Pathology (GAP 3)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo ""

PRETRAIN_CKPT=${1:-""}

if [ -z "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pre-trained checkpoint path required!"
    echo ""
    echo "Usage:"
    echo "  sbatch slurm_phase3b_finetune.sh <path_to_pretrained_checkpoint>"
    echo ""
    echo "Example:"
    echo "  BEST_CKPT=\$(ls -t models/phase3a_amos_pretrain/best_model-*.ckpt | head -1)"
    echo "  sbatch slurm_phase3b_finetune.sh \$BEST_CKPT"
    echo ""
    exit 1
fi

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Checkpoint file not found: $PRETRAIN_CKPT"
    exit 1
fi

echo "Pre-trained checkpoint: $PRETRAIN_CKPT"
echo ""

# Create logs directory
mkdir -p logs

# Navigate to project directory
cd /home/mete/abdomen-scanner || exit 1

# Activate environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify Python environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Check GPU status
echo "GPU Status:"
nvidia-smi
echo ""

# Set CUDA devices (for dual GPU setup)
export CUDA_VISIBLE_DEVICES=0,1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
source venv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPUs
echo "GPU Status:"
nvidia-smi
echo ""

# Set environment variables for distributed training
export MASTER_PORT=12356
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Verify MedSAM pseudo-labels exist (from Phase 2)
echo "=========================================="
echo "Verifying MedSAM Pseudo-labels (Phase 2)"
echo "=========================================="

MEDSAM_LABELS="data_processed/nifti_labels_medsam"

if [ ! -d "$MEDSAM_LABELS" ]; then
    echo "ERROR: MedSAM pseudo-labels not found!"
    echo ""
    echo "Please run Phase 2 first:"
    echo "  sbatch slurm_phase2_medsam.sh"
    echo ""
    echo "Expected directory: $MEDSAM_LABELS"
    exit 1
fi

LABEL_COUNT=$(find "$MEDSAM_LABELS" -name "*.nii.gz" | wc -l)
echo "✓ MedSAM labels found: $LABEL_COUNT"

if [ "$LABEL_COUNT" -lt 500 ]; then
    echo "WARNING: Expected ~735 cases, found $LABEL_COUNT labels"
    echo "Proceeding anyway..."
fi

echo ""

# Start fine-tuning
echo "=========================================="
echo "Starting Fine-tuning"
echo "=========================================="
echo "Dataset: TR_ABDOMEN_RAD_EMERGENCY (pathology labels)"
echo "Pre-trained on: AMOS 2022 (anatomical structures)"
echo "Epochs: 300 (7 days)"
echo "Batch size: 2 per GPU (4 total with DDP)"
echo "Strategy: Transfer Learning (AMOS → Pathology)"
echo ""
echo "Purpose: Leverage anatomical knowledge to detect rare pathology"
echo "         Classes 4-5 have only 54 examples each"
echo ""

python scripts/train_monai.py \
    --config configs/config.yaml \
    --experiment_name phase3b_pathology_finetune \
    --load_weights "$PRETRAIN_CKPT"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Fine-tuning Complete!"
    echo "=========================================="
    echo "Finished at: $(date)"
    echo ""
    echo "Fine-tuned model saved to:"
    echo "  models/phase3b_pathology_finetune/best_model-*.ckpt"
    echo ""
    echo "Model trained on:"
    echo "  Pre-training: AMOS 2022 (anatomical structures)"
    echo "  Fine-tuning:  Pathology detection (Classes 0-5)"
    echo ""
    echo "Critical Validation for Rare Classes (GAP 3):"
    echo "  Class 4 (Diverticulitis): 54 training examples"
    echo "  Class 5 (Appendicitis):   54 training examples"
    echo ""
    echo "Expected improvements from transfer learning:"
    echo "  - Better generalization for Classes 4-5"
    echo "  - Reduced overfitting on rare classes"
    echo "  - Improved anatomical context understanding"
    echo ""
    echo "Next step: Phase 4 - Inference & Evaluation"
    echo "  python scripts/infer_unet.py --checkpoint models/phase3b_pathology_finetune/best_model-*.ckpt"
    echo ""
    echo "Transfer learning applied successfully"
    echo "Model ready for inference on test set"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Fine-tuning Failed!"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if pre-trained checkpoint is valid:"
    echo "     python -c \"import torch; torch.load('$PRETRAIN_CKPT')\""
    echo ""
    echo "  2. Check if MedSAM labels are correct:"
    echo "     ls -lh $MEDSAM_LABELS | head"
    echo ""
    echo "Check logs:"
    echo "  logs/phase3b_finetune_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo "Job completed at $(date)"
