#!/bin/bash
# train.sh - Wrapper script for submitting SLURM training jobs on mesh-hpc
# Based on: https://github.com/JanosMozer/server-job-scheduler
#
# Usage: ./train.sh <experiment_name> [gpu_id]
#
# Examples:
#   ./train.sh phase1_dicom_conversion         # Auto-select GPU
#   ./train.sh phase2_medsam_inference 0       # Force GPU 0
#   ./train.sh phase3_unet_epoch300 1          # Force GPU 1

if [ -z "$1" ]; then
    echo "=========================================="
    echo "Abdomen Scanner - SLURM Job Submission"
    echo "=========================================="
    echo ""
    echo "Usage: ./train.sh <experiment_name> [gpu_id]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name  - Descriptive name for your experiment"
    echo "  gpu_id           - (Optional) Force specific GPU (0 or 1)"
    echo ""
    echo "Examples:"
    echo "  ./train.sh phase1_dicom_conversion"
    echo "  ./train.sh phase2_medsam_inference 0"
    echo "  ./train.sh phase3_unet_epoch300_lr0002 1"
    echo ""
    echo "Available SLURM scripts:"
    echo "  - slurm_test_gpu.sh           (Test GPU access)"
    echo "  - slurm_dicom_to_nifti.sh     (Phase 1: DICOM conversion)"
    echo "  - slurm_make_boxy_labels.sh   (Phase 1: Boxy labels)"
    echo "  - slurm_medsam_infer.sh       (Phase 2: MedSAM inference)"
    echo "  - slurm_train_unet.sh         (Phase 3: 3D U-Net training)"
    echo ""
    exit 1
fi

EXPERIMENT_NAME=$1
GPU_ID=$2

# Determine which SLURM script to use based on experiment name
if [[ "$EXPERIMENT_NAME" == *"test"* ]] || [[ "$EXPERIMENT_NAME" == *"gpu"* ]]; then
    SLURM_SCRIPT="slurm_test_gpu.sh"
elif [[ "$EXPERIMENT_NAME" == *"dicom"* ]] || [[ "$EXPERIMENT_NAME" == *"nifti"* ]]; then
    SLURM_SCRIPT="slurm_dicom_to_nifti.sh"
elif [[ "$EXPERIMENT_NAME" == *"boxy"* ]] || [[ "$EXPERIMENT_NAME" == *"label"* ]]; then
    SLURM_SCRIPT="slurm_make_boxy_labels.sh"
elif [[ "$EXPERIMENT_NAME" == *"medsam"* ]] || [[ "$EXPERIMENT_NAME" == *"phase2"* ]]; then
    SLURM_SCRIPT="slurm_medsam_infer.sh"
else
    # Default to U-Net training
    SLURM_SCRIPT="slurm_train_unet.sh"
fi

echo "=========================================="
echo "Submitting Job to mesh-hpc Cluster"
echo "=========================================="
echo "Experiment:    $EXPERIMENT_NAME"
echo "SLURM Script:  $SLURM_SCRIPT"
if [ -n "$GPU_ID" ]; then
    echo "Requested GPU: $GPU_ID (forced)"
else
    echo "GPU Selection: Automatic (least loaded)"
fi
echo "=========================================="
echo ""

# Submit job to SLURM
sbatch --job-name="$EXPERIMENT_NAME" \
       --export=ALL,REQUESTED_GPU="$GPU_ID" \
       "$SLURM_SCRIPT"

# Show current queue status
echo ""
echo "Job submitted! Current queue status:"
echo "---------------------------------------------------"
squeue -u $USER

echo ""
echo "To monitor your job:"
echo "  squeue -u \$USER              # Check job status"
echo "  tail -f logs/*.out           # Watch output log"
echo "  watch -n 1 nvidia-smi        # Monitor GPU usage"
echo "  scancel <JobID>              # Cancel a job"
echo ""
echo "Cluster Resources (mesh-hpc):"
echo "  - 2x NVIDIA RTX PRO 6000 (96GB VRAM each)"
echo "  - 128 CPU threads (Dual Xeon 6530P)"
echo "  - 128GB System RAM"
echo "  - CUDA 13.0 support"
echo ""
