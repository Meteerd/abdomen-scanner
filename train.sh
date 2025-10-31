#!/bin/bash
# train.sh - Simple wrapper to submit smart training jobs
# Usage: ./train.sh <experiment_name> [gpu_id]

if [ -z "$1" ]; then
    echo "Usage: ./train.sh <experiment_name> [gpu_id]"
    exit 1
fi

EXPERIMENT_NAME=$1
GPU_ID=$2

# Submit job to slurm
# The job name is set to the experiment name for easy identification
# We pass experiment name and optional GPU ID as environment variables to the slurm script
sbatch --job-name="$EXPERIMENT_NAME" \
       --export=ALL,WANDB_RUN_NAME="$EXPERIMENT_NAME",REQUESTED_GPU="$GPU_ID" \
       train_smart.slurm

echo "Job '$EXPERIMENT_NAME' submitted. Current queue status:"
squeue