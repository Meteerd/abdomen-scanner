#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=mesh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --oversubscribe

# ============================================================================
# GPU Test Job - First Time Cluster Setup
# ============================================================================
# This is a simple test to verify your cluster environment is working.
# Run this FIRST before submitting longer jobs.
#
# Usage: sbatch slurm_test_gpu.sh
# ============================================================================

echo "=========================================="
echo "GPU Test Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo ""

# Load conda
source ~/.bashrc
conda activate abdomen_scanner

# Test 1: Check nvidia-smi
echo "Test 1: GPU Detection with nvidia-smi"
echo "--------------------------------------"
nvidia-smi
echo ""

# Test 2: PyTorch CUDA
echo "Test 2: PyTorch CUDA Access"
echo "--------------------------------------"
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test tensor creation on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✓ Successfully created and multiplied tensors on GPU")
else:
    print("✗ CUDA not available!")
EOF

echo ""

# Test 3: MONAI
echo "Test 3: MONAI Import"
echo "--------------------------------------"
python << EOF
import monai
print(f"MONAI version: {monai.__version__}")
print(f"✓ MONAI successfully imported")
EOF

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Finished at: $(date)"
echo "=========================================="
