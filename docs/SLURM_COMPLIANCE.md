# SLURM Scripts - Compliance Complete ✅

**Date:** November 3, 2025  
**Status:** All scripts comply with mesh-hpc best practices

---

## Changes Applied

### 1. Log File Format (Critical Fix)
**Before:**
```bash
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
```

**After:**
```bash
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
```

**Benefit:** Uses job name (%x) + job ID (%j) for better organization

### 2. Oversubscribe Flag (Performance)
**Added to all scripts:**
```bash
#SBATCH --oversubscribe
```

**Benefit:** Better queue efficiency, allows flexible resource allocation

### 3. GPU Selection (Phase 1.5, 2, 3A, 3B)
**Added intelligent GPU selection:**
```bash
select_optimal_gpu() {
    # Selects GPU with lowest memory + utilization score
    # Uses nvidia-smi metrics
}

export CUDA_VISIBLE_DEVICES=$selected_gpu
```

**Benefit:** Automatic load balancing across GPUs

### 4. GPU Status Monitoring
**All GPU jobs now print:**
```bash
nvidia-smi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
```

**Benefit:** Easy debugging, track which GPU is assigned

---

## Compliance Checklist

| Check | Status | Scripts |
|-------|--------|---------|
| %x_%j log format | ✅ | All 6 scripts |
| --oversubscribe flag | ✅ | All 6 scripts |
| logs/ directory | ✅ | Created |
| GPU selection | ✅ | Phases 1.5, 2, 3A, 3B |
| nvidia-smi status | ✅ | All GPU jobs |
| CUDA_VISIBLE_DEVICES | ✅ | All GPU jobs |
| Error handling | ✅ | All scripts |
| Bash syntax | ✅ | Validated |

---

## Script Summary

### Phase 1: Data Preparation (CPU-only)
- **File:** `slurm_phase1_full.sh`
- **Resources:** 64 CPUs, 100GB RAM, 2 hours
- **Logs:** `./logs/Phase1_DataPrep_<jobid>.out/err`

### Phase 1.5: YOLO Validation (Single GPU)
- **File:** `slurm_phase1.5_yolo.sh`
- **Resources:** 1 GPU, 16 CPUs, 32GB RAM, 8 hours
- **GPU:** Auto-selected or specify with REQUESTED_GPU
- **Logs:** `./logs/Phase1.5_YOLO_<jobid>.out/err`

### Phase 2: MedSAM Inference (Dual GPU)
- **File:** `slurm_phase2_medsam.sh`
- **Resources:** 2 GPUs, 32 CPUs, 90GB RAM, 48 hours
- **GPU:** Uses both GPUs (0,1)
- **Logs:** `./logs/Phase2_MedSAM_<jobid>.out/err`

### Phase 2.5: Create Splits (CPU-only)
- **File:** `slurm_phase2.5_splits.sh`
- **Resources:** 8 CPUs, 16GB RAM, 30 minutes
- **Logs:** `./logs/Phase2.5_Splits_<jobid>.out/err`

### Phase 3.A: AMOS Pre-training (Dual GPU)
- **File:** `slurm_phase3a_pretrain.sh`
- **Resources:** 2 GPUs, 32 CPUs, 128GB RAM, 72 hours
- **GPU:** Uses both GPUs (0,1)
- **Logs:** `./logs/Phase3A_Pretrain_<jobid>.out/err`

### Phase 3.B: Pathology Fine-tuning (Dual GPU)
- **File:** `slurm_phase3b_finetune.sh`
- **Resources:** 2 GPUs, 32 CPUs, 128GB RAM, 168 hours
- **GPU:** Uses both GPUs (0,1)
- **Logs:** `./logs/Phase3B_Finetune_<jobid>.out/err`

---

## Usage Examples

### Standard Submission
```bash
sbatch slurm_phase1_full.sh
```

### Monitor Job
```bash
squeue -u $USER
tail -f ./logs/Phase1_DataPrep_*.out
```

### Custom GPU Selection (Phase 1.5)
```bash
sbatch --export=ALL,REQUESTED_GPU=0 slurm_phase1.5_yolo.sh
```

### Cancel Job
```bash
scancel <job_id>
```

---

## Best Practices from Documentation

✅ **Implemented:**
- mkdir -p ./logs before job submission
- %x_%j format for log files
- --oversubscribe for flexibility
- Intelligent GPU selection for multi-GPU systems
- nvidia-smi status printing
- CUDA_VISIBLE_DEVICES configuration
- Error handling with exit codes
- Descriptive job names

✅ **Project-Specific Additions:**
- Full path venv activation
- Python/PyTorch version verification
- Comprehensive error messages
- Next-step instructions in success messages

---

## Ready to Launch

**All systems compliant. Ready for Phase 1 execution:**
```bash
sbatch slurm_phase1_full.sh
```

**Logs will appear in:**
```
./logs/Phase1_DataPrep_<jobid>.out
./logs/Phase1_DataPrep_<jobid>.err
```
