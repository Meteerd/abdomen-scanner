# Quick Start: The 6-Phase Workflow

This project is **NOT** a single-command process. It is a 6-phase pipeline that requires sequential execution and manual checkpoints.

## Prerequisites

1. All data is on the cluster (`data_raw/dicom_files/`, `Temp/Information.xlsx`)
2. **AMOS 2022 Dataset** is downloaded and extracted to `data_raw/amos22/`
3. Python venv is activated: `source /home/mete/abdomen-scanner/venv/bin/activate`
4. W&B API key is set (optional): `export WANDB_API_KEY=...`

---

## Execution Workflow

### Step 1: Run Phase 1 (Data Curation)
```bash
sbatch slurm_phase1_full.sh
```
**Output:** `data_processed/nifti_images/` & `data_processed/nifti_labels_boxy/`  
**Time:** ~2 hours

### Step 2: Run Phase 1.5 (YOLO Label Validation)
```bash
sbatch slurm_phase1.5_yolo.sh
```
**Output:** `models/yolo/baseline_validation/results.csv`  
**Time:** ~8 hours

**CHECKPOINT:** Manually check `results.csv`. Do not proceed if mAP@0.5 < 0.70.

### Step 3: Run Phase 2 (Pseudo-Mask Generation)
```bash
sbatch slurm_phase2_medsam.sh
```
**Output:** `data_processed/nifti_labels_medsam/`  
**Time:** ~12 hours

### Step 4: Run Phase 2.5 (Manual QC)
```bash
python scripts/sample_rare_classes.py --samples_per_class 10
```
**Output:** `phase2_qc_checklist.txt`  
**Time:** 30 minutes (Manual)

**CHECKPOINT:** Manually open the NIfTI files listed in the checklist using ITK-SNAP or 3D Slicer. Do not proceed if >30% of rare class masks are low quality.

### Step 5: Run Phase 3.A (Anatomy Pre-training)
```bash
sbatch slurm_phase3a_pretrain.sh
```
**Output:** `models/phase3a_amos_pretrain/best_model-*.ckpt`  
**Time:** ~3 days

**CHECKPOINT:** Check W&B or logs. Do not proceed if validation Dice on AMOS is < 0.75.

### Step 6: Run Phase 3.B (Pathology Fine-tuning)
```bash
# Find the best checkpoint from Step 5
BEST_CKPT=$(ls -t models/phase3a_amos_pretrain/best_model-*.ckpt | head -1)
echo "Using checkpoint: $BEST_CKPT"

# Run fine-tuning
sbatch slurm_phase3b_finetune.sh $BEST_CKPT
```
**Output:** Final model in `models/phase3b_finetune/`  
**Time:** ~7 days

**Goal:** Final model with high Dice scores, especially for rare classes.

---

## Monitoring Progress

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
# Phase 1
tail -f logs/phase1_full_<job_id>.out

# Phase 1.5
tail -f logs/phase1.5_yolo_<job_id>.out

# Phase 2
tail -f logs/phase2_medsam_<job_id>.out

# Phase 3.A
tail -f logs/phase3a_pretrain_<job_id>.out

# Phase 3.B
tail -f logs/phase3b_finetune_<job_id>.out
```

### Check W&B Dashboard
Visit https://wandb.ai/ to see training curves, validation metrics, and sample predictions.

---

## Common Issues

### "Virtual environment not activated"
```bash
source /home/mete/abdomen-scanner/venv/bin/activate
```

### "MedSAM checkpoint not found"
```bash
# Download MedSAM checkpoint
wget -P models/ <medsam_checkpoint_url>
```

### Job stays PENDING
```bash
# Check partition availability
sinfo

# Check your job details
scontrol show job <job_id>
```

---

## Next Steps

After Phase 3.B completes:
1. **Evaluate on test set** - Run inference on held-out test cases
2. **Analyze results** - Calculate per-class Dice scores, visualize predictions
3. **Iterate** - If rare class performance is low, adjust class weights or sampling

For detailed information on each phase, see [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).

---

## Need Help?

1. [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - Full workflow details
2. [HPC_CLUSTER_SETUP.md](../Tutorials_For_Mete/HPC_CLUSTER_SETUP.md) - Cluster access
3. [SLURM_QUICK_REFERENCE.md](../Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md) - SLURM commands
