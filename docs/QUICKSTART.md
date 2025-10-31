# ⚡ Quick Start Guide - Abdomen Scanner

> **For impatient developers who want to start NOW**

## 🎯 Your Goal
Train a 3D U-Net to segment abdominal emergencies from CT scans using the mesh-hpc cluster.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [x] Tailscale connected
- [x] SSH access to mesh-hpc (`ssh mete@100.116.63.100`)
- [x] Data transferred to cluster (in progress)
- [x] W&B account (optional, for experiment tracking)

---

## 🚀 Three Commands to Start Training

### 1️⃣ Run Phase 1: Process Data (2 hours)
```bash
ssh mete@100.116.63.100
cd /home/mete/abdomen-scanner
sbatch slurm_phase1_full.sh
```

**What it does:** 
- DICOM → NIfTI conversion
- Create boxy labels with **z-axis anatomical validation** (GAP 1 fix)
- Maps 11 radiologist labels → 6 competition classes
- Filters ~5-10% of anatomically invalid annotations
- Split dataset (80/10/10)

**Monitor:** `squeue -u $USER` and `tail -f logs/phase1_*.out`

### 2️⃣ Run Phase 2: Generate Pseudo-Masks (18 hours)
```bash
# After Phase 1 completes
sbatch slurm_phase2_medsam.sh
```

**What it does:** MedSAM inference on both GPUs → high-quality 3D masks  
**Monitor:** `tail -f logs/phase2_*.out`

### 3️⃣ Run Phase 3: Train 3D U-Net (2-7 days)
```bash
# After Phase 2 completes
./train.sh phase3_unet_baseline
```

**What it does:** Multi-GPU training with PyTorch Lightning  
**Monitor:** `tail -f logs/Phase3_Training_*.out`

---

## 📊 Monitor Your Jobs

```bash
# Check job status
squeue -u $USER

# Watch output logs
tail -f logs/*.out

# Cancel a job
scancel <job-id>
```

---

## 🎓 First Time? Read These First

1. **[HPC_CLUSTER_SETUP.md](Tutorials_For_Mete/HPC_CLUSTER_SETUP.md)** - SSH, Tailscale, SLURM basics
2. **[PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)** - Full 3-phase workflow explained
3. **[SLURM_QUICK_REFERENCE.md](Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md)** - Common SLURM commands

---

## 🐛 Common Issues

### "conda: command not found"
```bash
source ~/.bashrc
conda activate abdomen_scanner
```

### "MedSAM checkpoint not found"
```bash
# Download MedSAM checkpoint
wget -P models/ <medsam_checkpoint_url>
```

### Job stays PENDING
- Check: `sinfo` (are GPUs available?)
- AI security team has priority (your job will queue)

---

## 📞 Need Help?

- **Cluster issues:** DM team lead
- **Code bugs:** Check `logs/*.err` files
- **SLURM questions:** Read [mesh-hpc SLURM guide](https://growmesh.notion.site/slurm-job-scheduler)

---

**Ready? SSH into the cluster and run Phase 1!** 🚀
