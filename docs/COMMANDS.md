# ⚡ Command Cheat Sheet - Abdomen Scanner

> **Quick reference for common operations**

---

## 🚀 Running Phases

```bash
# Phase 1: Process Data (2 hours)
sbatch slurm_phase1_full.sh

# Phase 2: MedSAM Inference (18 hours)
sbatch slurm_phase2_medsam.sh

# Phase 3: Train Model (2-7 days)
./train.sh phase3_unet_baseline
```

---

## 👀 Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output (live updates)
tail -f logs/phase1_*.out
tail -f logs/phase2_*.out
tail -f logs/Phase3_Training_*.out

# Check errors
tail -f logs/phase1_*.err
tail -f logs/phase2_*.err
tail -f logs/Phase3_Training_*.err

# GPU status
nvidia-smi

# Watch GPU usage (updates every 1 second)
watch -n 1 nvidia-smi
```

---

## 🛑 Job Control

```bash
# Cancel specific job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER

# Get detailed job info
scontrol show job <job-id>

# Check completed jobs
sacct
```

---

## 📁 File Operations

```bash
# Upload file to cluster
scp local_file.py mete@100.116.63.100:/home/mete/abdomen-scanner/scripts/

# Upload directory
scp -r local_dir/ mete@100.116.63.100:/home/mete/abdomen-scanner/

# Download file from cluster
scp mete@100.116.63.100:/home/mete/abdomen-scanner/models/best_model.pth ./

# Download directory
scp -r mete@100.116.63.100:/home/mete/abdomen-scanner/logs/ ./
```

---

## 🔧 Environment

```bash
# Activate conda environment
conda activate abdomen_scanner

# Check Python
which python
python --version

# Check PyTorch + CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check MONAI
python -c "import monai; print(f'MONAI: {monai.__version__}')"
```

---

## 📊 Data Versioning (DVC)

```bash
# Add processed data to DVC
dvc add data_processed/nifti_images
dvc add data_processed/nifti_labels_medsam

# Commit DVC metadata
git add data_processed/*.dvc .gitignore
git commit -m "feat: Version processed data"

# Push to remote storage (if configured)
dvc push
```

---

## 🔍 Quick Checks

```bash
# Count NIfTI files
ls data_processed/nifti_images/*.nii.gz | wc -l

# Check dataset splits
wc -l splits/*.txt

# Disk usage
du -sh data_processed/*

# Check SLURM queue
sinfo

# Your recent jobs
sacct -u $USER --starttime $(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## 🧪 Testing Scripts Locally

```bash
# Validate GAP 1 fix (z-axis boundary validation)
python scripts/test_gap1_fix.py

# Test Phase 1 scripts
python scripts/dicom_to_nifti.py --help
python scripts/make_boxy_labels.py --excel_path Temp/Information.xlsx --help
python scripts/split_dataset.py --help

# Test Phase 2 scripts
python scripts/medsam_infer.py --help
python scripts/aggregate_masks.py --help

# Test Phase 3 training
python scripts/train_monai.py --help
```

---

## 📝 Git Operations

```bash
# Check status
git status

# Add and commit
git add scripts/*.py
git commit -m "feat: Update training script"

# Push to remote
git push origin main

# Pull latest
git pull origin main

# View commit history
git log --oneline -10
```

---

## 🆘 Emergency

```bash
# Cancel all your jobs immediately
scancel -u $USER

# Check if cluster is down
ping 100.116.63.100

# Check SSH connection
ssh -v mete@100.116.63.100

# Kill hanging processes (on cluster)
pkill -u $USER python
```

---

## 📋 Project Structure Quick Check

```bash
# Verify all key files exist
cd /home/mete/abdomen-scanner
ls -l slurm_phase*.sh train.slurm train.sh
ls -l scripts/*.py
ls -l configs/config.yaml
ls -l data_raw/annotations/*.csv
```

---

## 🎯 W&B (Experiment Tracking)

```bash
# Set API key (once)
export WANDB_API_KEY=<your_key>

# Login
wandb login

# Check status
wandb status

# View runs
wandb runs
```

---

## 🚨 Common Issues

```bash
# If conda not found
source ~/.bashrc

# If GPU not visible
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1

# If Python module not found
conda activate abdomen_scanner
pip install <missing_module>

# If disk full
du -sh * | sort -h
```

---

**💡 Tip:** Bookmark this file for quick reference during development!
