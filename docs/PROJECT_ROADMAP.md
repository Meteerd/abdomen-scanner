# 🚀 Optimized Roadmap for Abdominal AI Segmentation

> **Status:** Phase 1 in progress (data transfer to HPC cluster)  
> **Last Updated:** October 31, 2025  
> **Hardware:** mesh-hpc cluster (2x RTX PRO 6000, 128 CPUs, 128GB RAM, 6-month access)

---

## 📋 Project Overview

Build a **weakly-supervised 3D segmentation pipeline** for abdominal emergencies from CT scans:
- **Input:** DICOM CT slices + 2D bounding box CSV annotations
- **Output:** Trained 3D U-Net for pixel-perfect multi-class segmentation
- **Key Innovation:** MedSAM transforms weak bounding boxes → high-quality pseudo-masks

**Target Pathologies (6 Competition Classes):**
1. **AAA/AAD:** Abdominal aortic aneurysm/dissection (n=9,783)
2. **Acute Pancreatitis** (n=6,923)
3. **Cholecystitis:** Acute cholecystitis/Gallbladder stone (n=6,265)
4. **Kidney/Ureteral Stones** (n=1,405)
5. **Diverticulitis** (n=54) ⚠️ **RARE CLASS**
6. **Appendicitis** (n=54) ⚠️ **RARE CLASS**

**Data Source:** `Temp/Information.xlsx` (TRAIININGDATA sheet - note 3 i's in sheet name)

---

## 🎯 Three-Phase Roadmap

### Phase 1: High-Performance Data Curation on HPC Cluster (Weeks 1-2)
**Core Decision:** ALL data processing happens on the cluster. Your local machine is for code development only.

**Bottleneck Solved:** Multi-hour processing → minutes (128 CPU cores)

#### Step 1.1: Centralize Raw Data on Cluster ✅ IN PROGRESS
```bash
# From Windows machine (one-time transfer)
scp -r "D:\AbdomenDataSet" mete@100.116.63.100:/home/mete/abdomen-scanner/data_raw/dicom_files/

# Upload annotations (Excel format)
scp Temp/Information.xlsx mete@100.116.63.100:/home/mete/abdomen-scanner/Temp/
```

**Critical:** Dataset uses Excel format (`Information.xlsx`), not CSV. The sheet name is `TRAIININGDATA` (with 3 i's - this is the actual name, not a typo).

**Why cluster?** Future re-processing or additional data extraction will be instant.

#### Step 1.2: Execute Full Pipeline via Single SLURM Job ✅ READY
```bash
# On cluster
cd /home/mete/abdomen-scanner
sbatch slurm_phase1_full.sh
```

**What it does:**
1. **DICOM → NIfTI conversion** (`dicom_to_nifti.py`)
   - Groups DICOM slices by case/series
   - Stacks into 3D volumes with proper spacing/orientation
   - Output: `data_processed/nifti_images/*.nii.gz`

2. **Generate 3D boxy labels** (`make_boxy_labels.py`) ✅ **GAP 1 FIX APPLIED**
   - Reads Excel file (TRAIININGDATA sheet) with 24,498 bounding box annotations
   - **11→6 Class Mapping:** Maps 11 radiologist labels to 6 competition classes
   - **Z-Axis Validation:** Uses Boundary Slice annotations (3,636 total) to validate anatomical extent
   - Only draws bboxes within valid anatomical z-range (prevents training on invalid labels)
   - Output: `data_processed/nifti_labels_boxy/*.nii.gz`
   
   **Critical Improvement:** Anatomical validation filters out ~5-10% of annotations that fall outside their organ's valid z-range, ensuring cleaner training data.

3. **Create dataset splits** (`split_dataset.py`)
   - 80% train / 10% val / 10% test
   - Reproducible (seed=42)
   - Output: `splits/train_cases.txt`, `val_cases.txt`, `test_cases.txt`

**SLURM Config:**
- **CPUs:** 64 cores (parallel processing)
- **Memory:** 100GB RAM
- **Time:** 2 hours
- **No GPU needed** (pure CPU task)

#### Step 1.3: Data Versioning with DVC ⏳ TODO
```bash
# After Phase 1 completes
dvc add data_processed/nifti_images
dvc add data_processed/nifti_labels_boxy
git add data_processed/*.dvc splits/*.txt
git commit -m "feat: Complete Phase 1 data processing"
git push
```

**Why DVC?** Version control for large datasets without bloating Git.

---

### Phase 2: Parallelized Pseudo-Mask Generation (Weeks 2-3)
**Core Decision:** Use BOTH GPUs simultaneously for 2x speedup.

#### Step 2.1: Set Up MedSAM Environment ✅ READY
```bash
# On cluster (if not already done)
conda activate abdomen_scanner
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download MedSAM checkpoint
wget -P models/ https://drive.google.com/uc?id=<MEDSAM_CHECKPOINT_ID>
# Or upload from local: scp medsam_vit_b.pth mete@100.116.63.100:/home/mete/abdomen-scanner/models/
```

**Do NOT fine-tune MedSAM.** Pre-trained model is sufficient for bbox prompts.

#### Step 2.2: High-Throughput Dual-GPU MedSAM Inference ✅ READY
```bash
# On cluster
sbatch slurm_phase2_medsam.sh
```

**What it does:**
1. Launches **2 parallel processes** (one per GPU)
   - GPU 0: First half of annotations
   - GPU 1: Second half of annotations
2. For each bbox annotation:
   - Load DICOM slice
   - Run MedSAM with bbox prompt
   - Save binary mask as PNG
3. **Aggregate 2D masks → 3D NIfTI labels**

**SLURM Config:**
- **GPUs:** 2 (RTX 6000)
- **CPUs:** 32 cores
- **Memory:** 90GB RAM
- **Time:** 48 hours (for 42,448 annotations)

**Expected Speed:**
- Single GPU: ~20 annotations/second
- Dual GPU: ~40 annotations/second
- Total time: ~18 hours for full dataset

#### Step 2.3: Version MedSAM Labels with DVC ⏳ TODO
```bash
dvc add data_processed/nifti_labels_medsam
git add data_processed/nifti_labels_medsam.dvc
git commit -m "feat: Complete Phase 2 MedSAM mask generation"
git push
```

---

### Phase 3: 3D U-Net Training & Optimization (Weeks 4-8)
**Core Decision:** Large patch sizes (192×192×160) to leverage 96GB VRAM per GPU.

#### Step 3.1: Strategy for Class Imbalance ✅ IMPLEMENTED
**Problem:** **SEVERE** imbalance (9,783 Class 1 annotations vs. only 54 for Class 5/6)

**Class Distribution:**
- **Class 1 (AAA/AAD):** 9,783 annotations (181× more than Class 5)
- **Class 2 (Pancreatitis):** 6,923 annotations
- **Class 3 (Cholecystitis):** 6,265 annotations  
- **Class 4 (Kidney/Ureteral):** 1,405 annotations
- **Class 5 (Diverticulitis):** 54 annotations ⚠️ **CRITICAL IMBALANCE**
- **Class 6 (Appendicitis):** 54 annotations ⚠️ **CRITICAL IMBALANCE**

**Solutions:**
1. **DiceCE Loss** - Combines Dice (segmentation) + Cross-Entropy (pixel-wise)
2. **Weighted Sampling** - `RandCropByPosNegLabeld` oversamples rare classes
3. **Class Weights** - Weight rare classes 100-180× more in loss function
4. **Transfer Learning (GAP 3)** - Pre-train on AMOS 2022/TotalSegmentator for Class 5 feature extraction

#### Step 3.2: Optimized MONAI Data Pipeline ✅ IMPLEMENTED
**Patch Size:** `[192, 192, 160]` (LARGE - standard is 96³)

**Why larger?** More spatial context → better segmentation of complex anatomy

**Transforms:**
- **Preprocessing:**
  - Spacing normalization: `[1.5, 1.5, 2.0]` mm
  - HU windowing: `[-175, 250]` (soft tissue)
  - Intensity normalization: Z-score

- **Augmentation:**
  - Random flips (x, y, z)
  - Random 90° rotations
  - Random intensity scaling/shifting
  - Positive/negative patch sampling (1:1 ratio)

#### Step 3.3: Multi-GPU Training with PyTorch Lightning ✅ IMPLEMENTED
```bash
# Submit training job
./train.sh phase3_unet_baseline

# Monitor
squeue -u $USER
tail -f logs/Phase3_Training_*.out
```

**Configuration:**
- **Model:** MONAI 3D U-Net
  - Channels: `[32, 64, 128, 256, 512]`
  - Dropout: 0.0
  - Residual units: 2 per block

- **Training:**
  - **Strategy:** DDP (DistributedDataParallel)
  - **GPUs:** 2 (effective batch size = 2 × 2 = 4)
  - **Epochs:** 500
  - **Optimizer:** AdamW (lr=2e-4, weight_decay=1e-5)
  - **Scheduler:** Cosine annealing
  - **Mixed precision:** FP16 (faster + lower memory)

- **Validation:**
  - Sliding window inference (overlap=0.5)
  - Dice metric (per-class + mean)
  - Save top 3 checkpoints

**SLURM Config:**
- **GPUs:** 2 (RTX 6000)
- **CPUs:** 32 cores (for data loading)
- **Memory:** 90GB RAM
- **Time:** 48 hours

**Logging:** Weights & Biases (W&B)
```bash
# Set W&B API key (once)
export WANDB_API_KEY=<your_key>
```

---

## 📊 Current Status & Next Actions

### ✅ Completed
- [x] All Python scripts implemented (Phase 1, 2, 3)
- [x] SLURM job scripts created
- [x] Config file (`config.yaml`) fully specified
- [x] PyTorch Lightning training infrastructure
- [x] Dual-GPU MedSAM inference setup
- [x] Documentation updated

### 🚧 In Progress
- [ ] **Step 1.1:** Data transfer to cluster (in progress)
  - Uploading `D:\AbdomenDataSet\Training-DataSets\` → cluster

### ⏳ TODO (Once Data Transfer Completes)
1. **Run Phase 1:** `sbatch slurm_phase1_full.sh`
2. **Version Phase 1 outputs:** DVC add
3. **Download MedSAM checkpoint** (if not already on cluster)
4. **Run Phase 2:** `sbatch slurm_phase2_medsam.sh`
5. **Version Phase 2 outputs:** DVC add
6. **Set W&B API key** (for experiment tracking)
7. **Run Phase 3:** `./train.sh phase3_unet_baseline`
8. **Monitor training:** Check W&B dashboard
9. **Evaluate on test set**
10. **Deploy model** for inference

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Processing** | SimpleITK, nibabel, pydicom | DICOM → NIfTI conversion |
| **Pseudo-Labeling** | MedSAM (Segment Anything) | Bbox → precise masks |
| **Deep Learning** | PyTorch 2.0+, MONAI 1.3+ | 3D medical imaging AI |
| **Multi-GPU Training** | PyTorch Lightning (DDP) | Distributed training |
| **Job Scheduler** | SLURM | HPC cluster management |
| **Version Control** | Git + DVC | Code + data versioning |
| **Experiment Tracking** | Weights & Biases (W&B) | Training metrics/logs |
| **Environment** | Conda | Dependency management |

---

## 📈 Expected Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| **Phase 1** | 1-2 days | Processed NIfTI volumes + splits |
| **Phase 2** | 1-2 days | High-quality 3D pseudo-masks |
| **Phase 3** | 5-7 days | Trained 3D U-Net model |
| **Validation** | 1 day | Test set metrics |
| **Total** | ~2 weeks | Production-ready segmentation model |

**Note:** Actual training time depends on convergence. Monitor validation Dice score.

---

## 🎓 Key Learning Resources

### Papers
- **MedSAM:** [Segment Anything in Medical Images](https://arxiv.org/abs/2304.12306)
- **3D U-Net:** [3D U-Net for Volumetric Segmentation](https://arxiv.org/abs/1606.06650)

### Documentation
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [SLURM Documentation](https://slurm.schedmd.com/)

### Your Cluster
- [mesh-hpc SLURM Guide](https://growmesh.notion.site/slurm-job-scheduler)

---

## 🚨 Critical Reminders

### Data Security
- ❌ **NEVER commit to Git:**
  - Raw DICOM files
  - Processed NIfTI volumes
  - Model weights (use DVC)
- ✅ **Always use `.gitignore`** for data directories

### HPC Etiquette
- ⏱️ **Set realistic time limits** in SLURM scripts
- 🔍 **Monitor jobs regularly:** `squeue -u $USER`
- ❌ **Cancel failed jobs immediately:** `scancel <job-id>`
- 💬 **Communicate with AI security team** (they have priority)

### Reproducibility
- 🎲 **Fix random seeds:** Already set to 42 in all scripts
- 📝 **Version everything:** DVC for data, Git for code
- 📊 **Track experiments:** Use W&B for all training runs

---

## 🎯 Success Metrics

### Phase 1
- ✅ All DICOM cases converted to NIfTI
- ✅ Boxy labels generated for all annotated cases
- ✅ Dataset splits created (80/10/10)

### Phase 2
- ✅ MedSAM masks for all 42,448 annotations
- ✅ 3D aggregated labels aligned with images
- ✅ Visual inspection confirms mask quality

### Phase 3
- 🎯 **Validation Dice Score:** > 0.75 (target)
- 🎯 **Test Set Performance:** Comparable to state-of-the-art
- 🎯 **Inference Speed:** < 5 seconds per case

---

**Last Updated:** October 31, 2025  
**Maintainer:** Mete (@Meteerd)  
**Status:** 🚀 Ready for execution once data transfer completes
