# 🏥 Abdominal Emergency AI Segmentation

> **State-of-the-art AI for automatic 3D segmentation of critical abdominal emergencies from CT scans**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![License](https://img.shields.io/badge/License-Private-orange.svg)]()

## 🎯 Overview

Automated **3D segmentation** of multiple critical abdominal emergencies (acute appendicitis, kidney stones, aortic aneurysms) from patient CT scans using **weakly-supervised learning**.

### 🚀 Key Innovation
Transforms **2D bounding box annotations** → **3D pixel-perfect segmentation masks** using MedSAM, eliminating the need for manual pixel-by-pixel annotation.

### ⚡ Quick Start
**New to the project?** → Read **[docs/QUICKSTART.md](docs/QUICKSTART.md)** for 3 commands to start training.

**Want the full picture?** → Read **[docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)** for detailed phase-by-phase workflow.

---

## 📊 Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Data Processing** | 🚧 Ready to execute | DICOM → NIfTI, boxy labels, dataset splits |
| **Phase 2: MedSAM Inference** | 📋 Scripts ready | Dual-GPU pseudo-mask generation |
| **Phase 3: 3D U-Net Training** | � Infrastructure ready | Multi-GPU distributed training with PyTorch Lightning |

**Current:** Data transfer to HPC cluster in progress

---

## 🖥️ Computing Environment

| Environment | Purpose | Hardware |
|-------------|---------|----------|
| **Local (Windows)** | Development & code editing | Anaconda + VS Code |
| **mesh-hpc Cluster** | All computation (data + training) | 2x RTX PRO 6000 (96GB VRAM), 128 CPUs, 128GB RAM |

**Job Scheduler:** SLURM  
**VPN Access:** Tailscale  
**Access Period:** 6 months

**⚠️ CRITICAL:** All data processing and training happens on the cluster. Local machine is for code development only.

---

## 📁 Repository Structure

```
abdomen-scanner/
├── README.md                    # You are here
├── LICENSE
├── .gitignore                   # Prevents committing large files
│
├── docs/                        # 📚 Documentation
│   ├── QUICKSTART.md            # ⚡ 3 commands to start
│   ├── PROJECT_ROADMAP.md       # 📋 Full 3-phase workflow
│   ├── SETUP_GUIDE.md           # 🛠️ Environment setup
│   ├── COMMANDS.md              # 💻 Command cheat sheet
│   └── UPDATE_SUMMARY.md        # 📝 Recent changes log
│
├── Tutorials_For_Mete/          # 🎓 Beginner guides
│   ├── BEGINNERS_SETUP.md       # First-time setup
│   ├── HPC_CLUSTER_SETUP.md     # Cluster access
│   └── SLURM_QUICK_REFERENCE.md # SLURM commands
│
├── environment.yml              # Conda environment specification
├── requirements.txt             # Pip packages
├── configs/
│   └── config.yaml              # Training hyperparameters (fully configured)
│
├── data_raw/                    # Raw DICOM files (NOT in Git)
│   ├── dicom_files/             # Transferred from Windows D:\ drive
│   └── annotations/
│       ├── README.md            # ⭐ Explains Excel vs CSV format
│       ├── TRAININGDATA.csv     # Placeholder only
│       └── COMPETITIONDATA.csv  # Placeholder only
│
├── Temp/                        # Temporary/working files
│   └── Information.xlsx         # ⭐ ACTUAL annotation data (TRAIININGDATA sheet)
│
├── data_processed/              # Processed volumes (NOT in Git, versioned with DVC)
│   ├── nifti_images/            # Phase 1: 3D CT volumes
│   ├── nifti_labels_boxy/       # Phase 1: Boxy annotations
│   ├── medsam_2d_masks/         # Phase 2: 2D MedSAM masks
│   └── nifti_labels_medsam/     # Phase 2: Aggregated 3D pseudo-masks
│
├── scripts/                     # ✅ All implemented and ready
│   ├── dicom_to_nifti.py        # Phase 1: DICOM → NIfTI conversion
│   ├── make_boxy_labels.py      # Phase 1: Generate boxy 3D labels
│   ├── split_dataset.py         # Phase 1: Create train/val/test splits
│   ├── medsam_infer.py          # Phase 2: MedSAM inference (dual-GPU)
│   ├── aggregate_masks.py       # Phase 2: 2D → 3D aggregation
│   ├── train_monai.py           # Phase 3: PyTorch Lightning training
│   └── utils/                   # Shared utilities
│
├── slurm_phase1_full.sh         # ✅ Phase 1 master script (64 CPUs)
├── slurm_phase2_medsam.sh       # ✅ Phase 2 dual-GPU inference
├── train.slurm                  # ✅ Phase 3 DDP training (2 GPUs)
├── train.sh                     # Job submission wrapper
│
├── splits/                      # Dataset splits (versioned in Git)
│   ├── train_cases.txt
│   ├── val_cases.txt
│   └── test_cases.txt
│
└── models/                      # Trained weights (NOT in Git, versioned with DVC)
```

---

## 🚀 Getting Started

### For Immediate Action
```bash
# Read this first
cat docs/QUICKSTART.md

# SSH to cluster
ssh mete@100.116.63.100

# Run Phase 1
cd /home/mete/abdomen-scanner
sbatch slurm_phase1_full.sh
```

### For Complete Understanding
1. **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 3 commands to start training
2. **[docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)** - Detailed 3-phase workflow
3. **[Tutorials_For_Mete/HPC_CLUSTER_SETUP.md](Tutorials_For_Mete/HPC_CLUSTER_SETUP.md)** - Cluster access & SLURM
4. **[Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md](Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md)** - Common commands

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch 2.0+ (CUDA 13.0) |
| **Medical AI** | MONAI 1.3+ |
| **Multi-GPU Training** | PyTorch Lightning (DDP) |
| **Data Processing** | pandas, pydicom, SimpleITK, nibabel |
| **Pseudo-Labeling** | MedSAM (Segment Anything Model) |
| **Job Scheduler** | SLURM (mesh-hpc cluster) |
| **Version Control** | Git + DVC (data versioning) |
| **Experiment Tracking** | Weights & Biases (W&B) |

---

## 📊 Three-Phase Workflow

### Phase 1: Data Processing (CPU-Only, 2 hours)
**Command:** `sbatch slurm_phase1_full.sh`
- DICOM → NIfTI conversion (preserves spacing/orientation)
- Generate 3D boxy labels from Excel annotations with **z-axis validation** (GAP 1 fix)
- Maps 11 radiologist labels → 6 competition classes
- Create reproducible train/val/test splits (80/10/10)

### Phase 2: Pseudo-Mask Generation (Dual-GPU, 18 hours)
**Command:** `sbatch slurm_phase2_medsam.sh`
- Run MedSAM on **both GPUs in parallel** (2x speedup)
- Process 42,448 annotations → high-quality 2D masks
- Aggregate 2D masks → 3D NIfTI label volumes

### Phase 3: 3D U-Net Training (Dual-GPU DDP, 2-7 days)
**Command:** `./train.sh phase3_unet_baseline`
- Train 3D U-Net with PyTorch Lightning
- **Large patches (192×192×160)** leverage 96GB VRAM
- DDP strategy across 2 GPUs
- Mixed precision (FP16) for speed
- DiceCE loss + weighted sampling for class imbalance

---

## 🎓 Key Features

### ✅ Fully Implemented
- Complete Phase 1-3 Python scripts
- SLURM job scripts for all phases
- PyTorch Lightning training infrastructure
- Dual-GPU parallelization for MedSAM
- Class imbalance handling (weighted loss + sampling)
- Large patch sizes optimized for 96GB VRAM
- Mixed precision training
- Experiment tracking (W&B integration)
- Comprehensive documentation

### 🔄 Optimized for HPC
- **128 CPU cores** for Phase 1 (minutes instead of hours)
- **Dual-GPU inference** for Phase 2 (2x speedup)
- **DDP training** for Phase 3 (efficient multi-GPU)
- **Large batch/patch sizes** (leverage 96GB VRAM per GPU)

---

## 📝 Configuration

All hyperparameters are in **`configs/config.yaml`**:
- Patch size: `[192, 192, 160]` (large!)
- Batch size: `2` per GPU (effective: 4)
- Learning rate: `2e-4` (AdamW)
- Loss: DiceCE with class weights `[0.5, 2.0, 2.0, 2.0]`
- Epochs: `500` (with cosine annealing)
- Mixed precision: `FP16`

---

## 🔒 Data Privacy & Security

### ⚠️ CRITICAL RULES

**NEVER commit to Git:**
- ❌ Raw DICOM files
- ❌ Processed NIfTI volumes  
- ❌ 2D mask images
- ❌ Model weights (use DVC)
- ❌ Patient identifiers
- ❌ Credentials or API keys

**DO commit to Git:**
- ✅ Python scripts
- ✅ SLURM job files
- ✅ Configuration files
- ✅ Documentation
- ✅ DVC metadata files (`.dvc`)
- ✅ Dataset split lists (`splits/*.txt`)

**Always:**
- Use `.gitignore` (already configured)
- Version data with DVC
- Keep data on secure cluster storage

---

## 🎯 Dataset

**Source:** TR_ABDOMEN_RAD_EMERGENCY (Koç et al. 2024)  
**Format:** DICOM CT scans + Excel annotations (`Temp/Information.xlsx`)  
**Size:** 24,498 bounding box annotations + 3,636 boundary slices across 735 cases

**Target Pathologies (6 Competition Classes):**
1. AAA/AAD (Abdominal aortic aneurysm/dissection) - 9,783 annotations
2. Acute Pancreatitis - 6,923 annotations
3. Cholecystitis - 6,265 annotations
4. Kidney/Ureteral Stones - 1,405 annotations
5. Diverticulitis - 54 annotations ⚠️ **RARE**
6. Appendicitis - 54 annotations ⚠️ **RARE**

**Note:** See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for complete class mapping details.

---

## 📞 Support & Resources

### Documentation
- **Quick Start:** [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Data Format:** [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) ⭐ **NEW**
- **Full Roadmap:** [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)
- **Commands Reference:** [docs/COMMANDS.md](docs/COMMANDS.md)
- **HPC Setup:** [Tutorials_For_Mete/HPC_CLUSTER_SETUP.md](Tutorials_For_Mete/HPC_CLUSTER_SETUP.md)
- **SLURM Commands:** [Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md](Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md)

### External Resources
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [MedSAM Paper](https://arxiv.org/abs/2304.12306)
- [mesh-hpc SLURM Guide](https://growmesh.notion.site/slurm-job-scheduler)

### Team
- **Project Lead:** Mete (@Meteerd)
- **Cluster:** mesh-hpc (6-month access)

---

## 📄 License

**Private / Proprietary**

This project contains medical imaging data and is not for public distribution.

---

## Acknowledgments

- **TR_ABDOMEN_RAD_EMERGENCY Dataset**
- **MedSAM** - Ma et al., 2023
- **MONAI Framework** - Project MONAI Consortium  
- **PyTorch Lightning** - Lightning AI
- **mesh-hpc cluster** - 6-month access

---

**Last Updated:** October 31, 2025  
**Version:** 1.0.0 (All phases implemented)  
**Status:** 🚀 Ready for execution once data transfer completes

**Next Step:** Read [docs/QUICKSTART.md](docs/QUICKSTART.md) and run Phase 1!

### 1️⃣ Clone the Repository

```bash
git clone https://gitlab.com/your-username/abdomen-scanner.git
cd abdomen-scanner
```

### 2️⃣ Set Up Local Environment (Windows)

**Prerequisites:**
- Anaconda3 installed
- Git installed

**Setup:**
```powershell
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate abdomen_scanner

# Verify installation
python verify_setup_local.py
```

**Expected output:** ✅ All checks passed!

### 3️⃣ Get Data Access

**⚠️ Data is NOT in Git repository!**

1. Contact project lead for TR_ABDOMEN_RAD_EMERGENCY dataset access
2. Place raw DICOM files in `data_raw/dicom_files/`
3. Place Excel annotations (`Information.xlsx`) in `Temp/` directory
   - **Note:** Sheet name is `TRAIININGDATA` (3 i's - not a typo!)
   - See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for details

### 4️⃣ Read Documentation

**New to the project?** Start here:
1. [`Tutorials_For_Mete/BEGINNERS_SETUP.md`](Tutorials_For_Mete/BEGINNERS_SETUP.md) - Anaconda & Python setup
2. [`Tutorials_For_Mete/HPC_CLUSTER_SETUP.md`](Tutorials_For_Mete/HPC_CLUSTER_SETUP.md) - mesh-hpc cluster access
3. [`Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md`](Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md) - SLURM commands

---

## 🤝 Team Workflow

### Development Cycle

```bash
# 1. Create a feature branch
git checkout -b feature/phase1-dicom-processing

# 2. Write/edit code locally (Windows)
# ... develop in VS Code ...

# 3. Test locally (if possible)
python scripts/dicom_to_nifti.py --test-mode

# 4. Commit changes
git add scripts/dicom_to_nifti.py
git commit -m "[Phase 1] Add DICOM to NIfTI conversion with spacing preservation"

# 5. Push to GitLab
git push origin feature/phase1-dicom-processing

# 6. Create Merge Request on GitLab
# ... go to GitLab website ...

# 7. After approval, merge to main
```

### Running on mesh-hpc Cluster

```bash
# SSH to cluster
ssh your-username@mesh-hpc

# Pull latest code
cd ~/abdomen-scanner
git pull origin main

# Submit job
./train.sh phase3_unet_training

# Monitor
squeue -u $USER
tail -f logs/train_*.out
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch 2.0+ (with CUDA 13.0) |
| **Medical AI** | MONAI 1.3+ |
| **Data Processing** | pandas, pydicom, SimpleITK, nibabel |
| **Image Processing** | OpenCV, Pillow, scikit-image |
| **Weakly-Supervised** | MedSAM (Segment Anything Model) |
| **Visualization** | matplotlib |
| **Job Scheduler** | SLURM (mesh-hpc cluster) |
| **Version Control** | Git / GitLab |

---

## 📊 Dataset Details

**Source:** TR_ABDOMEN_RAD_EMERGENCY (Koç et al. 2024)  
**Format:** DICOM CT scans with Excel annotations  
**Annotations:** `Temp/Information.xlsx` (TRAIININGDATA + COMPETITIONDATA sheets)

**Statistics:**
- 24,498 bounding box annotations
- 3,636 boundary slice annotations (for z-axis validation)
- 735 unique cases
- 11 radiologist labels mapped to 6 competition classes

**Target Pathologies (with annotation counts):**
1. AAA/AAD - 9,783 annotations ✅ Well-represented
2. Acute Pancreatitis - 6,923 annotations ✅ Well-represented
3. Cholecystitis - 6,265 annotations ✅ Well-represented
4. Kidney/Ureteral Stones - 1,405 annotations ⚠️ Moderate
5. Diverticulitis - 54 annotations 🚨 Critical imbalance
6. Appendicitis - 54 annotations 🚨 Critical imbalance

**Complete details:** See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

---

## 🔒 Data Privacy & Security

### ⚠️ CRITICAL RULES

**NEVER commit to Git:**
- ❌ Raw DICOM files
- ❌ Processed NIfTI volumes
- ❌ Model weights (use Git LFS if sharing)
- ❌ Patient identifiers
- ❌ SSH credentials or API keys

**DO commit to Git:**
- ✅ Python scripts
- ✅ SLURM job files
- ✅ Configuration files
- ✅ Documentation
- ✅ Small anonymized test data

**Always:**
- Keep data on secure cluster storage
- Follow your institution's data handling policies
- Use `.gitignore` to prevent accidents

---

## 📝 Contributing

### Branch Naming Convention
- `feature/phase1-dicom-loader` - New features
- `fix/spacing-calculation-bug` - Bug fixes
- `docs/update-setup-guide` - Documentation
- `experiment/medsam-threshold-test` - Experimental code

### Commit Message Format
```
[Phase X] Brief description

- Detailed change 1
- Detailed change 2
- Fixed issue #123
```

**Examples:**
- `[Phase 1] Add DICOM to NIfTI conversion with spacing preservation`
- `[Phase 2] Integrate MedSAM inference pipeline`
- `[Fix] Correct bounding box coordinate transformation`
- `[Docs] Update cluster setup guide with Tailscale instructions`

### Code Style
- Follow PEP 8
- Add docstrings to all functions
- Comment complex algorithms
- Keep functions under 50 lines when possible
- Use type hints where appropriate

---

## 📞 Support & Resources

### Documentation
- **Quick Start:** [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Full Roadmap:** [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)
- **Setup Guide:** [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)
- **Commands:** [docs/COMMANDS.md](docs/COMMANDS.md)
- **HPC Setup:** [Tutorials_For_Mete/HPC_CLUSTER_SETUP.md](Tutorials_For_Mete/HPC_CLUSTER_SETUP.md)
- **SLURM Guide:** [Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md](Tutorials_For_Mete/SLURM_QUICK_REFERENCE.md)

### External Resources
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [MedSAM Paper](https://arxiv.org/abs/2304.12306)
- [mesh-hpc SLURM Guide](https://growmesh.notion.site/slurm-job-scheduler)

### Team
- **Project Lead:** Mete (@Meteerd)
- **Cluster:** mesh-hpc (6-month access)

---

## 📄 License

**Private / Proprietary**

This project contains medical imaging data and is not for public distribution.

---

## 🙏 Acknowledgments

- **TR_ABDOMEN_RAD_EMERGENCY Dataset** - [Citation/Link]
- **MedSAM** - Ma et al., 2023
- **MONAI Framework** - Project MONAI Consortium
- **mesh-hpc cluster** - 6-month access provided by [Institution]
- **AI Security Project Team** - Priority cluster access coordination

---

**Last Updated:** October 30, 2025  
**Version:** 0.1.0 (Phase 1 Development)  
**Status:** 🚧 Active Development
