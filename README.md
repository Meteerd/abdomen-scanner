# 🏥 Abdominal Emergency AI Segmentation

> **State-of-the-art AI for automatic 3D segmentation of critical abdominal emergencies from CT scans**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-Private-orange.svg)]()

## 🎯 Overview

Automated **3D segmentation** of multiple critical abdominal emergencies (acute appendicitis, kidney stones, aortic aneurysms) from patient CT scans using **weakly-supervised learning**.

### 🚀 Key Innovation
Transforms **2D bounding box annotations** → **3D pixel-perfect segmentation masks** using MedSAM, eliminating the need for manual pixel-by-pixel annotation.

---

## 📊 Project Phases

### Phase 1: Data Processing ⚙️
- Convert DICOM slices → 3D NIfTI volumes  
- Generate 3D "boxy" labels from CSV annotations  
- **Status:** 🚧 In Development

### Phase 2: MedSAM Pseudo-Mask Generation 🤖
- Use MedSAM AI to create precise 2D segmentation masks  
- Aggregate masks into 3D NIfTI labels  
- **Status:** 📋 Planned

### Phase 3: 3D U-Net Training 🔥
- Train final segmentation model on mesh-hpc cluster  
- Distributed training with MONAI framework  
- **Status:** 📋 Planned

---

## 🖥️ Computing Environment

| Environment | Purpose | Hardware |
|-------------|---------|----------|
| **Local (Windows)** | Development & testing | Anaconda + VS Code |
| **mesh-hpc Cluster** | GPU training | 2x RTX PRO 6000 (96GB VRAM), 128 CPUs, 128GB RAM |

**Job Scheduler:** SLURM  
**VPN Access:** Tailscale

---

## 📁 Repository Structure

```
project_root/
├── README.md                    # You are here
├── .gitignore                   # Prevents committing large files
├── environment.yml              # Conda environment specification
├── requirements.txt             # Pip packages
│
├── data_raw/                    # Raw DICOM files (NOT in Git)
│   ├── dicom_files/
│   └── annotations/
│       ├── TRAININGDATA.csv
│       └── COMPETITIONDATA.csv
│
├── data_processed/              # Processed volumes (NOT in Git)
│   ├── nifti_images/
│   ├── nifti_labels_boxy/
│   └── nifti_labels_medsam/
│
├── scripts/                     # Python processing scripts
│   ├── dicom_to_nifti.py       # Phase 1: DICOM → NIfTI conversion
│   ├── make_boxy_labels.py     # Phase 1: Generate boxy 3D labels
│   ├── medsam_infer.py         # Phase 2: MedSAM inference
│   ├── aggregate_masks.py      # Phase 2: 2D → 3D aggregation
│   ├── train_monai.py          # Phase 3: 3D U-Net training
│   ├── split_dataset.py        # Create train/val/test splits
│   └── utils/                  # Shared utilities
│
├── slurm_*.sh                   # SLURM job scripts for cluster
├── train.sh                     # Job submission wrapper
│
├── configs/
│   └── config.yaml              # Training hyperparameters
│
├── models/                      # Trained weights (NOT in Git)
├── splits/                      # Dataset splits
│   ├── train_cases.txt
│   ├── val_cases.txt
│   └── test_cases.txt
│
└── Tutorials_For_Mete/          # Team setup guides
    ├── BEGINNERS_SETUP.md       # First-time Anaconda setup
    ├── HPC_CLUSTER_SETUP.md     # mesh-hpc cluster guide
    └── SLURM_QUICK_REFERENCE.md # SLURM commands cheat sheet
```

---

## 🚀 Getting Started

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
3. Place CSV annotations in `data_raw/annotations/`

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

## 📊 Dataset

**Source:** TR_ABDOMEN_RAD_EMERGENCY  
**Format:** DICOM CT scans with 2D bounding box annotations  
**Annotations:** CSV files (TRAININGDATA.csv, COMPETITIONDATA.csv)

**Target Pathologies:**
- Acute appendicitis
- Kidney stones
- Aortic aneurysms
- [Additional emergency conditions]

**Note:** Dataset details are in [`docs/DATASET.md`](docs/DATASET.md) (if available)

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

## 📞 Team

- **Project Lead:** [Your Name]
- **Phase 1 (Data Processing):** [Team Member]
- **Phase 2 (MedSAM Integration):** [Team Member]
- **Phase 3 (Model Training):** [Team Member]
- **Cluster Admin:** [Admin Name] - For SSH access & SLURM support

**Communication:** [Discord/Slack/Teams Channel Link]

---

## 🎓 Learning Resources

### Medical AI
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [Medical Segmentation Loss Functions](https://github.com/JunMa11/SegLoss)
- [3D Medical Imaging Basics](https://simpleitk.readthedocs.io/)

### Key Papers
- **MedSAM:** [Segment Anything in Medical Images](https://arxiv.org/abs/2304.12306)
- **3D U-Net:** [3D U-Net for Volumetric Segmentation](https://arxiv.org/abs/1606.06650)
- **MONAI:** [MONAI Framework Overview](https://monai.io/)

### Tools
- [SLURM Documentation](https://slurm.schedmd.com/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Git Workflow Guide](https://www.atlassian.com/git/tutorials/comparing-workflows)

---

## 🏆 Project Milestones

- [ ] **Phase 1:** Complete DICOM → NIfTI pipeline (Target: Nov 15, 2025)
- [ ] **Phase 2:** MedSAM pseudo-mask generation (Target: Dec 1, 2025)
- [ ] **Phase 3:** 3D U-Net training complete (Target: Dec 20, 2025)
- [ ] **Validation:** Model evaluation on test set
- [ ] **Deployment:** Clinical prototype

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
