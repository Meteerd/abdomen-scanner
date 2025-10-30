# Abdominal Emergency AI Segmentation - Setup Guide

## Project Overview

This project implements a **weakly-supervised 3D segmentation pipeline** for abdominal emergency diagnosis using CT scans. We leverage MedSAM to automatically generate high-quality segmentation masks from 2D bounding box annotations, then train a 3D U-Net for multi-class organ/pathology segmentation.

### Three-Phase Approach

1. **Phase 1: Data Processing** - Convert DICOM to NIfTI, create "boxy" 3D labels
2. **Phase 2: Pseudo-Mask Generation** - Use MedSAM to generate precise segmentation masks
3. **Phase 3: 3D Model Training** - Train 3D U-Net using MONAI framework

---

## Prerequisites

### Hardware Requirements
- **2x NVIDIA A6000 GPUs** (48GB VRAM each) for distributed training
- CUDA 12.1+ compatible drivers
- Sufficient disk space (~500GB recommended for processed data)

### Software Requirements
- **Anaconda3** (2025.06-0 or later) with Python 3.13 registered as default
- Windows 10/11 with PowerShell 5.1+
- Git (for version control)

---

## Installation Steps

### Step 0.1: Create Dedicated Conda Environment

This isolates all project dependencies to avoid conflicts with other Python projects.

#### Automated Setup (Recommended)

1. **Open Anaconda PowerShell Prompt** (from Start Menu)

2. **Navigate to project directory:**
   ```powershell
   cd "c:\Users\User\Desktop\Puzzles Software\Projects\OmniScan Medical\Abdomen-Scanner\project_root"
   ```

3. **Run the setup script:**
   ```powershell
   .\setup_environment.ps1
   ```

4. **Activate the environment:**
   ```powershell
   conda activate abdomen_scanner
   ```

#### Manual Setup (If Automated Fails)

If the automated script encounters issues, follow these manual steps:

```powershell
# Create base environment
conda create -n abdomen_scanner python=3.10 -y

# Activate it
conda activate abdomen_scanner

# Install PyTorch with CUDA support (for A6000 GPUs)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install conda packages
conda install pandas numpy scikit-learn -c conda-forge -y
conda install simpleitk nibabel pydicom -c conda-forge -y
conda install opencv pillow matplotlib tqdm pyyaml -c conda-forge -y

# Install pip packages
pip install monai[all]>=1.3.0
pip install torchmetrics>=1.0.0
```

---

### Step 0.2: Install Core Python Packages

The packages are organized by project phase:

#### Phase 1: Data Processing
- **pandas** - Parse CSV annotations (TRAININGDATA.csv, COMPETITIONDATA.csv)
- **pydicom** - Read DICOM slices
- **SimpleITK** - Convert DICOM → NIfTI (3D volumes)
- **nibabel** - Load/save NIfTI files
- **numpy** - Numerical operations for volume manipulation

#### Phase 2: MedSAM Integration
- **opencv-python** - Image preprocessing
- **Pillow** - Image transformations
- **scikit-image** - Advanced image operations
- **torch/torchvision** - Run MedSAM inference

#### Phase 3: 3D U-Net Training
- **PyTorch (2.0+)** - Deep learning framework
- **MONAI** - Medical imaging AI toolkit (data loaders, 3D U-Net, transforms)
- **torchmetrics** - Track Dice scores, IoU during training

#### Utilities
- **matplotlib** - Visualize data and results
- **tqdm** - Progress bars for long-running scripts
- **PyYAML** - Configuration management
- **scikit-learn** - Data splitting utilities

All packages are specified in `environment.yml` and will be installed automatically.

---

## Verification

After installation, verify your setup:

```powershell
# Ensure environment is activated
conda activate abdomen_scanner

# Run verification script
python verify_setup.py
```

The verification script checks:
- ✓ Python 3.10 installation
- ✓ All required packages installed
- ✓ PyTorch can access CUDA GPUs
- ✓ MONAI features available
- ✓ Project directory structure
- ✓ Raw data files present

### Expected Output

```
======================================================================
  ABDOMINAL EMERGENCY AI SEGMENTATION
  Environment Verification
======================================================================

======================================================================
  Python Version Check
======================================================================
✓ Python 3.10 (found 3.10.14)

======================================================================
  Core Package Installation
======================================================================
✓ pandas               (Data manipulation (CSV parsing))
✓ numpy                (Numerical operations)
✓ pydicom              (DICOM file reading)
...

======================================================================
  PyTorch & CUDA Configuration
======================================================================
✓ CUDA Available
✓ GPU Count (2 GPU(s) detected (need 2x A6000))
    GPU 0: NVIDIA RTX A6000 (48.0 GB)
    GPU 1: NVIDIA RTX A6000 (48.0 GB)
✓ CUDA Version (v12.1)
✓ PyTorch Version (v2.0.1)

✓ All checks passed! Environment is ready for development.
```

---

## VS Code Configuration

### Configure Python Interpreter

1. **Open Command Palette** (`Ctrl+Shift+P`)
2. Type: `Python: Select Interpreter`
3. Choose: `Python 3.10.x ('abdomen_scanner')`
   - Path: `C:\Users\User\anaconda3\envs\abdomen_scanner\python.exe`

### Recommended Extensions

Install these VS Code extensions for optimal development:
- **Python** (Microsoft) - Python language support
- **Pylance** (Microsoft) - Fast, feature-rich Python language server
- **Jupyter** (Microsoft) - Notebook support for experimentation
- **autoDocstring** - Automatic docstring generation

---

## Project Structure

```
project_root/
├── data_raw/                      # Original dataset (not in git)
│   ├── dicom_files/               # Raw DICOM slices
│   └── annotations/
│       ├── TRAININGDATA.csv       # Training set bounding boxes
│       └── COMPETITIONDATA.csv    # Additional annotations
│
├── data_processed/                # Processed data (not in git)
│   ├── nifti_images/              # 3D CT volumes (.nii.gz)
│   ├── nifti_labels_boxy/         # Phase 1: Boxy 3D labels
│   ├── nifti_labels_medsam/       # Phase 2: MedSAM refined masks
│   └── yolo_dataset/              # Optional: YOLO format data
│
├── scripts/                       # Python scripts for each phase
│   ├── dicom_to_nifti.py          # Phase 1: DICOM → NIfTI conversion
│   ├── make_boxy_labels.py        # Phase 1: Create boxy 3D labels
│   ├── medsam_infer.py            # Phase 2: MedSAM mask generation
│   ├── aggregate_masks.py         # Phase 2: Combine 2D → 3D masks
│   ├── train_monai.py             # Phase 3: 3D U-Net training
│   ├── split_dataset.py           # Create train/val/test splits
│   └── utils/                     # Shared utilities
│
├── configs/
│   └── config.yaml                # Training hyperparameters
│
├── models/                        # Saved model checkpoints
│
├── splits/                        # Dataset splits
│   ├── train_cases.txt
│   ├── val_cases.txt
│   └── test_cases.txt
│
├── environment.yml                # Conda environment specification
├── requirements.txt               # Pip requirements (reference)
└── verify_setup.py                # Environment verification script
```

---

## Next Steps

Once your environment is verified, proceed with Phase 1 development:

### Phase 1: Foundation Data Processing

1. **Convert DICOM to NIfTI:**
   ```powershell
   python scripts/dicom_to_nifti.py
   ```
   - Stacks 2D DICOM slices → 3D NIfTI volumes
   - Output: `data_processed/nifti_images/*.nii.gz`

2. **Generate Boxy Labels:**
   ```powershell
   python scripts/make_boxy_labels.py
   ```
   - Reads CSV bounding boxes
   - Creates 3D label volumes with rectangular masks
   - Output: `data_processed/nifti_labels_boxy/*.nii.gz`

3. **Split Dataset:**
   ```powershell
   python scripts/split_dataset.py
   ```
   - Creates train/val/test splits
   - Output: `splits/*.txt` files

---

## Troubleshooting

### Issue: "conda: command not found"
**Solution:** You're not in Anaconda PowerShell Prompt. Open it from Start Menu.

### Issue: CUDA not available in PyTorch
**Solutions:**
1. Check NVIDIA driver: `nvidia-smi` in terminal
2. Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch: `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y`

### Issue: Package import errors
**Solution:** Ensure environment is activated: `conda activate abdomen_scanner`

### Issue: Out of memory during training
**Solutions:**
1. Reduce batch size in `configs/config.yaml`
2. Reduce patch size (e.g., from [128,128,128] to [96,96,96])
3. Enable gradient checkpointing in training script

---

## Team Workflow

### Daily Development

1. **Activate environment:**
   ```powershell
   conda activate abdomen_scanner
   ```

2. **Work on your phase:**
   - Phase 1: Data processing scripts
   - Phase 2: MedSAM integration
   - Phase 3: Training pipeline

3. **Run verification regularly:**
   ```powershell
   python verify_setup.py
   ```

### Before Training

1. Ensure both A6000 GPUs are visible: `nvidia-smi`
2. Verify preprocessed data exists
3. Review `configs/config.yaml` hyperparameters
4. Test on small subset first

---

## Additional Resources

- **MONAI Documentation:** https://docs.monai.io/
- **MedSAM Paper:** https://arxiv.org/abs/2304.12306
- **PyTorch Distributed Training:** https://pytorch.org/tutorials/beginner/dist_overview.html

---

## Contact & Support

For questions or issues with the setup, contact the project lead or refer to the project documentation.

**Happy coding! Let's build something amazing.** 🚀
