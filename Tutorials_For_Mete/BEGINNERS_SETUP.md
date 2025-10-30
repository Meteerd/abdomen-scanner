z# 🚀 BEGINNER'S GUIDE: Setting Up Your Medical AI Environment

## Welcome! You're About to Set Up a Professional AI Development Environment

This guide assumes you've **never used Anaconda before**. I'll explain everything step-by-step.

---

## 📚 What You Need to Understand First

### What is a "Conda Environment"?
Think of it like a **separate folder** that contains:
- A specific version of Python (3.10 for your project)
- All the libraries your project needs (PyTorch, MONAI, etc.)
- **Isolated** from other Python projects so they don't interfere

**Why do we need this?**
- Different projects need different library versions
- Keeps your system Python clean
- Easy to delete and recreate if something breaks

### What is a "Package"?
A package is like an **app for Python**. Examples:
- `pandas` - Works with spreadsheet-like data (your CSV files)
- `pydicom` - Reads medical DICOM images
- `pytorch` - Deep learning framework (trains your AI)
- `monai` - Specialized for medical AI

---

## 🎯 Your Mission: Install 25+ Packages for Medical AI

Your project needs packages for three phases:

**Phase 1: Data Processing** 
- Read DICOM scans → Convert to 3D NIfTI volumes → Create labels
- Packages: `pydicom`, `SimpleITK`, `nibabel`, `pandas`

**Phase 2: MedSAM (AI Mask Generation)**
- Use MedSAM AI to automatically create precise segmentation masks
- Packages: `pytorch`, `opencv`, `pillow`

**Phase 3: 3D U-Net Training**
- Train your final AI model on 2x NVIDIA A6000 GPUs
- Packages: `monai`, `torchmetrics`, `pytorch` (with CUDA support)

---

## 🛠️ Step-by-Step Setup Instructions

### Step 1: Open Anaconda PowerShell Prompt

1. **Press Windows Key** (on your keyboard)
2. **Type:** `Anaconda PowerShell Prompt`
3. **Click** the green snake icon that appears
4. A black/blue terminal window will open

**What you'll see:**
```
(base) PS C:\Users\User>
```
- `(base)` = You're in Anaconda's default environment
- `PS` = PowerShell
- `C:\Users\User>` = Your current folder location

---

### Step 2: Navigate to Your Project Folder

**Copy and paste this command** (right-click in the terminal to paste):

```powershell
cd "c:\Users\User\Desktop\Puzzles Software\Projects\OmniScan Medical\Abdomen-Scanner\project_root"
```

**Press Enter**

**What this does:** Changes directory (`cd`) to your project folder, like opening a folder in File Explorer.

**You should now see:**
```
(base) PS c:\Users\User\Desktop\Puzzles Software\Projects\OmniScan Medical\Abdomen-Scanner\project_root>
```

---

### Step 3: Run the Automated Setup Script

Now we'll run a script I created that automatically installs everything.

**Type this command:**
```powershell
.\setup_environment.ps1
```

**Press Enter**

**What happens:**
1. ⚙️ Creates a new environment called `abdomen_scanner`
2. 📦 Downloads and installs Python 3.10
3. 📦 Downloads PyTorch with CUDA support (for your GPUs)
4. 📦 Installs 20+ packages (this takes 10-15 minutes)
5. ✅ Shows success message when done

**You'll see lots of text scrolling** - this is normal! It's downloading packages from the internet.

**☕ Grab a coffee - this takes about 10-15 minutes**

**⚠️ IMPORTANT NOTE:** This local environment is for **development and testing only**. You won't have GPU access locally. For actual training, you'll use the **mesh-hpc cluster** (see HPC_CLUSTER_SETUP.md after this setup is complete).

---

### Step 4: Activate Your New Environment

Once the setup finishes, you need to **"activate"** the environment (switch into it).

**Type this command:**
```powershell
conda activate abdomen_scanner
```

**Press Enter**

**What changes:**
```
(base) PS ...>          # BEFORE
(abdomen_scanner) PS ...>   # AFTER
```

See how `(base)` changed to `(abdomen_scanner)`? That means you're now **inside** your project environment!

**From now on, every time you work on this project:**
1. Open Anaconda PowerShell Prompt
2. Run: `conda activate abdomen_scanner`
3. Start coding!

---

### Step 5: Verify Everything Works

I created a verification script that checks if everything installed correctly.

**Type this command:**
```powershell
python verify_setup_local.py
```

**Press Enter**

**Note:** We use `verify_setup_local.py` (not `verify_setup.py`) because you don't have local GPUs - you'll use the remote cluster for that!

**What it checks:**
- ✓ Python 3.10 is installed
- ✓ All 25+ packages are present
- ✓ PyTorch installed (GPU check skipped - you'll use cluster)
- ✓ MONAI framework is ready
- ✓ Project folders exist
- ✓ SLURM job scripts exist

**Expected output:**
```
======================================================================
  ABDOMINAL EMERGENCY AI SEGMENTATION
  Local Environment Verification (Windows)
======================================================================

✓ Python 3.10 (found 3.10.14)
✓ pandas               (Data manipulation)
✓ pydicom              (DICOM file reading)
✓ torch                (PyTorch deep learning)
✓ monai                (Medical AI framework)
...
⚠️ Note: GPU checks skipped - you'll use the mesh-hpc cluster for GPU work.
   This local environment is for development and data preparation only.

✓ All checks passed! Environment is ready for development.
```

---

## 🎨 Configure VS Code (Your Code Editor)

Now let's tell VS Code to use your new environment:

### Step 6: Select Python Interpreter in VS Code

1. **In VS Code, press:** `Ctrl + Shift + P`
2. **Type:** `Python: Select Interpreter`
3. **Click** the option that appears
4. **Select:** `Python 3.10.x ('abdomen_scanner')`
   - It will show the path: `~\anaconda3\envs\abdomen_scanner\python.exe`

**Now VS Code knows to use your project's Python!**

---

## 📦 What Just Got Installed? (Package Breakdown)

Here's what each package does in plain English:

### Data Processing (Phase 1)
| Package | What It Does |
|---------|-------------|
| `pandas` | Read CSV files (your bounding box annotations) |
| `numpy` | Math operations on arrays (3D volume manipulation) |
| `pydicom` | Read DICOM files (CT scan slices) |
| `SimpleITK` | Convert DICOM → NIfTI (3D medical format) |
| `nibabel` | Save/load NIfTI files (.nii.gz) |

### Image Processing (Phase 2)
| Package | What It Does |
|---------|-------------|
| `opencv-python` | Image transformations and preprocessing |
| `Pillow` | Image loading and basic operations |
| `scikit-image` | Advanced image processing |

### AI & Deep Learning (Phase 3)
| Package | What It Does |
|---------|-------------|
| `torch` (PyTorch) | Deep learning framework - trains neural networks |
| `torchvision` | Computer vision utilities for PyTorch |
| `monai` | Medical imaging AI toolkit (3D U-Net, data loaders) |
| `torchmetrics` | Calculate accuracy metrics (Dice score, IoU) |

### Utilities
| Package | What It Does |
|---------|-------------|
| `matplotlib` | Create plots and visualizations |
| `tqdm` | Progress bars (shows "Processing... 45%") |
| `PyYAML` | Read configuration files (config.yaml) |
| `scikit-learn` | Split data into train/validation/test sets |

---

## 🔍 Troubleshooting Common Issues

### Problem 1: "conda: command not found"
**Solution:** You're not in Anaconda PowerShell Prompt. Close the terminal and open **Anaconda PowerShell Prompt** from Start Menu (not regular PowerShell).

### Problem 2: "cannot be loaded because running scripts is disabled"
**Solution:** Run this command first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try the setup script again.

### Problem 3: Setup script takes forever or freezes
**Solution:** 
1. Press `Ctrl+C` to cancel
2. Check your internet connection
3. Try manual installation (see SETUP_GUIDE.md)

### Problem 4: "No CUDA GPUs detected" in verification
**Solution:**
1. Check if GPUs are working: `nvidia-smi` (should show 2x A6000)
2. Reinstall PyTorch with CUDA:
   ```powershell
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

---

## 🎯 Next Steps After Setup

Once verification passes, you have **two options**:

### Option A: Quick Local Development & Testing
Test Phase 1 scripts locally (no GPU needed):
```powershell
# Test DICOM conversion locally (small subset)
python scripts/dicom_to_nifti.py --test-mode

# Test boxy label generation
python scripts/make_boxy_labels.py --test-mode
```

### Option B: Full Processing on Cluster (Recommended)
For the complete dataset, use the **mesh-hpc cluster**:

**Next:** Read `HPC_CLUSTER_SETUP.md` to learn how to:
1. Install Tailscale VPN
2. Get SSH credentials
3. Connect to mesh-hpc cluster
4. Submit SLURM jobs for GPU training

**The cluster has:**
- 2x NVIDIA RTX PRO 6000 GPUs (96GB VRAM each!)
- 128 CPU threads
- 128GB RAM
- CUDA 13.0 support

**You'll use SLURM to submit jobs like:**
```bash
# On the cluster
./train.sh my_experiment_name
```

This is where the **real training** happens! 🚀

---

## 📚 Learning Resources

Want to understand what you're building?

**Conda Basics:**
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- Common commands: `conda list`, `conda env list`, `conda deactivate`

**Medical AI:**
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [MedSAM Paper](https://arxiv.org/abs/2304.12306)

**PyTorch:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 🆘 Need Help?

If you get stuck at any point:

1. **Read the error message** - It usually tells you what's wrong
2. **Check SETUP_GUIDE.md** - More detailed troubleshooting
3. **Ask me!** - I'm here to help

---

## ✅ Checklist

Use this to track your progress:

- [ ] Anaconda installed
- [ ] Anaconda PowerShell Prompt opened
- [ ] Navigated to project folder (`cd ...`)
- [ ] Ran setup script (`.\setup_environment.ps1`)
- [ ] Activated environment (`conda activate abdomen_scanner`)
- [ ] Ran verification (`python verify_setup.py`)
- [ ] All checks passed ✓
- [ ] VS Code configured to use `abdomen_scanner` interpreter
- [ ] Ready to start Phase 1!

---

**You've got this! 🚀 Let's build some amazing medical AI!**
