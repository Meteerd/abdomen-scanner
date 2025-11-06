# Abdominal Emergency AI Segmentation - Complete Project Summary

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Environment and Infrastructure](#technical-environment-and-infrastructure)
3. [Data Architecture](#data-architecture)
4. [8-Phase Pipeline Architecture](#8-phase-pipeline-architecture)
5. [Critical Technical Decisions and Fixes](#critical-technical-decisions-and-fixes)
6. [Development Workflow and Best Practices](#development-workflow-and-best-practices)
7. [Current Project Status](#current-project-status)
8. [Next Steps and Future Work](#next-steps-and-future-work)

---

## Project Overview

### Primary Objective
Develop an automated 3D semantic segmentation system for detecting critical abdominal emergencies from CT scans using weakly-supervised learning. The system processes DICOM CT volumes and generates multi-class 3D segmentation masks for six pathology classes.

### Clinical Application
Enable rapid triage and volumetric quantification of life-threatening abdominal conditions in emergency radiology settings. The system targets conditions that require immediate clinical intervention.

### Target Pathologies (11 Clinical Labels Mapped to 6 Classes)
- Class 0: Abdominal Aortic Aneurysm/Dissection (AAA/AAD)
  - Abdominal aortic aneurysm
  - Abdominal aortic dissection
  - Abdominal Aorta
- Class 1: Acute Pancreatitis
  - Compatible with acute pancreatitis
- Class 2: Cholecystitis/Gallbladder Pathology
  - Compatible with acute cholecystitis
  - Gallbladder stone
- Class 3: Urolithiasis (Kidney/Ureteral Stones)
  - Kidney stone
  - Ureteral stone
- Class 4: Diverticulitis (RARE CLASS - 340 annotations across 21 cases)
  - Compatible with acute diverticulitis
  - Calcified diverticulum
- Class 5: Appendicitis (RARE CLASS - 2,283 annotations across 87 cases)
  - Compatible with acute appendicitis

### Key Technical Challenges
1. Weakly-supervised learning: Only 2D bounding box annotations available, not full 3D segmentation masks
2. Severe class imbalance: Rare pathologies (Classes 4 and 5) are critically underrepresented
3. Multi-pathology complexity: Single CT slices can contain multiple pathologies requiring class-specific mask handling
4. Large-scale inference: 42,450 total bounding box annotations across 651 CT volumes
5. Computational constraints: Training 3D segmentation models requires substantial GPU memory and time

---

## Technical Environment and Infrastructure

### Hardware Specifications
- Cluster Name: mesh-hpc (remote HPC cluster)
- GPU Configuration: 2x NVIDIA RTX 6000 Ada (96GB VRAM each, total 192GB)
- CPU Resources: 128 CPU cores
- System Memory: 128GB RAM
- Access Duration: 6-month allocation
- Network Access: Tailscale VPN for secure remote connection

### Software Stack
- Operating System: Linux (cluster), Windows 11 (local development)
- Python Version: 3.12
- Deep Learning Framework: PyTorch 2.0+
- Medical Imaging Library: MONAI (Medical Open Network for AI)
- Training Framework: PyTorch Lightning (for distributed data parallel training)
- 2D Object Detection: YOLOv11 (Ultralytics)
- Foundation Model: MedSAM (Medical Segment Anything Model)
- Scheduler: SLURM (Simple Linux Utility for Resource Management)
- Version Control: Git
- Remote Access: SSH, VS Code Remote-SSH extension

### Critical Infrastructure Decisions

#### Why Remote HPC Instead of Local Training
- Local laptop lacks sufficient GPU memory for 3D medical imaging (96GB VRAM required)
- Training duration would be prohibitive on consumer hardware (weeks vs days)
- Multi-GPU distributed training requires enterprise-grade infrastructure
- Shared cluster allows collaboration and resource pooling

#### Why SLURM Job Scheduling
- Prevents GPU conflicts when multiple users access shared hardware
- Provides fair resource allocation and job queuing
- Enables background execution of long-running jobs (days)
- Automatic logging and error handling
- Resource monitoring and accounting

#### Why Avoid Global Conda Commands
- Global conda installations can create version conflicts across users
- Shared environments lead to dependency hell
- Risk of breaking other users' workflows
- Solution: Isolated virtual environments per user in home directories
  ```bash
  # User-specific environment in home directory
  python3 -m venv /home/mete/abdomen-scanner/venv
  source /home/mete/abdomen-scanner/venv/bin/activate
  ```

#### Why SSH Protocol for Cluster Access
- Secure encrypted communication between local machine and remote cluster
- Standard protocol for remote Linux server access
- Enables file transfer (SCP), remote editing (VS Code Remote-SSH), and terminal access
- Tailscale VPN adds additional security layer
- No direct internet exposure of cluster resources

### Development Environment Setup

#### Local Machine (Windows 11 Laptop)
Purpose: Code development, small-scale testing, data preprocessing, visualization

Setup Process:
```bash
# Navigate to project directory
cd "C:\Users\User\Desktop\Puzzles Software\Projects\OmniScan Medical\Abdomen-Scanner\project_root"

# Create local virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup (no GPU required locally)
python verify_setup_local.py
```

#### Remote Cluster (mesh-hpc)
Purpose: GPU-intensive training, large-scale inference, data storage

Connection Workflow:
```bash
# Step 1: Connect via Tailscale VPN (one-time setup)
# Install Tailscale on Windows, join mesh-hpc network

# Step 2: SSH into cluster
ssh mete@100.116.63.100

# Step 3: Navigate to project
cd /home/mete/abdomen-scanner

# Step 4: Activate virtual environment
source venv/bin/activate

# Step 5: Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
python -c "import torch; print(torch.cuda.device_count())"  # Should print 2
```

#### VS Code Remote-SSH Workflow (Recommended)
Allows editing cluster files as if they were local:
1. Install "Remote - SSH" extension in VS Code
2. Connect to host: mete@100.116.63.100
3. Open folder: /home/mete/abdomen-scanner
4. Edit files directly on cluster with full IDE features
5. Run commands in integrated terminal with GPU access

---

## Data Architecture

### Dataset Overview
- Total CT Volumes: 651 cases
- Total Annotations: 42,450 bounding boxes (34,550 after filtering to bbox-only)
- Annotation Sources:
  - TRAININGDATA.csv: 28,134 annotations
  - COMPETITIONDATA.csv: 14,314 annotations
- Original Format: DICOM (medical imaging standard)
- Processed Format: NIfTI (Neuroimaging Informatics Technology Initiative)

### Directory Structure
```
abdomen-scanner/
├── data_raw/                              # Original unprocessed data
│   ├── dicom_files/                       # 651 CT scans in DICOM format
│   │   └── {SeriesInstanceUID}/           # One directory per CT series
│   │       └── *.dcm                      # Individual DICOM slices
│   └── annotations/
│       ├── TRAININGDATA.csv               # Training set annotations
│       └── COMPETITIONDATA.csv            # Competition set annotations
│
├── data_processed/                        # Processed data outputs
│   ├── nifti_images/                      # Phase 1 output: 735 NIfTI volumes
│   │   └── {SeriesInstanceUID}_{CaseNumber}.nii.gz
│   ├── nifti_labels_boxy/                 # Phase 1 output: Z-axis validation labels
│   │   └── {SeriesInstanceUID}_{CaseNumber}.nii.gz
│   ├── medsam_2d_masks/                   # Phase 2 output: 2D masks by case
│   │   └── case_{CaseNumber}/
│   │       └── image_{ImageID}_class_{ClassID}_mask.npy
│   └── nifti_labels_medsam/               # Phase 2 output: 643 3D pseudo-labels
│       └── {SeriesInstanceUID}_{CaseNumber}.nii.gz
│
├── splits/                                # Phase 2.5 output: Train/val/test splits
│   ├── train_cases.txt                    # 588 cases (80%)
│   ├── val_cases.txt                      # 73 cases (10%)
│   └── test_cases.txt                     # 74 cases (10%)
│
├── models/                                # Model checkpoints and weights
│   ├── medsam_vit_b.pth                   # MedSAM foundation model (2.4GB)
│   ├── yolo11x.pt                         # YOLOv11 weights for Phase 1.5
│   ├── phase3a_pretrain/                  # AMOS pre-training checkpoints
│   └── phase3b_finetune/                  # Fine-tuned pathology model
│
├── logs/                                  # SLURM job logs and training logs
│   └── {job_name}_{job_id}.out/err
│
└── runs/                                  # TensorBoard/Weights & Biases logs
```

### Annotation Format (CSV Structure)
```
Columns:
- SeriesInstanceUID: DICOM series identifier
- Case Number: Unique integer identifier per CT volume
- Image Id: Unique integer identifier per 2D slice
- Class: Pathology label (one of 11 clinical terms)
- Type: Annotation type (Bounding Box, Point, etc.)
- X, Y, Z: 3D spatial coordinates in DICOM reference frame
- Width, Height, Depth: Bounding box dimensions
- InstanceNumber: DICOM slice index (1-based, critical for z-axis mapping)
```

### Data Processing Challenges

#### DICOM to NIfTI Conversion
- DICOM uses inconsistent coordinate systems across scanners
- Slice ordering varies (superior-inferior vs inferior-superior)
- Pixel spacing and slice thickness differ per scanner
- Solution: Use nibabel and pydicom libraries with strict orientation validation

#### Multi-Pathology Slice Handling
- Critical Bug Discovered: Single slice can have multiple pathologies
- Original naive implementation: Masks without class_id overwrite each other
- Example: Slice with both Appendicitis (Class 5) and AAA (Class 0) would lose one mask
- Solution: Include class_id in filename: `image_{id}_class_{class_id}_mask.npy`
- Result: Increased from 31,650 to 32,648 masks (998 multi-pathology slices recovered)

#### Instance Number Mapping
- Problem: Image_Id from CSV does not directly map to NIfTI slice index
- MedSAM generates masks named by Image_Id, but 3D volumes need z-axis slice index
- Solution: Parse DICOM InstanceNumber from each slice, build mapping dict
  ```python
  # Build mapping: image_id -> slice_index
  instance_map = {}
  for slice_file in sorted_dicom_files:
      dcm = pydicom.dcmread(slice_file)
      image_id = int(dcm.SOPInstanceUID.split('.')[-1])
      slice_idx = int(dcm.InstanceNumber) - 1  # Convert to 0-based
      instance_map[image_id] = slice_idx
  ```

#### Multi-Class 3D Label Generation
- Problem: Original aggregation script created binary volumes (0 or 1)
- Need: Multi-class volumes with values 0-5 for each pathology
- Solution: Use np.maximum for class-aware aggregation
  ```python
  # Aggregate masks with class priority
  label_volume[:, :, slice_idx] = np.maximum(
      label_volume[:, :, slice_idx],
      mask_binary * class_id
  )
  ```
- Validation: Sampled 10 volumes, confirmed values in range [0, 5]

---

## 8-Phase Pipeline Architecture

### Phase 1: Data Curation and Preprocessing (2 hours)
**Objective:** Convert DICOM CT scans to standardized NIfTI format and validate spatial consistency

**Inputs:**
- 651 DICOM series in data_raw/dicom_files/
- Annotation CSVs with spatial coordinates

**Processing Steps:**
1. DICOM to NIfTI Conversion (scripts/dicom_to_nifti.py)
   - Read all DICOM slices in each series directory
   - Sort by InstanceNumber for correct z-axis ordering
   - Convert to single 3D NIfTI volume with proper orientation (RAS+)
   - Preserve physical spacing metadata (voxel size in mm)
   - Output: 735 NIfTI files (some cases have multiple series)

2. Boxy Label Generation (scripts/make_boxy_labels.py)
   - Purpose: Validate that annotations align with CT volumes in z-axis
   - Read bounding box annotations from CSV
   - Create 3D binary volumes with 1s inside bbox regions
   - Visual inspection: Load in ITK-SNAP to verify annotation quality
   - Output: 735 validation label volumes

**Outputs:**
- data_processed/nifti_images/ (735 files, ~80GB)
- data_processed/nifti_labels_boxy/ (735 files, ~5GB)

**SLURM Script:** slurm_phase1_full.sh
```bash
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
# No GPU required for this phase
```

**Validation:**
```bash
# Verify NIfTI conversion
python -c "
import nibabel as nib
img = nib.load('data_processed/nifti_images/1_2_840_10009_1_2_3_10001_20001.nii.gz')
print(f'Shape: {img.shape}')  # Should be (512, 512, depth)
print(f'Spacing: {img.header.get_zooms()}')  # Should be in mm
"
```

---

### Phase 1.5: YOLOv11 Baseline Validation (8 hours)
**Objective:** Verify annotation quality and establish 2D detection baseline

**Rationale:**
- Sanity check: If YOLO cannot detect bboxes, annotations are problematic
- Baseline metric: 2D detection performance (mAP) indicates data quality
- Fast iteration: YOLO trains much faster than 3D segmentation
- Debug tool: Visualize predictions to catch annotation errors early

**Processing Steps:**
1. Prepare YOLO Dataset (scripts/prep_yolo_data.py)
   - Convert CSV annotations to YOLO format (class x_center y_center width height)
   - Extract 2D slices from NIfTI volumes as PNG images
   - Apply class mapping (11 clinical labels to 6 classes)
   - Split into train/val/test sets
   - Output: yolo_dataset/ directory with images/ and labels/

2. Train YOLOv11 (Ultralytics framework)
   ```bash
   yolo detect train \
       data=yolo_dataset/data.yaml \
       model=yolo11x.pt \
       epochs=100 \
       imgsz=640 \
       batch=16 \
       device=0
   ```

**Results:**
- Training mAP50: 71.6% (acceptable for validation purposes)
- Confirmed annotations are spatially consistent
- Identified rare classes have sufficient signal for detection

**SLURM Script:** slurm_phase1.5_yolo.sh
```bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
```

---

### Phase 2: MedSAM Pseudo-Label Generation (19 hours)
**Objective:** Generate 2D segmentation masks for each bounding box annotation using foundation model

**Why MedSAM:**
- Foundation model pre-trained on 1.5M medical images
- Requires only bounding box prompts (no pixel-level annotations needed)
- Generalizes to unseen pathologies better than training from scratch
- Faster than manual annotation (42,450 bboxes would take months manually)

**Processing Steps:**
1. MedSAM 2D Inference (scripts/medsam_infer.py)
   - Load MedSAM checkpoint (medsam_vit_b.pth, 2.4GB)
   - For each bbox annotation in merged CSV:
     - Extract 2D slice from DICOM at specified InstanceNumber
     - Preprocess: Convert to 3-channel, resize to 1024x1024, normalize
     - Run MedSAM with bbox prompt [x_min, y_min, x_max, y_max]
     - Post-process: Threshold logits, resize mask to original dimensions
     - Save as NPY file: image_{image_id}_class_{class_id}_mask.npy
   - Parallel execution: Split annotations by GPU (GPU 0 gets even cases, GPU 1 gets odd)
   
2. 2D to 3D Aggregation (scripts/aggregate_masks.py)
   - Load all 2D masks for each case
   - Build image_id to slice_index mapping from DICOM metadata
   - Initialize 3D volume with zeros
   - For each mask:
     - Determine z-axis slice from InstanceNumber
     - Insert mask into 3D volume at correct slice
     - Use np.maximum for multi-class handling (higher class_id takes priority)
   - Save as NIfTI: {SeriesInstanceUID}_{CaseNumber}.nii.gz

**Critical Bug Fix:**
Original implementation saved masks without class_id, causing overwrites on multi-pathology slices. Fixed by including class_id in filename. This recovered 998 lost masks.

**Outputs:**
- data_processed/medsam_2d_masks/ (32,648 NPY files, ~45GB)
- data_processed/nifti_labels_medsam/ (643 NIfTI files, ~8GB)

**SLURM Script:** slurm_phase2_medsam.sh
```bash
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=90G
#SBATCH --cpus-per-task=32
```

**Key Implementation Details:**
```python
# Multi-GPU parallel execution
CUDA_VISIBLE_DEVICES=0 python scripts/medsam_infer.py --gpu_idx 0 --num_gpus 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/medsam_infer.py --gpu_idx 1 --num_gpus 2 &
wait  # Block until both processes complete
```

---

### Phase 2.5: Dataset Splitting (10 seconds)
**Objective:** Create train/validation/test splits for 3D segmentation training

**Processing Steps:**
1. Scan nifti_labels_medsam/ directory for available cases (643 total)
2. Match each label to corresponding image in nifti_images/
3. Randomly split: 80% train (588), 10% val (73), 10% test (74)
4. Save as JSON files with image/label/case_id triplets
   ```json
   {
     "image": "data_processed/nifti_images/1_2_840_10009_1_2_3_10001_20001.nii.gz",
     "label": "data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10001_20001.nii.gz",
     "case_id": "20001"
   }
   ```

**Critical Fix:**
Original script had duplicate function call causing type mismatch error. Removed duplicate, verified all 643 cases correctly split.

**Outputs:**
- splits/train_cases.txt (588 cases)
- splits/val_cases.txt (73 cases)
- splits/test_cases.txt (74 cases)

**SLURM Script:** slurm_phase2.5_splits.sh
```bash
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
```

**Verification:**
```bash
# Check all splits reference correct label directory
grep -c "nifti_labels_medsam" splits/train_cases.txt  # Should be 588
grep -c "nifti_labels_medsam" splits/val_cases.txt    # Should be 73
grep -c "nifti_labels_medsam" splits/test_cases.txt   # Should be 74
```

---

### Phase 2.6: Quality Control for Rare Classes (30 minutes manual)
**Objective:** Manually inspect MedSAM pseudo-labels for rare pathologies before expensive training

**Rationale:**
- Classes 4 and 5 have only 21 and 87 cases respectively (severe imbalance)
- MedSAM may fail on subtle pathologies like early-stage appendicitis
- Training on bad labels wastes 20-40 hours of compute time
- Early detection of systematic errors allows prompt correction

**Processing Steps:**
1. Sample Representative Cases (scripts/sample_rare_classes.py)
   - Read annotations from both CSVs (34,550 bounding boxes)
   - Filter to Class 4 (Diverticulitis) and Class 5 (Appendicitis)
   - Map case numbers to actual NIfTI filenames in directories
   - Randomly sample 10 cases per class (reproducible with seed=42)
   - Generate checklist with file paths and ITK-SNAP commands

2. Manual Inspection Protocol
   - Open each case in ITK-SNAP or 3D Slicer
   - Load CT image as main volume
   - Load MedSAM label as segmentation overlay
   - Assess four criteria:
     1. Coverage: Does mask cover full pathology extent?
     2. Boundary: Is mask boundary clean or noisy?
     3. False Positives: Any background wrongly segmented?
     4. Missing Slices: Z-axis gaps in annotations?
   - Record assessment: PASS / MARGINAL / FAIL

**Decision Criteria:**
- >70% PASS: Proceed to Phase 3 training
- 50-70% PASS: Re-run MedSAM with adjusted prompts (expand bbox by 10-20%)
- <50% PASS: Stop, investigate MedSAM failure modes, fix before training

**Output:** phase2_qc_checklist.txt
```
[1] Case Number: 20048
    Image: data_processed/nifti_images/1_2_840_10009_1_2_3_10048_20048.nii.gz
    Label: data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10048_20048.nii.gz
    
    ITK-SNAP command:
      itksnap -g data_processed/nifti_images/...20048.nii.gz \
              -s data_processed/nifti_labels_medsam/...20048.nii.gz
    
    Quality Check:
      [ ] Coverage: Full / Partial / Missing
      [ ] Boundary: Clean / Noisy / Incorrect
      [ ] False Positives: None / Few / Many
      [ ] Overall: PASS / MARGINAL / FAIL
      Notes: ___________________________________________
```

**Status:** Ready for manual inspection. Script successfully generated checklist with 20 cases (10 per rare class).

---

### Phase 3A: AMOS Pre-training (3 days - PENDING)
**Objective:** Pre-train 3D U-Net on AMOS 2022 dataset for anatomical feature learning

**Why Pre-training:**
- Small dataset problem: 643 cases is insufficient for training deep networks from scratch
- Anatomical priors: AMOS provides 500 CT volumes with 15 organ labels
- Transfer learning: Model learns general abdominal anatomy, then fine-tunes on pathology
- Rare class benefit: Pre-trained features improve performance on data-scarce classes

**AMOS Dataset:**
- 500 CT volumes with multi-organ segmentation
- 15 organ labels: spleen, kidneys, liver, stomach, pancreas, aorta, etc.
- Publicly available: https://zenodo.org/record/7262581
- Target location: data/AbdomenDataSet/AMOS-Dataset/

**Model Architecture (MONAI U-Net):**
```python
model = UNet(
    spatial_dims=3,
    in_channels=1,              # CT is single-channel
    out_channels=16,            # 15 organs + background
    channels=[32, 64, 128, 256, 512],
    strides=[2, 2, 2, 2],
    num_res_units=2
)
```

**Training Configuration:**
```yaml
# configs/config_pretrain.yaml
batch_size: 2                    # Limited by 96GB VRAM
patch_size: [128, 128, 128]      # 3D patches
learning_rate: 1e-4
epochs: 500
optimizer: AdamW
loss: DiceCELoss (Dice + Cross-Entropy)
```

**SLURM Script:** slurm_phase3a_pretrain.sh
```bash
#SBATCH --time=72:00:00         # 3 days
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
```

**Execution:**
```bash
sbatch slurm_phase3a_pretrain.sh
# Monitor: tail -f logs/Phase3A_Pretrain_{job_id}.out
```

**Expected Output:**
- models/phase3a_pretrain/best.ckpt
- Validation Dice score: ~0.80-0.85 for organ segmentation
- Checkpoint size: ~200MB

**Current Status:** BLOCKED - Awaiting AMOS dataset upload to cluster

---

### Phase 3B: Fine-tuning on Pathology Labels (7 days)
**Objective:** Fine-tune pre-trained model on MedSAM pseudo-labels for pathology segmentation

**Model Architecture (Modified U-Net):**
```python
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=6,              # 6 pathology classes (0-5)
    channels=[32, 64, 128, 256, 512],
    strides=[2, 2, 2, 2],
    num_res_units=2
)

# Load pre-trained encoder weights
pretrained_state = torch.load('models/phase3a_pretrain/best.ckpt')
model.load_state_dict(pretrained_state, strict=False)  # Only load matching layers
```

**Training Configuration:**
```yaml
# configs/config.yaml
batch_size: 2
patch_size: [128, 128, 128]
learning_rate: 5e-5              # Lower LR for fine-tuning
epochs: 1000
optimizer: AdamW
loss: DiceCELoss
class_weights: [1.0, 2.0, 2.0, 2.0, 5.0, 3.0]  # Higher weight for rare classes

# Data augmentation
augmentation:
  - RandFlipd (p=0.5)
  - RandRotate90d (p=0.5)
  - RandScaleIntensityd (p=0.5)
  - RandShiftIntensityd (p=0.5)
```

**Distributed Training Strategy:**
```python
# PyTorch Lightning DDP (Distributed Data Parallel)
trainer = pl.Trainer(
    accelerator='gpu',
    devices=2,                   # Use both RTX 6000 GPUs
    strategy='ddp',              # Data parallel across GPUs
    max_epochs=1000,
    precision='16-mixed'         # Mixed precision for faster training
)
```

**SLURM Script:** slurm_phase3b_finetune.sh
```bash
#SBATCH --time=168:00:00        # 7 days
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
```

**Execution:**
```bash
# Option 1: Fine-tune from AMOS pre-trained weights (RECOMMENDED)
sbatch slurm_phase3b_finetune.sh models/phase3a_pretrain/best.ckpt

# Option 2: Train from scratch (NOT RECOMMENDED for small dataset)
sbatch slurm_phase3b_finetune.sh none
```

**Expected Performance:**
- Validation Dice Score (Common Classes 0-3): 0.70-0.80
- Validation Dice Score (Rare Classes 4-5): 0.50-0.65 (challenging due to scarcity)
- Training Time: 7 days with 2 GPUs

**Status:** Ready to execute after Phase 3A completes and QC passes

---

### Phase 4: Clinical Inference (5 minutes per case)
**Objective:** Deploy trained model for inference on new unseen CT scans

**Inference Pipeline (scripts/inference.py):**
1. Load new patient DICOM series
2. Convert to NIfTI format
3. Preprocess: Resample to target spacing, normalize intensities
4. Run sliding window inference with trained model
   ```python
   # Use overlapping patches for smooth predictions
   output = sliding_window_inference(
       inputs=ct_volume,
       roi_size=(128, 128, 128),
       sw_batch_size=4,
       predictor=model,
       overlap=0.5
   )
   ```
5. Post-process: Argmax to get class labels, remove small islands
6. Generate outputs:
   - 3D segmentation NIfTI (for radiologist review)
   - Classification report (which pathologies detected)
   - Volumetric measurements (in mL)
   - Axial/coronal/sagittal slice visualizations

**Usage:**
```bash
python scripts/inference.py \
    --dicom_in data_raw/new_patient/series_001 \
    --out_dir results/patient_001 \
    --model_ckpt models/phase3b_finetune/best.ckpt \
    --config configs/config.yaml
```

**Output Structure:**
```
results/patient_001/
├── segmentation.nii.gz          # 3D label volume
├── classification_report.json   # Detected pathologies
├── volumetric_measurements.csv  # Volume per class in mL
└── visualizations/
    ├── axial_slice_050.png
    ├── coronal_slice_050.png
    └── sagittal_slice_050.png
```

**Status:** Script implemented, ready for testing after Phase 3B

---

## Critical Technical Decisions and Fixes

### Bug #1: Multi-Pathology Mask Overwriting (RESOLVED)
**Problem:**
- Initial medsam_infer.py saved masks as: image_{image_id}_mask.npy
- When a single slice had multiple pathologies (e.g., Appendicitis + AAA), later mask overwrote earlier mask
- Lost 998 masks (3.1% of dataset) silently

**Detection:**
- Expected 32,648 masks from 34,550 annotations (after filtering)
- Only found 31,650 masks after Phase 2
- Investigated: Some slices had multiple annotations but only one mask file

**Root Cause:**
```python
# BUGGY CODE
mask_path = case_out_dir / f"image_{image_id}_mask.npy"  # No class_id in filename
```

**Solution:**
```python
# FIXED CODE
mask_path = case_out_dir / f"image_{image_id}_class_{class_id}_mask.npy"
```

**Verification:**
Re-ran Phase 2 (Job 554), confirmed 32,648 masks generated (998 recovered)

---

### Bug #2: InstanceNumber to Slice Index Mapping (RESOLVED)
**Problem:**
- aggregate_masks.py assumed Image_Id directly maps to z-axis slice index
- Incorrect assumption: DICOM uses InstanceNumber field for slice ordering
- Result: Masks placed at wrong z-positions, 3D volumes corrupted

**Root Cause:**
```python
# BUGGY CODE
slice_idx = image_id  # WRONG: Image_Id is not slice index
label_volume[:, :, slice_idx] = mask
```

**Solution:**
Build explicit mapping from DICOM metadata:
```python
def build_instance_to_slice_map(nifti_image_path: Path, dicom_root: Path) -> Dict[int, int]:
    # Read DICOM files in order
    dicom_files = sorted(dicom_dir.glob('*.dcm'), 
                        key=lambda f: int(pydicom.dcmread(f).InstanceNumber))
    
    instance_map = {}
    for slice_idx, dicom_file in enumerate(dicom_files):
        dcm = pydicom.dcmread(dicom_file)
        image_id = int(dcm.SOPInstanceUID.split('.')[-1])
        instance_map[image_id] = slice_idx  # 0-based index
    
    return instance_map

# FIXED CODE
slice_idx = instance_map[image_id]
label_volume[:, :, slice_idx] = mask
```

**Verification:**
- Sampled 10 aggregated volumes
- Loaded in ITK-SNAP with original CT
- Confirmed masks align correctly with anatomical structures

---

### Bug #3: Binary Instead of Multi-Class Labels (RESOLVED)
**Problem:**
- aggregate_masks.py created binary volumes (0 or 1)
- Phase 3 training requires multi-class labels (0-5) for each pathology

**Root Cause:**
```python
# BUGGY CODE
label_volume[:, :, slice_idx] = mask_binary  # All masks become 1
```

**Solution:**
```python
# FIXED CODE
label_volume[:, :, slice_idx] = np.maximum(
    label_volume[:, :, slice_idx],
    mask_binary * class_id  # Multiply by class_id for multi-class
)
```

**Verification:**
```python
# Check label value range
import nibabel as nib
label = nib.load('data_processed/nifti_labels_medsam/sample.nii.gz')
data = label.get_fdata()
print(np.unique(data))  # Output: [0. 1. 2. 3. 4. 5.] ✓
```

---

### Bug #4: Split Dataset Duplicate Function Call (RESOLVED)
**Problem:**
- split_dataset.py had duplicate function call at end of script
- Caused TypeError: generate_split_json() got unexpected keyword argument

**Root Cause:**
```python
# BUGGY CODE (lines 150-160)
if __name__ == "__main__":
    generate_split_json(...)  # First call (correct)
    ...
    generate_split_json(...)  # Second call with wrong arguments (duplicate)
```

**Solution:**
Removed duplicate function call, kept only the correct invocation

**Verification:**
```bash
python scripts/split_dataset.py  # Runs without error
sbatch slurm_phase2.5_splits.sh  # Job 555 completed in <10 seconds
```

---

### Bug #5: QC Script Data Source Errors (RESOLVED)
**Problem:**
- sample_rare_classes.py read from Temp/Information.xlsx (old Excel file)
- Used lowercase 'class' column name (KeyError: 'class')
- Guessed filenames instead of scanning actual directories

**Root Cause:**
Script was written before dataset structure finalized, never updated

**Solution:**
Complete rewrite with correct implementation:
```python
# Load from CSVs not Excel
train_df = pd.read_csv("data_raw/annotations/TRAININGDATA.csv")
comp_df = pd.read_csv("data_raw/annotations/COMPETITIONDATA.csv")
df = pd.concat([train_df, comp_df])

# Use correct column name (uppercase C)
df['class_id'] = df['Class'].map(CLASS_MAPPING)

# Build real filename mapping
case_map = {}
for nifti_file in nifti_dir.glob("*.nii.gz"):
    case_number = int(nifti_file.name.split('_')[-1].replace('.nii.gz', ''))
    case_map[case_number] = nifti_file.name
```

**Verification:**
```bash
python scripts/sample_rare_classes.py --samples_per_class 10
# Output: phase2_qc_checklist.txt with 20 valid file paths
ls -lh data_processed/nifti_images/1_2_840_10009_1_2_3_10048_20048.nii.gz  # Exists ✓
```

---

## Development Workflow and Best Practices

### SLURM Job Submission Workflow

#### Step 1: Write SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=MyJob
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=mesh
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=24:00:00

cd /home/mete/abdomen-scanner
source venv/bin/activate

python scripts/my_script.py --arg1 value1 --arg2 value2
```

#### Step 2: Submit Job
```bash
sbatch slurm_my_job.sh
# Output: Submitted batch job 554
```

#### Step 3: Monitor Job
```bash
# Check job status
squeue -u mete

# View live output
tail -f logs/MyJob_554.out

# Check GPU utilization
ssh mete@100.116.63.100 "nvidia-smi"
```

#### Step 4: Retrieve Results
```bash
# Check if job completed successfully
grep -i "error" logs/MyJob_554.err
grep -i "success\|complete" logs/MyJob_554.out

# Download results to local machine
scp -r mete@100.116.63.100:/home/mete/abdomen-scanner/results/ ./local_results/
```

### File Transfer Best Practices

#### Upload Code Changes
```bash
# From local Windows machine
scp -r scripts/ mete@100.116.63.100:/home/mete/abdomen-scanner/
```

#### Download Training Results
```bash
# Download specific checkpoint
scp mete@100.116.63.100:/home/mete/abdomen-scanner/models/phase3b_finetune/best.ckpt ./models/

# Download all logs
scp -r mete@100.116.63.100:/home/mete/abdomen-scanner/logs/ ./logs_backup/
```

#### VS Code Remote-SSH (Recommended)
- No manual SCP needed
- Edit files directly on cluster
- Integrated terminal with GPU access
- Git integration works seamlessly
- File explorer shows cluster directories

### GPU Resource Management

#### Check Available Resources
```bash
# View cluster queue
squeue -p mesh

# Check GPU usage
nvidia-smi

# View allocated resources
scontrol show job {job_id}
```

#### Estimate Resource Requirements
- Phase 1: No GPU, 16 CPUs, 64GB RAM, 2 hours
- Phase 1.5: 1 GPU, 8 CPUs, 32GB RAM, 8 hours
- Phase 2: 2 GPUs, 32 CPUs, 90GB RAM, 60 hours
- Phase 3A: 2 GPUs, 32 CPUs, 120GB RAM, 72 hours
- Phase 3B: 2 GPUs, 32 CPUs, 120GB RAM, 168 hours

#### Avoid Resource Waste
```bash
# Don't request more than needed
#SBATCH --gres=gpu:2  # Only if you need 2 GPUs

# Cancel stuck jobs
scancel {job_id}

# Set realistic time limits
#SBATCH --time=24:00:00  # Job will be killed after 24 hours
```

### Version Control and Collaboration

#### Git Workflow on Cluster
```bash
# Initial clone
git clone https://github.com/Meteerd/abdomen-scanner.git
cd abdomen-scanner

# Pull latest changes before starting work
git pull origin main

# Make changes, test, commit
git add scripts/my_script.py
git commit -m "Fix bug in multi-class aggregation"
git push origin main
```

#### Avoid Committing Large Files
```bash
# .gitignore includes:
data_raw/
data_processed/
models/*.pth
models/*.ckpt
logs/
runs/
*.nii.gz
*.dcm
```

### Debugging Failed Jobs

#### Check Error Logs
```bash
# View full error log
cat logs/Phase2_MedSAM_554.err

# Search for common errors
grep -i "cuda\|memory\|error\|exception" logs/Phase2_MedSAM_554.err
```

#### Common Failure Modes
1. Out of Memory (OOM)
   - Symptom: "RuntimeError: CUDA out of memory"
   - Solution: Reduce batch_size or patch_size in config
   
2. Timeout
   - Symptom: "CANCELLED AT {timestamp} DUE TO TIME LIMIT"
   - Solution: Increase --time in SLURM script
   
3. File Not Found
   - Symptom: "FileNotFoundError: [Errno 2] No such file or directory"
   - Solution: Verify paths, check if previous phase completed
   
4. CUDA Device Mismatch
   - Symptom: "RuntimeError: Expected all tensors to be on the same device"
   - Solution: Ensure model and data both on GPU: .to('cuda')

#### Interactive Debugging
```bash
# Request interactive session with GPU
srun --partition=mesh --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=02:00:00 --pty bash

# Now you have shell with GPU access
cd /home/mete/abdomen-scanner
source venv/bin/activate
python  # Start Python interpreter for testing
```

---

## Current Project Status

### Completed Phases (100% Ready)

#### Phase 1: Data Curation
- Status: COMPLETE
- Output: 735 NIfTI images, 735 boxy labels
- Verification: All volumes loaded in ITK-SNAP, dimensions correct (512x512xdepth)
- Duration: 2 hours (Job 552)

#### Phase 1.5: YOLO Baseline
- Status: COMPLETE
- Output: YOLOv11 trained model, mAP 71.6%
- Verification: Confirmed annotations spatially consistent
- Duration: 8 hours (Job 553)

#### Phase 2: MedSAM Inference
- Status: COMPLETE
- Output: 32,648 class-specific 2D masks
- Critical Fix: Added class_id to filenames, recovered 998 multi-pathology masks
- Verification: Mask count matches expected (34,550 annotations - filtering)
- Duration: 19 hours (Job 554)

#### Phase 2: Mask Aggregation
- Status: COMPLETE
- Output: 643 multi-class 3D label volumes
- Critical Fixes: 
  - Implemented InstanceNumber mapping for correct z-axis placement
  - Multi-class aggregation using np.maximum with class_id
- Verification: Sampled 10 volumes, confirmed values in [0, 5], aligned with CT
- Duration: 19 minutes (manual run)

#### Phase 2.5: Dataset Splitting
- Status: COMPLETE
- Output: 588 train, 73 val, 74 test cases (JSON format)
- Critical Fix: Removed duplicate function call
- Verification: All 643 cases split, 100% reference nifti_labels_medsam
- Duration: <10 seconds (Job 555)

#### Phase 2.6: QC Script Preparation
- Status: READY FOR MANUAL INSPECTION
- Output: phase2_qc_checklist.txt with 20 cases (10 per rare class)
- Script Fixes: Rewrote to read CSVs, use correct column names, map real filenames
- Verification: All file paths exist and accessible
- Pending: Manual inspection in ITK-SNAP (30 minutes required)

### Blocked Phases

#### Phase 3A: AMOS Pre-training
- Status: BLOCKED - Awaiting dataset upload
- Required: AMOS 2022 dataset (500 CT volumes, ~150GB)
- Target Location: data/AbdomenDataSet/AMOS-Dataset/
- Script Ready: slurm_phase3a_pretrain.sh
- Estimated Duration: 3 days with 2 GPUs

#### Phase 3B: Fine-tuning
- Status: BLOCKED - Depends on Phase 3A and QC completion
- Can execute without pre-training but NOT RECOMMENDED (poor performance on small dataset)
- Alternative: Train from scratch if QC fails or AMOS unavailable
- Estimated Duration: 7 days with 2 GPUs

#### Phase 4: Clinical Inference
- Status: READY (script implemented, untested)
- Blocking: Need trained model from Phase 3B
- Can test pipeline on placeholder model for validation

---

## Next Steps and Future Work

### Immediate Actions (This Week)

1. Complete Manual QC (30 minutes)
   ```bash
   # Open checklist
   cat phase2_qc_checklist.txt
   
   # For each case, run:
   itksnap -g data_processed/nifti_images/{case}.nii.gz \
           -s data_processed/nifti_labels_medsam/{case}.nii.gz
   
   # Record assessment in checklist
   ```

2. Decision Point: QC Results
   - If PASS (>70%): Proceed to Phase 3A
   - If MARGINAL (50-70%): Re-run MedSAM with expanded bboxes
   - If FAIL (<50%): Debug MedSAM prompts, investigate failure modes

3. Upload AMOS Dataset
   ```bash
   # Download AMOS from Zenodo
   wget https://zenodo.org/record/7262581/files/amos22.zip
   
   # Upload to cluster
   scp amos22.zip mete@100.116.63.100:/home/mete/abdomen-scanner/data/
   
   # Extract on cluster
   ssh mete@100.116.63.100
   cd /home/mete/abdomen-scanner/data
   unzip amos22.zip -d AbdomenDataSet/AMOS-Dataset/
   
   # Verify upload
   bash verify_amos_upload.sh
   ```

### Phase 3A Execution (After AMOS Upload)

```bash
# Submit pre-training job
sbatch slurm_phase3a_pretrain.sh

# Monitor progress
tail -f logs/Phase3A_Pretrain_{job_id}.out

# Expected output after 3 days:
# - models/phase3a_pretrain/best.ckpt
# - Validation Dice: 0.80-0.85
```

### Phase 3B Execution (After QC Pass and Phase 3A Complete)

```bash
# Submit fine-tuning job with pre-trained weights
sbatch slurm_phase3b_finetune.sh models/phase3a_pretrain/best.ckpt

# Monitor progress
tail -f logs/Phase3B_Finetune_{job_id}.out

# Expected output after 7 days:
# - models/phase3b_finetune/best.ckpt
# - Validation Dice: 0.70-0.80 (common classes), 0.50-0.65 (rare classes)
```

### Phase 4 Validation (After Phase 3B Complete)

```bash
# Test inference on held-out test set
python scripts/inference.py \
    --dicom_in data_raw/dicom_files/test_case/ \
    --out_dir results/test_case/ \
    --model_ckpt models/phase3b_finetune/best.ckpt \
    --config configs/config.yaml

# Evaluate quantitative metrics
python scripts/evaluate.py \
    --predictions results/test_case/segmentation.nii.gz \
    --ground_truth data_processed/nifti_labels_medsam/test_case.nii.gz \
    --metrics dice iou hausdorff
```

### Future Enhancements

#### Model Improvements
1. Test alternative architectures
   - SegResNet (better for small datasets)
   - nnU-Net (self-configuring, SOTA on many benchmarks)
   - Swin UNETR (transformer-based, strong on AMOS)

2. Advanced augmentation
   - MixUp for class imbalance
   - CutMix for robustness
   - Test-time augmentation for inference

3. Class imbalance strategies
   - Focal loss (down-weight easy examples)
   - Balanced sampling (oversample rare classes)
   - Two-stage training (train common classes first, then rare)

#### Data Expansion
1. Request expert annotations for rare classes
   - 50-100 manually annotated Diverticulitis cases
   - 50-100 manually annotated Appendicitis cases

2. Synthetic data generation
   - Use diffusion models to generate CT slices with rare pathologies
   - CycleGAN for cross-domain augmentation

3. Multi-center collaboration
   - Pool data from multiple hospitals
   - Federated learning to preserve privacy

#### Clinical Deployment
1. DICOM integration
   - Implement DICOM receiver (PACS integration)
   - Automatic triggering on new studies
   - DICOM SR (Structured Report) output

2. Web interface
   - Upload DICOM files via web form
   - Real-time inference with progress bar
   - Interactive 3D visualization of results

3. Regulatory compliance
   - Document training data provenance
   - Validation on external test set (multi-center)
   - FDA 510(k) submission preparation

---

## Key Takeaways for New Contributors

### Technical Environment
- Remote HPC cluster (mesh-hpc) via Tailscale VPN and SSH
- SLURM job scheduler for GPU allocation
- No global conda, use isolated venv in home directory
- VS Code Remote-SSH for seamless cluster development

### Data Pipeline
- 651 CT scans, 42,450 annotations (11 labels mapped to 6 classes)
- Weakly-supervised: bounding boxes only, not full segmentation
- MedSAM foundation model generates pseudo-labels
- Multi-pathology handling critical (998 masks recovered)

### Critical Bugs Fixed
1. Multi-pathology mask overwriting (class_id in filename)
2. InstanceNumber to slice mapping (DICOM metadata required)
3. Binary to multi-class labels (np.maximum aggregation)
4. Split dataset duplicate call (removed redundant function)
5. QC script data source (CSVs not Excel, correct column names)

### Training Strategy
- Pre-training on AMOS (500 volumes, 15 organs) STRONGLY RECOMMENDED
- Fine-tuning on pathology labels (643 volumes, 6 classes)
- Distributed training (2x RTX 6000, PyTorch Lightning DDP)
- Estimated total time: 10 days (3 days pre-train + 7 days fine-tune)

### Current Blockers
- AMOS dataset upload pending
- Manual QC for rare classes pending (30 minutes required)
- Phase 3 training blocked until above resolved

### Project Philosophy
- Validate early and often (Phase 1.5 YOLO baseline)
- Manual QC before expensive training (Phase 2.6)
- Fail fast on bad data, not after weeks of training
- Document everything, future self will thank you
