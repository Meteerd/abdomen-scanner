# Abdominal Emergency AI Segmentation

Automated 3D segmentation of critical abdominal emergencies from CT scans using weakly-supervised learning.

**Tech Stack:** Python 3.12 | PyTorch 2.0+ | MONAI | Lightning | YOLOv11  
**Hardware:** mesh-hpc cluster (2x RTX 6000 96GB, 128 CPUs, 128GB RAM)

---

## 🚨 CRITICAL: Dataset Structure (READ FIRST)

**Training and Competition datasets contain COMPLETELY DIFFERENT CT scans despite sharing case numbers!**

- **Training-DataSets/case_20001** = Patient A (e.g., ureteral stone, 224 slices)
- **Competition-DataSets/case_20001** = Patient B (e.g., appendicitis, 404 slices)

**These are NOT the same patient!** Same case numbers, different scans, different pathologies.

**Solution:** All pipeline phases use TRAIN_/COMP_ prefixes to keep datasets separate:
- `TRAIN_20001.nii.gz` → Annotations from TRAININGDATA.csv
- `COMP_20001.nii.gz` → Annotations from COMPETITIONDATA.csv

**Expected Dataset Sizes:**
- Training-DataSets: 735 cases (case_20001 through case_20736, not all numbers exist)
- Competition-DataSets: 357 cases (case_20001 through case_20359)
- **Total: 1,092 unique CT scans** (NOT 735!)
- 356 case numbers overlap but contain different scans

See `data_raw/annotations/README.md` for complete data mapping documentation.

---

## Quick Start

**Essential Files:**
- `docs/QUICKSTART.md` - Execution workflow (6 phases)
- `docs/DATA_FORMAT.md` - Dataset structure and class mappings

**Complete workflow:**
```bash
# Phase 1: DICOM → NIfTI conversion (2h, creates 1,092 files)
sbatch slurm_phase1_full.sh
# Output: data_processed/nifti_images/
#   - TRAIN_20001.nii.gz, TRAIN_20002.nii.gz, ... (735 files)
#   - COMP_20001.nii.gz, COMP_20002.nii.gz, ... (357 files)

# Phase 1.5: YOLO validation (8h)
sbatch slurm_phase1.5_yolo.sh

# Phase 2: MedSAM pseudo-mask generation (12h)
sbatch slurm_phase2_medsam.sh
# Output: data_processed/medsam_2d_masks/
#   - TRAIN_case_20001/, TRAIN_case_20002/, ... (2D masks per case)
#   - COMP_case_20001/, COMP_case_20002/, ...
# Output: data_processed/nifti_labels_medsam/
#   - TRAIN_20001.nii.gz, COMP_20001.nii.gz, ... (3D aggregated labels)

# Phase 2.5: Create 3D training splits (5min)
sbatch slurm_phase2.5_splits.sh
# Output: splits/train_cases.txt, val_cases.txt, test_cases.txt
#   (contains both TRAIN_* and COMP_* case IDs)

# Phase 2.6: Manual QC (30min)
python scripts/sample_rare_classes.py

# Phase 3.A: AMOS pre-training (3d)
sbatch slurm_phase3a_pretrain.sh

# Phase 3.B: Fine-tuning on pathology (7d)
sbatch slurm_phase3b_finetune.sh [checkpoint]

# Phase 4: Clinical inference (5min per case)
python scripts/inference.py \
    --dicom_in data_raw/new_patient/series_001 \
    --out_dir results/patient_001 \
    --model_ckpt models/phase3b_finetune/best.ckpt \
    --config configs/config.yaml
```

---

## Project Status

| Phase | Duration | Description | Status |
|-------|----------|-------------|--------|
| **1: Data Curation** | 2h | DICOM→NIfTI + Z-axis validation | ✅ Ready |
| **1.5: YOLO Baseline** | 8h | Label validation (mAP>0.70) | ✅ Ready |
| **2: Pseudo-Masks** | 12h | MedSAM inference | ✅ Ready |
| **2.5: Create Splits** | 5min | JSON splits for 3D training | ✅ Ready |
| **2.6: Manual QC** | 30min | Rare class quality check | ✅ Ready |
| **3.A: Pre-training** | 3d | AMOS 2022 anatomy model | ⏳ Ready (awaiting data) |
| **3.B: Fine-tuning** | 7d | Pathology model | ✅ Ready |
| **4: Clinical Inference** | 5min | DICOM→Classification+Volume | ✅ Ready |

**Waiting on:** AMOS 2022 dataset upload to `data/AbdomenDataSet/AMOS-Dataset/`

---

## Repository Structure

```
abdomen-scanner/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── verify_setup_local.py        # Environment verification
├── verify_amos_upload.sh        # AMOS data verification
│
├── configs/
│   ├── config.yaml              # Phase 3.B (pathology training)
│   └── config_pretrain.yaml     # Phase 3.A (AMOS pre-training)
│
├── docs/
│   ├── QUICKSTART.md            # Execution workflow
│   └── DATA_FORMAT.md           # Dataset documentation
│
├── scripts/
│   ├── train_monai.py           # Main training script
│   ├── inference.py             # Phase 4 clinical inference
│   ├── transforms_amos.py       # AMOS preprocessing
│   ├── prepare_amos_dataset.py  # AMOS split generation
│   ├── dicom_to_nifti.py        # DICOM conversion
│   ├── aggregate_masks.py       # 2D→3D aggregation
│   ├── medsam_infer.py          # MedSAM inference
│   ├── prep_yolo_data.py        # Bbox to YOLO format
│   ├── sample_rare_classes.py   # QC sampling
│   ├── split_dataset.py         # Train/val/test splits
│   └── make_boxy_labels.py      # Z-axis validation labels
│
├── slurm_phase1_full.sh         # Phase 1 SLURM script
├── slurm_phase1.5_yolo.sh       # Phase 1.5 SLURM script
├── slurm_phase2_medsam.sh       # Phase 2 SLURM script
├── slurm_phase2.5_splits.sh     # Phase 2.5 SLURM script (create splits)
├── slurm_phase3a_pretrain.sh    # Phase 3.A SLURM script
└── slurm_phase3b_finetune.sh    # Phase 3.B SLURM script
```

**Data directories** (not in Git):
- `data/` - Processed data, meta.csv
- `data_raw/` - Raw DICOM files
- `models/` - Trained checkpoints
- `splits/` - Train/val/test case lists

---
│   └── annotations/
│       ├── README.md            # ⭐ Explains Excel vs CSV format
│       ├── TRAININGDATA.csv     # Placeholder only
## Dataset

### TR_ABDOMEN_RAD_EMERGENCY Dataset

**CRITICAL:** This dataset has two completely separate subdirectories with different scans:

| Directory | Cases | Annotations | Bboxes | Source CSV |
|-----------|-------|-------------|--------|------------|
| **Training-DataSets/** | 735 | 28,134 | 24,498 | TRAININGDATA.csv |
| **Competition-DataSets/** | 357 | 14,314 | 10,052 | COMPETITIONDATA.csv |
| **Case Number Overlap** | 356 | - | - | Different scans! |
| **TOTAL (Unique Scans)** | **1,092** | **42,448** | **34,550** | Both CSVs |

**Key Insight:** 356 case numbers (20001-20356) exist in BOTH directories but contain **completely different patients**:
- Training-DataSets/20001: Ureteral stone patient (Image IDs: 100007, 100008, 100010)
- Competition-DataSets/20001: Appendicitis patient (Image IDs: 100014, 100017-100091)

**Data Source:**
- DICOM files: `data/AbdomenDataSet/Training-DataSets/` and `Competition-DataSets/`
- Annotations: `data_raw/annotations/TRAININGDATA.csv` and `COMPETITIONDATA.csv`
- Each CSV maps Case Number + Image ID to bounding box coordinates

**Pipeline Handling:**
- All phases use `TRAIN_` and `COMP_` prefixes to prevent mixing datasets
- CSV annotations tagged with `dataset_source` column ('TRAIN' or 'COMP')
- Matching enforced by BOTH case number AND dataset source

### Target Classes (6 Pathologies)

**Mapped from 11 radiologist labels → 6 competition classes:**

| Class | Pathology | Training | Competition | TOTAL | Notes |
|-------|-----------|----------|-------------|-------|-------|
| 1 | AAA/AAD | 7,952 | 1,831 | **9,783** | Well-represented |
| 2 | Pancreatitis | 5,842 | 1,081 | **6,923** | Well-represented |
| 3 | Cholecystitis | 5,398 | 867 | **6,265** | Well-represented |
| 4 | Kidney/Ureteral Stones | 1,251 | 154 | **1,405** | Moderate |
| 5 | Diverticulitis | 0 | 0 | **0** | ⚠️ RARE (see note) |
| 6 | Appendicitis | 54 | 2,229 | **2,283** | 95% in Competition! |

**Critical Note on Appendicitis:** OLD pipeline only processed Training (4 cases). NEW pipeline processes both datasets (87 cases total = 4 Training + 83 Competition). This recovers 95% of missing appendicitis cases!

**Class Imbalance Solution:**
- AMOS 2022 pre-training (500 CT scans, 15 anatomical structures)
- Aggressive class weights: [0.5, 1.0, 1.0, 1.0, 5.0, 100.0, 100.0]
- Rare class oversampling during training

**Complete documentation:** `data_raw/annotations/README.md` and `docs/DATA_FORMAT.md`

---

## Setup

**Cluster Access:**
```bash
ssh mete@100.116.63.100
cd /home/mete/abdomen-scanner
source venv/bin/activate
```

**Verify Environment:**
```bash
python verify_setup_local.py
```

**AMOS Dataset Upload:**
1. Upload to `data/AbdomenDataSet/AMOS-Dataset/` via WinSCP
2. Verify: `./verify_amos_upload.sh`
3. Prepare: `python scripts/prepare_amos_dataset.py`

---

## Troubleshooting

**SLURM Job Issues:**
```bash
squeue -u mete                    # Check job status
scancel <job_id>                  # Cancel job
tail -f logs/*.out                # View logs
```

**Out of Memory:**
- Reduce `batch_size` in config.yaml
- Reduce `patch_size` to [160, 160, 128]

**AMOS Dataset Not Found:**
```bash
# Verify upload
ls -l data/AbdomenDataSet/AMOS-Dataset/ | head
find data/AbdomenDataSet/AMOS-Dataset/ -type d | wc -l
```

---

## License

Private / Proprietary - Medical imaging data, not for public distribution

---

## Acknowledgments

- TR_ABDOMEN_RAD_EMERGENCY Dataset (Koç et al. 2024)
- AMOS 2022 Dataset (https://amos22.grand-challenge.org/)
- MedSAM (Ma et al. 2023)
- YOLOv11 (Ultralytics)
- MONAI Framework
- PyTorch Lightning
- mesh-hpc cluster access

**Version:** 1.0.0  
**Status:** Ready for execution  
**Last Updated:** November 3, 2025

## Acknowledgments

- **TR_ABDOMEN_RAD_EMERGENCY Dataset** - Koç et al., 2024
- **AMOS 2022 Dataset** - Ji et al., 2022
- **MedSAM** - Ma et al., 2023
- **MONAI Framework** - Project MONAI Consortium
- **YOLOv11** - Ultralytics
- **mesh-hpc cluster** - 6-month access

---

**Last Updated:** November 1, 2025  
**Version:** 1.0.0 (All 6 phases implemented)  
**Status:** Ready for execution
