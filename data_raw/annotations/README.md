# Annotations Directory

## 🚨 CRITICAL: Two Completely Different Datasets

**TR_ABDOMEN_RAD_EMERGENCY contains TWO separate datasets with overlapping case numbers but DIFFERENT scans!**

### Dataset Structure

```
data/AbdomenDataSet/
├── Training-DataSets/          ← 735 unique CT scans
│   ├── 20001/                  ← Patient A (e.g., ureteral stone, 224 slices)
│   ├── 20002/
│   └── 20736/
└── Competition-DataSets/       ← 357 unique CT scans
    ├── 20001/                  ← Patient B (different from Training!) (e.g., appendicitis, 404 slices)
    ├── 20002/
    └── 20359/
```

**CRITICAL INSIGHT:** Case numbers 20001-20356 exist in BOTH directories but are **completely different patients**:
- Different DICOM files (different Image IDs)
- Different number of slices
- Different pathologies
- Different anatomy

### CSV Annotation Files

**Location:** `data_raw/annotations/`

| File | Rows | Bboxes | Cases | Maps To |
|------|------|--------|-------|---------|
| **TRAININGDATA.csv** | 28,134 | 24,498 | 735 | Training-DataSets/ |
| **COMPETITIONDATA.csv** | 14,314 | 10,052 | 357 | Competition-DataSets/ |
| **TOTAL** | 42,448 | 34,550 | 1,092 | Both directories |

**CSV Columns:**
- `Case Number` - Case ID (e.g., 20001) - **SHARED across both datasets!**
- `Image Id` - DICOM filename as integer (e.g., 100007 = file "100007.dcm")
- `Type` - Either "Bounding Box" or "Boundary Slice"
- `Class` - Pathology or anatomy name
- `Data` - Coordinates for bboxes ("xmin,ymin-xmax,ymax"), NaN for boundaries

### Example: Case 20001 (Exists in Both Datasets)

**Training-DataSets/case_20001:**
```
data/AbdomenDataSet/Training-DataSets/20001/
├── 100007.dcm
├── 100008.dcm
└── 100010.dcm  (3 DICOM files total)
```

**TRAININGDATA.csv (Case Number=20001):**
```csv
Case Number,Image Id,Type,Class,Data
20001,100007,Bounding Box,ureteral stone,"251,290-262,302"
20001,100008,Bounding Box,ureteral stone,"251,291-261,301"
20001,100010,Boundary Slice,Kidney-Bladder,
```

**Competition-DataSets/case_20001:**
```
data/AbdomenDataSet/Competition-DataSets/20001/
├── 100014.dcm
├── 100017.dcm
├── 100018.dcm
└── ... (20 DICOM files total - COMPLETELY DIFFERENT from Training!)
```

**COMPETITIONDATA.csv (Case Number=20001):**
```csv
Case Number,Image Id,Type,Class,Data
20001,100014,Boundary Slice,Colon,0
20001,100017,Boundary Slice,Abdominal Aorta,0
20001,100063,Bounding Box,Compatible with acute appendicitis,"177,255-196,266"
20001,100064,Bounding Box,Compatible with acute appendicitis,"180,252-197,267"
...
```

**Key Differences:**
- Different Image IDs (100007 vs 100014-100091)
- Different pathologies (ureteral stone vs appendicitis)
- Different number of annotations (3 vs 21)
- **These are NOT the same patient!**

### Annotation Types

1. **Bounding Box**: 2D rectangular annotation marking pathology on a specific DICOM slice
   - Example: `Type="Bounding Box"`, `Class="Kidney stone"`, `Data="251,290-262,302"`
   - Format: "xmin,ymin-xmax,ymax" (pixel coordinates)
   - Used for training segmentation models

2. **Boundary Slice**: Marks anatomical structure extent in z-axis (3D range validation)
   - Example: `Type="Boundary Slice"`, `Class="Pancreas"`, `Data=NaN`
   - Each anatomy has 2 boundaries per case (cranial + caudal limits)
   - Ensures bounding boxes are only drawn within valid anatomical regions
   - Critical for GAP 1 fix (z-axis validation)

### Pipeline Data Flow

**Phase 1: DICOM → NIfTI Conversion**

Script: `scripts/dicom_to_nifti.py`

```bash
Input:  data/AbdomenDataSet/Training-DataSets/20001/*.dcm
        data/AbdomenDataSet/Competition-DataSets/20001/*.dcm

Output: data_processed/nifti_images/TRAIN_20001.nii.gz    (735 files)
        data_processed/nifti_images/COMP_20001.nii.gz     (357 files)
        TOTAL: 1,092 NIfTI files
```

**Key Logic:**
- Scans `Training-DataSets/` → creates `TRAIN_{case_number}.nii.gz`
- Scans `Competition-DataSets/` → creates `COMP_{case_number}.nii.gz`
- Prefix prevents confusion between datasets
- Sorts slices by InstanceNumber, stacks into 3D volume

**Phase 1.5: Generate Boxy Labels (with Z-Axis Validation)**

Script: `scripts/make_boxy_labels.py`

```python
# Load and tag CSV annotations
train_df = pd.read_csv('data_raw/annotations/TRAININGDATA.csv')
train_df['dataset_source'] = 'TRAIN'  # Tag each row

comp_df = pd.read_csv('data_raw/annotations/COMPETITIONDATA.csv')
comp_df['dataset_source'] = 'COMP'    # Tag each row

combined_df = pd.concat([train_df, comp_df])  # Safe to merge now!

# Process each NIfTI file
for nifti_file in ["TRAIN_20001.nii.gz", "COMP_20001.nii.gz", ...]:
    # Parse filename
    if nifti_file.startswith('TRAIN_'):
        case_number = 20001
        dataset_source = 'TRAIN'
    elif nifti_file.startswith('COMP_'):
        case_number = 20001
        dataset_source = 'COMP'
    
    # Filter annotations by BOTH case number AND dataset
    case_annotations = combined_df[
        (combined_df['Case Number'] == case_number) &
        (combined_df['dataset_source'] == dataset_source)
    ]
    
    # Get anatomical boundaries (also filtered by dataset)
    boundaries = get_anatomical_boundaries(
        case_number, boundary_df, dataset_source
    )
    
    # Draw bounding boxes (only within valid z-range)
    for bbox in case_annotations:
        image_id = bbox['Image Id']
        class_name = bbox['Class']
        coords = parse_bbox(bbox['Data'])  # "251,290-262,302"
        
        # Check if bbox is within anatomical z-range
        if is_valid_z_position(class_name, image_id, boundaries):
            draw_bbox_on_slice(label_volume, image_id, coords)
    
    # Save with matching prefix
    save(label_volume, f"{dataset_source}_{case_number}.nii.gz")
```

```bash
Output: data_processed/nifti_labels_boxy/TRAIN_20001.nii.gz  (matches input)
        data_processed/nifti_labels_boxy/COMP_20001.nii.gz   (matches input)
```

**Phase 2: MedSAM Pseudo-Mask Generation**

Script: `scripts/medsam_infer.py`

```python
# Load annotations (already has 'dataset' column)
train_df = pd.read_csv('TRAININGDATA.csv')
train_df['dataset'] = 'Training-DataSets'

comp_df = pd.read_csv('COMPETITIONDATA.csv')
comp_df['dataset'] = 'Competition-DataSets'

df = pd.concat([train_df, comp_df])

# Build DICOM manifest with dataset tracking
manifest = {}
for dicom_file in scan_all_dicoms():
    case_num = extract_case_number(dicom_file)
    image_id = extract_image_id(dicom_file)  # From filename
    dataset = 'Training-DataSets' or 'Competition-DataSets'
    
    manifest[(case_num, image_id, dataset)] = dicom_file

# Process each annotation
for annotation in df:
    case_num = annotation['Case Number']
    image_id = annotation['Image Id']
    dataset = annotation['dataset']
    
    # Look up DICOM using 3-tuple key (case, image, dataset)
    dicom_path = manifest[(case_num, image_id, dataset)]
    
    # Generate mask and save
    dataset_prefix = 'TRAIN' if dataset == 'Training-DataSets' else 'COMP'
    save_mask(f"data_processed/medsam_2d_masks/{dataset_prefix}_case_{case_num}/")
```

```bash
Output: data_processed/medsam_2d_masks/TRAIN_case_20001/image_100007_class_4_mask.npy
        data_processed/medsam_2d_masks/COMP_case_20001/image_100014_class_6_mask.npy
```

**Phase 2 (continued): Aggregate 2D → 3D Labels**

Script: `scripts/aggregate_masks.py`

```bash
Input:  data_processed/medsam_2d_masks/TRAIN_case_20001/*.npy (2D masks)
        data_processed/nifti_images/TRAIN_20001.nii.gz (reference geometry)

Output: data_processed/nifti_labels_medsam/TRAIN_20001.nii.gz (3D label volume)
```

**Phase 2.5: Create Training Splits**

Script: `scripts/split_dataset.py`

```bash
Input:  data_processed/nifti_images/TRAIN_*.nii.gz (735 files)
        data_processed/nifti_images/COMP_*.nii.gz (357 files)

Output: splits/train_cases.txt   (e.g., "TRAIN_20001", "COMP_20050", ...)
        splits/val_cases.txt
        splits/test_cases.txt
```

### Critical Mapping Rules

**Rule 1: Filename determines dataset source**
```
TRAIN_20001.nii.gz → Filter TRAININGDATA.csv (dataset_source='TRAIN')
COMP_20001.nii.gz  → Filter COMPETITIONDATA.csv (dataset_source='COMP')
```

**Rule 2: CSV filtering requires BOTH conditions**
```python
# WRONG (will mix datasets!)
annotations = df[df['Case Number'] == 20001]

# CORRECT (keeps datasets separate)
annotations = df[
    (df['Case Number'] == 20001) &
    (df['dataset_source'] == 'TRAIN')  # or 'COMP'
]
```

**Rule 3: Image ID comes from DICOM filename**
```
DICOM file: data/.../20001/100007.dcm
CSV row:    Case Number=20001, Image Id=100007
Match:      Image Id (100007) == DICOM filename stem
```

**Rule 4: All pipeline phases use consistent prefixes**
- Phase 1: `TRAIN_20001.nii.gz`, `COMP_20001.nii.gz`
- Phase 1.5: `TRAIN_20001.nii.gz` (boxy labels)
- Phase 2: `TRAIN_case_20001/` (MedSAM masks), `TRAIN_20001.nii.gz` (aggregated)
- Phase 2.5: `TRAIN_20001`, `COMP_20001` (split files)
- Phase 3: Training loop reads prefixed filenames directly

### Scripts Using This Data

✅ **All scripts updated to handle dataset separation:**
- `scripts/dicom_to_nifti.py` - Creates TRAIN_/COMP_ prefixed NIfTI files
- `scripts/make_boxy_labels.py` - Tags CSV with `dataset_source`, filters by both case + dataset
- `scripts/medsam_infer.py` - Uses `dataset` column, 3-tuple manifest keys
- `scripts/aggregate_masks.py` - Matches TRAIN_case_/COMP_case_ to TRAIN_/COMP_ files
- `scripts/split_dataset.py` - Recognizes and preserves TRAIN_/COMP_ prefixes
- `scripts/train_monai.py` - Loads data using prefixed filenames

### Class Mapping

**11 Radiologist Labels → 6 Competition Classes:**

| Class | Radiologist Labels | Training | Competition | TOTAL | Recovery |
|-------|-------------------|----------|-------------|-------|----------|
| 0 | Background | N/A | N/A | N/A | - |
| 1 | Abdominal aortic aneurysm/dissection | 7,952 | 1,831 | **9,783** | ✓ |
| 2 | Compatible with acute pancreatitis | 5,842 | 1,081 | **6,923** | ✓ |
| 3 | Compatible with acute cholecystitis, Gallbladder stone | 5,398 | 867 | **6,265** | ✓ |
| 4 | Kidney stone, ureteral stone | 1,251 | 154 | **1,405** | ✓ |
| 5 | Compatible with acute diverticulitis, Calcified diverticulum | 0 | 0 | **0** | - |
| 6 | Compatible with acute appendicitis | 54 | 2,229 | **2,283** | ⚠️ **95% recovered!** |

**CRITICAL: Class 6 (Appendicitis) Recovery**
- OLD pipeline: Only processed Training → 4 cases (54 bboxes)
- NEW pipeline: Processes both datasets → 87 cases (2,283 bboxes)
- **Recovery: 2,175% increase in annotations, 2,075% increase in cases!**
- 95% of appendicitis cases were in Competition dataset (ignored by old pipeline)

**Class Mapping Code:**
```python
CLASS_MAPPING = {
    'Abdominal aortic aneurysm/dissection': 1,
    'Compatible with acute pancreatitis': 2,
    'Compatible with acute cholecystitis': 3,
    'Gallbladder stone': 3,  # Maps to same class
    'Kidney stone': 4,
    'ureteral stone': 4,     # Maps to same class
    'Compatible with acute diverticulitis': 5,
    'Calcified diverticulum': 5,  # Maps to same class
    'Compatible with acute appendicitis': 6,
}
```

### Reading the Data

**Option 1: Load CSVs directly (current implementation)**
```python
import pandas as pd

# Load both CSV files and tag with dataset source
train_df = pd.read_csv('data_raw/annotations/TRAININGDATA.csv')
train_df['dataset_source'] = 'TRAIN'

comp_df = pd.read_csv('data_raw/annotations/COMPETITIONDATA.csv')
comp_df['dataset_source'] = 'COMP'

# Merge safely (now each row is tagged)
df = pd.concat([train_df, comp_df], ignore_index=True)

# Filter by case AND dataset
case_20001_train = df[
    (df['Case Number'] == 20001) &
    (df['dataset_source'] == 'TRAIN')
]

case_20001_comp = df[
    (df['Case Number'] == 20001) &
    (df['dataset_source'] == 'COMP')
]
```

**Option 2: Load from Excel (legacy, for reference)**
```python
import pandas as pd

# Read Excel sheets (note: "TRAIININGDATA" has 3 i's)
train_df = pd.read_excel('Temp/Information.xlsx', sheet_name='TRAIININGDATA')
comp_df = pd.read_excel('Temp/Information.xlsx', sheet_name='COMPETITIONDATA')
```

### Verification Commands

**Check dataset separation:**
```bash
# Count NIfTI files
ls data_processed/nifti_images/TRAIN_*.nii.gz | wc -l  # Should be 735
ls data_processed/nifti_images/COMP_*.nii.gz | wc -l   # Should be 357

# Check CSV annotations
wc -l data_raw/annotations/TRAININGDATA.csv      # 28,135 lines (28,134 + header)
wc -l data_raw/annotations/COMPETITIONDATA.csv   # 14,315 lines (14,314 + header)

# Verify no Image ID overlap for case 20001
python3 << 'EOF'
import pandas as pd
train = pd.read_csv('data_raw/annotations/TRAININGDATA.csv')
comp = pd.read_csv('data_raw/annotations/COMPETITIONDATA.csv')

train_ids = set(train[train['Case Number']==20001]['Image Id'])
comp_ids = set(comp[comp['Case Number']==20001]['Image Id'])

print(f"Training Image IDs: {sorted(train_ids)}")
print(f"Competition Image IDs: {sorted(comp_ids)[:5]}...")
print(f"Overlap: {len(train_ids & comp_ids)} (should be 0)")
EOF
```

### Data Quality Assurance

**Pre-Processing Checks:**
1. ✓ CSV files exist in `data_raw/annotations/`
2. ✓ Both Training-DataSets/ and Competition-DataSets/ directories exist
3. ✓ Case numbers match between DICOM directories and CSV files
4. ✓ Image IDs match DICOM filenames (e.g., 100007.dcm → Image Id=100007)
5. ✓ No Image ID overlap between datasets (for same case numbers)

**Post-Processing Validation:**
1. ✓ Total NIfTI files = 1,092 (735 TRAIN + 357 COMP)
2. ✓ All bounding boxes matched to correct dataset (34,550 total)
3. ✓ Rare class counts correct (especially appendicitis: 87 cases)
4. ✓ Train/val/test splits contain both TRAIN_ and COMP_ prefixes

### Documentation

See [docs/DATA_FORMAT.md](../../docs/DATA_FORMAT.md) for complete details on:
- Class mapping definitions
- Z-axis boundary validation methodology
- Class distribution analysis
- Training strategies for imbalanced classes

See [README.md](../../README.md) for:
- Complete pipeline workflow
- Expected outputs per phase
- SLURM job execution commands

---

**Version:** 2.0 (Dataset Separation Fix Applied - November 6, 2025)  
**Critical Fix:** Training and Competition datasets now processed separately throughout entire pipeline  
**Impact:** Recovered 2,175% more appendicitis annotations (54 → 2,283 bboxes)
