# 📊 Dataset Format & Class Mapping

> **Critical information about the TR_ABDOMEN_RAD_EMERGENCY dataset**

---

## 📁 Data Source

**File:** `Temp/Information.xlsx`  
**Format:** Microsoft Excel with 2 sheets

### Sheets:
1. **TRAIININGDATA** ⚠️ (Note: 3 i's - this is the actual sheet name)
   - 28,134 total rows
   - 24,498 bounding box annotations
   - 3,636 boundary slice annotations
   - 735 unique cases

2. **COMPETITIONDATA**
   - 14,314 total rows
   - 10,052 bounding box annotations
   - 4,262 boundary slice annotations
   - Competition evaluation data

---

## 📋 Annotation Types

### 1. Bounding Box Annotations
**Type:** `"Bounding Box"`  
**Purpose:** 2D rectangular annotations marking pathology on each slice

**Format:**
| Column | Description | Example |
|--------|-------------|---------|
| `Case Number` | Case ID | 20001 |
| `Image Id` | Slice identifier | 100007 |
| `Type` | Annotation type | "Bounding Box" |
| `Class` | Pathology name | "Kidney stone" |
| `Data` | Coordinates | "251,290-262,302" |

**Data Format:** `"xmin,ymin-xmax,ymax"` (pixel coordinates)

### 2. Boundary Slice Annotations
**Type:** `"Boundary Slice"`  
**Purpose:** Mark start/end of anatomical structures in z-axis (3D extent)

**Format:**
| Column | Description | Example |
|--------|-------------|---------|
| `Case Number` | Case ID | 20003 |
| `Image Id` | Boundary slice ID | 100497 |
| `Type` | Annotation type | "Boundary Slice" |
| `Class` | Anatomy name | "Pancreas" |
| `Data` | Not used | NaN |

**Critical:** Each anatomy has **2 boundary slices** (start + end) per case.

**Example - Case 20003:**
```
Pancreas: z=[100497, 100594] → 98 slices
Kidney-Bladder: z=[100503, 100838] → 336 slices
appendix: z=[100729, 100748] → 20 slices
```

---

## 🎯 Class Mapping: 11 Radiologist Labels → 6 Competition Classes

Based on Koç et al. (2024) Table 2:

### Competition Class 1: AAA/AAD (n=9,783) ✅ Well-represented
**Radiologist Labels:**
- `Abdominal aortic aneurysm` (n=8,977)
- `Abdominal aortic dissection` (n=806)

**Anatomy:** `Abdominal Aorta`

---

### Competition Class 2: Acute Pancreatitis (n=6,923) ✅ Well-represented
**Radiologist Labels:**
- `Compatible with acute pancreatitis` (n=6,923)

**Anatomy:** `Pancreas`

---

### Competition Class 3: Cholecystitis (n=6,265) ✅ Well-represented
**Radiologist Labels:**
- `Compatible with acute cholecystitis` (n=4,925)
- `Gallbladder stone` (n=1,340)

**Anatomy:** `Gall bladder`

---

### Competition Class 4: Kidney/Ureteral Stones (n=1,405) ⚠️ Moderate
**Radiologist Labels:**
- `Kidney stone` (n=1,114)
- `ureteral stone` (n=291)

**Anatomy:** `Kidney-Bladder`

---

### Competition Class 5: Diverticulitis (n=54) 🚨 CRITICAL IMBALANCE
**Radiologist Labels:**
- `Calcified diverticulum` (n=36)
- `Compatible with acute diverticulitis` (n=18)

**Anatomy:** `Colon`

**⚠️ Problem:** Only 54 annotations (181× fewer than Class 1)  
**Solution:** Transfer learning pre-training (GAP 3) on public datasets

---

### Competition Class 6: Appendicitis (n=54) 🚨 CRITICAL IMBALANCE
**Radiologist Labels:**
- `Compatible with acute appendicitis` (n=54)

**Anatomy:** `appendix`

**⚠️ Problem:** Only 54 annotations (181× fewer than Class 1)  
**Solution:** Heavy oversampling + weighted loss

---

## 📊 Class Distribution Visualization

```
Class 1 (AAA/AAD):         ████████████████████ 9,783 (39.9%)
Class 2 (Pancreatitis):    ██████████████       6,923 (28.3%)
Class 3 (Cholecystitis):   █████████████        6,265 (25.6%)
Class 4 (Kidney/Ureteral): ███                  1,405 (5.7%)
Class 5 (Diverticulitis):  ░                       54 (0.2%) ⚠️
Class 6 (Appendicitis):    ░                       54 (0.2%) ⚠️
```

**Imbalance Ratio:** 181:1 (Class 1 vs Class 5/6)

---

## 🔬 Anatomical Boundaries

These define valid z-axis ranges for each pathology class:

| Anatomy | Purpose | Typical Span |
|---------|---------|--------------|
| `Abdominal Aorta` | AAA/AAD validation | ~140-200 slices |
| `Pancreas` | Pancreatitis validation | ~80-120 slices |
| `Gall bladder` | Cholecystitis validation | ~50-80 slices |
| `Kidney-Bladder` | Kidney stone validation | ~300-400 slices |
| `Colon` | Diverticulitis validation | ~300-500 slices |
| `appendix` | Appendicitis validation | ~15-30 slices |

**Critical (GAP 1 Fix):** Bounding boxes are only drawn within their anatomy's valid z-range, preventing anatomically impossible labels.

---

## 🛠️ Implementation

### Python Class Mapping Dictionary
```python
CLASS_MAPPING = {
    'background': 0,
    
    # Competition Class 1: AAA/AAD
    'Abdominal aortic aneurysm': 1,
    'Abdominal aortic dissection': 1,
    
    # Competition Class 2: Acute Pancreatitis
    'Compatible with acute pancreatitis': 2,
    
    # Competition Class 3: Cholecystitis
    'Compatible with acute cholecystitis': 3,
    'Gallbladder stone': 3,
    
    # Competition Class 4: Kidney/Ureteral Stones
    'Kidney stone': 4,
    'ureteral stone': 4,
    
    # Competition Class 5: Diverticulitis
    'Compatible with acute diverticulitis': 5,
    'Calcified diverticulum': 5,
    
    # Competition Class 6: Appendicitis
    'Compatible with acute appendicitis': 6,
}
```

### Reading Data
```python
import pandas as pd

# Read training annotations
df = pd.read_excel('Temp/Information.xlsx', sheet_name='TRAIININGDATA')

# Filter bounding boxes
bboxes = df[df['Type'] == 'Bounding Box']

# Filter boundaries
boundaries = df[df['Type'] == 'Boundary Slice']

# Get case annotations
case_20003 = df[df['Case Number'] == 20003]
```

---

## 📝 Critical Notes

### 1. Sheet Name Typo
⚠️ **The sheet name is `TRAIININGDATA` (3 i's)** - this is NOT a typo, it's the actual name in the Excel file.

### 2. Image ID vs Slice Index
- `Image Id` is the DICOM slice identifier (e.g., 100007)
- Slice index in NIfTI volume may differ
- Need proper mapping (handled in `dicom_to_nifti.py`)

### 3. Coordinate Format
- Bounding boxes use `xmin,ymin-xmax,ymax` format
- Coordinates are in **pixel space** (not mm)
- Origin is top-left (standard image coordinates)

### 4. Boundary Validation
- Each anatomy should have **exactly 2 boundaries** per case
- Some cases may have only 1 (edge case - handle gracefully)
- Boundaries define **inclusive** z-range `[start, end]`

---

## 🔍 Data Validation

Run validation test:
```bash
python scripts/test_gap1_fix.py
```

This verifies:
- ✅ All 11 pathology classes map correctly
- ✅ Anatomical boundaries extract properly
- ✅ Z-axis validation filters invalid annotations
- ✅ Class distribution matches expectations

---

## 📚 References

- **Paper:** Koç et al. (2024) - Abdominal emergency segmentation dataset
- **Dataset:** TR_ABDOMEN_RAD_EMERGENCY
- **Source:** Provided as `Information.xlsx`

---

**Last Updated:** October 31, 2025  
**Validation Status:** ✅ Tested with `test_gap1_fix.py`
