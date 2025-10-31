# Annotations Directory

## ⚠️ Important: Dataset Format Changed

**Original placeholder files:** `TRAININGDATA.csv`, `COMPETITIONDATA.csv`  
**Actual data format:** Excel file at `Temp/Information.xlsx`

### Why the Change?
The actual TR_ABDOMEN_RAD_EMERGENCY dataset provided by Koç et al. (2024) uses Excel format with two sheets:
1. **TRAIININGDATA** (note: 3 i's - this is the actual sheet name)
2. **COMPETITIONDATA**

### Excel Structure

**Location:** `/home/mete/abdomen-scanner/Temp/Information.xlsx`

**TRAIININGDATA sheet:**
- 28,134 total rows
- 24,498 bounding box annotations
- 3,636 boundary slice annotations  
- 735 unique cases

**Columns:**
- `Case Number` - Case ID (e.g., 20001)
- `Image Id` - DICOM slice identifier (e.g., 100007)
- `Type` - Either "Bounding Box" or "Boundary Slice"
- `Class` - Pathology or anatomy name
- `Data` - Coordinates for bboxes ("xmin,ymin-xmax,ymax"), NaN for boundaries

### Annotation Types

1. **Bounding Box**: 2D rectangular annotation marking pathology
   - Example: `Type="Bounding Box"`, `Class="Kidney stone"`, `Data="251,290-262,302"`

2. **Boundary Slice**: Marks anatomical structure extent in z-axis
   - Example: `Type="Boundary Slice"`, `Class="Pancreas"`, `Data=NaN`
   - Each anatomy has 2 boundaries per case (start + end)

### Scripts Using This Data

✅ **Updated to use Excel format:**
- `scripts/make_boxy_labels.py` - Uses `--excel_path Temp/Information.xlsx`
- `scripts/test_gap1_fix.py` - Validates Excel data structure

### Class Mapping

**11 Radiologist Labels → 6 Competition Classes:**

| Class | Radiologist Labels | Count | Notes |
|-------|-------------------|-------|-------|
| 1 | Abdominal aortic aneurysm/dissection | 9,783 | Well-represented |
| 2 | Compatible with acute pancreatitis | 6,923 | Well-represented |
| 3 | Compatible with acute cholecystitis, Gallbladder stone | 6,265 | Well-represented |
| 4 | Kidney stone, ureteral stone | 1,405 | Moderate |
| 5 | Compatible with acute diverticulitis, Calcified diverticulum | 54 | ⚠️ RARE |
| 6 | Compatible with acute appendicitis | 54 | ⚠️ RARE |

### Reading the Data

```python
import pandas as pd

# Read Excel file
df = pd.read_excel('Temp/Information.xlsx', sheet_name='TRAIININGDATA')

# Get bounding boxes
bboxes = df[df['Type'] == 'Bounding Box']

# Get boundaries
boundaries = df[df['Type'] == 'Boundary Slice']
```

### Documentation

See [docs/DATA_FORMAT.md](../../docs/DATA_FORMAT.md) for complete details on:
- Excel structure
- Class mapping
- Z-axis boundary validation
- Class distribution analysis

---

**Note:** The CSV files in this directory are placeholders only. All scripts have been updated to use the Excel format.
