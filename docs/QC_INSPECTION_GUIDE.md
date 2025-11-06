# Quality Control Inspection Guide - ITK-SNAP Tutorial

## Step 1: Install ITK-SNAP

### Windows Installation:
1. Locate the downloaded ITK-SNAP installer (likely in Downloads folder)
2. The file should be named something like: `itksnap-3.8.0-win64.exe`
3. Double-click the installer
4. Follow installation wizard (accept defaults)
5. ITK-SNAP will be installed to: `C:\Program Files\ITK-SNAP 3.8\bin\`
6. A desktop shortcut should be created

### Verify Installation:
- Double-click ITK-SNAP icon on desktop
- If it opens successfully, installation is complete
- Close ITK-SNAP for now

---

## Step 2: Transfer NIfTI Files from Cluster to Local Machine

You need to download the specific image and label files from the cluster to your local Windows machine.

### Option A: Download All Required Files at Once (Recommended)

Open PowerShell on your Windows machine:

```powershell
# Create local directory for QC files
cd "C:\Users\User\Desktop"
mkdir QC_Files
cd QC_Files

# Create subdirectories
mkdir images
mkdir labels

# Download all 20 cases (10 Diverticulitis + 10 Appendicitis)
# This will take 5-10 minutes depending on file sizes

# Class 4 (Diverticulitis) - 10 cases
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10048_20048.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10048_20048.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10050_20050.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10050_20050.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10054_20054.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10054_20054.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10055_20055.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10055_20055.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10057_20057.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10057_20057.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10059_20059.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10059_20059.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10063_20064.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10063_20064.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10086_20088.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10086_20088.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10227_20237.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10227_20237.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10239_20249.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10239_20249.nii.gz labels/

# Class 5 (Appendicitis) - 10 cases
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10005_20005.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10005_20005.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10013_20013.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10013_20013.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10015_20015.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10015_20015.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10016_20016.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10016_20016.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10020_20020.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10020_20020.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10033_20033.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10033_20033.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10036_20036.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10036_20036.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10040_20040.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10040_20040.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10100_20102.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10100_20102.nii.gz labels/

scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10224_20234.nii.gz images/
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10224_20234.nii.gz labels/
```

### Option B: Download One Case at a Time (Start Here if New to This)

```powershell
# Create local directory
cd "C:\Users\User\Desktop"
mkdir QC_Files
cd QC_Files

# Download first case (20048 - Diverticulitis)
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_images/1_2_840_10009_1_2_3_10048_20048.nii.gz .
scp mete@100.116.63.100:/home/mete/abdomen-scanner/data_processed/nifti_labels_medsam/1_2_840_10009_1_2_3_10048_20048.nii.gz .
```

### Verify Download:
```powershell
# List downloaded files
dir
# You should see two .nii.gz files (image and label)
```

---

## Step 3: Open Files in ITK-SNAP

### Method 1: Using ITK-SNAP GUI (Easiest for Beginners)

1. Launch ITK-SNAP from desktop shortcut

2. Load CT Image (Main Volume):
   - Click: **File → Open Main Image...**
   - Navigate to: `C:\Users\User\Desktop\QC_Files\images\`
   - Select: `1_2_840_10009_1_2_3_10048_20048.nii.gz`
   - Click: **Next** (ITK-SNAP will auto-detect format)
   - Click: **Finish**
   - You should now see 3 views: Axial (top-left), Coronal (top-right), Sagittal (bottom-left)

3. Load MedSAM Label (Segmentation Overlay):
   - Click: **Segmentation → Open Segmentation...**
   - Navigate to: `C:\Users\User\Desktop\QC_Files\labels\`
   - Select: `1_2_840_10009_1_2_3_10048_20048.nii.gz`
   - Click: **Next**
   - Click: **Finish**
   - The segmentation mask should now appear as colored overlay on the CT

### Method 2: Using Command Line (Faster for Multiple Cases)

Open PowerShell and run:

```powershell
# Navigate to ITK-SNAP installation
cd "C:\Program Files\ITK-SNAP 3.8\bin"

# Launch with image and label
.\ITK-SNAP.exe -g "C:\Users\User\Desktop\QC_Files\images\1_2_840_10009_1_2_3_10048_20048.nii.gz" -s "C:\Users\User\Desktop\QC_Files\labels\1_2_840_10009_1_2_3_10048_20048.nii.gz"
```

---

## Step 4: Navigate and Inspect in ITK-SNAP

### Understanding the Interface:

```
+------------------+------------------+
|   Axial View     |  Coronal View    |
|  (Top-down)      |  (Front view)    |
+------------------+------------------+
|  Sagittal View   |   3D View        |
|  (Side view)     |  (Optional)      |
+------------------+------------------+
```

### Basic Navigation Controls:

**Mouse Controls:**
- **Left-click and drag**: Pan the image
- **Right-click and drag**: Zoom in/out
- **Middle-click** (or scroll wheel): Change slice (move through z-axis)
- **Ctrl + Left-click**: Crosshair placement (synchronizes all views)

**Keyboard Shortcuts:**
- **Arrow keys**: Move through slices
- **Page Up/Down**: Jump 10 slices
- **Q/W**: Adjust window level (brightness)
- **A/S**: Adjust window width (contrast)
- **Z/X**: Zoom in/out

### Adjusting Image Display:

1. **Window/Level Adjustment** (Important for CT):
   - Click: **Tools → Image Contrast**
   - Try preset: **Abdominal Soft Tissue**
   - Or manually adjust sliders:
     - Window: 400
     - Level: 40
   - This highlights soft tissue pathology

2. **Segmentation Overlay Visibility**:
   - Toggle overlay on/off: **Segmentation → Toggle Segmentation Visibility** (or press 'S')
   - Adjust opacity: Use slider in segmentation panel (right side)
   - Recommended opacity: 40-60% (can see both CT and mask)

3. **Label Properties**:
   - Click: **Segmentation → Label Editor**
   - You'll see labels 0-5 (background + 5 pathology classes)
   - Change colors if needed for better visibility

---

## Step 5: Quality Assessment Criteria

For EACH case, systematically check these four criteria:

### 1. Coverage Assessment

**Question:** Does the mask cover the full extent of the pathology?

**How to Check:**
- Scroll through all slices where pathology appears
- Compare mask boundaries with abnormal tissue on CT
- Look for: inflammatory changes, fluid collections, wall thickening

**Rating:**
- **Full**: Mask covers 90-100% of visible pathology
- **Partial**: Mask covers 50-89% of pathology (missing edges or slices)
- **Missing**: Mask covers <50% or completely wrong location

**Example Issues:**
- Appendicitis: Mask only covers proximal appendix, misses inflamed tip
- Diverticulitis: Mask misses pericolic fat stranding (inflammation around colon)

### 2. Boundary Accuracy

**Question:** Are the mask boundaries clean and accurate?

**How to Check:**
- Zoom in on mask edges (right-click and drag up)
- Check if boundaries align with anatomical structures
- Look for jagged/pixelated edges vs smooth contours

**Rating:**
- **Clean**: Smooth boundaries, follows anatomical structures
- **Noisy**: Rough/jagged edges but generally correct location
- **Incorrect**: Boundaries far from actual pathology margins

**Example Issues:**
- Over-segmentation: Mask includes adjacent normal bowel
- Under-segmentation: Mask stops short of pathology boundary

### 3. False Positives

**Question:** Is there background tissue wrongly labeled as pathology?

**How to Check:**
- Look at areas OUTSIDE the known pathology location
- Check if any normal structures are labeled
- Scroll to slices above/below pathology

**Rating:**
- **None**: No false positive labels
- **Few**: 1-2 small spurious labels (can be manually cleaned)
- **Many**: Widespread false positives (major problem)

**Example Issues:**
- Normal colon loops labeled as diverticulitis
- Small bowel labeled as appendicitis
- Vascular structures wrongly segmented

### 4. Missing Slices (Z-axis Continuity)

**Question:** Are there gaps in the z-axis where mask should appear but doesn't?

**How to Check:**
- Find slices where pathology is clearly visible on CT
- Scroll through entire pathology region
- Note slices where mask appears vs disappears

**Rating:**
- **None**: Continuous labeling through all pathology slices
- **Minor Gaps**: 1-2 slices missing (not critical)
- **Major Gaps**: Multiple consecutive slices missing (problematic)

**Example Issues:**
- Appendicitis labeled on 5 slices, but visible on 10 slices (50% missing)
- Diverticulitis mask appears, disappears, reappears (fragmented)

---

## Step 6: Record Your Assessment

Open the checklist file: `phase2_qc_checklist.txt`

For Case 20048 (first Diverticulitis case), fill in:

```
[1] Case Number: 20048

    Quality Check:
      [X] Coverage: Full / Partial / Missing
      [X] Boundary: Clean / Noisy / Incorrect
      [X] False Positives: None / Few / Many
      [X] Overall: PASS / MARGINAL / FAIL
      Notes: Mask covers sigmoid colon inflammation well. 
             Minor edge noise in pericolic region.
             2 slices missing at superior extent.
```

### Overall Rating Guidelines:

**PASS:**
- Coverage: Full or Partial (>80%)
- Boundary: Clean or mildly Noisy
- False Positives: None or Few (easily removable)
- Z-axis: Complete or minor gaps only

**MARGINAL:**
- Coverage: Partial (50-80%)
- Boundary: Noisy but recognizable
- False Positives: Few scattered labels
- Z-axis: Some gaps but main pathology captured

**FAIL:**
- Coverage: Missing or <50%
- Boundary: Incorrect location
- False Positives: Many widespread labels
- Z-axis: Major gaps or completely wrong slices

---

## Step 7: Repeat for All 20 Cases

### Efficient Workflow:

1. Keep ITK-SNAP open
2. For next case, click: **File → Open Main Image...** (loads new CT)
3. Then: **Segmentation → Open Segmentation...** (loads new label)
4. Inspect using same 4 criteria
5. Record in checklist
6. Repeat

### Time Estimate:
- First case: 5-10 minutes (learning the interface)
- Subsequent cases: 2-3 minutes each
- Total time: 30-40 minutes for all 20 cases

### Pro Tips:
- Take screenshots of problematic cases (Print Screen key)
- Note specific slice numbers where issues occur
- If unsure, mark as MARGINAL and add detailed notes

---

## Step 8: Calculate Pass Rate and Make Decision

After inspecting all 20 cases, fill in the summary section:

```
SUMMARY
================================================================================

Class 4 (Diverticulitis):
  Total checked: 10
  PASS:     7 / 10
  MARGINAL: 2 / 10
  FAIL:     1 / 10
  Pass rate: 70%

Class 5 (Appendicitis):
  Total checked: 10
  PASS:     8 / 10
  MARGINAL: 1 / 10
  FAIL:     1 / 10
  Pass rate: 80%

OVERALL DECISION:
  [X] PASS: Proceed to Phase 3
  [ ] MARGINAL: Re-run MedSAM with adjusted prompts
  [ ] FAIL: Fix MedSAM inference before Phase 3

Reviewer: Mete    Date: 2025-11-05
```

### Decision Logic:

**PASS (>70% good):**
- Action: Proceed to Phase 3A (AMOS pre-training) or Phase 3B (fine-tuning)
- Rationale: Label quality sufficient for training, model will learn despite minor noise

**MARGINAL (50-70% good):**
- Action: Consider re-running MedSAM Phase 2
- Adjustments: Expand bounding boxes by 10-20% to capture more context
- Re-run only failed cases or all cases depending on systematic errors

**FAIL (<50% good):**
- Action: STOP, do not proceed to Phase 3
- Investigation needed:
  - Check if DICOM to NIfTI conversion correct
  - Verify bounding box annotations are accurate
  - Test MedSAM on different pathologies
  - Consider alternative segmentation method

---

## Troubleshooting

### ITK-SNAP Won't Open Files:

**Problem:** "Cannot open file" error

**Solutions:**
1. Check file extension is `.nii.gz` (compressed NIfTI)
2. Try unzipping first: 
   ```powershell
   # Install 7-Zip, then:
   7z x 1_2_840_10009_1_2_3_10048_20048.nii.gz
   # This creates .nii file (uncompressed)
   ```
3. Verify file downloaded completely (check file size > 0 KB)

### Images Look Too Dark/Bright:

**Problem:** Can't see anatomical structures

**Solution:**
- Adjust Window/Level: **Tools → Image Contrast**
- Use CT presets:
  - Abdominal Soft Tissue: W=400, L=40
  - Bone: W=2000, L=300
  - Lung: W=1500, L=-600

### Can't See Segmentation Overlay:

**Problem:** Loaded label but no color overlay visible

**Solutions:**
1. Press 'S' key to toggle segmentation visibility
2. Check opacity slider (should be 40-60%)
3. Verify label file has values >0:
   - Click: **Segmentation → Label Inspector**
   - Should show labels 0-5

### Files Are Too Large to Download:

**Problem:** Each NIfTI file is 100+ MB

**Solution:**
- Download one case at a time
- Inspect, then delete before downloading next
- Or use compression: SCP automatically compresses during transfer

### Not Sure if Pathology Visible:

**Problem:** Don't have medical training, can't identify pathology

**Guidance:**
- You're not diagnosing, just checking if MASK aligns with ABNORMAL areas
- Look for:
  - Diverticulitis: Thickened colon wall, stranding in fat around colon
  - Appendicitis: Dilated tube-like structure in right lower abdomen, fluid around it
- If unsure, mark as MARGINAL and note uncertainty

---

## What Happens Next?

### If QC PASSES:

1. Upload completed checklist to cluster:
   ```powershell
   scp "C:\Users\User\Desktop\QC_Files\phase2_qc_checklist_completed.txt" mete@100.116.63.100:/home/mete/abdomen-scanner/
   ```

2. Wait for AMOS dataset upload (blocking Phase 3A)

3. Once AMOS ready, submit Phase 3A:
   ```bash
   ssh mete@100.116.63.100
   cd /home/mete/abdomen-scanner
   sbatch slurm_phase3a_pretrain.sh
   ```

4. After 3 days, submit Phase 3B:
   ```bash
   sbatch slurm_phase3b_finetune.sh models/phase3a_pretrain/best.ckpt
   ```

### If QC FAILS:

1. Document specific failure modes in checklist

2. Meet with team to decide:
   - Re-run MedSAM with adjusted parameters
   - Request expert annotations for rare classes
   - Consider alternative segmentation method

3. Do NOT proceed to Phase 3 until labels pass QC

---

## Quick Reference: Common ITK-SNAP Commands

| Action | Shortcut |
|--------|----------|
| Toggle segmentation | S |
| Next slice | Up arrow or scroll up |
| Previous slice | Down arrow or scroll down |
| Zoom in | Z or right-drag up |
| Zoom out | X or right-drag down |
| Adjust brightness | Q/W |
| Adjust contrast | A/S |
| Reset view | R |
| Crosshair mode | Spacebar |

---

## Support

If you encounter issues:
1. Check ITK-SNAP documentation: http://www.itksnap.org/docs/
2. Verify file paths are correct
3. Ensure Tailscale VPN is connected before SCP
4. Test with one case first before downloading all 20

Good luck with the QC inspection!
