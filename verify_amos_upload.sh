#!/bin/bash
# Quick verification script for AMOS dataset upload

echo "=========================================="
echo "AMOS Dataset Upload Verification"
echo "=========================================="
echo ""

# Check AMOS directory
AMOS_DIR="data/AbdomenDataSet/AMOS-Dataset"
META_CSV="data/meta.csv"

if [ ! -d "$AMOS_DIR" ]; then
    echo "❌ AMOS directory NOT found: $AMOS_DIR"
    echo ""
    echo "Expected location:"
    echo "  /home/mete/abdomen-scanner/data/AbdomenDataSet/AMOS-Dataset/"
    echo ""
    echo "Please upload the dataset using WinSCP or scp."
    exit 1
else
    echo "✓ AMOS directory found: $AMOS_DIR"
fi

# Count case folders
NUM_CASES=$(find "$AMOS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "✓ Case folders found: $NUM_CASES"

if [ $NUM_CASES -lt 100 ]; then
    echo "  ⚠️  Warning: Expected ~1200+ cases, found only $NUM_CASES"
    echo "  Upload may be incomplete."
fi

# Check meta.csv
if [ ! -f "$META_CSV" ]; then
    echo "❌ meta.csv NOT found: $META_CSV"
    echo ""
    
    # Check if it's in Temp
    if [ -f "Temp/meta.csv" ]; then
        echo "  Found in Temp/meta.csv"
        echo "  Copying to data/meta.csv..."
        mkdir -p data
        cp Temp/meta.csv data/meta.csv
        echo "✓ meta.csv copied to data/"
    else
        echo "  meta.csv not found in Temp/ either."
        echo "  Please upload meta.csv to data/"
        exit 1
    fi
else
    echo "✓ meta.csv found: $META_CSV"
    NUM_ROWS=$(wc -l < "$META_CSV")
    echo "  Rows in meta.csv: $NUM_ROWS"
fi

# Sample first few cases
echo ""
echo "Sample case folders:"
find "$AMOS_DIR" -mindepth 1 -maxdepth 1 -type d | head -10

# Check if a case has files
FIRST_CASE=$(find "$AMOS_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -n "$FIRST_CASE" ]; then
    echo ""
    echo "Contents of first case folder:"
    echo "  $FIRST_CASE"
    ls -lh "$FIRST_CASE" | head -10
fi

# Check meta.csv format
echo ""
echo "Meta.csv header:"
head -1 "$META_CSV"

echo ""
echo "Meta.csv first 3 data rows:"
head -4 "$META_CSV" | tail -3

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "AMOS directory: ✓ ($NUM_CASES folders)"
echo "meta.csv: ✓ ($NUM_ROWS rows)"
echo ""

if [ $NUM_CASES -lt 100 ]; then
    echo "⚠️  Dataset appears incomplete. Please verify upload."
    echo ""
    echo "Next steps:"
    echo "1. Check if WinSCP transfer is still in progress"
    echo "2. Verify source data has ~1200+ folders"
    echo "3. Re-upload if necessary"
else
    echo "✓ Dataset appears complete!"
    echo ""
    echo "Next steps:"
    echo "1. Run dataset preparation:"
    echo "   python scripts/prepare_amos_dataset.py"
    echo ""
    echo "2. Start Phase 3.A pre-training:"
    echo "   sbatch slurm_phase3a_pretrain_clean.sh"
fi

echo ""
