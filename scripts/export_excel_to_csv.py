"""
Helper Script: Export Excel Annotations to CSV for MedSAM

Purpose:
- Converts Temp/Information.xlsx (Excel) to CSV format for MedSAM inference
- Filters only Bounding Box annotations (MedSAM doesn't need Boundary Slices)
- Outputs standard CSV format expected by medsam_infer.py

Usage:
    python scripts/export_excel_to_csv.py --excel_path Temp/Information.xlsx --out_csv data_raw/annotations/TRAININGDATA_exported.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def export_bboxes_to_csv(excel_path: Path, out_csv: Path, sheet_name: str = 'TRAIININGDATA'):
    """
    Export bounding box annotations from Excel to CSV format.
    
    Args:
        excel_path: Path to Information.xlsx
        out_csv: Output CSV path
        sheet_name: Sheet name (default: TRAIININGDATA - note 3 i's)
    """
    print(f"Reading Excel file: {excel_path}")
    print(f"Sheet: {sheet_name}")
    
    # Read Excel
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    print(f"Total rows in Excel: {len(df)}")
    
    # Filter only bounding boxes
    bbox_df = df[df['Type'] == 'Bounding Box'].copy()
    
    print(f"Bounding Box annotations: {len(bbox_df)}")
    print(f"Unique cases: {bbox_df['Case Number'].nunique()}")
    
    # Show class distribution
    print("\nClass distribution:")
    for cls, count in bbox_df['Class'].value_counts().items():
        print(f"  {cls:50s} (n={count})")
    
    # Create output directory
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV
    bbox_df.to_csv(out_csv, index=False)
    
    print(f"\n✅ Exported {len(bbox_df)} bounding boxes to: {out_csv}")
    print(f"   Format: Standard CSV with columns: {list(bbox_df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Excel annotations to CSV for MedSAM"
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="Temp/Information.xlsx",
        help="Path to Information.xlsx"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data_raw/annotations/TRAININGDATA_exported.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default="TRAIININGDATA",
        help="Sheet name (default: TRAIININGDATA - note 3 i's)"
    )
    
    args = parser.parse_args()
    
    excel_path = Path(args.excel_path)
    out_csv = Path(args.out_csv)
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    print("=" * 70)
    print("Excel → CSV Export for MedSAM")
    print("=" * 70)
    
    export_bboxes_to_csv(excel_path, out_csv, args.sheet_name)
    
    print("\n" + "=" * 70)
    print("✅ Export complete!")
    print("=" * 70)
    print("\nNext step:")
    print(f"  Use this CSV in Phase 2: --master_csv {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
