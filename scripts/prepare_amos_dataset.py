#!/usr/bin/env python3
"""
Prepare AMOS 2022 Dataset for Training

This script processes the AMOS dataset structure and creates:
1. Train/val/test splits from meta.csv
2. Case inventory with pathology information
3. Validation that all required files exist

Input:
    - data/AbdomenDataSet/AMOS-Dataset/ (folders: s0000/, s0001/, ...)
    - data/meta.csv (metadata with splits)

Output:
    - splits/amos_train_cases.txt
    - splits/amos_val_cases.txt  
    - splits/amos_test_cases.txt
    - data/amos_inventory.csv (detailed case inventory)
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict
import json


def read_meta_csv(meta_path: str) -> pd.DataFrame:
    """Read and parse the meta.csv file"""
    print(f"Reading metadata from: {meta_path}")
    df = pd.read_csv(meta_path, delimiter=';')
    print(f"  Total cases: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    return df


def find_case_files(amos_dir: str, case_id: str) -> Dict[str, str]:
    """
    Find image and label files for a given case ID
    
    Returns dict with 'image' and 'label' paths (or None if not found)
    """
    case_folder = Path(amos_dir) / case_id
    
    if not case_folder.exists():
        return {'image': None, 'label': None, 'exists': False}
    
    # AMOS dataset structure (after aggregation):
    # - ct.nii.gz (CT image)
    # - label.nii.gz (aggregated multi-class label from segmentations/)
    result = {'exists': True, 'image': None, 'label': None}
    
    ct_path = case_folder / 'ct.nii.gz'
    label_path = case_folder / 'label.nii.gz'
    
    if ct_path.exists():
        result['image'] = str(ct_path)
    
    if label_path.exists():
        result['label'] = str(label_path)
    
    return result


def create_splits(meta_df: pd.DataFrame, amos_dir: str, output_dir: str):
    """Create train/val/test split files from meta.csv"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by split column
    splits = meta_df.groupby('split')
    
    statistics = {
        'total_cases': len(meta_df),
        'splits': {},
        'missing_cases': [],
        'pathology_distribution': {}
    }
    
    # Create split files
    for split_name, split_df in splits:
        print(f"\nProcessing {split_name} split:")
        print(f"  Cases: {len(split_df)}")
        
        split_file = output_path / f"amos_{split_name}_cases.txt"
        
        # Store data dicts with full paths instead of just IDs
        valid_case_dicts = []
        missing_cases = []
        
        # Verify each case exists
        for idx, row in split_df.iterrows():
            case_id = row['image_id']
            case_info = find_case_files(amos_dir, case_id)
            
            # Check for both image and label (for training/validation)
            if (split_name in ['train', 'val'] and 
                case_info['exists'] and 
                case_info['image'] and 
                case_info['label']):
                
                # Create data dictionary with full paths
                valid_case_dicts.append({
                    "image": case_info['image'],
                    "label": case_info['label'],
                    "case_id": case_id
                })
            # Check for image only (for testing)
            elif (split_name == 'test' and
                  case_info['exists'] and
                  case_info['image']):
                  
                valid_case_dicts.append({
                    "image": case_info['image'],
                    "label": None,  # No label for test set
                    "case_id": case_id
                })
            else:
                missing_cases.append(case_id)
                statistics['missing_cases'].append({
                    'case_id': case_id,
                    'split': split_name
                })
        
        # Write JSON strings to split file (one per line)
        with open(split_file, 'w') as f:
            for case_dict in valid_case_dicts:
                f.write(json.dumps(case_dict) + "\n")
        
        print(f"  Valid cases: {len(valid_case_dicts)}")
        print(f"  Missing cases: {len(missing_cases)}")
        if missing_cases:
            print(f"  Missing: {missing_cases[:5]}...")
        print(f"  Saved to: {split_file}")
        
        # Store statistics
        statistics['splits'][split_name] = {
            'total': len(split_df),
            'valid': len(valid_case_dicts),
            'missing': len(missing_cases)
        }
        
        # Pathology distribution for this split
        pathology_counts = split_df['pathology'].value_counts().to_dict()
        statistics['pathology_distribution'][split_name] = pathology_counts
    
    return statistics


def create_inventory(meta_df: pd.DataFrame, amos_dir: str, output_path: str):
    """Create detailed inventory CSV with file paths and metadata"""
    
    print("\nCreating detailed inventory...")
    
    inventory = []
    
    for idx, row in meta_df.iterrows():
        case_id = row['image_id']
        case_info = find_case_files(amos_dir, case_id)
        
        inventory.append({
            'case_id': case_id,
            'split': row['split'],
            'age': row['age'],
            'gender': row['gender'],
            'institute': row['institute'],
            'study_type': row['study_type'],
            'manufacturer': row['manufacturer'],
            'scanner_model': row['scanner_model'],
            'kvp': row['kvp'],
            'pathology': row['pathology'],
            'pathology_location': row['pathology_location'],
            'image_path': case_info.get('image', ''),
            'label_path': case_info.get('label', ''),
            'exists': case_info.get('exists', False),
            'has_image': case_info.get('image') is not None,
            'has_label': case_info.get('label') is not None
        })
    
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(output_path, index=False)
    print(f"  Inventory saved to: {output_path}")
    
    # Print summary
    print(f"\n  Total cases: {len(inventory_df)}")
    print(f"  Cases with images: {inventory_df['has_image'].sum()}")
    print(f"  Cases with labels: {inventory_df['has_label'].sum()}")
    print(f"  Missing folders: {(~inventory_df['exists']).sum()}")
    
    return inventory_df


def print_summary(statistics: Dict, inventory_df: pd.DataFrame):
    """Print comprehensive summary"""
    
    print("\n" + "="*60)
    print("AMOS DATASET PREPARATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal cases in metadata: {statistics['total_cases']}")
    
    print("\nSplit Distribution:")
    for split_name, split_stats in statistics['splits'].items():
        print(f"  {split_name:10s}: {split_stats['valid']:4d} valid / {split_stats['total']:4d} total")
    
    print(f"\nMissing cases: {len(statistics['missing_cases'])}")
    if statistics['missing_cases']:
        print("  First 10 missing:")
        for item in statistics['missing_cases'][:10]:
            print(f"    {item['case_id']} ({item['split']})")
    
    print("\nPathology Distribution:")
    for split_name, pathology_dist in statistics['pathology_distribution'].items():
        print(f"\n  {split_name} split:")
        for pathology, count in sorted(pathology_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {pathology:20s}: {count:4d} cases")
    
    print("\nFiles created:")
    print(f"  - splits/amos_train_cases.txt")
    print(f"  - splits/amos_val_cases.txt")
    print(f"  - splits/amos_test_cases.txt")
    print(f"  - data/amos_inventory.csv")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Prepare AMOS 2022 dataset for training')
    parser.add_argument('--amos_dir', type=str, 
                        default='data/AbdomenDataSet/AMOS-Dataset',
                        help='Path to AMOS dataset directory (contains s0000/, s0001/, ...)')
    parser.add_argument('--meta_csv', type=str,
                        default='data/meta.csv',
                        help='Path to meta.csv file')
    parser.add_argument('--output_dir', type=str,
                        default='splits',
                        help='Output directory for split files')
    parser.add_argument('--inventory_output', type=str,
                        default='data/amos_inventory.csv',
                        help='Output path for inventory CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.amos_dir):
        print(f"ERROR: AMOS directory not found: {args.amos_dir}")
        print("Please ensure the dataset is uploaded to the correct location.")
        return 1
    
    if not os.path.exists(args.meta_csv):
        print(f"ERROR: Meta CSV not found: {args.meta_csv}")
        return 1
    
    print("="*60)
    print("AMOS 2022 Dataset Preparation")
    print("="*60)
    print(f"AMOS directory: {args.amos_dir}")
    print(f"Meta CSV: {args.meta_csv}")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Read metadata
    meta_df = read_meta_csv(args.meta_csv)
    
    # Create split files
    statistics = create_splits(meta_df, args.amos_dir, args.output_dir)
    
    # Create inventory
    inventory_df = create_inventory(meta_df, args.amos_dir, args.inventory_output)
    
    # Print summary
    print_summary(statistics, inventory_df)
    
    print("\nDataset preparation complete!")
    return 0


if __name__ == '__main__':
    exit(main())
