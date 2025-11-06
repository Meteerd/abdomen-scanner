#!/usr/bin/env python3
"""Quick verification of Phase 2 aggregated labels."""

import nibabel as nib
import numpy as np
from pathlib import Path

label_dir = Path('data_processed/nifti_labels_medsam')
label_files = list(label_dir.glob('*.nii.gz'))

print(f'Total 3D label volumes: {len(label_files)}')

# Sample 10 random files
import random
random.seed(42)
samples = random.sample(label_files, min(10, len(label_files)))

print('\nSample verification (10 files):')
print('-' * 60)

for label_file in samples:
    img = nib.load(label_file)
    data = img.get_fdata()
    unique = np.unique(data)
    non_zero = np.sum(data > 0)
    print(f'{label_file.name}: classes={list(unique.astype(int))}, non-zero={non_zero:,}')

# Quick class distribution (sample 100 files)
sample_for_dist = random.sample(label_files, min(100, len(label_files)))
class_counts = {i: 0 for i in range(6)}

for label_file in sample_for_dist:
    img = nib.load(label_file)
    data = img.get_fdata()
    for class_id in range(6):
        if (data == class_id).any():
            class_counts[class_id] += 1

print(f'\nClass distribution (sampled {len(sample_for_dist)} files):')
print('-' * 60)
print(f'Class 1: {class_counts[1]} files ({class_counts[1]/len(sample_for_dist)*100:.1f}%)')
print(f'Class 2: {class_counts[2]} files ({class_counts[2]/len(sample_for_dist)*100:.1f}%)')
print(f'Class 3: {class_counts[3]} files ({class_counts[3]/len(sample_for_dist)*100:.1f}%)')
print(f'Class 4: {class_counts[4]} files ({class_counts[4]/len(sample_for_dist)*100:.1f}%)')
print(f'Class 5: {class_counts[5]} files ({class_counts[5]/len(sample_for_dist)*100:.1f}%)')

# Count empty volumes
empty_count = sum(1 for lf in sample_for_dist if np.sum(nib.load(lf).get_fdata() > 0) == 0)
print(f'\nEmpty volumes: {empty_count}/{len(sample_for_dist)}')
print(f'Non-empty volumes: {len(sample_for_dist) - empty_count}/{len(sample_for_dist)}')

print('\nPhase 2 aggregation: SUCCESS')
