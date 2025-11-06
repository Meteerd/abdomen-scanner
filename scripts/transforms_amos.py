# AMOS 2022 Dataset Transforms
# Based on arXiv:2206.08023v3

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, RandCropByPosNegLabeld, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
)

# AMOS-specific constants from paper (should remain hardcoded)
AMOS_A_MIN = -991
AMOS_A_MAX = 362
AMOS_B_MIN = -7.38  # (-991 - 50) / 141
AMOS_B_MAX = 2.21   # (362 - 50) / 141

def get_amos_train_transforms(config: dict):
    """
    Preprocessing and augmentation for AMOS 2022 dataset training.
    Based on the paper (arXiv:2206.08023v3).
    
    AMOS preprocessing (hardcoded from paper):
    - HU clipping: [-991, 362]
    - Normalization: (x - 50) / 141
    - Spacing: [1.5, 1.5, 2.0] mm
    
    Args:
        config: Configuration dictionary with 'aug' section
    """
    # Get augmentation parameters from config
    patch_size = tuple(config['aug']['patch_size'])
    
    # Calculate pos/neg for desired ratio
    # pos_neg_ratio = 3.0 means 3 positive : 1 negative samples
    pos_neg_ratio = config['aug']['pos_neg_ratio']
    num_samples = 4  # Total samples per iteration
    pos = int(round(num_samples * (pos_neg_ratio / (pos_neg_ratio + 1))))
    neg = num_samples - pos
    
    flip_prob = config['aug']['flip_prob']
    rotate90_prob = config['aug']['rotate90_prob']
    scale_intensity_prob = config['aug'].get('scale_intensity_prob', 0.15)
    shift_intensity_prob = config['aug'].get('shift_intensity_prob', 0.15)
    scale_intensity_factor = config['aug']['scale_intensity']
    shift_intensity_offset = config['aug']['shift_intensity']
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=AMOS_A_MIN, 
            a_max=AMOS_A_MAX,
            b_min=AMOS_B_MIN,
            b_max=AMOS_B_MAX,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,      # FIXED: Now uses config value
            pos=pos,                      # FIXED: Now calculated from pos_neg_ratio
            neg=neg,                      # FIXED: Now calculated from pos_neg_ratio
            num_samples=num_samples,
        ),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),  # FIXED
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),  # FIXED
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=2),  # FIXED
        RandRotate90d(keys=["image", "label"], prob=rotate90_prob, max_k=3), # FIXED
        RandScaleIntensityd(keys=["image"], factors=scale_intensity_factor, prob=scale_intensity_prob),  # FIXED
        RandShiftIntensityd(keys=["image"], offsets=shift_intensity_offset, prob=shift_intensity_prob),  # FIXED
    ])

def get_amos_val_transforms():
    """
    Validation transforms for AMOS 2022 dataset (no augmentation).
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=AMOS_A_MIN,   # Use constants
            a_max=AMOS_A_MAX,
            b_min=AMOS_B_MIN,
            b_max=AMOS_B_MAX,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])
