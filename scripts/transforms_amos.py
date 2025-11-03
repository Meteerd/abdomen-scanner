# AMOS 2022 Dataset Transforms
# Based on arXiv:2206.08023v3

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, RandCropByPosNegLabeld, ScaleIntensityRanged,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
)

def get_amos_train_transforms():
    """
    Preprocessing and augmentation for AMOS 2022 dataset training.
    Based on the paper (arXiv:2206.08023v3).
    
    AMOS preprocessing:
    - HU clipping: [-991, 362]
    - Normalization: (x - 50) / 141
    - Spacing: [1.5, 1.5, 2.0] mm
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=-991, 
            a_max=362,
            b_min=-7.38,  # (-991 - 50) / 141
            b_max=2.21,   # (362 - 50) / 141
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1, 
            neg=1, 
            num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
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
            a_min=-991, 
            a_max=362,
            b_min=-7.38,
            b_max=2.21,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])
