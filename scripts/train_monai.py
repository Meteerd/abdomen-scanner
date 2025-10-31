"""
Phase 3 - 3D U-Net Training with PyTorch Lightning & MONAI

Purpose:
- Train 3D U-Net for multi-class abdominal segmentation.
- Uses PyTorch Lightning for simplified multi-GPU training (DDP).
- MONAI for medical imaging transforms and models.
- Optimized for mesh-hpc cluster with 2x RTX 6000 (96GB VRAM each).

Usage:
    # Single GPU
    python scripts/train_monai.py --config configs/config.yaml
    
    # Multi-GPU (via SLURM with DDP)
    ./train.sh phase3_unet_baseline
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

# MONAI imports
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord, Activations, AsDiscrete
)
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_case_ids(split_file: Path) -> List[str]:
    """Load case IDs from split text file."""
    with open(split_file, 'r') as f:
        case_ids = [line.strip() for line in f if line.strip()]
    return case_ids


def prepare_data_dicts(case_ids: List[str], image_dir: Path, label_dir: Path) -> List[Dict]:
    """
    Prepare data dictionaries for MONAI dataset.
    
    Args:
        case_ids: List of case identifiers
        image_dir: Directory containing NIfTI images
        label_dir: Directory containing NIfTI labels
        
    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    data_dicts = []
    
    for case_id in case_ids:
        image_path = image_dir / f"{case_id}.nii.gz"
        label_path = label_dir / f"{case_id}.nii.gz"
        
        if image_path.exists() and label_path.exists():
            data_dicts.append({
                "image": str(image_path),
                "label": str(label_path),
            })
    
    return data_dicts


def get_transforms(config: Dict, mode: str = 'train'):
    """
    Get MONAI transforms for train/val.
    
    Args:
        config: Configuration dictionary
        mode: 'train' or 'val'
        
    Returns:
        MONAI Compose transform
    """
    # Extract config parameters
    patch_size = config.get('aug', {}).get('patch_size', [192, 192, 160])
    target_spacing = config.get('preproc', {}).get('target_spacing', [1.5, 1.5, 2.0])
    intensity_clip = config.get('preproc', {}).get('intensity_clip', [0.5, 99.5])
    pos_neg_ratio = config.get('aug', {}).get('pos_neg_ratio', 1.0)
    
    # Common transforms
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,  # Typical CT HU range
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    
    if mode == 'train':
        # Training transforms with augmentation
        transforms = common_transforms + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=pos_neg_ratio,  # Ratio of positive (foreground) samples
                neg=1.0,
                num_samples=2,  # Number of patches per image
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    else:
        # Validation transforms (no augmentation)
        transforms = common_transforms + [
            ToTensord(keys=["image", "label"]),
        ]
    
    return Compose(transforms)


class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for 3D segmentation.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Model config
        model_config = config.get('model', {})
        in_channels = model_config.get('in_channels', 1)
        out_channels = model_config.get('out_channels', 4)
        channels = model_config.get('channels', [32, 64, 128, 256, 512])
        strides = model_config.get('strides', [2, 2, 2, 2])
        dropout = model_config.get('dropout', 0.0)
        
        # Create 3D U-Net
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
            num_res_units=2,
        )
        
        # Loss function - DiceCE Loss (handles class imbalance well)
        loss_config = config.get('loss', {}).get('dice_ce', {})
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            ce_weight=torch.tensor([0.5, 2.0, 2.0, 2.0]),  # Weight rare classes more
        )
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Post-processing
        self.post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=out_channels)])
        self.post_label = Compose([AsDiscrete(to_onehot=out_channels)])
        
        # Validation outputs (for epoch-level metrics)
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Forward pass
        outputs = self.forward(images)
        
        # Compute loss
        loss = self.loss_fn(outputs, labels)
        
        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Sliding window inference for large volumes
        patch_size = self.config.get('aug', {}).get('patch_size', [192, 192, 160])
        outputs = sliding_window_inference(
            inputs=images,
            roi_size=patch_size,
            sw_batch_size=2,
            predictor=self.forward,
            overlap=0.5,
        )
        
        # Compute loss
        loss = self.loss_fn(outputs, labels)
        
        # Compute Dice metric
        outputs_processed = [self.post_pred(i) for i in outputs]
        labels_processed = [self.post_label(i) for i in labels]
        
        self.dice_metric(y_pred=outputs_processed, y=labels_processed)
        
        # Store for epoch-level aggregation
        self.validation_step_outputs.append({"val_loss": loss})
        
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        # Aggregate Dice metric
        dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        # Aggregate validation loss
        avg_val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        
        # Log
        self.log("val_loss", avg_val_loss, prog_bar=True, sync_dist=True)
        self.log("val_dice", dice, prog_bar=True, sync_dist=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer_config = self.config.get('optimizer', {})
        lr = optimizer_config.get('lr', 2e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler (cosine annealing)
        max_epochs = self.config.get('train', {}).get('epochs', 300)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net with PyTorch Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="phase3_unet_baseline",
        help="Experiment name for logging"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    
    # Paths
    paths = config.get('paths', {})
    nifti_images = Path(paths.get('nifti_images', 'data_processed/nifti_images'))
    labels_medsam = Path(paths.get('labels_medsam', 'data_processed/nifti_labels_medsam'))
    train_split = Path(paths.get('train_split', 'splits/train_cases.txt'))
    val_split = Path(paths.get('val_split', 'splits/val_cases.txt'))
    models_dir = Path(paths.get('models_dir', 'models'))
    
    # Load splits
    print(f"Loading dataset splits...")
    train_case_ids = load_case_ids(train_split)
    val_case_ids = load_case_ids(val_split)
    print(f"Train cases: {len(train_case_ids)}")
    print(f"Val cases: {len(val_case_ids)}")
    
    # Prepare data dictionaries
    train_files = prepare_data_dicts(train_case_ids, nifti_images, labels_medsam)
    val_files = prepare_data_dicts(val_case_ids, nifti_images, labels_medsam)
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    
    # Create transforms
    train_transforms = get_transforms(config, mode='train')
    val_transforms = get_transforms(config, mode='val')
    
    # Create datasets
    print("Creating datasets...")
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    
    # Create data loaders
    train_config = config.get('train', {})
    batch_size = train_config.get('batch_size', 2)
    num_workers = train_config.get('num_workers', 8)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Use batch_size=1 for validation with sliding window
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    
    # Create model
    print("Creating model...")
    model = SegmentationModel(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir / args.experiment_name,
        filename="best_model-{epoch:02d}-{val_dice:.4f}",
        save_top_k=3,
        monitor="val_dice",
        mode="max",
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger (W&B)
    use_wandb = os.getenv("WANDB_API_KEY") is not None
    if use_wandb:
        wandb_logger = WandbLogger(
            project=os.getenv("WANDB_PROJECT", "abdomen-segmentation"),
            name=args.experiment_name,
            save_dir="logs",
        )
        loggers = [wandb_logger]
    else:
        print("Warning: W&B API key not set. Logging disabled.")
        loggers = []
    
    # Trainer
    max_epochs = train_config.get('epochs', 300)
    val_every = train_config.get('val_every', 1)
    mixed_precision = train_config.get('mixed_precision', True)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=2,  # Use both GPUs
        strategy="ddp",  # Distributed Data Parallel
        precision="16-mixed" if mixed_precision else 32,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=loggers,
        check_val_every_n_epoch=val_every,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train
    print(f"\nStarting training for {max_epochs} epochs...")
    print(f"Experiment: {args.experiment_name}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Strategy: DDP (2 GPUs)")
    print("")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {models_dir / args.experiment_name}")


if __name__ == "__main__":
    main()
