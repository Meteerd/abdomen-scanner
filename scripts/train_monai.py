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
import json
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


def load_data_dicts_from_split(split_file: Path) -> List[Dict]:
    """
    Load data dictionaries directly from split file.
    
    The file should contain one JSON string per line:
    {"image": "/path/to/img.nii.gz", "label": "/path/to/lbl.nii.gz", "case_id": "s0000"}
    
    This works for both AMOS (Phase 3.A) and processed data (Phase 3.B).
    """
    data_dicts = []
    with open(split_file, 'r') as f:
        for line in f:
            if line.strip():
                data_dicts.append(json.loads(line))
    return data_dicts


def get_transforms(config: Dict, mode: str = 'train', use_amos: bool = False):
    """
    Get MONAI transforms for train/val.
    
    Args:
        config: Configuration dictionary
        mode: 'train' or 'val'
        use_amos: If True, use AMOS-specific preprocessing (for Phase 3.A)
        
    Returns:
        MONAI Compose transform
    """
    # Use AMOS-specific transforms if specified
    if use_amos:
        from transforms_amos import get_amos_train_transforms, get_amos_val_transforms
        if mode == 'train':
            return get_amos_train_transforms(config)  # FIXED: Pass config
        else:
            return get_amos_val_transforms()
    
    # Extract config parameters for pathology training (Phase 3.B)
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
        
        # Loss function - DiceCE Loss
        loss_config = config.get('loss', {}).get('dice_ce', {})
        
        # Load class weights from config (allows different weights for pre-training vs fine-tuning)
        class_weights = loss_config.get('class_weights', None)
        if class_weights:
            print(f"Applying class weights to loss: {class_weights}")
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        # DiceCELoss combines Dice loss and Cross-Entropy loss
        # weight: class weights for CE loss
        # lambda_dice and lambda_ce: weighting between Dice and CE components
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            weight=class_weights,  # Correct parameter name for class weights
            lambda_dice=loss_config.get('lambda_dice', 1.0),
            lambda_ce=loss_config.get('lambda_ce', 1.0),
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
        lr = float(optimizer_config.get('lr', 2e-4))
        weight_decay = float(optimizer_config.get('weight_decay', 1e-5))
        
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
    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint for transfer learning (GAP 3)"
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
    train_split = Path(paths.get('train_split', 'splits/train_cases.txt'))
    val_split = Path(paths.get('val_split', 'splits/val_cases.txt'))
    models_dir = Path(paths.get('models_dir', 'models'))
    
    # Load data dictionaries directly from split files
    print(f"Loading dataset splits...")
    train_files = load_data_dicts_from_split(train_split)
    val_files = load_data_dicts_from_split(val_split)
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    
    # Determine if using AMOS dataset (check config for 'amos' or 'pretrain')
    use_amos = 'amos' in str(args.config).lower() or 'pretrain' in str(args.config).lower()
    if use_amos:
        print("Using AMOS-specific preprocessing (Phase 3.A)")
    
    # Create transforms
    train_transforms = get_transforms(config, mode='train', use_amos=use_amos)
    val_transforms = get_transforms(config, mode='val', use_amos=use_amos)
    
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
    
    # Load pre-trained weights if provided (GAP 3: Transfer Learning)
    if args.load_weights is not None:
        checkpoint_path = Path(args.load_weights)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"GAP 3: Transfer Learning Enabled")
        print(f"{'='*60}")
        print(f"Loading pre-trained weights from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (new layers): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys (removed layers): {len(unexpected_keys)}")
        
        print(f"Pre-trained weights loaded successfully")
        print(f"{'='*60}\n")
    
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
