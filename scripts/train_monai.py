"""
PLACEHOLDER ONLY — NO CODE.

Purpose:
- Training entrypoint for MONAI 3D UNet (single GPU or DDP when launched via torchrun).

Inputs (intended):
- --config configs/config.yaml

Implementation notes:
- Parse YAML; build transforms, datasets, loaders.
- DiceCE loss; AdamW optimizer; AMP; checkpointing to models/.
- If distributed: use DistributedSampler and DDP wrapper.
"""
