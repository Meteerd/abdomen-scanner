#!/usr/bin/env python3
"""
Real-time training metrics visualization for YOLO
Usage: python plot_training_metrics.py /path/to/results.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_metrics(csv_path, save_path=None):
    """Plot training metrics from YOLO results.csv"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
    if 'train/dfl_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: mAP metrics
    ax2 = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', 
                linewidth=2, marker='o', color='green')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', 
                linewidth=2, marker='s', color='blue')
    ax2.axhline(y=0.70, color='r', linestyle='--', label='Target (70%)', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('Mean Average Precision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Precision & Recall
    ax3 = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', 
                linewidth=2, marker='o', color='purple')
    if 'metrics/recall(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', 
                linewidth=2, marker='s', color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision & Recall')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Learning rate
    ax4 = axes[1, 1]
    lr_cols = [col for col in df.columns if 'lr/' in col]
    for col in lr_cols:
        ax4.plot(df['epoch'], df[col], label=col.replace('lr/', ''), linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.savefig(csv_path.parent / 'training_metrics.png', dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to: {csv_path.parent / 'training_metrics.png'}")
    
    # Print current metrics
    if len(df) > 0:
        latest = df.iloc[-1]
        print(f"\n📊 Latest Metrics (Epoch {int(latest['epoch'])}):")
        print(f"   mAP@0.5: {latest.get('metrics/mAP50(B)', 0):.3f}")
        print(f"   mAP@0.5:0.95: {latest.get('metrics/mAP50-95(B)', 0):.3f}")
        print(f"   Precision: {latest.get('metrics/precision(B)', 0):.3f}")
        print(f"   Recall: {latest.get('metrics/recall(B)', 0):.3f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_training_metrics.py <results.csv>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    plot_metrics(csv_path)
