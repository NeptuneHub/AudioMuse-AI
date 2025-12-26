#!/usr/bin/env python3
"""
Checkpoint Inspector for Student CLAP Training

Utility to inspect checkpoints and show training progress.
"""

import torch
import sys
from pathlib import Path
import argparse
from datetime import datetime

def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint file and display information."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"─" * 50)
        
        # Basic info
        epoch = checkpoint.get('epoch', 'Unknown')
        timestamp = checkpoint.get('timestamp', None)
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            print(f"📅 Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"🔄 Epoch: {epoch}")
        
        # Training metrics
        if 'train_metrics' in checkpoint:
            metrics = checkpoint['train_metrics']
            print(f"📊 Training Loss: {metrics.get('avg_loss', 'N/A'):.6f}")
            print(f"📈 Cosine Similarity: {metrics.get('avg_cosine_sim', 'N/A'):.4f}")
            print(f"🎵 Songs Processed: {metrics.get('num_songs', 'N/A'):,}")
        
        # Validation metrics
        if 'val_cosine_sim' in checkpoint:
            print(f"✅ Validation Cosine Sim: {checkpoint['val_cosine_sim']:.4f}")
        
        if 'best_val_cosine' in checkpoint:
            print(f"🏆 Best Val Cosine Sim: {checkpoint['best_val_cosine']:.4f}")
        
        if 'patience_counter' in checkpoint:
            print(f"⏰ Patience Counter: {checkpoint['patience_counter']}")
        
        # Model info
        if 'model_state_dict' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            print(f"🧠 Model Parameters: {total_params:,}")
        
        print(f"✅ Checkpoint is valid and can be resumed")
        return True
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        return False

def list_checkpoints(checkpoint_dir: str = "checkpoints"):
    """List all available checkpoints."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob("*.pth"))
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"📁 Checkpoints in {checkpoint_dir}:")
    print(f"─" * 60)
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for checkpoint in checkpoints:
        try:
            data = torch.load(checkpoint, map_location='cpu')
            epoch = data.get('epoch', '?')
            val_sim = data.get('val_cosine_sim', data.get('best_val_cosine', None))
            timestamp = data.get('timestamp', checkpoint.stat().st_mtime)
            dt = datetime.fromtimestamp(timestamp)
            
            val_str = f"(val: {val_sim:.4f})" if val_sim else ""
            print(f"  {checkpoint.name:<25} Epoch {epoch:<3} {dt.strftime('%m/%d %H:%M')} {val_str}")
            
        except:
            print(f"  {checkpoint.name:<25} ❌ Invalid")

def main():
    parser = argparse.ArgumentParser(description='Inspect Student CLAP checkpoints')
    parser.add_argument('--list', action='store_true', help='List all checkpoints')
    parser.add_argument('--inspect', type=str, help='Inspect specific checkpoint')
    parser.add_argument('--latest', action='store_true', help='Inspect latest checkpoint')
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.inspect:
        inspect_checkpoint(args.inspect)
    elif args.latest:
        latest_path = Path("checkpoints/latest.pth")
        if latest_path.exists():
            inspect_checkpoint(str(latest_path))
        else:
            print("❌ No latest checkpoint found")
    else:
        print("Use --list, --inspect <path>, or --latest")

if __name__ == '__main__':
    main()