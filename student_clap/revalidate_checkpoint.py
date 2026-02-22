#!/usr/bin/env python3
"""Re-run validation for an existing Student CLAP checkpoint and update it.

Usage:
  python student_clap/revalidate_checkpoint.py --ckpt student_clap/checkpoints/checkpoint_epoch_1.pth

Options:
  --ckpt     Path to checkpoint (.pth)
  --config   Path to config.yaml (default: student_clap/config.yaml)
  --dry-run  Run validation but don't write the checkpoint
  --update-latest  Also update `latest.pth` in the same folder
"""
import argparse
import time
import yaml
import torch
import logging
from pathlib import Path

# Ensure repository root is importable when running this script directly
import os, sys
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Local imports (repo root must be CWD)
from student_clap.models.student_onnx_model import StudentCLAPTrainer
from student_clap.data.dataset import StudentCLAPDataset
from student_clap.train_real import validate_real

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Checkpoint path (.pth)')
    p.add_argument('--config', default='student_clap/config.yaml', help='Path to config.yaml')
    p.add_argument('--dry-run', action='store_true', help="Don't overwrite the checkpoint file; only print metrics")
    p.add_argument('--update-latest', action='store_true', help='Also write metrics to latest.pth in the same folder')
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    config = yaml.safe_load(open(args.config))
    # Resolve any relative paths in config['paths'] relative to the config file location
    config_file_path = Path(args.config).resolve()
    config_dir = config_file_path.parent
    if isinstance(config.get('paths'), dict):
        for key, val in list(config['paths'].items()):
            if isinstance(val, str) and not os.path.isabs(val):
                resolved = (config_dir / val).resolve()
                config['paths'][key] = str(resolved)
    logger.info(f"Loaded config: {args.config} (resolved paths relative to {config_dir})")

    # Build trainer and load model weights
    trainer = StudentCLAPTrainer(config)
    ckpt = torch.load(str(ckpt_path), map_location='cpu')

    # Restore model weights
    if 'model_state_dict' not in ckpt:
        logger.error('Provided file does not look like a training checkpoint (no model_state_dict).')
        raise SystemExit(1)

    trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    logger.info(f"Model weights loaded (device={trainer.device})")

    # If optimizer state exists, optionally restore (not required for validation)
    try:
        if 'optimizer_state_dict' in ckpt:
            try:
                trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                logger.info('Optimizer state restored (for completeness)')
            except Exception:
                logger.info('Optimizer state not restored (incompatible)')
    except Exception:
        pass

    epoch = int(ckpt.get('epoch', 1))

    # Build validation dataset
    val_dataset = StudentCLAPDataset(config, split='val', epoch=epoch)

    # Run validation
    logger.info(f"Running validation for checkpoint: {ckpt_path.name} (epoch={epoch})")
    start = time.time()
    val_metrics = validate_real(trainer, val_dataset, config, epoch=epoch)
    elapsed = time.time() - start
    logger.info(f"Validation finished in {elapsed:.1f}s")

    # Summarize metrics
    val_mse = val_metrics.get('mse')
    val_cos = val_metrics.get('cosine_similarity', {}).get('mean')
    logger.info(f"Validation summary — val_mse: {val_mse}, val_cosine: {val_cos}")

    # Decide primary validation metric based on config
    loss_fn = config.get('training', {}).get('loss_function', 'mse')
    if loss_fn in ('cosine', 'kl'):
        val_metric = val_cos
        val_metric_name = 'val_cosine'
    else:
        val_metric = val_mse
        val_metric_name = 'val_mse'

    # Print train cosine (already stored inside checkpoint['train_metrics'] if present)
    train_cos = None
    if 'train_metrics' in ckpt:
        train_cos = ckpt['train_metrics'].get('avg_cosine_sim') or ckpt['train_metrics'].get('avg_cosine')
    logger.info(f"Train cosine (from checkpoint.train_metrics): {train_cos}")

    # Update checkpoint dict with validation results
    ckpt_updates = {
        'last_val_mse': val_mse,
        'val_mse': val_mse,
        'val_cosine': val_cos,
        'val_metric': val_metric,
        'val_metric_name': val_metric_name,
        'timestamp': time.time()
    }
    if 'val_semantic_error' in val_metrics:
        ckpt_updates['val_semantic_error'] = val_metrics['val_semantic_error']

    # Show what would be updated
    logger.info('Checkpoint will be updated with:')
    for k, v in ckpt_updates.items():
        logger.info(f"  {k}: {v}")

    if args.dry_run:
        logger.info('Dry-run: not writing checkpoint file')
        return

    # Merge and save
    ckpt.update(ckpt_updates)
    torch.save(ckpt, str(ckpt_path))
    logger.info(f"Wrote updated checkpoint: {ckpt_path}")

    if args.update_latest:
        latest_path = ckpt_path.parent / 'latest.pth'
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
        except Exception:
            pass
        torch.save(ckpt, str(latest_path))
        logger.info(f"Also updated latest.pth: {latest_path}")


if __name__ == '__main__':
    main()
