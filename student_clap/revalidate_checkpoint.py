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
    p.add_argument('--teacher-model-type', choices=['clap', 'mulan'], help='Override teacher model type for validation')
    p.add_argument('--teacher-model', help='Override teacher model path/identifier for validation')
    p.add_argument('--no-cache', action='store_true', help='Disable mel/teacher cache during validation (forces recompute)')
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    # Resolve config path (support running from inside student_clap/ or repo root)
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        # Try resolving relative to this script folder (student_clap/)
        script_dir = Path(__file__).resolve().parent
        alt = (script_dir / config_path).resolve()
        if alt.exists():
            config_path = alt

    config = yaml.safe_load(open(config_path))
    # Resolve any relative paths in config['paths'] relative to the config file location
    config_file_path = config_path.resolve()
    config_dir = config_file_path.parent
    if isinstance(config.get('paths'), dict):
        for key, val in list(config['paths'].items()):
            if isinstance(val, str) and not os.path.isabs(val):
                resolved = (config_dir / val).resolve()
                config['paths'][key] = str(resolved)
    logger.info(f"Loaded config: {config_path} (resolved paths relative to {config_dir})")

    # Build trainer (we may override config below if checkpoint contains it)
    trainer = StudentCLAPTrainer(config)

    ckpt = torch.load(str(ckpt_path), map_location='cpu')

    # If the checkpoint includes a stored config, merge it to ensure revalidation
    # uses the same teacher/model settings as training.
    ckpt_config = ckpt.get('config')
    trained_teacher_type = None
    if isinstance(ckpt_config, dict):
        # Keep only the config keys that are stable across environments (teacher model + distillation settings).
        # Avoid overriding dataset paths, checkpoints, logs, etc. that are environment-specific.
        safe_ckpt_config = {}
        if 'paths' in ckpt_config and isinstance(ckpt_config['paths'], dict):
            safe_ckpt_config['paths'] = {}
            for k in ['teacher_model', 'teacher_model_type']:
                if k in ckpt_config['paths']:
                    safe_ckpt_config['paths'][k] = ckpt_config['paths'][k]
        if 'training' in ckpt_config and isinstance(ckpt_config['training'], dict):
            safe_ckpt_config['training'] = {}
            for k in ['use_teacher_embedding_cache', 'global_mixup', 'mixup_alpha', 'augmentation_enabled']:
                if k in ckpt_config['training']:
                    safe_ckpt_config['training'][k] = ckpt_config['training'][k]
        if 'model' in ckpt_config and isinstance(ckpt_config['model'], dict):
            safe_ckpt_config['model'] = ckpt_config['model']

        def _merge_dicts(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _merge_dicts(base[k], v)
                else:
                    base[k] = v
            return base

        # Detect teacher type from checkpoint config (may be missing)
        trained_paths = ckpt_config.get('paths', {}) if isinstance(ckpt_config.get('paths'), dict) else {}
        trained_teacher_type = trained_paths.get('teacher_model_type')
        if trained_teacher_type is None and 'teacher_model' in trained_paths:
            trained_teacher_model = str(trained_paths['teacher_model']).lower()
            if any(x in trained_teacher_model for x in ['mulan', 'muq', 'muq-mulan', 'mulan-large']):
                trained_teacher_type = 'mulan'
            else:
                trained_teacher_type = 'clap'

        config = _merge_dicts(config, safe_ckpt_config)
        logger.info("Merged selected checkpoint config into current config (ensures teacher/model settings match training)")

        # Rebuild trainer with merged config if the model architecture / settings changed
        trainer = StudentCLAPTrainer(config)

    # If we know what teacher type the checkpoint was trained with, warn when
    # the user overrides it to a different teacher (different embedding space).
    if trained_teacher_type is not None:
        current_teacher_type = config.get('paths', {}).get('teacher_model_type')
        if current_teacher_type is None and 'teacher_model' in config.get('paths', {}):
            tm = str(config['paths']['teacher_model']).lower()
            current_teacher_type = 'mulan' if any(x in tm for x in ['mulan', 'muq', 'muq-mulan', 'mulan-large']) else 'clap'
        if current_teacher_type and current_teacher_type != trained_teacher_type:
            logger.warning(
                f"⚠️ Checkpoint was trained with teacher_type='{trained_teacher_type}', "
                f"but validation is configured with teacher_type='{current_teacher_type}'. "
                "Cosine similarity will NOT be directly comparable (different embedding spaces)."
            )

    # Apply CLI overrides (useful to force MuLan + disable cache)
    if args.teacher_model_type:
        config.setdefault('paths', {})['teacher_model_type'] = args.teacher_model_type
        logger.info(f"Overriding teacher_model_type to: {args.teacher_model_type}")
    if args.teacher_model:
        config.setdefault('paths', {})['teacher_model'] = args.teacher_model
        logger.info(f"Overriding teacher_model to: {args.teacher_model}")
    if args.no_cache:
        config.setdefault('training', {})['use_teacher_embedding_cache'] = False
        logger.info("Disabling teacher embedding cache for validation (no cached mel/embeddings)")

    # If we changed config via CLI overrides, rebuild trainer to reflect changes
    trainer = StudentCLAPTrainer(config)

    # Restore model weights
    if 'model_state_dict' not in ckpt:
        logger.error('Provided file does not look like a training checkpoint (no model_state_dict).')
        raise SystemExit(1)

    trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    logger.info(f"Model weights loaded (device={trainer.device})")

    # If the loaded weights are in bfloat16 (common when training with AMP),
    # adjust validation run to avoid dtype mismatches.
    has_bf16 = any(p.dtype == torch.bfloat16 for p in trainer.model.parameters())
    logger.info(f"Model contains bfloat16 weights: {has_bf16}")

    # If running on MPS, avoid using AMP/autocast entirely (it can produce
    # mixed-dtype inputs that crash conv layers).  Cast model back to float32.
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        trainer.model.to(dtype=torch.float32)
        trainer.use_amp = False
        logger.info("Running on MPS: forcing float32 model and disabling AMP/autocast")
    elif has_bf16 and torch.cuda.is_available():
        trainer.use_amp = True
        trainer.amp_device_type = 'cuda'
        logger.info("Detected bfloat16 weights; enabling AMP autocast for validation on CUDA")
    elif has_bf16:
        # On CPU or non-CUDA/MPS, cast back to float32 for safety
        trainer.model.to(dtype=torch.float32)
        trainer.use_amp = False
        logger.info("Detected bfloat16 weights but no CUDA available; casting model to float32 for validation")

    logger.info(f"Validation AMP settings: use_amp={trainer.use_amp}, amp_device_type={getattr(trainer,'amp_device_type',None)}")

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
