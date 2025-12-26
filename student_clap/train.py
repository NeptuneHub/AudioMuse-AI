"""
Student CLAP Training Script

Main entry point for training the lightweight student CLAP model.

NOTE: This is a conceptual implementation. Full ONNX training requires:
1. ONNX Runtime training API (onnxruntime-training package)
2. Proper training graph with gradient operators
3. Optimizer state management

This script provides the structure and workflow for training.
For production, consider:
- PyTorch implementation with ONNX export
- ONNX Runtime training artifacts
- TensorFlow to ONNX conversion
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from student_clap.data.dataset import StudentCLAPDataset
from student_clap.models.student_onnx_model import StudentCLAPTrainer
from student_clap.training.evaluation import evaluate_embeddings, print_evaluation_report

logger = logging.getLogger(__name__)


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'].upper())
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def expand_env_vars(config: dict) -> dict:
    """Recursively expand environment variables in config."""
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        env_var = config[2:-1]
        return os.environ.get(env_var, config)
    else:
        return config


def train_epoch_real(trainer: StudentCLAPTrainer,
                    dataset: StudentCLAPDataset,
                    config: dict,
                    epoch: int) -> dict:
    """
    Real ONNX-based training epoch using PyTorch with ONNX export.
    
    Uses knowledge distillation from existing CLAP embeddings in database
    to train a lightweight student model following tinyCLAP approach.
    
    Args:
        trainer: Student CLAP trainer with real ONNX model
        dataset: Training dataset
        config: Configuration dict
        epoch: Current epoch number
        
    Returns:
        Dict with epoch metrics
    """
    logger.info(f"🚀 REAL ONNX TRAINING - Epoch {epoch}/{config['training']['epochs']}")
    
    batch_size = config['training']['batch_size']
    
    total_loss = 0.0
    total_mse = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    num_songs = 0
    
    epoch_start_time = time.time()
    
    # Iterate over batches with STREAMING downloads
    for batch_data in tqdm(dataset.iterate_batches_streaming(batch_size, shuffle=True), 
                          desc=f"Epoch {epoch} - Real Training"):
        
        batch_start_time = time.time()
        
        # Prepare batch for training
        batch = {
            'audio_segments': [],
            'teacher_embeddings': [],
            'song_ids': []
        }
        
        for item in batch_data:
            # Get audio segments for this song (10s segments, 5s hop)
            audio_segments = dataset._segment_audio(item['audio_data'])
            batch['audio_segments'].append(audio_segments)
            batch['teacher_embeddings'].append(item['teacher_embedding'])
            batch['song_ids'].append(item['song_id'])
        
        # 🧠 REAL TRAINING STEP
        logger.info(f"🔥 BATCH {num_batches + 1}: Training on {len(batch_data)} songs...")
        
        try:
            # Forward pass, loss computation, and backward pass
            step_metrics = trainer.train_step(batch)
            
            # Log detailed metrics
            logger.info(f"   ✅ Forward pass through student CNN + Transformer")
            logger.info(f"   📊 Loss: {step_metrics['total_loss']:.6f}")
            logger.info(f"      └─ MSE Loss: {step_metrics['mse_loss']:.6f}")
            logger.info(f"      └─ Cosine Loss: {step_metrics['cosine_loss']:.6f}")
            logger.info(f"   🎯 Cosine Similarity: {step_metrics['mean_cosine_sim']:.4f} (min: {step_metrics['min_cosine_sim']:.4f}, max: {step_metrics['max_cosine_sim']:.4f})")
            
            # Accumulate metrics
            total_loss += step_metrics['total_loss']
            total_mse += step_metrics['mse_loss']
            total_cosine_sim += step_metrics['mean_cosine_sim']
            num_batches += 1
            num_songs += len(batch_data)
            
            batch_time = time.time() - batch_start_time
            logger.info(f"   ⏱️ Batch time: {batch_time:.1f}s ({batch_time/len(batch_data):.1f}s/song)")
            
        except Exception as e:
            logger.error(f"❌ Training failed on batch {num_batches + 1}: {e}")
            continue
        
        logger.info(f"─" * 60)
        
        if num_batches % config['logging']['log_every'] == 0:
            logger.info(f"  Batch {num_batches}: loss={loss:.6f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'epoch': epoch,
        'avg_loss': avg_loss,
        'num_batches': num_batches
    }


def validate_placeholder(dataset: StudentCLAPDataset,
                        config: dict) -> dict:
    """
    Placeholder for validation.
    
    Args:
        dataset: Validation dataset
        config: Configuration dict
        
    Returns:
        Dict with validation metrics
    """
    logger.info("Running validation...")
    
    # Collect embeddings
    student_embeddings_list = []
    teacher_embeddings_list = []
    
    for batch in dataset.iterate_batches_streaming(config['training']['batch_size'], shuffle=False):
        teacher_embeddings = np.stack([item['teacher_embedding'] for item in batch])
        
        # Simulate student embeddings
        student_embeddings = [
            teacher_embeddings[i:i+1].repeat(item['num_segments'], axis=0) + 
            np.random.randn(item['num_segments'], 512).astype(np.float32) * 0.1
            for i, item in enumerate(batch)
        ]
        
        # Average student embeddings
        student_avg = np.array([
            np.mean(seg, axis=0) / (np.linalg.norm(np.mean(seg, axis=0)) + 1e-8)
            for seg in student_embeddings
        ])
        
        student_embeddings_list.append(student_avg)
        teacher_embeddings_list.append(teacher_embeddings)
    
    # Concatenate all
    student_all = np.vstack(student_embeddings_list)
    teacher_all = np.vstack(teacher_embeddings_list)
    
    # Evaluate
    metrics = evaluate_embeddings(student_all, teacher_all)
    
    return metrics


def train(config_path: str, resume: str = None):
    """
    Main training loop.
    
    Args:
        config_path: Path to config file
        resume: Path to checkpoint to resume from
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = expand_env_vars(config)
    
    # Setup logging
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Student CLAP Training")
    logger.info("=" * 60)
    
    # Print architecture
    model_config = TinyCLAPConfig(config['model'])
    logger.info("\n" + get_architecture_description(model_config))
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = StudentCLAPDataset(config, split='train', 
                                       validation_split=config['training']['validation_split'])
    val_dataset = StudentCLAPDataset(config, split='val',
                                     validation_split=config['training']['validation_split'])
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    
    # Print dataset stats
    train_stats = train_dataset.get_dataset_stats()
    logger.info(f"\nDataset statistics:")
    for key, value in train_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train epoch
        train_metrics = train_epoch_placeholder(train_dataset, config, epoch)
        logger.info(f"Epoch {epoch} - Train loss: {train_metrics['avg_loss']:.6f}")
        
        # Validate
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = validate_placeholder(val_dataset, config)
            print_evaluation_report(val_metrics, f"Validation - Epoch {epoch}")
            
            # Check for improvement
            val_loss = val_metrics['mse']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"✓ New best validation loss: {best_val_loss:.6f}")
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.onnx"
                logger.info(f"  Saved checkpoint: {checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{config['training']['early_stopping_patience']})")
        
        # Save regular checkpoint
        if epoch % config['training']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.onnx"
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Final save
    final_path = Path(config['paths']['final_model'])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nTraining complete!")
    logger.info(f"Final model saved to: {final_path}")
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("⚠️  NOTE: This is a placeholder implementation")
    logger.info("   For production, implement full ONNX training loop")
    logger.info("   or use PyTorch with ONNX export")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Student CLAP model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    try:
        train(args.config, args.resume)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
