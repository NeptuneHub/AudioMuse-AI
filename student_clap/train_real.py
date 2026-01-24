"""
Student CLAP Training Script

Main entry point for training the lightweight student CLAP model.
Implements real ONNX-based training using PyTorch with knowledge distillation
from existing CLAP embeddings stored in the database.
"""

import os

import sys
import yaml
import logging
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure parent directory is in sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Text distillation imports
from student_clap.models.student_text_model import StudentCLAPText
from student_clap.data.text_sampler import sample_text_queries
from student_clap.data.clap_text_embedder import CLAPTextEmbedder
from student_clap.data.text_tokenizer import get_tokenizer

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
    # Print device, precision, LR, WD at epoch start
    device = trainer.device
    lr = trainer.optimizer.param_groups[0]['lr']
    wd = trainer.optimizer.param_groups[0].get('weight_decay', None)
    logger.info(f"🚀 REAL ONNX TRAINING - Epoch {epoch}/{config['training']['epochs']} | Device: {device} | LR: {lr} | WD: {wd}")
    
    batch_size = config['training']['batch_size']
    
    total_loss = 0.0
    total_mse = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    num_songs = 0
    
    # Calculate total batches for progress tracking
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    epoch_start_time = time.time()
    
    logger.info(f"📊 EPOCH {epoch}/{config['training']['epochs']} - Processing {len(dataset)} songs in ~{total_batches} batches")
    
    # Iterate over batches with STREAMING downloads
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch_idx, batch_data in enumerate(tqdm(dataset.iterate_batches_streaming(batch_size, shuffle=True), 
                          desc=f"Epoch {epoch} - Real Training")):
        
        batch_start_time = time.time()
        
        # Prepare batch for training
        batch = {
            'audio_segments': [],
            'teacher_embeddings': [],
            'teacher_segment_embeddings': [],
            'song_ids': []
        }
        
        for item in batch_data:
            # Get audio segments for this song (already segmented by dataset)
            audio_segments = item['audio_segments']
            batch['audio_segments'].append(audio_segments)
            batch['teacher_embeddings'].append(item['teacher_embedding'])
            batch['teacher_segment_embeddings'].append(item.get('teacher_segment_embeddings'))
            batch['song_ids'].append(item['item_id'])
        
        # 🧠 REAL TRAINING STEP
        logger.info(f"🔥 BATCH {num_batches + 1}/{total_batches} (EPOCH {epoch}/{config['training']['epochs']}): Training on {len(batch_data)} songs...")
        
        try:
            # --- Linear LR warmup for epoch 1 ---
            if epoch == 1:
                warmup_lr = (batch_idx + 1) / total_batches * config['training']['learning_rate']
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"[LR WARMUP] Epoch 1, Batch {batch_idx+1}/{total_batches}: LR set to {warmup_lr:.6f}")
            # Forward pass, loss computation, and backward pass
            step_metrics = trainer.train_step(batch)
            
            # Log detailed metrics
            accumulation_info = f" [acc {step_metrics['accumulation_step']}/{trainer.gradient_accumulation_steps}]"
            update_info = " 🔄 WEIGHTS UPDATED!" if step_metrics['will_update'] else ""
            num_training_samples = step_metrics.get('num_training_samples', len(batch_data))
            logger.info(f"   ✅ Forward pass through student CNN + Transformer{accumulation_info}{update_info}")
            logger.info(f"   📈 Training samples: {num_training_samples} (from {len(batch_data)} songs)")
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
            
            # 🧹 AGGRESSIVE MEMORY CLEANUP (prevent 15GB buildup)
            import gc
            import torch
            
            # Clear only the large tensors, keep variables we need
            if 'batch' in locals():
                if 'audio_segments' in batch:
                    del batch['audio_segments']  # This is the heavy data
                del batch
            if 'step_metrics' in locals():
                del step_metrics
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache (MPS and CPU)
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"   🧹 Memory cleaned after batch")
            
            # 📊 Log memory usage (Mac Mini has 16GB)
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_used_gb = (memory.total - memory.available) / (1024**3)
                memory_total_gb = memory.total / (1024**3)
                logger.info(f"   💾 Memory: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory.percent:.1f}%)")
            except ImportError:
                pass  # psutil not available
            epoch_progress = (num_batches + 1) / total_batches * 100
            total_progress = ((epoch - 1) + (num_batches + 1) / total_batches) / config['training']['epochs'] * 100
            
            # Estimate time remaining
            elapsed_time = time.time() - epoch_start_time
            if num_batches > 0:
                avg_batch_time = elapsed_time / (num_batches + 1)
                eta_epoch = avg_batch_time * (total_batches - num_batches - 1)
                logger.info(f"   ⏱️ Batch: {batch_time:.1f}s ({batch_time/len(batch_data):.1f}s/song)")
                logger.info(f"   📈 Progress: {epoch_progress:.1f}% epoch, {total_progress:.1f}% total (ETA: {eta_epoch/60:.1f}min)")
            else:
                logger.info(f"   ⏱️ Batch time: {batch_time:.1f}s ({batch_time/len(batch_data):.1f}s/song)")
            
        except Exception as e:
            logger.error(f"❌ Training failed on batch {num_batches + 1}: {e}")
            continue
        
        logger.info(f"─" * 60)
    
    # Compute averages BEFORE updating scheduler (ReduceLROnPlateau needs the metric)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    
    # Scheduler stepping is handled after validation (we want to monitor validation cosine for generalization).
    # Do not step scheduler here on training metric to avoid reducing LR based on training improvements.
    current_lr = trainer.optimizer.param_groups[0]['lr']
    
    epoch_time = time.time() - epoch_start_time
    
    # Show mel cache stats at end of epoch 1
    if epoch == 1:
        cache_stats = dataset.mel_cache.get_stats()
        logger.info(f"📦 MEL CACHE STATS (END OF EPOCH 1):")
        logger.info(f"   🎵 Total cached: {cache_stats['total_cached']} songs")
        logger.info(f"   💾 Cache size: {cache_stats['cache_size_gb']:.1f} GB")
        logger.info(f"   📊 Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    
    logger.info(f"🎯 EPOCH {epoch}/{config['training']['epochs']} COMPLETE:")
    logger.info(f"   📈 Average Loss: {avg_loss:.6f}")
    logger.info(f"   📊 Average MSE: {avg_mse:.6f}")
    logger.info(f"   🎯 Average Cosine Sim: {avg_cosine_sim:.4f}")
    logger.info(f"   📚 Songs processed: {num_songs}/{len(dataset)} ({num_batches}/{total_batches} batches)")
    logger.info(f"   ⏱️ Epoch time: {epoch_time/60:.1f}min ({epoch_time/num_songs:.1f}s/song)")
    logger.info(f"   📖 Learning rate: {current_lr:.2e}")
    
    # Training progress summary
    training_progress = epoch / config['training']['epochs'] * 100
    logger.info(f"🚀 OVERALL TRAINING PROGRESS: {training_progress:.1f}% ({epoch}/{config['training']['epochs']} epochs)")
    
    return {
        'epoch': epoch,
        'avg_loss': avg_loss,
        'avg_mse': avg_mse,
        'avg_cosine_sim': avg_cosine_sim,
        'num_batches': num_batches,
        'num_songs': num_songs,
        'epoch_time': epoch_time,
        'learning_rate': current_lr
    }


def validate_real(trainer: StudentCLAPTrainer,
                 dataset: StudentCLAPDataset,
                 config: dict,
                 epoch: int = 1) -> dict:
    """
    Real validation using trained student model.
    
    Args:
        trainer: Student CLAP trainer with trained model
        dataset: Validation dataset
        config: Configuration dict
        epoch: Current epoch number for logging
        
    Returns:
        Dict with validation metrics
    """
    logger.info(f"🔍 Running REAL validation (Epoch {epoch})...")
    
    trainer.model.eval()
    trainer.model.to(trainer.device)
    
    # Collect embeddings
    student_embeddings_list = []
    teacher_embeddings_list = []
    song_ids = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataset.iterate_batches_streaming(config['training']['batch_size'], shuffle=False),
                              desc=f"Validation (Epoch {epoch})"):
            
            # Prepare batch
            batch = {
                'audio_segments': [],
                'teacher_embeddings': [],
                'song_ids': []
            }
            
            for item in batch_data:
                audio_segments = item['audio_segments']
                if not isinstance(audio_segments, torch.Tensor):
                    audio_segments = torch.from_numpy(audio_segments)
                audio_segments = audio_segments.to(device=trainer.device)
                batch['audio_segments'].append(audio_segments)
                batch['teacher_embeddings'].append(item['teacher_embedding'])
                batch['song_ids'].append(item['item_id'])
            
            # Forward pass without gradients
            student_embeddings = []
            teacher_embeddings_batch = []  # Only include teachers for processed students
            
            for i, audio_segments in enumerate(batch['audio_segments']):
                # audio_segments are PRE-COMPUTED mel spectrograms! (num_segments, 1, 128, time)
                if not isinstance(audio_segments, torch.Tensor):
                    audio_segments = torch.from_numpy(audio_segments)
                audio_segments = audio_segments.to(device=trainer.device)
                # ⚠️ SKIP SONGS WITH ONLY 1 SEGMENT (BatchNorm requires at least 2 samples)
                if audio_segments.shape[0] < 2:
                    logger.warning(f"⚠️ Skipping song {batch['song_ids'][i]} in validation - only {audio_segments.shape[0]} segment")
                    continue
                
                # Process segments in chunks to reduce memory usage
                chunk_size = config['model'].get('segment_batch_size', 10)
                segment_embeddings_list = []
                for chunk_start in range(0, audio_segments.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, audio_segments.shape[0])
                    chunk = audio_segments[chunk_start:chunk_end]
                    chunk_embeddings = trainer.model(chunk)  # (chunk_size, 512)
                    segment_embeddings_list.append(chunk_embeddings)
                
                segment_embeddings = torch.cat(segment_embeddings_list, dim=0)  # (num_segments, 512)
                
                # Average across segments to get single embedding per song (same as training!)
                avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)  # (1, 512)
                
                # Re-normalize after averaging
                avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=1)
                
                student_embeddings.append(avg_embedding.cpu().numpy())
                teacher_embeddings_batch.append(batch['teacher_embeddings'][i])  # Only add teacher if student was processed
            
            # Stack and store (only if we have valid embeddings)
            if student_embeddings:
                student_batch = np.vstack(student_embeddings)
                teacher_batch = np.stack(teacher_embeddings_batch)
                
                student_embeddings_list.append(student_batch)
                teacher_embeddings_list.append(teacher_batch)
                song_ids.extend([batch['song_ids'][i] for i in range(len(batch['song_ids'])) if batch['audio_segments'][i].shape[0] >= 2])
    
    # Concatenate all embeddings
    student_all = np.vstack(student_embeddings_list)
    teacher_all = np.vstack(teacher_embeddings_list)
    
    # Evaluate
    metrics = evaluate_embeddings(student_all, teacher_all)
    metrics['num_songs'] = len(song_ids)

    # Restore model to training mode to avoid surprising downstream code
    try:
        trainer.model.train()
        logger.info("🔁 Restored model to train mode after validation")
    except Exception:
        pass

    return metrics


def train(config_path: str, resume: str = None):
    """
    Main training loop with real ONNX implementation.
    
    Args:
        config_path: Path to config file
        resume: Path to checkpoint to resume from
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # === DISTILLATION ENABLE FLAGS ===
    audio_enabled = config.get('distillation', {}).get('audio_enabled', True)
    text_enabled = config.get('distillation', {}).get('text_enabled', True)
    if text_enabled:
        text_json_path = Path(config['paths']['text_json'])
        text_teacher_path = Path(config['paths']['teacher_model_text'])
        text_checkpoint_dir = Path(config['paths']['checkpoints'])
        text_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Categories to sample (pick any 3, or make configurable)
        text_categories = ['Genre_Style', 'Instrumentation_Vocal', 'Emotion_Mood']
        # Teacher text embedder and tokenizer
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.vocab_size
        text_cfg = config.get('model_text', {})
        embedding_dim = text_cfg.get('embedding_dim', 512)
        hidden_dim = text_cfg.get('hidden_dim', 256)
        num_layers = text_cfg.get('num_layers', 2)
        nhead = text_cfg.get('nhead', 4)
        student_text_model = StudentCLAPText(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead
        )
        text_optimizer = torch.optim.Adam(student_text_model.parameters(), lr=3e-4)
        text_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(text_optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
        teacher_text_embedder = CLAPTextEmbedder(str(text_teacher_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        student_text_model.to(device)
    
    # Expand environment variables
    config = expand_env_vars(config)
    
    # Setup logging
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("🚀 Student CLAP REAL ONNX Training")
    logger.info("=" * 60)
    
    # Initialize trainer with real ONNX model
    logger.info("\n🏗️ Building Student CLAP model...")
    trainer = StudentCLAPTrainer(config)

    # Log initial learning rate and weight decay for stage 1 optimizer
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    initial_wd = trainer.optimizer.param_groups[0].get('weight_decay', None)
    logger.info(f"🔧 Stage 1 initial learning rate: {initial_lr:.6f} | weight_decay: {initial_wd}")

    # Print model info
    model_info = trainer.model.count_parameters()
    logger.info(f"\n📊 Model Architecture:")
    logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"   Estimated size: {model_info['estimated_size_mb']:.1f} MB")
    logger.info(f"   Device: {trainer.device}")
    
    # Initialize start_epoch early (before datasets need it)
    start_epoch = 1
    best_val_cosine = 0.0
    patience_counter = 0
    last_val_cosine = None  # Store last validation cosine (None if not run yet)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for resume argument first
    audio_resume_path = None
    text_resume_path = None
    if resume:
        logger.info(f"\n⏮️ Manual resume requested: {resume}")
        # User may provide either audio or text checkpoint manually
        if 'text' in resume:
            text_resume_path = resume
        else:
            audio_resume_path = resume
    else:
        # Auto-detect latest checkpoint for audio and text
        if audio_enabled:
            audio_latest_path = checkpoint_dir / "latest.pth"
            if audio_latest_path.exists() and audio_latest_path.is_file():
                logger.info(f"\n🔍 Auto-detected existing audio checkpoint: {audio_latest_path}")
                audio_resume_path = str(audio_latest_path)
        if text_enabled:
            text_latest_path = checkpoint_dir / "last_text.pth"
            if text_latest_path.exists() and text_latest_path.is_file():
                logger.info(f"\n🔍 Auto-detected existing text checkpoint: {text_latest_path}")
                text_resume_path = str(text_latest_path)
        if not audio_resume_path and not text_resume_path:
            logger.info(f"\n🆕 No existing checkpoints found - starting fresh training")
    # Load audio checkpoint if available
    if audio_enabled and audio_resume_path:
        try:
            logger.info(f"📂 Loading audio checkpoint: {audio_resume_path}")
            checkpoint = torch.load(audio_resume_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            # Attempt to restore optimizer state; if missing or failing, keep fresh optimizer and apply config LR/WD
            if 'optimizer_state_dict' in checkpoint:
                try:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Ensure LR and weight_decay align with config (override to config values)
                    new_lr = config['training']['learning_rate']
                    new_wd = config['training']['weight_decay']
                    for pg in trainer.optimizer.param_groups:
                        pg['lr'] = new_lr
                        pg['weight_decay'] = new_wd
                    logger.info(f"✓ Optimizer restored from checkpoint and LR/WD overridden to config (lr={new_lr}, wd={new_wd})")
                except Exception as e:
                    logger.warning(f"⚠️ Could not restore optimizer state cleanly: {e}; using fresh optimizer with config values")
                    for pg in trainer.optimizer.param_groups:
                        pg['lr'] = config['training']['learning_rate']
                        pg['weight_decay'] = config['training']['weight_decay']
            else:
                logger.info("No optimizer state in checkpoint — using fresh optimizer (config LR/WD applied)")
                for pg in trainer.optimizer.param_groups:
                    pg['lr'] = config['training']['learning_rate']
                    pg['weight_decay'] = config['training']['weight_decay']

            # Attempt to restore scheduler; if missing or failing, create a new one driven by validation (mode='max')
            if 'scheduler_state_dict' in checkpoint:
                try:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("✓ Scheduler restored from checkpoint")
                except Exception as e:
                    logger.warning(f"⚠️ Could not restore scheduler state: {e}")
                    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        trainer.optimizer,
                        mode='max',
                        factor=0.1,
                        patience=3,
                        threshold=0.005,
                        threshold_mode='rel',
                        min_lr=1e-6
                    )
                    logger.info("✓ Created new scheduler (mode=max) due to restore failure")
            else:
                trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    trainer.optimizer,
                    mode='max',
                    factor=0.1,
                    patience=3,
                    threshold=0.005,
                    threshold_mode='rel',
                    min_lr=1e-6
                )
                logger.info("No scheduler state in checkpoint — created fresh scheduler (mode=max)")

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_cosine = checkpoint.get('best_val_cosine', 0.0)
            patience_counter = checkpoint.get('patience_counter', 0)
            logger.info(f"✅ Successfully resumed audio from epoch {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"   📈 Best cosine similarity so far: {best_val_cosine:.4f}")
            logger.info(f"   ⏰ Patience counter: {patience_counter}/{config['training']['early_stopping_patience']}")
            logger.info(f"   🎯 Will continue from epoch {start_epoch}")
            # Confirm optimizer param groups (LR & weight_decay) after resume
            try:
                pg = trainer.optimizer.param_groups[0]
                logger.info(f"   🔧 Optimizer after resume: lr={pg.get('lr')} | weight_decay={pg.get('weight_decay')}")
            except Exception:
                pass
            if start_epoch > (config['training']['epochs'] + config['training'].get('stage2_epochs', 0)):
                logger.info(f"🎉 Audio training already completed! (reached {config['training']['epochs'] + config['training'].get('stage2_epochs', 0)} epochs)")
                final_onnx_path = Path(config['paths']['final_model'])
                trainer.export_to_onnx(str(final_onnx_path))
                logger.info(f"✅ Exported final audio ONNX model: {final_onnx_path}")
                return
        except Exception as e:
            logger.error(f"❌ Failed to load audio checkpoint: {e}")
            logger.info("🔄 Starting audio training from scratch...")
            start_epoch = 1
            best_val_cosine = 0.0
            patience_counter = 0
    # Load text checkpoint if available
    if text_enabled and text_resume_path:
        try:
            logger.info(f"📂 Loading text checkpoint: {text_resume_path}")
            checkpoint = torch.load(text_resume_path, map_location=device)
            student_text_model.load_state_dict(checkpoint['model_state_dict'])
            text_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            try:
                text_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"✅ Text scheduler state restored")
            except Exception as e:
                logger.warning(f"⚠️ Could not restore text scheduler state: {e}")
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"✅ Successfully resumed text from epoch {checkpoint['epoch']}")
            if start_epoch > config['training']['epochs']:
                logger.info(f"🎉 Text training already completed! (reached {config['training']['epochs']} epochs)")
                # Export final text ONNX model to /student_clap/models/final_text_model.onnx
                final_text_onnx_path = Path(config['paths']['final_model_text'])
                final_text_onnx_path.parent.mkdir(parents=True, exist_ok=True)
                student_text_model.export_to_onnx(str(final_text_onnx_path), device=device)
                logger.info(f"✅ Exported final text ONNX model: {final_text_onnx_path}")
                return
        except Exception as e:
            logger.error(f"❌ Failed to load text checkpoint: {e}")
            logger.info("🔄 Starting text training from scratch...")
            start_epoch = 1
    
    # Create datasets NOW that we know start_epoch
    logger.info("\n📁 Creating datasets...")
    
    # 🔄 Check for mel cache checkpoint (in case previous training failed)
    cache_checkpoint_path = Path(config['paths']['checkpoints']) / 'mel_cache_checkpoint.pkl'
    if cache_checkpoint_path.exists():
        logger.info(f"💡 Found mel cache checkpoint: {cache_checkpoint_path}")
        logger.info("   Note: Mel cache will be automatically restored from this if needed")
        logger.info("   (The dataset class handles cache restoration automatically)")
    
    train_dataset = StudentCLAPDataset(config, split='train', 
                                       validation_split=config['training']['validation_split'],
                                       epoch=start_epoch)
    val_dataset = StudentCLAPDataset(config, split='val',
                                     validation_split=config['training']['validation_split'],
                                     epoch=start_epoch)
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    
    # Print dataset stats
    train_stats = train_dataset.get_dataset_stats()
    logger.info(f"\n📈 Dataset statistics:")
    for key, value in train_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Two-stage training setup (like tinyCLAP)
    stage1_epochs = config['training']['epochs']  # 15 epochs
    stage2_epochs = config['training'].get('stage2_epochs', 5)  # 5 additional epochs
    total_epochs = stage1_epochs + stage2_epochs  # 20 total
    
    # Training loop
    logger.info("\n" + "=" * 60)
    if start_epoch == 1:
        logger.info("🎓 Starting FRESH ONNX Training (TWO-STAGE)...")
    else:
        logger.info(f"🔄 RESUMING ONNX Training from epoch {start_epoch}...")
    logger.info("   📚 Using existing CLAP embeddings from database")
    logger.info("   🏗️ Knowledge distillation: Teacher (268MB) → Student (~20-40MB)")
    logger.info("   🎵 Music-specialized compression following tinyCLAP")
    logger.info("   🎯 STAGE 1: Epochs 1-{} - Train entire model".format(stage1_epochs))
    logger.info("   🎯 STAGE 2: Epochs {}-{} - Freeze encoder, refine projection only".format(stage1_epochs + 1, total_epochs))
    if start_epoch > 1:
        progress_pct = (start_epoch - 1) / total_epochs * 100
        logger.info(f"   📊 Training progress: {progress_pct:.1f}% complete ({start_epoch-1}/{total_epochs} epochs done)")
    logger.info("=" * 60)
    
    # 💾 Save mel cache checkpoint before training starts
    logger.info("\n💾 Creating mel cache checkpoint before training...")
    cache_checkpoint_path = Path(config['paths']['checkpoints']) / 'mel_cache_checkpoint.pkl'
    try:
        # Save cache state from both datasets
        import pickle
        cache_data = {
            'train_cache': dict(train_dataset.mel_cache) if hasattr(train_dataset, 'mel_cache') else {},
            'val_cache': dict(val_dataset.mel_cache) if hasattr(val_dataset, 'mel_cache') else {},
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'config': config
        }
        with open(cache_checkpoint_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"✅ Mel cache checkpoint saved: {cache_checkpoint_path}")
        if hasattr(train_dataset, 'mel_cache'):
            logger.info(f"   Train cache: {len(train_dataset.mel_cache)} songs")
        if hasattr(val_dataset, 'mel_cache'):
            logger.info(f"   Val cache: {len(val_dataset.mel_cache)} songs")
        logger.info("   💡 If training fails, you can restore this cache to avoid recomputing!")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save mel cache checkpoint: {e}")
        logger.warning("   Training will continue, but cache won't be preserved on failure")
    
    best_val_cosine = 0.0
    patience_counter = 0
    training_start_time = time.time()
    stage2_triggered = False
    
    audio_enabled = config.get('distillation', {}).get('audio_enabled', True)
    for epoch in range(start_epoch, total_epochs + 1):
        # Log scheduler settings at start of epoch
        try:
            sched = trainer.scheduler
            sched_mode = getattr(sched, 'mode', 'N/A')
            sched_factor = getattr(sched, 'factor', 'N/A')
            sched_patience = getattr(sched, 'patience', 'N/A')
            sched_threshold = getattr(sched, 'threshold', 'N/A')
            sched_threshold_mode = getattr(sched, 'threshold_mode', 'N/A')
            sched_min_lr = getattr(sched, 'min_lrs', None)
            if sched_min_lr is not None:
                # min_lrs can be a list
                sched_min_lr = sched_min_lr[0] if isinstance(sched_min_lr, (list, tuple)) else sched_min_lr
            curr_lr = trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer, 'optimizer') else 'N/A'
            logger.info(f"📊 Scheduler @ epoch {epoch}: mode={sched_mode}, factor={sched_factor}, patience={sched_patience}, threshold={sched_threshold}, threshold_mode={sched_threshold_mode}, min_lr={sched_min_lr}, current_lr={curr_lr}")
        except Exception as e:
            logger.warning(f"⚠️ Could not read scheduler settings: {e}")

        # === TEXT DISTILLATION EPOCH ===
        if text_enabled:
            logger.info(f"\n=== TEXT DISTILLATION: Epoch {epoch} ===")
            n_text_samples = config.get('dataset', {}).get('sample_size', 0)
            if n_text_samples == 0:
                n_text_samples = 50000
            text_queries = sample_text_queries(text_json_path, text_categories, n_samples=n_text_samples)
            batch_size = config.get('model', {}).get('segment_batch_size', 5)
            total_batches = (len(text_queries) + batch_size - 1) // batch_size
            logger.info(f"[TEXT] Using batch_size={batch_size}, n_text_samples={n_text_samples}, total_batches={total_batches}, actual_queries={len(text_queries)}")
            student_text_model.train()
            total_text_loss = 0.0
            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(text_queries))
                batch_texts = text_queries[start:end]
                logger.info(f"[DEBUG] Batch {batch_idx+1}/{total_batches}: {len(batch_texts)} queries (should be <= {batch_size})")
                enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=77, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                with torch.no_grad():
                    teacher_emb = teacher_text_embedder.encode(input_ids.cpu().numpy(), attention_mask.cpu().numpy())
                    teacher_emb = torch.from_numpy(teacher_emb).to(device).float()
                student_emb = student_text_model(input_ids, attention_mask)
                cosine_sim = torch.nn.functional.cosine_similarity(student_emb, teacher_emb, dim=1)
                loss = -cosine_sim.mean()
                text_optimizer.zero_grad()
                loss.backward()
                text_optimizer.step()
                total_text_loss += loss.item()
                for i, query in enumerate(batch_texts):
                    logger.info(f"[TEXT][Batch {batch_idx+1}/{total_batches}] Query: '{query}' | Cosine Loss: {-cosine_sim[i].item():.6f} | Cosine Sim: {cosine_sim[i].item():.6f}")
                min_sim = cosine_sim.min().item()
                max_sim = cosine_sim.max().item()
                logger.info(f"[TEXT][Batch {batch_idx+1}/{total_batches}] Min Cosine Sim: {min_sim:.6f} | Max Cosine Sim: {max_sim:.6f}")
            avg_text_loss = total_text_loss / total_batches
            text_scheduler.step(avg_text_loss)
            logger.info(f"[TEXT] Epoch {epoch}: Avg distillation loss: {avg_text_loss:.6f}")

            # Save text model checkpoint BEFORE audio checkpoint, so epoch is correct
            text_ckpt = {
                'epoch': epoch,
                'model_state_dict': student_text_model.state_dict(),
                'optimizer_state_dict': text_optimizer.state_dict(),
                'scheduler_state_dict': text_scheduler.state_dict(),
                'avg_text_loss': avg_text_loss,
                'config': config,
                'timestamp': time.time()
            }
            text_ckpt_path = text_checkpoint_dir / f"checkpoint_text_epoch_{epoch}.pth"
            torch.save(text_ckpt, text_ckpt_path)
            last_text_ckpt_path = text_checkpoint_dir / "last_text.pth"
            torch.save(text_ckpt, last_text_ckpt_path)
            logger.info(f"[TEXT] Saved checkpoint: {text_ckpt_path}")
            logger.info(f"[TEXT] Updated last_text.pth: {last_text_ckpt_path}")
        # STAGE 2: Switch to projection-only training after stage1_epochs (audio only)
        if audio_enabled and epoch == stage1_epochs + 1 and not stage2_triggered:
            logger.info("\n" + "=" * 60)
            logger.info("🔄 SWITCHING TO STAGE 2: Projection-only refinement")
            logger.info("=" * 60)

            # Freeze encoder layers
            trainer._freeze_encoder()

            # Get current (final stage1) learning rate
            stage1_lr = trainer.optimizer.param_groups[0]['lr']
            default_stage2_lr = config['training'].get('stage2_learning_rate', 0.001)
            # Use max(stage1_lr, 0.001) as stage2_lr, but if stage1_lr < 0.001, use stage1_lr and trigger reduction
            if stage1_lr < default_stage2_lr:
                stage2_lr = stage1_lr
                logger.info(f"   ⚠️ Stage 1 LR ({stage1_lr:.2e}) < default stage2 LR ({default_stage2_lr:.2e}), using stage1 LR for stage2 start!")
                trigger_reduction = True
            else:
                stage2_lr = default_stage2_lr
                trigger_reduction = False

            trainer.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=stage2_lr,
                weight_decay=config['training']['weight_decay']
            )

            stage2_sched_cfg = config['training'].get('stage2_lr_scheduler', {})
            s2_mode = stage2_sched_cfg.get('mode', 'max')
            s2_factor = stage2_sched_cfg.get('factor', 0.1)
            s2_patience = stage2_sched_cfg.get('patience', 3)
            s2_threshold = stage2_sched_cfg.get('threshold', 0.005)
            s2_threshold_mode = stage2_sched_cfg.get('threshold_mode', 'rel')
            s2_min = stage2_sched_cfg.get('min_lr', 1e-6)

            trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode=s2_mode,
                factor=s2_factor,
                patience=s2_patience,
                threshold=s2_threshold,
                threshold_mode=s2_threshold_mode,
                min_lr=s2_min
            )
            logger.info(f"📉 Stage 2 LR Scheduler reset: ReduceLROnPlateau (factor={s2_factor}, patience={s2_patience}, threshold={s2_threshold})")
            logger.info(f"   📈 Stage 2 learning rate: {stage2_lr:.2e}")
            logger.info(f"   📊 Training {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,} parameters (projection head only)")
            logger.info("   🎯 This refines the embedding alignment while keeping learned features intact")
            logger.info("=" * 60 + "\n")

            # If stage1_lr < 0.001, trigger scheduler step to reduce LR immediately
            if trigger_reduction:
                logger.info(f"   🚨 Triggering immediate LR reduction in stage 2 (ReduceLROnPlateau.step)")
                # Step with a high loss to force reduction
                trainer.scheduler.step(1.0)

            stage2_triggered = True

        # --- LOG LR REDUCTION EVENT ---
        if hasattr(trainer.scheduler, '_last_lr') and hasattr(trainer.scheduler, 'last_epoch'):
            # Check if learning rate was reduced in the previous epoch
            if epoch > start_epoch:
                prev_lr = getattr(trainer.scheduler, '_last_lr', [None])[0]
                curr_lr = trainer.optimizer.param_groups[0]['lr']
                if prev_lr is not None and curr_lr < prev_lr:
                    logger.info("\n" + "!"*60)
                    logger.info(f"!!! LEARNING RATE REDUCED at start of EPOCH {epoch}: {prev_lr:.6f} -> {curr_lr:.6f} !!!")
                    logger.info("!"*60 + "\n")
        # ✅ NO dataset recreation needed - iterate_batches_streaming handles everything!
        # The datasets are already created before the loop and stream data lazily.
        # Recreating them would:
        #   1. Waste time
        #   2. Risk memory leaks (old datasets not cleaned)
        #   3. Re-query the cache database unnecessarily
        
        # Update config with current stage info for logging
        current_stage = 1 if epoch <= stage1_epochs else 2
        config_with_stage = config.copy()
        config_with_stage['training'] = config['training'].copy()
        config_with_stage['training']['epochs'] = total_epochs
        config_with_stage['current_stage'] = current_stage

        # Only run audio distillation if enabled
        if audio_enabled:
            # Train epoch with REAL implementation
            train_metrics = train_epoch_real(trainer, train_dataset, config_with_stage, epoch)
            if audio_enabled:
                # 💾 SAVE CHECKPOINT AFTER EVERY EPOCH (for resume capability)
                logger.info(f"💾 Saving checkpoint after epoch {epoch}...")
                epoch_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                latest_checkpoint_path = checkpoint_dir / "latest.pth"
                epoch_checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'best_val_cosine': best_val_cosine,
                    'patience_counter': patience_counter,
                    'config': config,
                    'timestamp': time.time()
                }
                torch.save(epoch_checkpoint_data, epoch_checkpoint_path)
                # Remove symlink if it exists before saving (prevents overwriting old epoch files)
                if latest_checkpoint_path.exists() or latest_checkpoint_path.is_symlink():
                    latest_checkpoint_path.unlink()
                torch.save(epoch_checkpoint_data, latest_checkpoint_path)  # Save as real file, not symlink
                logger.info(f"✅ Checkpoint saved: {epoch_checkpoint_path}")
                logger.info(f"✅ Latest checkpoint updated: {latest_checkpoint_path}")
        # Now save text model checkpoint (after both audio and text are trained, matching audio logic)
        if text_enabled:
            text_ckpt = {
                'epoch': epoch,
                'model_state_dict': student_text_model.state_dict(),
                'optimizer_state_dict': text_optimizer.state_dict(),
                'scheduler_state_dict': text_scheduler.state_dict(),
                'avg_text_loss': avg_text_loss,
                'config': config,
                'timestamp': time.time()
            }
            text_ckpt_path = text_checkpoint_dir / f"checkpoint_text_epoch_{epoch}.pth"
            torch.save(text_ckpt, text_ckpt_path)
            last_text_ckpt_path = text_checkpoint_dir / "last_text.pth"
            torch.save(text_ckpt, last_text_ckpt_path)
            logger.info(f"[TEXT] Saved checkpoint: {text_ckpt_path}")
            logger.info(f"[TEXT] Updated last_text.pth: {last_text_ckpt_path}")
            # Export text model to ONNX after every epoch (only here, not in audio section)
            text_onnx_path = text_checkpoint_dir / f"text_model_epoch_{epoch}.onnx"
            student_text_model.export_to_onnx(str(text_onnx_path), device=device)
            logger.info(f"[TEXT] Exported ONNX: {text_onnx_path}")
        
        # Validate every epoch (only if audio distillation is enabled)
        if audio_enabled:
            val_metrics = validate_real(trainer, val_dataset, config, epoch)
            val_cosine = val_metrics['cosine_similarity']['mean']
            last_val_cosine = val_cosine
            # Log concise validation summary (this will appear in training.log)
            logger.info(f"🔍 Validation completed (Epoch {epoch}) — mean cosine: {val_cosine:.4f} (n={val_metrics.get('num_songs','N/A')})")
            print_evaluation_report(val_metrics, f"Validation - Epoch {epoch}")

            # Update the per-epoch checkpoint to include the last validation cosine so files reflect validation results
            try:
                # Store validation result under both keys to ensure downstream tools/scripts find it
                epoch_checkpoint_data['last_val_cosine'] = last_val_cosine
                epoch_checkpoint_data['val_cosine_sim'] = last_val_cosine
                torch.save(epoch_checkpoint_data, epoch_checkpoint_path)
                if latest_checkpoint_path.exists() or latest_checkpoint_path.is_symlink():
                    latest_checkpoint_path.unlink()
                torch.save(epoch_checkpoint_data, latest_checkpoint_path)
                logger.info(f"✅ Updated epoch checkpoint with validation metrics: {epoch_checkpoint_path} (val_cosine={last_val_cosine:.6f})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update epoch checkpoint with validation metrics: {e}")

            # Step scheduler on validation metric (we monitor cosine similarity - higher is better)
            try:
                trainer.scheduler.step(val_cosine)
                logger.info(f"Scheduler stepped using validation cosine: {val_cosine:.4f}")
            except Exception as e:
                logger.warning(f"Failed to step scheduler on validation metric: {e}")

            # Check for improvement (use cosine similarity as main metric)
            if val_cosine > best_val_cosine:
                best_val_cosine = val_cosine
                patience_counter = 0
                logger.info(f"✓ New best validation cosine similarity: {best_val_cosine:.4f}")
                # Save best checkpoint
                best_checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pth"
                best_checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'val_cosine_sim': val_cosine,
                    'best_val_cosine': best_val_cosine,
                    'patience_counter': patience_counter,
                    'config': config,
                    'timestamp': time.time()
                }
                torch.save(best_checkpoint_data, best_checkpoint_path)
                logger.info(f"  💾 Saved best checkpoint: {best_checkpoint_path}")
                # Export audio model to ONNX
                onnx_path = checkpoint_dir / f"best_model_epoch_{epoch}.onnx"
                trainer.export_to_onnx(str(onnx_path))
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{config['training']['early_stopping_patience']})")
        
        if audio_enabled:
            # Save checkpoint after EVERY epoch (crash recovery)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_metrics': train_metrics,
                'best_val_cosine': best_val_cosine,
                'last_val_cosine': last_val_cosine,
                'val_cosine_sim': last_val_cosine,
                'patience_counter': patience_counter,
                'config': config,
                'timestamp': time.time()
            }
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
            # Export text model to ONNX after every epoch (only if text_enabled)
            if text_enabled:
                text_onnx_path = checkpoint_dir / f"text_model_epoch_{epoch}.onnx"
                student_text_model.export_to_onnx(str(text_onnx_path), device=device)
            # Update latest.pth as a real file (not symlink to avoid corruption)
            latest_path = checkpoint_dir / "latest.pth"
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            torch.save(checkpoint_data, latest_path)
            logger.info(f"🔗 Updated latest checkpoint: {latest_path}")
            # Save additional checkpoint every 5 epochs (backup)
            if epoch % 5 == 0:
                backup_path = checkpoint_dir / f"backup_epoch_{epoch}.pth"
                torch.save(checkpoint_data, backup_path)
                logger.info(f"📦 Backup checkpoint: {backup_path}")
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"\n⏹️ Early stopping triggered after {epoch} epochs")
            break
    
    # Training complete
    total_training_time = time.time() - training_start_time
    logger.info(f"\n🎉 Training complete!")
    logger.info(f"   ⏱️ Total training time: {total_training_time/3600:.1f} hours")
    logger.info(f"   🏆 Best validation cosine similarity: {best_val_cosine:.4f}")
    
    # Final model export
    final_model_path = Path(config['paths']['final_model'])
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save final PyTorch model
    final_pth_path = final_model_path.with_suffix('.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'best_val_cosine_sim': best_val_cosine,
        'total_epochs': epoch
    }, final_pth_path)
    
    # Export final ONNX models (audio and text)
    trainer.export_to_onnx(str(final_model_path))
    logger.info(f"🎯 Final ONNX model: {final_model_path}")
    logger.info(f"🎯 Final PyTorch model: {final_pth_path}")
    if text_enabled:
        final_text_onnx_path = Path(config['paths']['final_model_text'])
        final_text_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        student_text_model.export_to_onnx(str(final_text_onnx_path), device=device)
        logger.info(f"🎯 Final Text ONNX model: {final_text_onnx_path}")
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ REAL ONNX TRAINING COMPLETE!")
    logger.info("   🎵 Student model ready for production deployment")
    logger.info("   📦 Drop-in replacement for existing CLAP audio encoder")
    logger.info("   🚀 Expected 5-10x size reduction, 2-5x speed improvement")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Student CLAP model with REAL ONNX implementation')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    try:
        train(args.config, args.resume)
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)