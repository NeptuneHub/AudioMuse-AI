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

    # Resolve teacher backend/device/provider for clearer logs (audio teacher)
    teacher_descr = 'n/a'
    try:
        clap = getattr(dataset, 'clap_embedder', None)
        if clap is None:
            teacher_descr = 'missing'
        else:
            backend = getattr(clap, '_backend', 'onnx')
            if backend == 'torch':
                tdev = getattr(clap, '_device', None)
                teacher_descr = f"pt/{tdev}"
            else:
                sess = getattr(clap, 'session', None)
                provs = []
                try:
                    provs = sess.get_providers() if sess is not None else []
                except Exception:
                    provs = []
                provider = provs[0] if provs else 'cpu'
                teacher_descr = f"onnx/{provider}"
    except Exception:
        teacher_descr = 'unknown'

    logger.info(f"🚀 REAL ONNX TRAINING - Epoch {epoch}/{config['training']['epochs']} | StudentDevice: {device} | Teacher: {teacher_descr} | LR: {lr} | WD: {wd}")
    
    batch_size = config['training']['batch_size']
    log_every = config.get('logging', {}).get('log_every', 10)

    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    total_semantic = 0.0
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
            'teacher_mel_segments': [],
            'song_ids': []
        }
        
        for item in batch_data:
            # Get audio segments for this song (already segmented by dataset)
            audio_segments = item['audio_segments']
            batch['audio_segments'].append(audio_segments)
            batch['teacher_embeddings'].append(item['teacher_embedding'])
            batch['teacher_segment_embeddings'].append(item.get('teacher_segment_embeddings'))
            batch['teacher_mel_segments'].append(item.get('teacher_mel_segments'))
            batch['song_ids'].append(item['item_id'])

        # --- Mixup augmentation ---
        global_mixup = config['training'].get('global_mixup', False)
        mixup_alpha = config['training'].get('mixup_alpha', 0.0)
        use_global_mixup = (global_mixup and mixup_alpha and mixup_alpha > 0
                            and len(batch['audio_segments']) > 1 and dataset.split == 'train')

        if not use_global_mixup:
            # --- OLD song-level Mixup augmentation (preserved for rollback) ---
            if mixup_alpha and mixup_alpha > 0 and len(batch['audio_segments']) > 1 and dataset.split == 'train':
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                bsz = len(batch['audio_segments'])
                idx = np.random.permutation(bsz)
                # Ensure no self-mix (roll if any fixed points)
                if np.any(idx == np.arange(bsz)):
                    idx = np.roll(idx, 1)
                mixed_audio = []
                mixed_teacher = []
                mixed_teacher_segment_embs = []
                mixed_song_ids = []
                for i in range(bsz):
                    j = int(idx[i])
                    A = batch['audio_segments'][i]
                    B = batch['audio_segments'][j]

                    if isinstance(A, np.ndarray):
                        A_t = torch.from_numpy(A).float().to(device)
                    elif isinstance(A, torch.Tensor):
                        A_t = A.to(device)
                    else:
                        A_t = torch.tensor(A, dtype=torch.float32, device=device)

                    if isinstance(B, np.ndarray):
                        B_t = torch.from_numpy(B).float().to(device)
                    elif isinstance(B, torch.Tensor):
                        B_t = B.to(device)
                    else:
                        B_t = torch.tensor(B, dtype=torch.float32, device=device)

                    min_seg = min(A_t.shape[0], B_t.shape[0]) if A_t.shape[0] and B_t.shape[0] else 0
                    if min_seg == 0:
                        mixed_seg_t = A_t
                    else:
                        A_sub = A_t[:min_seg]
                        B_sub = B_t[:min_seg]
                        mixed_seg_t = lam * A_sub + (1.0 - lam) * B_sub
                        if mixed_seg_t.shape[0] == 1:
                            mixed_seg_t = torch.cat([mixed_seg_t, mixed_seg_t], dim=0)

                    mixed_audio.append(mixed_seg_t)

                    use_teacher_emb_cache = config['training'].get('use_teacher_embedding_cache', True)

                    if use_teacher_emb_cache:
                        embA = batch['teacher_embeddings'][i]
                        embB = batch['teacher_embeddings'][j]
                        if isinstance(embA, np.ndarray):
                            embA_t = torch.from_numpy(embA).float().to(device)
                        elif isinstance(embA, torch.Tensor):
                            embA_t = embA.to(device)
                        else:
                            embA_t = torch.tensor(embA, dtype=torch.float32, device=device)

                        if isinstance(embB, np.ndarray):
                            embB_t = torch.from_numpy(embB).float().to(device)
                        elif isinstance(embB, torch.Tensor):
                            embB_t = embB.to(device)
                        else:
                            embB_t = torch.tensor(embB, dtype=torch.float32, device=device)

                        mixed_teacher.append(lam * embA_t + (1.0 - lam) * embB_t)

                        tsegA = batch['teacher_segment_embeddings'][i]
                        tsegB = batch['teacher_segment_embeddings'][j]
                        if tsegA is not None and tsegB is not None:
                            min_emb_seg = min(len(tsegA), len(tsegB))
                            mixed_tsegs = []
                            for k in range(min_emb_seg):
                                a_k = tsegA[k]
                                b_k = tsegB[k]
                                a_t = torch.from_numpy(a_k).float().to(device) if isinstance(a_k, np.ndarray) else (a_k.to(device) if isinstance(a_k, torch.Tensor) else torch.tensor(a_k, dtype=torch.float32, device=device))
                                b_t = torch.from_numpy(b_k).float().to(device) if isinstance(b_k, np.ndarray) else (b_k.to(device) if isinstance(b_k, torch.Tensor) else torch.tensor(b_k, dtype=torch.float32, device=device))
                                mixed_tsegs.append(lam * a_t + (1.0 - lam) * b_t)
                            mixed_teacher_segment_embs.append(mixed_tsegs)
                        else:
                            mixed_teacher_segment_embs.append(None)
                    else:
                        # When teacher embedding cache is disabled, mix teacher mel
                        # segments and compute teacher embeddings from the mixed mel.
                        try:
                            tmelA = batch.get('teacher_mel_segments', [None] * bsz)[i]
                            tmelB = batch.get('teacher_mel_segments', [None] * bsz)[j]
                            if tmelA is not None and tmelB is not None:
                                min_tseg = min(len(tmelA), len(tmelB))
                                mixed_tmel = (lam * tmelA[:min_tseg] + (1.0 - lam) * tmelB[:min_tseg]).astype(np.float32)
                                # Teacher embeddings from mixed teacher mel
                                teacher_emb, teacher_seg_embs = dataset.clap_embedder.compute_embeddings_from_mel(mixed_tmel)
                            else:
                                # Fallback: teacher mel not available, use student mel
                                mixed_np = mixed_seg_t.detach().cpu().numpy()
                                if mixed_np.ndim == 3:
                                    mixed_np = mixed_np[:, np.newaxis, :, :]
                                teacher_emb, teacher_seg_embs = dataset.clap_embedder.compute_embeddings_from_mel(mixed_np)
                            if teacher_emb is None:
                                raise RuntimeError("CLAP failed to compute embeddings for mixed mel")
                            mixed_teacher.append(torch.from_numpy(teacher_emb).float().to(device))
                            if teacher_seg_embs is not None:
                                mixed_teacher_segment_embs.append([torch.from_numpy(x).float().to(device) for x in teacher_seg_embs])
                            else:
                                mixed_teacher_segment_embs.append(None)
                        except Exception as e:
                            logger.error(f"[MIXUP][CACHE-OFF] Failed to compute teacher embeddings for mixed sample: {e}")
                            mixed_teacher.append(torch.zeros((config['model']['embedding_dim'],), dtype=torch.float32, device=device))
                            mixed_teacher_segment_embs.append(None)

                    mixed_song_ids.append(f"{batch['song_ids'][i]}+{batch['song_ids'][j]}")

                batch['audio_segments'] = mixed_audio
                batch['teacher_embeddings'] = mixed_teacher
                batch['teacher_segment_embeddings'] = mixed_teacher_segment_embs
                batch['song_ids'] = mixed_song_ids
                logger.info(f"[MIXUP] Applied Mixup: alpha={mixup_alpha}, lam={lam:.4f}")

        # 🧠 REAL TRAINING STEP
        # Include STAGE and current LR in batch log so resume/stage behavior is visible in a single line
        try:
            curr_lr = trainer.optimizer.param_groups[0]['lr']
            lr_str = f"{curr_lr:.1e}"
        except Exception:
            lr_str = "N/A"
        # Get current logit_scale (temperature) if using learnable scale
        temp_str = ""
        if getattr(trainer, 'use_logit_scale', False) and hasattr(trainer.model, 'logit_scale'):
            import math
            logit_val = trainer.model.logit_scale.detach().cpu().item()
            temp_multiplier = math.exp(logit_val)
            temp_str = f", T={temp_multiplier:.2f}"
        # Log gating alpha for fusion model (per-dim: show mean/min/max)
        if hasattr(trainer.model, 'alpha'):
            gate = torch.sigmoid(trainer.model.alpha.detach())
            if gate.dim() == 0:
                temp_str += f", gate={gate.item():.4f}"
            else:
                temp_str += f", gate_mean={gate.mean().item():.4f}/min={gate.min().item():.4f}/max={gate.max().item():.4f}"
        stage_num = config.get('current_stage', 1)
        logger.info(f"🔥 BATCH {num_batches + 1}/{total_batches} (EPOCH {epoch}/{config['training']['epochs']}) [STAGE {stage_num}, LR {lr_str}{temp_str}]: Training on {len(batch_data)} songs...")

        try:
            # --- Linear LR warmup for epoch 1 (if enabled) ---
            if epoch == 1 and config['training'].get('warmup_enabled', True):
                warmup_lr = (batch_idx + 1) / total_batches * config['training']['learning_rate']
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"[LR WARMUP] Epoch 1, Batch {batch_idx+1}/{total_batches}: LR set to {warmup_lr:.6f}")

            if use_global_mixup:
                # --- GLOBAL SEGMENT-LEVEL MIXUP ---
                # Flatten all segments from all songs into a single tensor, then mix.
                # Memory-efficient: concatenate directly from batch instead of
                # building an intermediate per-segment list (avoids an extra copy).
                use_teacher_emb_cache = config['training'].get('use_teacher_embedding_cache', True)

                # Build flat tensors directly from the batch (no intermediate list)
                student_parts = []
                teacher_emb_parts = []   # only used when cache ON
                teacher_mel_parts = []   # only used when cache OFF

                for i in range(len(batch['audio_segments'])):
                    mel_segs = batch['audio_segments'][i]
                    if isinstance(mel_segs, np.ndarray):
                        mel_segs = torch.from_numpy(mel_segs).float()
                    if isinstance(mel_segs, torch.Tensor):
                        mel_segs = mel_segs.cpu()
                    student_parts.append(mel_segs)  # already (num_segs, 1, 128, T)

                    if not use_teacher_emb_cache:
                        tmel_list = batch.get('teacher_mel_segments', [])
                        if i < len(tmel_list) and tmel_list[i] is not None:
                            teacher_mel_parts.append(tmel_list[i])  # numpy (num_segs, 1, 64, T)
                    else:
                        teacher_seg_embs = batch['teacher_segment_embeddings'][i]
                        teacher_song_emb = batch['teacher_embeddings'][i]
                        n_segs = mel_segs.shape[0]
                        for s in range(n_segs):
                            if teacher_seg_embs is not None and s < len(teacher_seg_embs):
                                t_emb = teacher_seg_embs[s]
                            else:
                                t_emb = teacher_song_emb
                            if isinstance(t_emb, np.ndarray):
                                t_emb = torch.from_numpy(t_emb).float()
                            if isinstance(t_emb, torch.Tensor):
                                t_emb = t_emb.cpu()
                            teacher_emb_parts.append(t_emb)

                # Concatenate student segments into one tensor (no extra copy vs stacking a flat list)
                mel_stack = torch.cat(student_parts, dim=0)  # (total_segments, 1, 128, T)
                del student_parts
                # Free batch student data immediately — mel_stack is the only reference now
                del batch['audio_segments']

                total_segments = mel_stack.shape[0]

                if total_segments >= 2:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = np.random.permutation(total_segments)
                    if np.any(perm == np.arange(total_segments)):
                        perm = np.roll(perm, 1)
                    perm_t = torch.from_numpy(perm).long()

                    # Mix student mel in-place style: create mixed, then free original
                    mixed_mel = lam * mel_stack + (1.0 - lam) * mel_stack[perm_t]
                    del mel_stack

                    if use_teacher_emb_cache:
                        # --- Cache ON: mix teacher embeddings directly ---
                        teacher_stack = torch.stack(teacher_emb_parts, dim=0)
                        del teacher_emb_parts
                        mixed_teacher = lam * teacher_stack + (1.0 - lam) * teacher_stack[perm_t]
                        del teacher_stack
                    else:
                        # --- Cache OFF: mix teacher mel, compute teacher embeddings ---
                        del teacher_emb_parts
                        # Free batch teacher mel data — we'll use our own concat
                        if 'teacher_mel_segments' in batch:
                            del batch['teacher_mel_segments']

                        if len(teacher_mel_parts) > 0:
                            tmel_stack = np.concatenate(teacher_mel_parts, axis=0)  # (total, 1, 64, T)
                            del teacher_mel_parts
                            # Safety: truncate to match student segment count
                            # (torchaudio vs librosa resamplers may differ by ±1 segment)
                            if tmel_stack.shape[0] != total_segments:
                                min_seg = min(tmel_stack.shape[0], total_segments)
                                logger.warning(
                                    f"[GLOBAL MIXUP] Segment count mismatch: "
                                    f"student={total_segments}, teacher={tmel_stack.shape[0]}, "
                                    f"truncating to {min_seg}"
                                )
                                tmel_stack = tmel_stack[:min_seg]
                                mixed_mel = mixed_mel[:min_seg]
                                perm = perm[perm < min_seg]  # keep only valid indices
                                # re-do permutation if truncation broke it
                                if len(perm) != min_seg:
                                    perm = np.random.permutation(min_seg)
                                    if np.any(perm == np.arange(min_seg)):
                                        perm = np.roll(perm, 1)
                                total_segments = min_seg
                            mixed_tmel = (lam * tmel_stack + (1.0 - lam) * tmel_stack[perm]).astype(np.float32)
                            del tmel_stack
                            if mixed_tmel.ndim == 3:
                                mixed_tmel = mixed_tmel[:, np.newaxis, :, :]
                            avg_emb, seg_embs = dataset.clap_embedder.compute_embeddings_from_mel(mixed_tmel)
                            del mixed_tmel
                            mixed_teacher = torch.stack([torch.from_numpy(e).float() for e in seg_embs], dim=0)
                        else:
                            # Fallback: teacher mel not available, use student mel for teacher
                            logger.warning("[GLOBAL MIXUP] teacher_mel_segments not available, falling back to student-mel-based teacher")
                            mixed_np = mixed_mel.numpy()
                            if mixed_np.ndim == 3:
                                mixed_np = mixed_np[:, np.newaxis, :, :]
                            avg_emb, seg_embs = dataset.clap_embedder.compute_embeddings_from_mel(mixed_np)
                            mixed_teacher = torch.stack([torch.from_numpy(e).float() for e in seg_embs], dim=0)
                    del perm_t

                    logger.info(f"[GLOBAL MIXUP] Applied: alpha={mixup_alpha}, lam={lam:.4f}, "
                               f"total_segments={total_segments} (from {len(batch_data)} songs)")

                    should_log_details = (batch_idx % log_every == 0)
                    step_metrics = trainer.train_step_global_mixup(mixed_mel, mixed_teacher, compute_diagnostics=should_log_details)
                else:
                    logger.warning("[GLOBAL MIXUP] Skipped: fewer than 2 total segments")
                    should_log_details = (batch_idx % log_every == 0)
                    step_metrics = trainer.train_step(batch, compute_diagnostics=should_log_details)
            else:
                # Forward pass, loss computation, and backward pass
                should_log_details = (batch_idx % log_every == 0)
                step_metrics = trainer.train_step(batch, compute_diagnostics=should_log_details)
            
            # Log detailed metrics
            accumulation_info = f" [acc {step_metrics['accumulation_step']}/{trainer.gradient_accumulation_steps}]"
            update_info = " 🔄 WEIGHTS UPDATED!" if step_metrics['will_update'] else ""
            num_training_samples = step_metrics.get('num_training_samples', len(batch_data))
            logger.info(f"   ✅ Forward pass through student CNN + Transformer{accumulation_info}{update_info}")
            logger.info(f"   📈 Training samples: {num_training_samples} (from {len(batch_data)} songs)")
            logger.info(f"   📊 Loss: {step_metrics['total_loss']:.6f}")
            logger.info(f"      └─ MSE Loss: {step_metrics['mse_loss']:.6f}")
            sem_loss = step_metrics.get('semantic_loss', 0.0) or 0.0
            if sem_loss > 0:
                logger.info(f"      └─ Semantic Loss: {sem_loss:.6f}")
            logger.info(f"      └─ Cosine Loss: {step_metrics['cosine_loss']:.6f}")
            if step_metrics.get('kl_loss', 0.0) > 0:
                logger.info(f"      └─ KL Loss (raw): {step_metrics['kl_loss']:.6f}")
            # Per-query semantic alignment diagnostics
            sem_details = step_metrics.get('semantic_details')
            if sem_details:
                logger.info(f"   🧠 SEMANTIC ALIGNMENT (Top Discrepancies):")
                for rank, d in enumerate(sem_details['top_discrepancies'], 1):
                    name = d['name'].title()
                    sign = '+' if d['diff'] >= 0 else ''
                    warn = ' ⚠️' if abs(d['diff']) > 0.15 else ''
                    logger.info(f"     {rank}. [{name}]  Diff: {sign}{d['diff']:.2f} (T: {d['teacher']:.2f}, S: {d['student']:.2f}){warn}")
                logger.info(f"     ✨ Avg Query Alignment: {sem_details['avg_query_alignment']:.2f} (Teacher-Student Sim)")
            logger.info(f"   🎯 Cosine Similarity: {step_metrics['mean_cosine_sim']:.4f} (min: {step_metrics['min_cosine_sim']:.4f}, max: {step_metrics['max_cosine_sim']:.4f})")

            # Accumulate metrics
            total_loss += step_metrics['total_loss']
            total_mse += step_metrics['mse_loss']
            total_kl += step_metrics.get('kl_loss', 0.0)
            total_semantic += sem_loss
            total_cosine_sim += step_metrics['mean_cosine_sim']
            num_batches += 1
            num_songs += len(batch_data)
            
            batch_time = time.time() - batch_start_time
            
            # 🧹 AGGRESSIVE MEMORY CLEANUP (prevent 15GB buildup)
            import gc
            
            # Clear only the large tensors, keep variables we need
            if 'batch' in locals():
                if 'audio_segments' in batch:
                    del batch['audio_segments']  # This is the heavy data
                if 'raw_audio_segments' in batch:
                    del batch['raw_audio_segments']
                if 'teacher_mel_segments' in batch:
                    del batch['teacher_mel_segments']  # Teacher mel segments used for mixup
                del batch
            
            # Step CosineAnnealingLR per-batch for smooth decay (both Stage 1 and Stage 2),
            # but only when the optimizer actually updated parameters (accounts for gradient accumulation).
            try:
                if hasattr(trainer, 'scheduler'):
                    sched = trainer.scheduler
                    if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                        if 'step_metrics' in locals() and step_metrics.get('will_update', False):
                            sched.step()
                            new_lr = trainer.optimizer.param_groups[0]['lr']
                            last_epoch = getattr(sched, 'last_epoch', 'N/A')
                            logger.info(f"🔁 CosineAnnealingLR step → LR={new_lr:.6e} (last_epoch={last_epoch})")
                        else:
                            logger.info("CosineAnnealingLR not stepped this batch (optimizer update not performed due to accumulation)")
                    else:
                        logger.info(f"Scheduler is {sched.__class__.__name__}; per-batch Cosine step skipped")
            except Exception as e:
                logger.warning(f"⚠️ Failed to step per-batch scheduler: {e}")

            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache (MPS and CPU)
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up temporary metrics object
            if 'step_metrics' in locals():
                del step_metrics

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
    avg_kl = total_kl / num_batches if num_batches > 0 else 0.0
    avg_semantic = total_semantic / num_batches if num_batches > 0 else 0.0
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
    if avg_kl > 0:
        logger.info(f"   📊 Average KL (raw): {avg_kl:.6f}")
    if avg_semantic > 0:
        logger.info(f"   📝 Average Semantic Loss: {avg_semantic:.6f}")
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
        'avg_kl': avg_kl,
        'avg_semantic': avg_semantic,
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
    val_song_student_embs = []  # Song-level means for semantic error
    val_song_teacher_embs = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataset.iterate_batches_streaming(config['training']['batch_size'], shuffle=False),
                              desc=f"Validation (Epoch {epoch})"):
            
            # Prepare batch
            batch = {
                'audio_segments': [],
                'teacher_embeddings': [],
                'teacher_segment_embeddings': [],
                'song_ids': []
            }
            
            for item in batch_data:
                audio_segments = item['audio_segments']
                if not isinstance(audio_segments, torch.Tensor):
                    audio_segments = torch.from_numpy(audio_segments)
                audio_segments = audio_segments.to(device=trainer.device)
                batch['audio_segments'].append(audio_segments)
                batch['teacher_embeddings'].append(item['teacher_embedding'])
                batch['teacher_segment_embeddings'].append(item.get('teacher_segment_embeddings'))
                batch['song_ids'].append(item['item_id'])
            
            # Forward pass without gradients
            student_embeddings = []
            teacher_embeddings_batch = []  # Will include per-segment and averaged teachers
            
            for i, audio_segments in enumerate(batch['audio_segments']):
                # audio_segments are PRE-COMPUTED mel spectrograms! (num_segments, 1, 128, time)
                if not isinstance(audio_segments, torch.Tensor):
                    audio_segments = torch.from_numpy(audio_segments)
                audio_segments = audio_segments.to(device=trainer.device)

                # ⚠️ SKIP SONGS WITH ONLY 1 SEGMENT - consistent with training behavior
                if audio_segments.shape[0] < 2:
                    logger.warning(f"⚠️ Skipping song {batch['song_ids'][i]} in validation - only {audio_segments.shape[0]} segment")
                    continue

                # Process segments in chunks to reduce memory usage
                chunk_size = config['model'].get('segment_batch_size', 10)
                segment_embeddings_list = []
                use_amp = getattr(trainer, 'use_amp', False)
                amp_device_type = getattr(trainer, 'amp_device_type', None) or ( 'cuda' if torch.cuda.is_available() else 'cpu' )
                for chunk_start in range(0, audio_segments.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, audio_segments.shape[0])
                    chunk = audio_segments[chunk_start:chunk_end]
                    with torch.amp.autocast(device_type=amp_device_type, dtype=torch.bfloat16, enabled=use_amp):
                        chunk_embeddings = trainer.model(chunk)  # (chunk_size, 512)
                    chunk_embeddings = chunk_embeddings.float()
                    segment_embeddings_list.append(chunk_embeddings)
                
                segment_embeddings = torch.cat(segment_embeddings_list, dim=0)  # (num_segments, 512)
                segment_embeddings = torch.nn.functional.normalize(segment_embeddings, p=2, dim=1)
                
                # Convert to numpy once
                segment_embeddings_np = segment_embeddings.cpu().numpy()
                num_segs = segment_embeddings_np.shape[0]

                # Get teacher segment embeddings if available
                teacher_seg_embs = batch['teacher_segment_embeddings'][i]
                teacher_song_emb = batch['teacher_embeddings'][i]

                # For each student segment, compare to corresponding teacher segment embedding if available,
                # otherwise compare to the song-level teacher embedding (mirrors training behavior)
                if teacher_seg_embs is not None and len(teacher_seg_embs) >= 1:
                    # Ensure teacher segment embeddings are numpy arrays
                    tseg = [ (e.numpy() if hasattr(e, 'numpy') else e) for e in teacher_seg_embs ]
                    for s_idx in range(num_segs):
                        student_embeddings.append(segment_embeddings_np[s_idx])
                        if s_idx < len(tseg):
                            teacher_embeddings_batch.append(tseg[s_idx])
                        else:
                            teacher_embeddings_batch.append(teacher_song_emb)
                else:
                    for s_idx in range(num_segs):
                        student_embeddings.append(segment_embeddings_np[s_idx])
                        teacher_embeddings_batch.append(teacher_song_emb)

                # Also include the averaged embedding for this song (same as training)
                avg_embedding = np.mean(segment_embeddings_np, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                student_embeddings.append(avg_embedding)
                teacher_embeddings_batch.append(teacher_song_emb)

                # Collect song-level embeddings for semantic error
                val_song_student_embs.append(avg_embedding)
                teacher_song_np = teacher_song_emb.numpy() if hasattr(teacher_song_emb, 'numpy') else teacher_song_emb
                val_song_teacher_embs.append(teacher_song_np)

                logger.info(f"   🔎 Validation: song {batch['song_ids'][i]} -> segments={num_segs}, pairs_added={num_segs + 1}")
            
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

    # Additional per-song diagnostics for better visibility
    try:
        # Compute per-song cosine similarities (normalized dot product)
        s_norm = student_all / (np.linalg.norm(student_all, axis=1, keepdims=True) + 1e-8)
        t_norm = teacher_all / (np.linalg.norm(teacher_all, axis=1, keepdims=True) + 1e-8)
        per_song_cosines = np.sum(s_norm * t_norm, axis=1)
        # Summary stats
        p10, p50, p90 = np.percentile(per_song_cosines, [10, 50, 90])
        logger.info(f"🔬 Validation per-song cosine (n={len(per_song_cosines)}): mean={metrics['cosine_similarity']['mean']:.4f}, p10={p10:.4f}, median={p50:.4f}, p90={p90:.4f}")
        # Warn if validation set is very small — results will be noisy and perhaps misleading
        if len(per_song_cosines) < 10:
            logger.warning("⚠️ Validation set is small (<10 songs). Validation metrics will be noisy and may not reflect generalization.")
        # Attach per-song values to metrics for optional post-analysis
        metrics['per_song_cosines'] = per_song_cosines.tolist()
    except Exception as e:
        logger.warning(f"⚠️ Could not compute detailed validation diagnostics: {e}")

    # Compute val_kl (raw KL divergence on embedding distributions) using trainer's temperature
    try:
        import math
        if getattr(trainer, 'use_logit_scale', False) and hasattr(trainer.model, 'logit_scale'):
            T_val = trainer.model.logit_scale.exp().item()
        else:
            T_val = float(getattr(trainer, 'loss_temperature', 1.0))
        s_t = torch.from_numpy(s_norm).float()
        t_t = torch.from_numpy(t_norm).float()
        p_s = torch.nn.functional.log_softmax(s_t / T_val, dim=-1)
        p_t = torch.nn.functional.softmax(t_t / T_val, dim=-1)
        val_kl = torch.nn.functional.kl_div(p_s, p_t, reduction='batchmean').item()
        metrics['val_kl'] = val_kl
        logger.info(f"🔬 val_kl (raw, T={T_val:.2f}): {val_kl:.6f}")
    except Exception as e:
        logger.warning(f"⚠️ Could not compute val_kl: {e}")

    # Compute val_semantic_error (KLD) if text anchors are available
    if trainer.text_anchors is not None and len(val_song_student_embs) > 0:
        try:
            with torch.no_grad():
                s = torch.from_numpy(np.vstack(val_song_student_embs)).float().to(trainer.device)
                t = torch.from_numpy(np.vstack(val_song_teacher_embs)).float().to(trainer.device)
                s = torch.nn.functional.normalize(s, p=2, dim=1)
                t = torch.nn.functional.normalize(t, p=2, dim=1)
                s_sim = torch.mm(s, trainer.text_anchors_t)
                t_sim = torch.mm(t, trainer.text_anchors_t)
                tau = trainer.semantic_temperature
                s_log_prob = torch.nn.functional.log_softmax(s_sim / tau, dim=-1)
                t_prob = torch.nn.functional.softmax(t_sim / tau, dim=-1)
                val_semantic_error = torch.nn.functional.kl_div(s_log_prob, t_prob, reduction='batchmean').item()
            metrics['val_semantic_error'] = val_semantic_error
            logger.info(f"🔬 val_semantic_error (KLD): {val_semantic_error:.6f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not compute val_semantic_error: {e}")

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
        # Prefer CUDA -> MPS (macOS) -> CPU for student text model
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
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

    # Log loss settings (temperature / logit_scale / focal weighting)
    try:
        logit = None
        if getattr(trainer, 'use_logit_scale', False) and hasattr(trainer.model, 'logit_scale'):
            logit = float(trainer.model.logit_scale.detach().cpu().item())
        logger.info(f"🎚️ Loss settings: temperature={trainer.loss_temperature}, use_logit_scale={trainer.use_logit_scale}, init_logit_scale={logit}, focal_gamma={trainer.focal_gamma}, focal_low={trainer.focal_low}, focal_high={trainer.focal_high}")
    except Exception:
        logger.info("🎚️ Loss settings: (unavailable)")

    # Print model info
    model_info = trainer.model.count_parameters()
    logger.info(f"\n📊 Model Architecture:")
    logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"   Estimated size: {model_info['estimated_size_mb']:.1f} MB")
    logger.info(f"   Device: {trainer.device}")
    
    # --- Semantic alignment anchors (query-based distillation) ---
    # Compute text anchors ONCE at startup, then free the text model to save memory
    import json
    import gc
    query_json_path = Path(os.path.dirname(config_path)) / "query.json"
    semantic_queries = None
    if query_json_path.exists() and trainer.lambda_semantic > 0:
        with open(query_json_path) as f:
            semantic_queries = json.load(f)["semantic_anchors"]
        logger.info(f"📝 Loaded {len(semantic_queries)} semantic queries from {query_json_path}")
        # Load text model temporarily just to compute anchors
        text_teacher_path = Path(config['paths']['teacher_model_text'])
        if text_enabled:
            _tok = tokenizer
            _emb = teacher_text_embedder
        else:
            _tok = get_tokenizer()
            _emb = CLAPTextEmbedder(str(text_teacher_path))
        enc = _tok(semantic_queries, padding=True, truncation=True, max_length=77, return_tensors='pt')
        with torch.no_grad():
            text_anchors_np = _emb.encode(enc['input_ids'].numpy(), enc['attention_mask'].numpy())
        text_anchors = torch.from_numpy(text_anchors_np).float()
        text_anchors = torch.nn.functional.normalize(text_anchors, p=2, dim=1)
        trainer.set_text_anchors(text_anchors, query_names=semantic_queries)
        logger.info(f"📝 Text anchors computed: {text_anchors.shape} — freeing text model from memory")
        # Free text model + tokenizer immediately (only needed if we created them)
        if not text_enabled:
            del _tok, _emb
            gc.collect()
    else:
        logger.info("📝 Semantic alignment loss disabled (lambda_semantic=0 or query.json not found)")

    # Initialize start_epoch early (before datasets need it)
    start_epoch = 1
    loss_fn = config['training'].get('loss_function', 'mse')
    val_metric_name = 'val_cosine' if loss_fn in ('cosine', 'kl') else 'val_mse'
    patience_counter = 0
    last_val_metric = None  # Store last validation metric (None if not run yet)
    
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
                    logger.info("✓ Optimizer state restored from checkpoint (LR/WD preserved from checkpoint)")
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
                    trainer._scheduler_restored = True
                    logger.info("✓ Scheduler restored from checkpoint")
                except Exception as e:
                    logger.warning(f"⚠️ Could not restore scheduler state: {e}")
                    lr_cfg = config['training'].get('lr_scheduler', {})
                    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        trainer.optimizer,
                        mode=lr_cfg.get('mode', 'max'),
                        factor=lr_cfg.get('factor', 0.1),
                        patience=lr_cfg.get('patience', 10),
                        threshold=lr_cfg.get('threshold', 0.005),
                        threshold_mode=lr_cfg.get('threshold_mode', 'rel'),
                        min_lr=lr_cfg.get('min_lr', 1e-6)
                    )
                    logger.info(f"✓ Created new scheduler from config (patience={lr_cfg.get('patience', 10)}) due to restore failure")
            else:
                lr_cfg = config['training'].get('lr_scheduler', {})
                trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    trainer.optimizer,
                    mode=lr_cfg.get('mode', 'max'),
                    factor=lr_cfg.get('factor', 0.1),
                    patience=lr_cfg.get('patience', 10),
                    threshold=lr_cfg.get('threshold', 0.005),
                    threshold_mode=lr_cfg.get('threshold_mode', 'rel'),
                    min_lr=lr_cfg.get('min_lr', 1e-6)
                )
                logger.info(f"No scheduler state in checkpoint — created fresh scheduler from config (patience={lr_cfg.get('patience', 10)})")

            start_epoch = checkpoint.get('epoch', 0) + 1
            # Restore best metric from checkpoint (backwards compatible with old best_val_mse key)
            if 'best_val_metric' in checkpoint:
                best_val_metric = checkpoint['best_val_metric']
            elif loss_fn == 'mse':
                best_val_metric = checkpoint.get('best_val_mse', float('inf'))
            else:
                best_val_metric = float('-inf')  # No prior cosine metric, start fresh
            patience_counter = checkpoint.get('patience_counter', 0)
            logger.info(f"✅ Successfully resumed audio from epoch {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"   📈 Best validation metric so far: {best_val_metric:.6f} (loss_function={loss_fn})")
            logger.info(f"   ⏰ Patience counter: {patience_counter}/{config['training'].get('lr_scheduler', {}).get('patience', 10)}")
            logger.info(f"   🎯 Will continue from epoch {start_epoch}")
            # Confirm optimizer param groups (LR & weight_decay) after resume
            try:
                pg = trainer.optimizer.param_groups[0]
                logger.info(f"   🔧 Optimizer after resume: lr={pg.get('lr')} | weight_decay={pg.get('weight_decay')}")
            except Exception:
                pass

            # If resuming inside Stage 2 (i.e., start_epoch > stage1), enforce Stage 2 state using the
            # idempotent helper below which reads all params from config. This ensures we always follow
            # the values in config.yaml when restarting.
            try:
                # Ensure stage match config and epoch
                def ensure_stage_for_epoch(epoch_val):
                    try:
                        s1 = config['training']['epochs']
                        if epoch_val > s1:
                            logger.info("🔁 Resuming inside Stage 2 (post-stage1). Enforcing Stage 2 state from config...")
                            try:
                                trainer._freeze_encoder()
                            except Exception as e:
                                logger.warning(f"⚠️ Could not call trainer._freeze_encoder(): {e}")
                            # Build projection-only optimizer with values from config
                            try:
                                trainer.optimizer = torch.optim.Adam(
                                    filter(lambda p: p.requires_grad, trainer.model.parameters()),
                                    lr=config['training'].get('stage2_learning_rate', 0.000003),
                                    weight_decay=config['training']['weight_decay']
                                )
                                # If a scheduler was restored from checkpoint, preserve it instead of rebuilding
                                if getattr(trainer, '_scheduler_restored', False):
                                    logger.info("✓ Preserving restored scheduler from checkpoint for Stage 2 (not rebuilding from config)")
                                else:
                                    lr_cfg = config['training'].get('lr_scheduler', {})
                                    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        trainer.optimizer,
                                        mode=lr_cfg.get('mode', 'max'),
                                        factor=lr_cfg.get('factor', 0.1),
                                        patience=lr_cfg.get('patience', 10),
                                        threshold=lr_cfg.get('threshold', 0.005),
                                        threshold_mode=lr_cfg.get('threshold_mode', 'rel'),
                                        min_lr=lr_cfg.get('min_lr', 1e-6)
                                    )
                                    logger.info(f"✅ Rebuilt projection-only optimizer and scheduler for Stage 2 resume (patience={lr_cfg.get('patience', 10)})")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not rebuild projection-only optimizer: {e}")

                            # Big visible stage header (include selected scheduler name)
                            sched_name = trainer.scheduler.__class__.__name__ if hasattr(trainer, 'scheduler') else 'N/A'
                            logger.info("\n" + "="*60)
                            logger.info(f"===========================STAGE 2 - {sched_name}==========================")
                            logger.info("="*60 + "\n")
                        else:
                            # Ensure a full-model optimizer is present with config values for LR and WD
                            try:
                                logger.info("🔁 Resuming inside Stage 1 or before stage cutoff. Enforcing Stage 1 optimizer from config...")
                                trainer.optimizer = torch.optim.Adam(
                                    trainer.model.parameters(),
                                    lr=config['training']['learning_rate'],
                                    weight_decay=config['training']['weight_decay']
                                )
                                lr_cfg = config['training'].get('lr_scheduler', {})
                                trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    trainer.optimizer,
                                    mode=lr_cfg.get('mode', 'max'),
                                    factor=lr_cfg.get('factor', 0.1),
                                    patience=lr_cfg.get('patience', 10),
                                    threshold=lr_cfg.get('threshold', 0.005),
                                    threshold_mode=lr_cfg.get('threshold_mode', 'rel'),
                                    min_lr=lr_cfg.get('min_lr', 1e-6)
                                )
                                logger.info(f"✅ Rebuilt full-model optimizer and scheduler from config (patience={lr_cfg.get('patience', 10)})")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not rebuild full-model optimizer: {e}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error in ensure_stage_for_epoch: {e}")

                ensure_stage_for_epoch(start_epoch)
            except Exception as e:
                logger.warning(f"⚠️ Error enforcing stage from epoch on resume: {e}")

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
            best_val_metric = float('-inf') if loss_fn in ('cosine', 'kl') else float('inf')
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
    
    train_dataset = StudentCLAPDataset(config, split='train', epoch=start_epoch)
    val_dataset = StudentCLAPDataset(config, split='val', epoch=start_epoch)
    
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

    # If Stage 1 uses CosineAnnealingLR, (re)create scheduler with actual dataset size for per-batch stepping.
    # We check config (not isinstance) because checkpoint resume may have overwritten the scheduler.
    lr_sched_cfg = config['training'].get('lr_scheduler', {})
    if lr_sched_cfg.get('use_cosine_annealing', False):
        lr_min = float(lr_sched_cfg.get('min_lr', 1e-6))
        batch_size = int(config['training'].get('batch_size', 1))
        batches_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
        t_max = int(stage1_epochs) * batches_per_epoch
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=t_max, eta_min=lr_min
        )
        logger.info(f"📉 Stage 1 CosineAnnealingLR initialized with T_max={t_max} ({stage1_epochs} epochs × {batches_per_epoch} batches)")
    
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
    
    # For cosine/kl: higher is better → start from -inf. For MSE: lower is better → start from +inf.
    best_val_metric = float('-inf') if loss_fn in ('cosine', 'kl') else float('inf')
    training_start_time = time.time()
    
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

            # Additional diagnostics: scheduler class, T_max (if Cosine) and last_epoch
            sched_name = sched.__class__.__name__ if hasattr(sched, '__class__') else str(type(sched))
            t_max = getattr(sched, 'T_max', getattr(sched, 't_max', 'N/A'))
            last_epoch = getattr(sched, 'last_epoch', 'N/A')

            logger.info(f"📊 Scheduler @ epoch {epoch}: class={sched_name}, T_max={t_max}, last_epoch={last_epoch}, mode={sched_mode}, factor={sched_factor}, patience={sched_patience}, threshold={sched_threshold}, threshold_mode={sched_threshold_mode}, min_lr={sched_min_lr}, current_lr={curr_lr}")
        except Exception as e:
            logger.warning(f"⚠️ Could not read scheduler settings: {e}")

        # Big visible stage header for humans
        try:
            current_stage = 1 if epoch <= config['training']['epochs'] else 2
            logger.info("\n" + "="*60)
            logger.info(f"===========================STAGE {current_stage}==========================")
            logger.info("="*60 + "\n")
        except Exception:
            pass

        # Text anchors are pre-computed once at startup (no per-epoch recomputation needed)

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
        if audio_enabled and epoch == stage1_epochs + 1:
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
            s2_min = float(stage2_sched_cfg.get('min_lr', 1e-6))

            # If configured to reuse Stage 1 scheduler, create ReduceLROnPlateau here.
            if stage2_sched_cfg.get('use_stage1_scheduler', False):
                lr_cfg = config['training'].get('lr_scheduler', {})
                trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    trainer.optimizer,
                    mode=lr_cfg.get('mode', 'max'),
                    factor=lr_cfg.get('factor', 0.1),
                    patience=lr_cfg.get('patience', 10),
                    threshold=lr_cfg.get('threshold', 0.005),
                    threshold_mode=lr_cfg.get('threshold_mode', 'rel'),
                    min_lr=float(lr_cfg.get('min_lr', 1e-6))
                )
                logger.info(f"📉 Stage 2 LR Scheduler reset: ReduceLROnPlateau (reusing Stage 1 settings)")
            else:
                # Compute t_max = stage2_epochs * batches_per_epoch for per-batch stepping
                batch_size = config['training'].get('batch_size', 1)
                batches_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
                t_max = stage2_epochs * batches_per_epoch

                # Use CosineAnnealingLR for stage 2 for smooth decay over the total number of stage2 steps
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer,
                    T_max=t_max,
                    eta_min=s2_min
                )
                logger.info(f"📉 Stage 2 LR Scheduler reset: CosineAnnealingLR (T_max={t_max}, eta_min={s2_min})")
            logger.info(f"   📈 Stage 2 learning rate: {stage2_lr:.2e}")
            logger.info(f"   📊 Training {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,} parameters (projection head only)")
            logger.info("   🎯 This refines the embedding alignment while keeping learned features intact")
            sched_name = trainer.scheduler.__class__.__name__ if hasattr(trainer, 'scheduler') else 'N/A'
            logger.info("\n" + "="*60)
            logger.info(f"===========================STAGE 2 - {sched_name}==========================")

            # If stage1_lr < default_stage2_lr, advance scheduler once to nudge LR downwards
            if trigger_reduction:
                logger.info(f"   🚨 Triggering immediate scheduler step in stage 2")
                try:
                    trainer.scheduler.step()
                except Exception:
                    pass



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
                    'best_val_mse': best_val_metric if loss_fn == 'mse' else 0,
                    'best_val_metric': best_val_metric,
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
            val_mse = val_metrics['mse']
            val_cosine = val_metrics['cosine_similarity']['mean']
            # Select primary validation metric based on loss_function
            if loss_fn in ('cosine', 'kl'):
                val_metric = val_cosine
                val_metric_name = 'val_cosine'
            else:
                val_metric = val_mse
                val_metric_name = 'val_mse'
            last_val_metric = val_metric
            # Log concise validation summary (this will appear in training.log)
            val_sem = val_metrics.get('val_semantic_error', 'N/A')
            val_kl = val_metrics.get('val_kl', 0.0)
            logger.info(f"🔍 Validation completed (Epoch {epoch}) — {val_metric_name}: {val_metric:.6f} | val_mse: {val_mse:.6f} | val_cosine: {val_cosine:.4f} | val_kl: {val_kl:.6f} | val_semantic_error: {val_sem} (n={val_metrics.get('num_songs','N/A')})")
            print_evaluation_report(val_metrics, f"Validation - Epoch {epoch}")

            # Update the per-epoch checkpoint to include validation metrics
            try:
                epoch_checkpoint_data['last_val_mse'] = val_mse
                epoch_checkpoint_data['val_mse'] = val_mse
                epoch_checkpoint_data['val_cosine'] = val_cosine
                epoch_checkpoint_data['val_kl'] = val_kl
                epoch_checkpoint_data['train_kl'] = train_metrics.get('avg_kl', 0.0)
                epoch_checkpoint_data['val_metric'] = val_metric
                epoch_checkpoint_data['val_metric_name'] = val_metric_name
                if 'val_semantic_error' in val_metrics:
                    epoch_checkpoint_data['val_semantic_error'] = val_metrics['val_semantic_error']
                torch.save(epoch_checkpoint_data, epoch_checkpoint_path)
                if latest_checkpoint_path.exists() or latest_checkpoint_path.is_symlink():
                    latest_checkpoint_path.unlink()
                torch.save(epoch_checkpoint_data, latest_checkpoint_path)
                logger.info(f"✅ Updated epoch checkpoint with validation metrics: {epoch_checkpoint_path} ({val_metric_name}={val_metric:.6f})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update epoch checkpoint with validation metrics: {e}")

            # Step scheduler on validation metric
            try:
                sched = trainer.scheduler
                sched_cls = sched.__class__.__name__
                if sched_cls == 'ReduceLROnPlateau':
                    mode = getattr(sched, 'mode', None)
                    if loss_fn in ('cosine', 'kl'):
                        # Cosine/KL: higher is better
                        val_to_step = val_metric if mode == 'max' else -val_metric
                    else:
                        # MSE: lower is better
                        val_to_step = -val_metric if mode == 'max' else val_metric
                    sched.step(val_to_step)
                    logger.info(f"Scheduler (ReduceLROnPlateau) stepped using {val_metric_name}: {val_metric:.6f} (mode={mode})")
                elif sched_cls == 'CosineAnnealingLR':
                    # CosineAnnealingLR is driven per-batch during training; do NOT call
                    # step() here at validation time as it would double-count steps.
                    logger.debug("Skipping validation-time scheduler.step() for CosineAnnealingLR (per-batch stepping active)")
                else:
                    # Generic fallback: step once per epoch
                    sched.step()
                    logger.info(f"Scheduler ({sched_cls}) stepped (per-epoch)")
            except Exception as e:
                logger.warning(f"Failed to step scheduler on validation metric: {e}")

            # Check for improvement
            if loss_fn in ('cosine', 'kl'):
                is_better = val_metric > best_val_metric
            else:
                is_better = val_metric < best_val_metric

            if is_better:
                best_val_metric = val_metric
                patience_counter = 0
                logger.info(f"✓ New best {val_metric_name}: {best_val_metric:.6f}")
                # Save best checkpoint
                best_checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pth"
                best_checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'val_mse': val_mse,
                    'val_cosine': val_cosine,
                    'val_kl': val_kl,
                    'train_kl': train_metrics.get('avg_kl', 0.0),
                    'val_metric': val_metric,
                    'val_metric_name': val_metric_name,
                    'best_val_metric': best_val_metric,
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
                logger.info(f"No improvement ({patience_counter}/{config['training'].get('lr_scheduler', {}).get('patience', 10)})")
        
        if audio_enabled:
            # Save checkpoint after EVERY epoch (crash recovery)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_metrics': train_metrics,
                'best_val_mse': best_val_metric if loss_fn == 'mse' else val_metrics.get('mse', 0),
                'last_val_mse': val_metrics.get('mse', 0) if audio_enabled else 0,
                'val_mse': val_metrics.get('mse', 0) if audio_enabled else 0,
                'val_cosine': val_metrics.get('cosine_similarity', {}).get('mean', 0) if audio_enabled else 0,
                'val_kl': val_metrics.get('val_kl', 0) if audio_enabled else 0,
                'train_kl': train_metrics.get('avg_kl', 0),
                'val_metric': last_val_metric if audio_enabled else 0,
                'val_metric_name': val_metric_name if audio_enabled else 'N/A',
                'best_val_metric': best_val_metric,
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
            # Optionally export ONNX for every epoch (can be slow)
            try:
                if config['training'].get('export_onnx_every_epoch', False):
                    onnx_epoch_path = checkpoint_dir / f"model_epoch_{epoch}.onnx"
                    trainer.export_to_onnx(str(onnx_epoch_path))
                    logger.info(f"✅ Exported ONNX for epoch {epoch}: {onnx_epoch_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to export ONNX for epoch {epoch}: {e}")
            # Save additional checkpoint every 5 epochs (backup)
            if epoch % 5 == 0:
                backup_path = checkpoint_dir / f"backup_epoch_{epoch}.pth"
                torch.save(checkpoint_data, backup_path)
                logger.info(f"📦 Backup checkpoint: {backup_path}")
    
    # Training complete
    total_training_time = time.time() - training_start_time
    logger.info(f"\n🎉 Training complete!")
    logger.info(f"   ⏱️ Total training time: {total_training_time/3600:.1f} hours")
    logger.info(f"   🏆 Best validation {val_metric_name if audio_enabled else 'metric'}: {best_val_metric:.6f}")
    
    # Final model export
    final_model_path = Path(config['paths']['final_model'])
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save final PyTorch model
    final_pth_path = final_model_path.with_suffix('.pth')
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'best_val_mse': best_val_metric if loss_fn == 'mse' else 0,
        'best_val_metric': best_val_metric,
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