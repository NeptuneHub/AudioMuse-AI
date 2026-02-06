"""
Student CLAP ONNX Model Implementation

Implements student CLAP audio encoder using EfficientAT MobileNet architecture with PyTorch training
and export to pure ONNX for inference. EfficientAT uses efficient CNNs trained via Transformer-to-CNN
knowledge distillation for superior audio tagging performance.

Architecture:
- EfficientAT MobileNet: Pre-trained on AudioSet
- Projection: backbone_dim -> 512 dimensional embedding space
- Model size: ~5-15 MB depending on width multiplier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Import EfficientAT MobileNet
import warnings
# Silently ignore the torchvision ConvNormActivation deprecation message here (module-level warning)
warnings.filterwarnings("ignore", message="Don't use ConvNormActivation directly")
from models.efficientat import get_model as get_efficientat_model

logger = logging.getLogger(__name__)


class Projection(nn.Module):
    """Projection head to map backbone features to embedding space."""

    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class StudentCLAPAudio(nn.Module):
    """
    Student CLAP audio encoder using EfficientAT MobileNet architecture.

    Uses EfficientAT (Transformer-to-CNN Knowledge Distillation) for efficient
    audio encoding with AudioSet pre-trained weights.

    Architecture:
    - EfficientAT MobileNet: Pre-trained on AudioSet
    - Projection head: backbone_dim -> 512 dimensional embedding space

    Available models (pretrained on AudioSet):
    - mn10_as: 4.88M params (width_mult=1.0)
    - mn05_as: ~2.5M params (width_mult=0.5)
    - mn20_as: ~15M params (width_mult=2.0)

    Designed to match the teacher CLAP's 512-dimensional embedding space.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Audio preprocessing params
        self.sample_rate = config['audio']['sample_rate']
        self.n_mels = config['audio']['n_mels']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length_stft']
        self.fmin = config['audio']['fmin']
        self.fmax = config['audio']['fmax']

        # Model params
        self.embedding_dim = config['model']['embedding_dim']
        self.dropout = config['model'].get('dropout', 0.3)
        self.pretrained_name = config['model'].get('efficientat_model', 'mn10_as')
        self.use_pretrained = config['model'].get('use_pretrained', True)
        self.use_gradient_checkpointing = config['model'].get('use_gradient_checkpointing', False)
        self.segment_batch_size = config['model'].get('segment_batch_size', 10)

        self.build_model()

    def build_model(self):
        """
        Build the student model architecture using EfficientAT MobileNet.

        EfficientAT models are pre-trained on AudioSet via Transformer-to-CNN
        knowledge distillation, providing excellent audio representations.
        """
        logger.info(f"Building EfficientAT model:")
        logger.info(f"  Model (requested): {self.pretrained_name}")
        logger.info(f"  Use pretrained: {self.use_pretrained}")
        logger.info(f"  n_mels: {self.n_mels}")
        logger.info(f"  Dropout: {self.dropout}")

        # Load EfficientAT MobileNet
        # Note: EfficientAT expects input_dim_f (frequency) and input_dim_t (time)
        # We'll compute these based on our mel spectrogram settings
        pretrained = self.pretrained_name if self.use_pretrained else None

        self.backbone = get_efficientat_model(
            num_classes=527,  # AudioSet classes (will be ignored, we use features)
            pretrained_name=pretrained,
            head_type="mlp",
            se_dims="c",  # Channel-wise squeeze-excitation
            input_dim_f=self.n_mels,
            input_dim_t=1000,  # Will be dynamically handled
        )

        # Backwards compatibility aliases used in older code/checkpoints
        # Keep `base` and `phinet` pointing to the same backbone reference
        self.base = self.backbone
        self.phinet = self.backbone

        # Expose what pretrained was actually loaded by the backbone (if any)
        loaded = getattr(self.backbone, '_loaded_pretrained', None)
        logger.info(f"  Loaded pretrained (backbone): {loaded}")

        # Determine backbone output dimension by running a dummy forward pass
        with torch.no_grad():
            # EfficientAT expects (batch, 1, n_mels, time) input
            dummy_input = torch.randn(1, 1, self.n_mels, 1000)
            _, features = self.backbone(dummy_input)
            backbone_dim = features.shape[-1]

        logger.info(f"  Backbone output dim: {backbone_dim}")

        # Projection head to map backbone features to embedding space
        self.projection_head = Projection(backbone_dim, self.embedding_dim, p=self.dropout)

        # Log model stats
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        projection_params = sum(p.numel() for p in self.projection_head.parameters())

        # Expose loaded pretrained name for clarity in logs
        loaded_pretrained = getattr(self.backbone, '_loaded_pretrained', None)
        self.loaded_pretrained = loaded_pretrained

        logger.info(f"Built Student CLAP model with EfficientAT:")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Backbone (requested={self.pretrained_name}, loaded={loaded_pretrained}): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        logger.info(f"  Projection head: {projection_params:,}")
        logger.info(f"  Output embedding dim: {self.embedding_dim}")

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through student model.

        Args:
            mel_spec: Mel-spectrogram of shape (batch, 1, n_mels, time) or (batch, n_mels, time)

        Returns:
            embeddings: L2-normalized embeddings of shape (batch, 512)
        """
        # Ensure correct input shape: (batch, 1, n_mels, time)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)  # Add channel dimension

        # EfficientAT forward returns (logits, features)
        if self.training and self.use_gradient_checkpointing:
            def backbone_forward(x):
                return self.backbone(x)
            _, audio_features = torch.utils.checkpoint.checkpoint(
                backbone_forward, mel_spec, use_reentrant=False
            )
        else:
            _, audio_features = self.backbone(mel_spec)
        embeddings = self.projection_head(audio_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram from raw audio.
        Must match the teacher CLAP's preprocessing exactly.

        Args:
            audio: Raw audio tensor of shape (batch, samples) at 48kHz

        Returns:
            mel_spec: Mel-spectrogram of shape (batch, 1, 128, time)
        """

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0
        ).to(audio.device)

        mel_spec = mel_transform(audio)

        mel_spec = torch.log(mel_spec + 1e-7)

        mel_spec = mel_spec.unsqueeze(1)

        return mel_spec

    def process_audio_segments(self, audio_segments: torch.Tensor) -> torch.Tensor:
        """
        Process multiple audio segments and return averaged embedding.
        This matches the teacher CLAP's segmentation and averaging strategy.

        Args:
            audio_segments: Tensor of shape (num_segments, samples) where each
                          segment is 10 seconds (480,000 samples) at 48kHz

        Returns:
            averaged_embedding: Single 512-dim L2-normalized embedding
        """

        model_device = next(self.parameters()).device
        audio_segments = audio_segments.to(model_device)

        mel_specs = self.compute_mel_spectrogram(audio_segments)

        segment_embeddings = self.forward(mel_specs)

        averaged_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)

        averaged_embedding = F.normalize(averaged_embedding, p=2, dim=1)

        return averaged_embedding

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters for size analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        size_mb = total_params * 4 / (1024 * 1024)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_size_mb': size_mb
        }

class StudentCLAPTrainer:
    """
    ONNX-compatible trainer for Student CLAP using PyTorch.

    Trains the student model to match teacher embeddings from database,
    then exports to ONNX for production deployment.
    """


    def __init__(self, config: Dict):
        self.config = config

        # --- Device autodetection, always use float32 ---
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = StudentCLAPAudio(config).to(self.device)

        # --- Loss scaling options (temperature or learnable logit_scale) ---
        self.use_logit_scale = bool(config['training'].get('use_logit_scale', False))
        self.loss_temperature = float(config['training'].get('loss_temperature', 1.0))
        # Focal-weighting params
        self.focal_gamma = float(config['training'].get('loss_focal_gamma', 0.0))
        self.focal_low = float(config['training'].get('loss_focal_low_threshold', 0.4))
        self.focal_high = float(config['training'].get('loss_focal_high_threshold', 0.5))

        if self.use_logit_scale:
            init_val = float(config['training'].get('init_logit_scale', 1.0))
            # Attach learnable logit_scale to model so it is saved in model_state_dict
            self.model.logit_scale = nn.Parameter(torch.tensor(float(init_val)))
            logger.info(f"🔧 Using learnable logit_scale (init={init_val})")
        else:
            logger.info(f"🔧 Using static temperature for loss: {self.loss_temperature}")

        if self.focal_gamma > 0.0:
            logger.info(f"🎯 Using focal weighting on cosine (gamma={self.focal_gamma}, low={self.focal_low}, high={self.focal_high})")
        else:
            logger.info("🎯 No focal weighting (gamma=0)")

        # Support configurable optimizer: 'adam' (default) or 'adamw'
        optimizer_type = config['training'].get('optimizer', 'adam').lower()
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            logger.info("🔧 Using AdamW optimizer")
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )

        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.accumulation_counter = 0

        # Use validation-driven scheduler (mode='max' because we maximize cosine similarity)
        lr_sched_cfg = config['training'].get('lr_scheduler', {})
        lr_mode = lr_sched_cfg.get('mode', 'max')
        lr_factor = lr_sched_cfg.get('factor', 0.1)
        lr_patience = lr_sched_cfg.get('patience', 10)
        lr_threshold = lr_sched_cfg.get('threshold', 1e-4)
        lr_threshold_mode = lr_sched_cfg.get('threshold_mode', 'rel')
        lr_min = float(lr_sched_cfg.get('min_lr', 1e-6))

        if lr_sched_cfg.get('use_cosine_annealing', False):
            # Placeholder T_max=1; will be re-initialized in train_real.py once dataset size is known
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1, eta_min=lr_min
            )
            logger.info(f"📉 LR Scheduler: CosineAnnealingLR (placeholder, will be re-initialized with actual T_max)")
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=lr_mode,
                factor=lr_factor,
                patience=lr_patience,
                threshold=lr_threshold,
                threshold_mode=lr_threshold_mode,
                min_lr=lr_min
            )
            logger.info(f"📉 LR Scheduler: ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience}, threshold={lr_threshold}, mode={lr_mode})")

        self.training_strategy = config['training'].get('training_strategy', 'averaged')
        self.segment_batch_size = config['model'].get('segment_batch_size', 10)

        self.projection_only = config['training'].get('projection_only', False)
        if self.projection_only:
            logger.info("🔒 STAGE 2: Freezing encoder, training projection head only")
            self._freeze_encoder()

        logger.info(f"Initialized Student CLAP trainer on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        logger.info(f"Training strategy: {self.training_strategy}")

    #

    def _freeze_encoder(self):
        """Freeze encoder layers, keep only projection head trainable (Stage 2).

        This implementation is robust to different student architectures: it
        disables gradients for all parameters, then enables them back for the
        projection head (and optional `logit_scale` if present). It also places
        the encoder in `eval()` so BatchNorm uses running stats collected in
        Stage 1, which stabilizes outputs when the encoder is frozen.
        """

        # First, disable gradients everywhere
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable only projection head params
        if hasattr(self.model, 'projection_head') and self.model.projection_head is not None:
            for param in self.model.projection_head.parameters():
                param.requires_grad = True
            # Ensure projection head is in training mode (we will update it)
            try:
                self.model.projection_head.train()
            except Exception:
                pass
        else:
            logger.warning("⚠️ projection_head not found on model when attempting to freeze encoder")

        # If using learnable logit_scale, allow it to be trained during stage 2
        if hasattr(self.model, 'logit_scale') and isinstance(getattr(self.model, 'logit_scale'), torch.nn.Parameter):
            self.model.logit_scale.requires_grad = True

        # Set encoder to eval() to use running BatchNorm stats collected during Stage 1
        encoder_flag_set = False
        # Prefer explicit `backbone` attribute (EfficientAT), then older names 'base' or 'phinet'
        for attr_name in ('backbone', 'base', 'phinet'):
            if hasattr(self.model, attr_name):
                try:
                    getattr(self.model, attr_name).eval()
                    encoder_flag_set = True
                    logger.info(f"🔒 Encoder ({attr_name}) set to eval() for Stage 2")
                    break
                except Exception:
                    pass

        if not encoder_flag_set:
            # Fallback: set whole model to eval but re-enable projector training
            self.model.eval()
            logger.warning("⚠️ Could not find encoder module by name; set entire model to eval() as fallback")
            if hasattr(self.model, 'projection_head'):
                try:
                    self.model.projection_head.train()
                except Exception:
                    pass

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"   📊 Trainable params: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")

    def compute_loss(self,
                    student_embeddings: torch.Tensor,
                    teacher_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute knowledge distillation loss following tinyCLAP approach.

        Uses NEGATIVE COSINE SIMILARITY as the primary loss (like tinyCLAP paper).
        This directly optimizes for embedding alignment, which is what matters for retrieval.

        Args:
            student_embeddings: Averaged embeddings from student model (batch, 512)
            teacher_embeddings: Teacher embeddings from database (batch, 512)

        Returns:
            total_loss: Combined loss for backpropagation
            loss_dict: Individual loss components for logging
        """

        # Always use default float32 for all tensors
        if not isinstance(teacher_embeddings, torch.Tensor):
            teacher_embeddings = torch.from_numpy(teacher_embeddings).to(self.device)
        else:
            teacher_embeddings = teacher_embeddings.to(self.device)
        student_embeddings = student_embeddings.to(self.device)

        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
        student_embeddings = F.normalize(student_embeddings, p=2, dim=1)

        cosine_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1)

        # Apply temperature or learnable logit_scale (scaling applied to loss logits)
        if getattr(self, 'use_logit_scale', False):
            # Clamp logit_scale to [0, ln(50)] to prevent runaway growth (like OpenAI CLIP)
            # This keeps T (temperature multiplier) in range [1, 50]
            import math
            max_logit_scale = math.log(50)  # ~3.912
            with torch.no_grad():
                self.model.logit_scale.clamp_(0, max_logit_scale)
            scale = self.model.logit_scale.exp()
            scaled = cosine_sim * scale
            # Log the *effective* multiplier (exp of the stored logit_scale parameter)
            scale_value = float(scale.detach().cpu().item())
        else:
            scaled = cosine_sim / float(self.loss_temperature)
            scale_value = float(self.loss_temperature)

        # Focal-style weighting (based on raw cosine_sim)
        if self.focal_gamma > 0.0:
            # base weight (emphasize low-cosine examples)
            weights = (1.0 - cosine_sim).clamp(min=0.0) ** float(self.focal_gamma)
            # apply triangular window: full weight for <= low, zero for >= high, linear interp between
            low = float(self.focal_low)
            high = float(self.focal_high)
            if high > low:
                interp = torch.clamp((high - cosine_sim) / (high - low), min=0.0, max=1.0)
            else:
                interp = torch.ones_like(cosine_sim)
            weights = weights * interp
            # normalize weights to keep loss scale comparable
            weights_sum = weights.sum()
            if weights_sum.item() > 0:
                weights = weights / weights_sum * weights.numel()
        else:
            weights = torch.ones_like(cosine_sim)

        # Per-sample loss and weighted average (avoid divide-by-zero)
        per_sample_loss = -scaled
        denom = weights.sum().clamp_min(1e-6)
        total_loss = (per_sample_loss * weights).sum() / denom

        with torch.no_grad():
            mse_loss = F.mse_loss(student_embeddings, teacher_embeddings)

        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': -total_loss.item(),
            'mean_cosine_sim': cosine_sim.mean().item(),
            'min_cosine_sim': cosine_sim.min().item(),
            'max_cosine_sim': cosine_sim.max().item(),
            'loss_scale': scale_value,
            'focal_gamma': float(self.focal_gamma),
            'focal_weighted_samples': int((weights > 0).sum().item())
        }

        return total_loss, loss_dict

    def train_step(self, batch: Dict) -> Dict:
        """
        Single training step on a batch.

        # Always use default float32 for training
        self.model.to(self.device)
        self.model.train()
            step_metrics: Dictionary with loss and performance metrics
        """

        self.model.to(self.device)
        self.model.train()

        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        student_embeddings = []
        teacher_embeddings = []

        for i, (mel_segments, teacher_emb, teacher_segment_embs) in enumerate(zip(
            batch['audio_segments'],
            batch['teacher_embeddings'],
            batch.get('teacher_segment_embeddings', [None] * len(batch['audio_segments']))
        )):

            # Always use default float32 for all input tensors
            if not isinstance(mel_segments, torch.Tensor):
                mel_segments = torch.from_numpy(mel_segments)
            mel_segments = mel_segments.to(self.device)

            # Move teacher_emb and teacher_segment_embs to correct device/dtype if tensor
            if isinstance(teacher_emb, np.ndarray):
                teacher_emb = torch.from_numpy(teacher_emb)
            if isinstance(teacher_emb, torch.Tensor):
                teacher_emb = teacher_emb.to(self.device)
            if teacher_segment_embs is not None:
                teacher_segment_embs = [torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in teacher_segment_embs]
                teacher_segment_embs = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in teacher_segment_embs]

            if mel_segments.shape[0] < 2:
                logger.warning(f"⚠️ Skipping song {batch['song_ids'][i]} - only {mel_segments.shape[0]} segment (BatchNorm needs ≥2)")
                continue

            if self.training_strategy == "segments":
                # Process segments in chunks to reduce memory usage
                chunk_size = self.segment_batch_size
                segment_embeddings_list = []
                for chunk_start in range(0, mel_segments.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, mel_segments.shape[0])
                    chunk = mel_segments[chunk_start:chunk_end]
                    # Ensure chunk is on correct device/dtype
                    chunk = chunk.to(self.device)
                    chunk_embeddings = self.model.forward(chunk)
                    segment_embeddings_list.append(chunk_embeddings)
                
                segment_embeddings = torch.cat(segment_embeddings_list, dim=0)
                
                for seg_idx, seg_emb in enumerate(segment_embeddings):
                    student_embeddings.append(seg_emb.unsqueeze(0))
                    if teacher_segment_embs is not None and seg_idx < len(teacher_segment_embs):
                        teacher_embeddings.append(teacher_segment_embs[seg_idx])
                    else:
                        teacher_embeddings.append(teacher_emb)

            elif self.training_strategy == "averaged":
                # Process segments in chunks to reduce memory usage
                chunk_size = self.segment_batch_size
                segment_embeddings_list = []
                for chunk_start in range(0, mel_segments.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, mel_segments.shape[0])
                    chunk = mel_segments[chunk_start:chunk_end]
                    chunk = chunk.to(self.device)
                    chunk_embeddings = self.model.forward(chunk)
                    segment_embeddings_list.append(chunk_embeddings)
                
                segment_embeddings = torch.cat(segment_embeddings_list, dim=0)
                
                avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)
                avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
                student_embeddings.append(avg_embedding)
                teacher_embeddings.append(teacher_emb)

            elif self.training_strategy == "both":
                # Process segments in chunks to reduce memory usage
                chunk_size = self.segment_batch_size
                segment_embeddings_list = []
                for chunk_start in range(0, mel_segments.shape[0], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, mel_segments.shape[0])
                    chunk = mel_segments[chunk_start:chunk_end]
                    chunk = chunk.to(self.device)
                    chunk_embeddings = self.model.forward(chunk)
                    segment_embeddings_list.append(chunk_embeddings)
                
                segment_embeddings = torch.cat(segment_embeddings_list, dim=0)

                for seg_idx, seg_emb in enumerate(segment_embeddings):
                    student_embeddings.append(seg_emb.unsqueeze(0))
                    if teacher_segment_embs is not None and seg_idx < len(teacher_segment_embs):
                        teacher_embeddings.append(teacher_segment_embs[seg_idx])
                    else:
                        teacher_embeddings.append(teacher_emb)

                avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)
                avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
                student_embeddings.append(avg_embedding)
                teacher_embeddings.append(teacher_emb)

            else:
                raise ValueError(f"Unknown training_strategy: {self.training_strategy}")




        if len(student_embeddings) == 0 or len(teacher_embeddings) == 0:
            logger.warning("⚠️ Skipping batch: no valid samples (all skipped, e.g. only 1 segment)")
            return {
                'loss': None,
                'total_loss': None,
                'mse_loss': None,
                'cosine_loss': None,
                'mean_cosine_sim': None,
                'min_cosine_sim': None,
                'max_cosine_sim': None,
                'cosine_similarity': None,
                'num_training_pairs': 0,
                'num_training_samples': 0,
                'accumulation_step': self.accumulation_counter,
                'will_update': False,
            }

        # Concatenate and ensure all embeddings are on correct device/dtype
        student_embeddings = torch.cat(student_embeddings, dim=0).to(self.device)
        teacher_embeddings = [torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in teacher_embeddings]
        teacher_embeddings = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in teacher_embeddings]
        teacher_embeddings = torch.cat([e.unsqueeze(0) if e.dim() == 1 else e for e in teacher_embeddings], dim=0).to(self.device)

        loss, loss_dict = self.compute_loss(student_embeddings, teacher_embeddings)

        loss = loss / self.gradient_accumulation_steps

        loss.backward()

        self.accumulation_counter += 1

        will_update = False
        if self.accumulation_counter >= self.gradient_accumulation_steps:

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])

            self.optimizer.step()

            self.accumulation_counter = 0
            will_update = True

        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'total_loss': loss.item() * self.gradient_accumulation_steps,
            'mse_loss': loss_dict['mse_loss'],
            'cosine_loss': loss_dict['cosine_loss'],
            'mean_cosine_sim': loss_dict['mean_cosine_sim'],
            'min_cosine_sim': loss_dict['min_cosine_sim'],
            'max_cosine_sim': loss_dict['max_cosine_sim'],
            'cosine_similarity': loss_dict['mean_cosine_sim'],
            'num_training_pairs': len(student_embeddings),
            'num_training_samples': len(student_embeddings),
            'accumulation_step': self.accumulation_counter,
            'will_update': will_update,
        }

    def train_step_global_mixup(self, mixed_mel: torch.Tensor, mixed_teacher: torch.Tensor) -> Dict:
        """
        Training step for global segment-level mixup.
        Receives pre-mixed mel spectrograms and teacher embeddings as flat tensors.

        Args:
            mixed_mel: (N_total, 1, 128, T) - mixed mel spectrograms
            mixed_teacher: (N_total, 512) - mixed teacher segment embeddings
        """
        self.model.to(self.device)
        self.model.train()

        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        total_segments = mixed_mel.shape[0]
        mixed_mel = mixed_mel.to(self.device)
        mixed_teacher = mixed_teacher.to(self.device)

        # Process all segments through the model in chunks
        chunk_size = self.segment_batch_size
        student_emb_list = []
        for chunk_start in range(0, total_segments, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_segments)
            chunk = mixed_mel[chunk_start:chunk_end]
            chunk_emb = self.model.forward(chunk)
            student_emb_list.append(chunk_emb)

        student_embeddings = torch.cat(student_emb_list, dim=0)

        loss, loss_dict = self.compute_loss(student_embeddings, mixed_teacher)

        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        self.accumulation_counter += 1

        will_update = False
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()
            self.accumulation_counter = 0
            will_update = True

        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'total_loss': loss.item() * self.gradient_accumulation_steps,
            'mse_loss': loss_dict['mse_loss'],
            'cosine_loss': loss_dict['cosine_loss'],
            'mean_cosine_sim': loss_dict['mean_cosine_sim'],
            'min_cosine_sim': loss_dict['min_cosine_sim'],
            'max_cosine_sim': loss_dict['max_cosine_sim'],
            'cosine_similarity': loss_dict['mean_cosine_sim'],
            'num_training_pairs': total_segments,
            'num_training_samples': total_segments,
            'accumulation_step': self.accumulation_counter,
            'will_update': will_update,
        }

    def export_to_onnx(self, output_path: str):
        """
        Export trained model to ONNX format for production deployment.

        Args:
            output_path: Path to save the ONNX model
        """
        self.model.eval()

        dummy_input = torch.randn(1, 1, 128, 1000, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['embedding'],
            dynamic_axes={
                'mel_spectrogram': {3: 'time_frames'},
                'embedding': {0: 'batch_size'}
            }
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        logger.info(f"✅ Successfully exported Student CLAP to ONNX: {output_path}")

        self._test_onnx_model(output_path, dummy_input)

    def _test_onnx_model(self, onnx_path: str, dummy_input: torch.Tensor):
        """Test that the ONNX model produces correct outputs."""

        with torch.no_grad():
            pytorch_output = self.model(dummy_input).cpu().numpy()

        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'mel_spectrogram': dummy_input.cpu().numpy()})[0]

        max_diff = np.abs(pytorch_output - onnx_output).max()
        logger.info(f"ONNX vs PyTorch max difference: {max_diff:.6f}")

        if max_diff < 1e-5:
            logger.info("✅ ONNX model verification passed")
        else:
            logger.warning(f"⚠️ ONNX model verification failed: max_diff={max_diff}")
