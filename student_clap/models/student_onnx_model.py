"""
Student CLAP ONNX Model Implementation

Implements student CLAP audio encoder using PhiNet mobile-optimized architecture with PyTorch training
and export to pure ONNX for inference. PhiNet uses inverted residual blocks with depthwise separable
convolutions and squeeze-and-excitation modules for efficient mobile deployment.

Architecture:
- PhiNet: 5 PhiBlocks with inverted residuals and SE modules
- Channels: [16, 24, 32, 64, 96]
- Transformer: 2 layers for temporal modeling
- Projection: 384 -> 256 -> 512 dimensional embedding space
- Model size: ~10-15 MB (2-3M parameters)
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

logger = logging.getLogger(__name__)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 24):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PhiBlock(nn.Module):
    """
    PhiNet inverted residual block with depthwise separable convolution and SE module.

    Implements the parameterized design from PhiNet paper (Francesco Paissan et al., 2022).
    Uses expansion factor calculated from network width multipliers alpha and beta.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2,
                 expansion: int = 6, use_se: bool = True):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.use_se = use_se

        if expansion != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand = nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        if use_se:
            self.se = SEBlock(hidden_dim)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.expand(x)

        out = self.depthwise(out)

        if self.use_se:
            out = self.se(out)

        out = self.project(out)

        if self.use_residual:
            out = out + identity

        return out

class ConvBlock(nn.Module):
    """Standard convolutional block (kept for compatibility, not used in PhiNet)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class StudentCLAPAudio(nn.Module):
    """
    Student CLAP audio encoder using parameterized PhiNet architecture.

    Follows tinyCLAP's design (NO TRANSFORMER):
    - Parameterized PhiNet with N inverted residual blocks
    - 1x1 convolution to intermediate dimension (2048)
    - Global average pooling
    - MLP projection: 2048 -> 256 -> 512

    tinyCLAP's successful configuration (6.2M parameters):
    - alpha=3.00, beta=0.75, t0=4, N=7

    Architecture:
    - PhiNet: Depthwise separable convolutions + Squeeze-and-Excitation
    - NO TRANSFORMER (unlike our previous implementation)
    - Simple global pooling + MLP projection

    Designed to match the teacher CLAP's 512-dimensional embedding space.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.sample_rate = config['audio']['sample_rate']
        self.n_mels = config['audio']['n_mels']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length_stft']
        self.fmin = config['audio']['fmin']
        self.fmax = config['audio']['fmax']

        self.embedding_dim = config['model']['embedding_dim']
        self.phinet_alpha = config['model']['phinet_alpha']
        self.phinet_beta = config['model']['phinet_beta']
        self.phinet_t0 = config['model']['phinet_t0']
        self.phinet_N = config['model']['phinet_N']
        self.hidden_dim = config['model']['hidden_dim']

        self.build_model()

    def build_model(self):
        """
        Build the student model architecture using parameterized PhiNet design.

        Follows PhiNet paper (Paissan et al., 2022) with configurable width/shape multipliers:
        - alpha: Width multiplier for all layers
        - beta: Shape multiplier (controls block configuration)
        - t0: Initial resolution divisor
        - N: Number of PhiNet blocks

        tinyCLAP's successful config: alpha=3.00, beta=0.75, t0=4, N=7 (6.2M params)
        """
        alpha = self.phinet_alpha
        beta = self.phinet_beta
        t0 = self.phinet_t0
        N = self.phinet_N

        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)

        base_channels = []
        if N > 0:
            base_channels.append(24)
        if N > 1:
            base_channels.append(48)
        if N > 2:
            base_channels.append(96)
        if N > 3:
            base_channels.append(144)
        if N > 4:
            base_channels.append(192)
        if N > 5:
            base_channels.append(288)
        if N > 6:
            base_channels.append(384)

        for i in range(len(base_channels), N):
            base_channels.append(base_channels[-1] + 96)

        channels = [make_divisible(c * alpha * beta) for c in base_channels]

        stem_channel = make_divisible(16 * alpha)

        logger.info(f"Building parameterized PhiNet with:")
        logger.info(f"  alpha={alpha}, beta={beta}, t0={t0}, N={N}")
        logger.info(f"  Stem channel: {stem_channel}")
        logger.info(f"  Block channels: {channels}")

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channel),
            nn.ReLU6(inplace=True)
        )

        blocks = []
        in_ch = stem_channel

        for i in range(N):
            out_ch = channels[i]

            stride = 2 if i < N else 1
            expansion = 6

            blocks.append(PhiBlock(in_ch, out_ch, stride=stride, expansion=expansion, use_se=True))
            in_ch = out_ch

        self.cnn_stem = nn.ModuleList([self.stem] + blocks)

        self.final_channels = channels[-1] if channels else stem_channel

        self.intermediate_dim = 1024
        self.pn_block = nn.Conv2d(
            self.final_channels,
            self.intermediate_dim,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.intermediate_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Built Student CLAP model with parameterized PhiNet (NO TRANSFORMER):")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  PhiNet: {N} blocks with channels {channels}")
        logger.info(f"  Intermediate dim (pn_block): {self.intermediate_dim}")
        logger.info(f"  Output embedding dim: {self.embedding_dim}")

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through student model (tinyCLAP architecture: NO TRANSFORMER).

        Args:
            mel_spec: Mel-spectrogram of shape (batch, 1, n_mels, time)
                     where n_mels=128, time varies by segment length

        Returns:
            embeddings: L2-normalized embeddings of shape (batch, 512)
        """

        x = mel_spec
        for block in self.cnn_stem:
            x = block(x)

        x = self.pn_block(x)

        x = x.mean((-1, -2))

        embeddings = self.projection_head(x)

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
        audio_segments = audio_segments.to(model_device, dtype=torch.float32)

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

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = StudentCLAPAudio(config).to(self.device).float()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.accumulation_counter = 0

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            min_lr=1e-6
        )
        logger.info(f"📉 LR Scheduler: ReduceLROnPlateau (factor=0.1, patience=10)")

        self.training_strategy = config['training'].get('training_strategy', 'averaged')

        self.projection_only = config['training'].get('projection_only', False)
        if self.projection_only:
            logger.info("🔒 STAGE 2: Freezing encoder, training projection head only")
            self._freeze_encoder()

        logger.info(f"Initialized Student CLAP trainer on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        logger.info(f"Training strategy: {self.training_strategy}")

    def _freeze_encoder(self):
        """Freeze encoder layers, keep only projection head trainable (Stage 2)."""

        for param in self.model.cnn_stem.parameters():
            param.requires_grad = False

        for param in self.model.transformer.parameters():
            param.requires_grad = False

        for param in self.model.projection_head.parameters():
            param.requires_grad = True

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

        if not isinstance(teacher_embeddings, torch.Tensor):
            teacher_embeddings = torch.from_numpy(teacher_embeddings).to(dtype=torch.float32, device=self.device)
        else:
            teacher_embeddings = teacher_embeddings.to(dtype=torch.float32, device=self.device)

        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)

        cosine_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1)
        total_loss = -cosine_sim.mean()

        with torch.no_grad():
            mse_loss = F.mse_loss(student_embeddings, teacher_embeddings)

        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': -total_loss.item(),
            'mean_cosine_sim': cosine_sim.mean().item(),
            'min_cosine_sim': cosine_sim.min().item(),
            'max_cosine_sim': cosine_sim.max().item()
        }

        return total_loss, loss_dict

    def train_step(self, batch: Dict) -> Dict:
        """
        Single training step on a batch.

        Args:
            batch: Dictionary with:
                - 'audio_segments': List of audio segment tensors per song
                - 'teacher_embeddings': Teacher embeddings from database
                - 'song_ids': Song IDs for logging

        Returns:
            step_metrics: Dictionary with loss and performance metrics
        """
        self.model.train()
        self.model.float()

        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        student_embeddings = []
        teacher_embeddings = []

        for i, (mel_segments, teacher_emb, teacher_segment_embs) in enumerate(zip(
            batch['audio_segments'],
            batch['teacher_embeddings'],
            batch.get('teacher_segment_embeddings', [None] * len(batch['audio_segments']))
        )):

            if not isinstance(mel_segments, torch.Tensor):
                mel_segments = torch.from_numpy(mel_segments).to(dtype=torch.float32, device=self.device)
            else:
                mel_segments = mel_segments.to(dtype=torch.float32, device=self.device)

            if mel_segments.shape[0] < 2:
                logger.warning(f"⚠️ Skipping song {batch['song_ids'][i]} - only {mel_segments.shape[0]} segment (BatchNorm needs ≥2)")
                continue

            if self.training_strategy == "segments":

                segment_embeddings = self.model.forward(mel_segments)
                for seg_idx, seg_emb in enumerate(segment_embeddings):
                    student_embeddings.append(seg_emb.unsqueeze(0))
                    if teacher_segment_embs is not None and seg_idx < len(teacher_segment_embs):
                        teacher_embeddings.append(teacher_segment_embs[seg_idx])
                    else:
                        teacher_embeddings.append(teacher_emb)

            elif self.training_strategy == "averaged":

                segment_embeddings = self.model.forward(mel_segments)
                avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)
                avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
                student_embeddings.append(avg_embedding)
                teacher_embeddings.append(teacher_emb)

            elif self.training_strategy == "both":

                segment_embeddings = self.model.forward(mel_segments)

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

        student_embeddings = torch.cat(student_embeddings, dim=0)
        teacher_embeddings = np.stack(teacher_embeddings)

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
