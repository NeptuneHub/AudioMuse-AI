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
    
    def __init__(self, channels: int, reduction: int = 4):
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
    """PhiNet inverted residual block with depthwise separable convolution and SE module."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, expansion: int = 4):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase (pointwise)
        if expansion != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        self.se = SEBlock(hidden_dim)
        
        # Projection phase (pointwise)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        out = self.expand(x)
        
        # Depthwise
        out = self.depthwise(out)
        
        # SE applied before projection
        out = self.se(out)
        
        # Projection
        out = self.project(out)
        
        # Residual connection
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
    Student CLAP audio encoder using PhiNet mobile-optimized architecture.
    
    Architecture follows PhiNet design with inverted residual blocks:
    - 5 PhiBlocks with channels [16, 24, 32, 64, 96]
    - Depthwise separable convolutions for efficiency
    - Squeeze-and-Excitation modules for channel attention
    - 2 Transformer layers for temporal modeling
    - Projection head: 384 -> 256 -> 512
    
    Designed to match the teacher CLAP's 512-dimensional embedding space.
    Model size: ~10-15 MB (2-3M parameters) - very lightweight and fast!
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Audio preprocessing parameters (must match teacher CLAP)
        self.sample_rate = config['audio']['sample_rate']  # 48000
        self.n_mels = config['audio']['n_mels']  # 128
        self.n_fft = config['audio']['n_fft']  # 1024
        self.hop_length = config['audio']['hop_length_stft']  # 480
        self.fmin = config['audio']['fmin']  # 0
        self.fmax = config['audio']['fmax']  # 14000
        
        # Model architecture parameters
        self.embedding_dim = config['model']['embedding_dim']  # Must be 512
        self.cnn_channels = config['model']['cnn_channels']  # [32, 64, 128]
        self.transformer_layers = config['model']['transformer_layers']  # 2
        self.attention_heads = config['model']['attention_heads']  # 4
        self.hidden_dim = config['model']['hidden_dim']  # 256
        
        # Build the model architecture
        self.build_model()
        
    def build_model(self):
        """Build the student model architecture using PhiNet mobile-optimized design."""
        
        # 1. PhiNet stem: Initial convolution
        # Input: (batch, 1, 128, time)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),  # -> (batch, 16, 64, time/2)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 2. PhiNet blocks: Inverted residual with depthwise separable conv + SE
        # Each block reduces spatial dimensions by 2x
        self.block1 = PhiBlock(16, 24, stride=2, expansion=4)   # -> (batch, 24, 32, time/4)
        self.block2 = PhiBlock(24, 32, stride=2, expansion=4)   # -> (batch, 32, 16, time/8)
        self.block3 = PhiBlock(32, 64, stride=2, expansion=4)   # -> (batch, 64, 8, time/16)
        self.block4 = PhiBlock(64, 96, stride=2, expansion=4)   # -> (batch, 96, 4, time/32)
        
        # Group as cnn_stem for compatibility with freezing logic
        self.cnn_stem = nn.ModuleList([self.stem, self.block1, self.block2, 
                                       self.block3, self.block4])
        
        # Calculate the flattened feature dimension after PhiNet
        # After 5 reductions: (96, 4, time/32)
        # Flattened: 96 * 4 = 384 features per time step
        self.cnn_output_dim = 96 * 4  # 384
        
        # 3. Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_output_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.transformer_layers
        )
        
        # 4. Projection head to 512-dimensional embedding
        # Maps from 384 -> 256 -> 512 to reduce dimensions gradually
        self.projection_head = nn.Sequential(
            nn.Linear(self.cnn_output_dim, self.hidden_dim),  # 384 -> 256
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.embedding_dim),  # 256 -> 512
        )
        
        logger.info(f"Built Student CLAP model with PhiNet mobile-optimized architecture:")
        logger.info(f"  PhiNet: 5 PhiBlocks [16, 24, 32, 64, 96] with inverted residuals + SE")
        logger.info(f"  Transformer layers: {self.transformer_layers}")
        logger.info(f"  Attention heads: {self.attention_heads}")
        logger.info(f"  Output embedding dim: {self.embedding_dim}")
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through student model.
        
        Args:
            mel_spec: Mel-spectrogram of shape (batch, 1, n_mels, time)
                     where n_mels=128, time varies by segment length
                     
        Returns:
            embeddings: L2-normalized embeddings of shape (batch, 512)
        """
        batch_size = mel_spec.shape[0]
        
        # 1. PhiNet feature extraction with inverted residuals
        # Input: (batch, 1, 128, time) -> Output: (batch, 96, 4, time/32)
        x = mel_spec
        for block in self.cnn_stem:
            x = block(x)
        cnn_features = x
        
        # 2. Reshape for transformer: (batch, 96, 4, time/32) -> (batch, time/32, 384)
        b, c, h, w = cnn_features.shape
        x = cnn_features.permute(0, 3, 1, 2)  # (batch, time/32, 96, 4)
        x = x.reshape(b, w, c * h)  # (batch, time/32, 384)
        
        # 3. Transformer for temporal modeling
        x = self.transformer(x)  # (batch, time/32, 384)
        
        # 4. Global average pooling over time
        x = x.mean(dim=1)  # (batch, 384)
        
        # 5. Projection to 512-dimensional embedding
        embeddings = self.projection_head(x)  # (batch, 512)
        
        # 6. L2 normalization (critical for cosine similarity)
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
        # Use torchaudio.transforms.MelSpectrogram with same params as teacher
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0  # Power spectrogram
        ).to(audio.device)
        
        # Compute mel-spectrogram
        mel_spec = mel_transform(audio)  # (batch, n_mels, time)
        
        # Convert to log-scale (like teacher CLAP)
        mel_spec = torch.log(mel_spec + 1e-7)
        
        # Add channel dimension: (batch, n_mels, time) -> (batch, 1, n_mels, time)
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
        # Ensure tensor is on the same device as model and correct dtype
        model_device = next(self.parameters()).device
        audio_segments = audio_segments.to(model_device, dtype=torch.float32)
        
        # Compute mel-spectrograms for all segments
        mel_specs = self.compute_mel_spectrogram(audio_segments)  # (num_segments, 1, 128, time)
        
        # Forward pass through model
        segment_embeddings = self.forward(mel_specs)  # (num_segments, 512)
        
        # Average embeddings across segments (same as teacher)
        averaged_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)  # (1, 512)
        
        # Re-normalize after averaging
        averaged_embedding = F.normalize(averaged_embedding, p=2, dim=1)
        
        return averaged_embedding
        
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters for size analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in MB (4 bytes per float32 parameter)
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
        # Use Apple Silicon MPS if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Initialize model
        self.model = StudentCLAPAudio(config).to(self.device).float()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Gradient accumulation setup
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.accumulation_counter = 0
        
        # Learning rate scheduler - tinyCLAP uses ReduceLROnPlateau (reduces only when stuck)
        # This is MUCH better than CosineAnnealing which aggressively decays LR over time
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',          # Minimize loss
            factor=0.1,          # Reduce LR by 10x when plateau
            patience=10,         # Wait 10 epochs before reducing
            min_lr=1e-6
        )
        logger.info(f"📉 LR Scheduler: ReduceLROnPlateau (factor=0.1, patience=10)")
        
        # Training strategy
        self.training_strategy = config['training'].get('training_strategy', 'averaged')
        
        # Two-stage training support (like tinyCLAP)
        self.projection_only = config['training'].get('projection_only', False)
        if self.projection_only:
            logger.info("🔒 STAGE 2: Freezing encoder, training projection head only")
            self._freeze_encoder()
        
        logger.info(f"Initialized Student CLAP trainer on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        logger.info(f"Training strategy: {self.training_strategy}")
        
    def _freeze_encoder(self):
        """Freeze encoder layers, keep only projection head trainable (Stage 2)."""
        # Freeze CNN stem
        for param in self.model.cnn_stem.parameters():
            param.requires_grad = False
        
        # Freeze transformer
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        
        # Keep projection head trainable
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
        # Convert teacher embeddings to torch tensors if needed
        if not isinstance(teacher_embeddings, torch.Tensor):
            teacher_embeddings = torch.from_numpy(teacher_embeddings).to(dtype=torch.float32, device=self.device)
        else:
            teacher_embeddings = teacher_embeddings.to(dtype=torch.float32, device=self.device)
        
        # Ensure teacher embeddings are L2-normalized
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
        
        # tinyCLAP loss: Negative cosine similarity (directly optimize for alignment)
        # This is better than MSE because it focuses on direction, not magnitude
        cosine_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1)
        total_loss = -cosine_sim.mean()  # Negative because we want to MAXIMIZE similarity
        
        # Keep MSE for monitoring only (not used in loss)
        with torch.no_grad():
            mse_loss = F.mse_loss(student_embeddings, teacher_embeddings)
        
        # Collect metrics
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': -total_loss.item(),  # Positive value for logging (1 - sim)
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
        self.model.float()  # Ensure model is in float32 mode
        
        # Only zero gradients at the start of accumulation cycle
        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()
        
        # Process ALL songs in batch together (like old approach - fast!)
        student_embeddings = []
        teacher_embeddings = []
        
        for i, (mel_segments, teacher_emb, teacher_segment_embs) in enumerate(zip(
            batch['audio_segments'], 
            batch['teacher_embeddings'],
            batch.get('teacher_segment_embeddings', [None] * len(batch['audio_segments']))
        )):
            # mel_segments is already computed mel spectrograms: (num_segments, 1, n_mels, time)
            # Convert to tensor if needed
            if not isinstance(mel_segments, torch.Tensor):
                mel_segments = torch.from_numpy(mel_segments).to(dtype=torch.float32, device=self.device)
            else:
                mel_segments = mel_segments.to(dtype=torch.float32, device=self.device)
            
            # ⚠️ SKIP SONGS WITH ONLY 1 SEGMENT (BatchNorm requires at least 2 samples)
            if mel_segments.shape[0] < 2:
                logger.warning(f"⚠️ Skipping song {batch['song_ids'][i]} - only {mel_segments.shape[0]} segment (BatchNorm needs ≥2)")
                continue
            
            # Training strategy determines what we train on
            if self.training_strategy == "segments":
                # Train on individual segments
                # Process ALL segments at once (single forward pass!)
                segment_embeddings = self.model.forward(mel_segments)  # (num_segments, 512)
                for seg_idx, seg_emb in enumerate(segment_embeddings):
                    student_embeddings.append(seg_emb.unsqueeze(0))  # (1, 512)
                    if teacher_segment_embs is not None and seg_idx < len(teacher_segment_embs):
                        teacher_embeddings.append(teacher_segment_embs[seg_idx])
                    else:
                        teacher_embeddings.append(teacher_emb)
                    
            elif self.training_strategy == "averaged":
                # Process segments and average embeddings
                # Process ALL segments at once (single forward pass!)
                segment_embeddings = self.model.forward(mel_segments)  # (num_segments, 512)
                avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)  # (1, 512)
                avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
                student_embeddings.append(avg_embedding)
                teacher_embeddings.append(teacher_emb)
                
            elif self.training_strategy == "both":
                # Train on individual segments AND averaged embedding
                # Process ALL segments at once (single forward pass = shared computation graph!)
                segment_embeddings = self.model.forward(mel_segments)  # (num_segments, 512)
                
                # Add individual segments to training
                for seg_idx, seg_emb in enumerate(segment_embeddings):
                    student_embeddings.append(seg_emb.unsqueeze(0))  # (1, 512)
                    if teacher_segment_embs is not None and seg_idx < len(teacher_segment_embs):
                        teacher_embeddings.append(teacher_segment_embs[seg_idx])
                    else:
                        teacher_embeddings.append(teacher_emb)
                
                # Average from the SAME segment embeddings (shares computation graph!)
                avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)  # (1, 512)
                avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
                student_embeddings.append(avg_embedding)
                teacher_embeddings.append(teacher_emb)
            
            else:
                raise ValueError(f"Unknown training_strategy: {self.training_strategy}")
        
        # Stack all embeddings from entire batch (FAST APPROACH like devel!)
        student_embeddings = torch.cat(student_embeddings, dim=0)
        teacher_embeddings = np.stack(teacher_embeddings)
        
        # Compute loss for entire batch at once
        loss, loss_dict = self.compute_loss(student_embeddings, teacher_embeddings)
        
        # Scale loss by accumulation steps
        loss = loss / self.gradient_accumulation_steps
        
        # Single backward pass for entire batch
        loss.backward()
        
        # Update accumulation counter
        self.accumulation_counter += 1
        
        # Only update weights when we've accumulated enough gradients
        will_update = False
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            
            # Optimizer step
            self.optimizer.step()
            
            # Reset accumulation counter
            self.accumulation_counter = 0
            will_update = True
        
        # Return metrics
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
        
        # Create dummy input for ONNX export
        # Input: mel-spectrogram of shape (1, 1, 128, time_frames)
        # For 10-second audio at 48kHz with hop_length=480: time_frames = 1000
        dummy_input = torch.randn(1, 1, 128, 1000, device=self.device)
        
        # Export to ONNX with opset 17 for maximum compatibility
        # Opset 17 supports all modern PyTorch operators including:
        # - scaled_dot_product_attention (requires opset >= 14)
        # - unflatten (requires opset >= 13)
        # - All transformer and attention operators
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,  # Use opset 17 for full PyTorch operator support
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['embedding'],
            dynamic_axes={
                'mel_spectrogram': {3: 'time_frames'},  # Variable time dimension
                'embedding': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"✅ Successfully exported Student CLAP to ONNX: {output_path}")
        
        # Test the ONNX model
        self._test_onnx_model(output_path, dummy_input)
        
    def _test_onnx_model(self, onnx_path: str, dummy_input: torch.Tensor):
        """Test that the ONNX model produces correct outputs."""
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = self.model(dummy_input).cpu().numpy()
        
        # Get ONNX output
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'mel_spectrogram': dummy_input.cpu().numpy()})[0]
        
        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        logger.info(f"ONNX vs PyTorch max difference: {max_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("✅ ONNX model verification passed")
        else:
            logger.warning(f"⚠️ ONNX model verification failed: max_diff={max_diff}")