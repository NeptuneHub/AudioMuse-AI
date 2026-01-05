"""
Student CLAP ONNX Model Implementation

Implements lightweight student CLAP audio encoder using ONNX with PyTorch training
and export to pure ONNX for inference. Based on tinyCLAP architecture.
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


class StudentCLAPAudio(nn.Module):
    """
    Lightweight student CLAP audio encoder based on tinyCLAP architecture.
    
    Designed to match the teacher CLAP's 512-dimensional embedding space
    while being significantly smaller (~20-40MB vs 268MB).
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
        """Build the student model architecture following tinyCLAP design."""
        
        # 1. CNN Stem for feature extraction
        self.cnn_stem = nn.Sequential(
            # Conv block 1: Input channels = n_mels (128), output = 32
            nn.Conv2d(1, self.cnn_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.cnn_channels[0]),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2: 32 -> 64
            nn.Conv2d(self.cnn_channels[0], self.cnn_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.cnn_channels[1]),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3: 64 -> 128
            nn.Conv2d(self.cnn_channels[1], self.cnn_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.cnn_channels[2]),
            nn.GELU(),
        )
        
        # Calculate the flattened feature dimension after CNN
        # For input mel-spec of shape (1, 128, T), after CNN operations:
        # After conv1: (32, 64, T/2) -> MaxPool: (32, 32, T/4)  
        # After conv2: (64, 16, T/8) -> MaxPool: (64, 8, T/16)
        # After conv3: (128, 4, T/32)
        # Flattened: 128 * 4 = 512 features per time step
        self.cnn_output_dim = self.cnn_channels[2] * 4  # 128 * 4 = 512
        
        # 2. Transformer encoder for temporal modeling
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
        
        # 3. Projection head to 512-dimensional embedding
        self.projection_head = nn.Sequential(
            nn.Linear(self.cnn_output_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.embedding_dim),  # 512 dims
        )
        
        logger.info(f"Built Student CLAP model:")
        logger.info(f"  CNN channels: {self.cnn_channels}")
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
        
        # 1. CNN feature extraction
        # Input: (batch, 1, 128, time) -> Output: (batch, 128, 4, time/32)
        cnn_features = self.cnn_stem(mel_spec)
        
        # 2. Reshape for transformer: (batch, 128, 4, time/32) -> (batch, time/32, 512)
        batch_size, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.permute(0, 3, 1, 2)  # (batch, time/32, 128, 4)
        sequence_features = cnn_features.reshape(batch_size, width, -1)  # (batch, time/32, 512)
        
        # 3. Transformer encoding
        transformer_output = self.transformer(sequence_features)  # (batch, time/32, 512)
        
        # 4. Global average pooling over time dimension
        pooled_features = torch.mean(transformer_output, dim=1)  # (batch, 512)
        
        # 5. Projection to embedding space
        embeddings = self.projection_head(pooled_features)  # (batch, 512)
        
        # 6. L2 normalization (essential for CLAP compatibility)
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
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
        
        # Loss weights
        self.mse_weight = config['training']['mse_weight']
        self.cosine_weight = config['training']['cosine_weight']
        
        logger.info(f"Initialized Student CLAP trainer on {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
        
    def compute_loss(self, 
                    student_embeddings: torch.Tensor, 
                    teacher_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute knowledge distillation loss following tinyCLAP approach.
        
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
        
        # 1. MSE Loss: Minimize Euclidean distance
        mse_loss = F.mse_loss(student_embeddings, teacher_embeddings)
        
        # 2. Cosine Similarity Loss: Ensure directional alignment
        cosine_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1)
        cosine_loss = 1.0 - cosine_sim.mean()  # 1 - mean similarity
        
        # 3. Combined loss
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
        
        # Collect metrics
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'cosine_loss': cosine_loss.item(),
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
        
        # Process each song in the batch
        student_embeddings = []
        teacher_embeddings = []
        
        for i, (mel_segments, teacher_emb) in enumerate(zip(batch['audio_segments'], batch['teacher_embeddings'])):
            # mel_segments is already computed mel spectrograms: (num_segments, 1, n_mels, time)
            # Convert to tensor if needed
            if not isinstance(mel_segments, torch.Tensor):
                mel_segments = torch.from_numpy(mel_segments).to(dtype=torch.float32, device=self.device)
            else:
                mel_segments = mel_segments.to(dtype=torch.float32, device=self.device)
            
            # Forward pass through model directly (mels already computed!)
            segment_embeddings = self.model.forward(mel_segments)  # (num_segments, 512)
            
            # Average embeddings across segments (same as teacher CLAP)
            avg_embedding = torch.mean(segment_embeddings, dim=0, keepdim=True)  # (1, 512)
            
            # Re-normalize after averaging
            avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
            
            student_embeddings.append(avg_embedding)
            teacher_embeddings.append(teacher_emb)
        
        # Stack embeddings
        student_embeddings = torch.cat(student_embeddings, dim=0)  # (batch_size, 512)
        teacher_embeddings = np.stack(teacher_embeddings)  # (batch_size, 512)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(student_embeddings, teacher_embeddings)
        
        # Scale loss by accumulation steps (important for gradient accumulation)
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update accumulation counter
        self.accumulation_counter += 1
        
        # Only update weights when we've accumulated enough gradients
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            
            # Optimizer step
            self.optimizer.step()
            
            # Reset accumulation counter
            self.accumulation_counter = 0
        
        # Scale loss_dict back to original scale for logging
        scaled_loss_dict = {k: v * self.gradient_accumulation_steps if 'loss' in k else v for k, v in loss_dict.items()}
        scaled_loss_dict['accumulation_step'] = self.accumulation_counter
        scaled_loss_dict['will_update'] = (self.accumulation_counter == 0)
        
        return scaled_loss_dict
        
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
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
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