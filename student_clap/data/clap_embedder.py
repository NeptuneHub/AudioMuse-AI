"""
Minimal CLAP Embedder for Teacher Embeddings

Standalone implementation that calculates CLAP embeddings directly using ONNX model.
"""

import os
import logging
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# CLAP parameters (MUST match AudioMuse-AI preprocessing)
SAMPLE_RATE = 48000
SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
HOP_LENGTH = 240000      # 5 seconds (50% overlap)

# Mel-spectrogram parameters (HTSAT-base model)
# These MUST match laion_clap's audio_cfg exactly:
#   window_size=1024, hop_size=480, mel_bins=64, fmin=50, fmax=14000
N_FFT = 1024
HOP_LENGTH_STFT = 480
N_MELS = 64
F_MIN = 50
F_MAX = 14000


class CLAPEmbedder:
    """
    Minimal CLAP embedder that supports either ONNX (`.onnx`) or PyTorch
    split checkpoints (`clap_audio_model.pt`). When a PyTorch checkpoint is
    provided the embedder runs the teacher using `laion_clap` on CPU.

    This class also supports batched segment inference via `segment_batch_size`.
    """
    
    def __init__(self, model_path: str, segment_batch_size: int = 1, use_amp: bool = False):
        """
        Initialize CLAP embedder.

        Args:
            model_path: Path to `clap_audio_model.onnx` or `clap_audio_model.pt`
            segment_batch_size: number of segments to run in a single forward pass
            use_amp: if True, use BF16 autocast on CUDA for inference
        """
        self.use_amp = use_amp
        if not os.path.exists(model_path):
            raise RuntimeError(f"CLAP model not found: {model_path}")

        # Normalize batch size
        try:
            self.segment_batch_size = max(1, int(segment_batch_size))
        except Exception:
            self.segment_batch_size = 1

        # Backend selection by file extension
        lower = model_path.lower()
        if lower.endswith(('.pt', '.pth')):
            # Use PyTorch (laion_clap) for teacher
            try:
                import torch
                import laion_clap
            except Exception as e:
                raise RuntimeError(f"PyTorch CLAP backend requested but missing dependency: {e}")

            logger.info(f"CLAP audio (PyTorch) loaded: {model_path}")
            logger.info(f"✅ Using PyTorch backend for teacher; segment_batch_size={self.segment_batch_size}")

            # Try to load minimal split checkpoint (audio_branch + audio_projection) or full checkpoint
            state = torch.load(model_path, map_location='cpu')
            clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            clap.eval()

            # If split checkpoint saved by split_clap_pt.py
            if isinstance(state, dict) and ('audio_branch' in state or 'audio_projection' in state):
                try:
                    if 'audio_branch' in state:
                        clap.model.audio_branch.load_state_dict(state['audio_branch'], strict=False)
                    if 'audio_projection' in state:
                        clap.model.audio_projection.load_state_dict(state['audio_projection'], strict=False)
                except Exception:
                    # fallback to load_ckpt for other checkpoint formats
                    try:
                        clap.load_ckpt(model_path)
                    except Exception:
                        raise
            else:
                # attempt to load as full CLAP checkpoint
                try:
                    clap.load_ckpt(model_path)
                except Exception as e:
                    raise RuntimeError(f"Unable to load PyTorch CLAP checkpoint: {e}")

            # Create lightweight audio wrapper (same forward signature used elsewhere)
            import torch.nn as nn
            class AudioCLAPWrapper(nn.Module):
                def __init__(self, clap_model):
                    super().__init__()
                    self.audio_branch = clap_model.model.audio_branch
                    self.audio_projection = clap_model.model.audio_projection

                def forward(self, mel_spec: 'torch.Tensor'):
                    x = mel_spec.transpose(1, 3)  # (batch, 64, time, 1)
                    x = self.audio_branch.bn0(x)
                    x = x.transpose(1, 3)  # (batch, 1, time, 64)
                    x = self.audio_branch.reshape_wav2img(x)
                    audio_output = self.audio_branch.forward_features(x)
                    audio_embed = self.audio_projection(audio_output['embedding'])
                    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
                    return audio_embed

            self._backend = 'torch'
            self.audio_wrapper = AudioCLAPWrapper(clap)
            self.audio_wrapper.eval()

            # Prefer CUDA -> MPS (macOS) -> CPU for PyTorch backend
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            self._device = device
            self.audio_wrapper.to(self._device)
            logger.info(f"✅ PyTorch teacher device: {self._device}")

        else:
            # Default: ONNX backend (unchanged behavior)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.log_severity_level = 3
            available_providers = ort.get_available_providers()
            # Prefer CUDA -> Metal/CoreML -> CPU for ONNXRuntime providers
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info(f"CLAP model loaded: {model_path}")
                logger.info(f"✅ Using CUDAExecutionProvider for ONNX teacher model")
            elif 'MetalExecutionProvider' in available_providers:
                providers = ['MetalExecutionProvider', 'CPUExecutionProvider']
                logger.info(f"CLAP model loaded: {model_path}")
                logger.info(f"✅ Using MetalExecutionProvider (onnxruntime-metal) for ONNX teacher model")
            elif 'CoreMLExecutionProvider' in available_providers:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                logger.info(f"CLAP model loaded: {model_path}")
                logger.info(f"✅ Using CoreMLExecutionProvider for ONNX teacher model")
            else:
                providers = ['CPUExecutionProvider']
                logger.info(f"CLAP model loaded: {model_path}")
                logger.info(f"✅ Using CPUExecutionProvider for ONNX teacher model")

            self._backend = 'onnx'
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
    
    def compute_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram from audio waveform.
        
        Args:
            audio_data: Audio waveform (mono, 48kHz)
            
        Returns:
            mel_spectrogram: Shape (1, n_mels, time_frames) — i.e. (1, 64, T)
                             Standard (frequency, time) layout so that
                             resample_mel_frequency operates on axis-0 correctly.
        """
        # Compute mel-spectrogram — librosa returns (n_mels, time_frames)
        mel = librosa.feature.melspectrogram(
            y=audio_data,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH_STFT,
            win_length=N_FFT,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=2.0,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX
        )
        
        # Convert to log scale
        mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
        
        # Keep (n_mels, time_frames) — do NOT transpose.
        # resample_mel_frequency expects axis-0 = frequency, axis-1 = time.
        # The model expects (batch, 1, time, n_mels); the .T in
        # compute_embeddings_from_mel handles that conversion.
        
        # Add channel dimension: (1, n_mels, time_frames)
        mel = mel[np.newaxis, :, :]
        
        return mel.astype(np.float32)
    
    def analyze_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], float, int, Optional[list]]:
        """
        Analyze an audio file and return averaged CLAP embedding + individual segment embeddings.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (averaged_embedding, duration_seconds, num_segments, segment_embeddings_list)
            Returns (None, 0, 0, None) if analysis fails
        """
        try:
            # Pre-check: skip files that PySoundFile can't read (avoids slow audioread fallback)
            try:
                sf.info(audio_path)
            except Exception:
                logger.warning(f"⚠️ Skipping {audio_path} — not readable by soundfile (would trigger slow audioread fallback)")
                return None, 0, 0, None

            # Load audio at 48kHz
            audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # Quantize to int16 and back (match CLAP preprocessing)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767.0).astype(np.int16)
            audio_data = (audio_data / 32767.0).astype(np.float32)
            
            duration_sec = len(audio_data) / SAMPLE_RATE
            
            # Create overlapping segments
            segments = []
            total_length = len(audio_data)
            
            if total_length <= SEGMENT_LENGTH:
                # Pad short audio
                padded = np.pad(audio_data, (0, SEGMENT_LENGTH - total_length), mode='constant')
                segments.append(padded)
            else:
                # Create overlapping segments
                for start in range(0, total_length - SEGMENT_LENGTH + 1, HOP_LENGTH):
                    segment = audio_data[start:start + SEGMENT_LENGTH]
                    segments.append(segment)
                
                # Add final segment if needed
                last_start = len(segments) * HOP_LENGTH
                if last_start < total_length:
                    last_segment = audio_data[-SEGMENT_LENGTH:]
                    segments.append(last_segment)
            
            num_segments = len(segments)
            
            # Compute mel for all segments first and use the batched inference path
            mel_inputs = [self.compute_mel_spectrogram(seg)[0] for seg in segments]  # (n_mels, time)
            # Stack into (num_segments, 1, n_mels, time)
            mel_batch = np.stack(mel_inputs).astype(np.float32)
            mel_batch = mel_batch[:, np.newaxis, :, :]

            # Use compute_embeddings_from_mel to run batched inference (resampling + backend aware)
            avg_embedding, segment_embeddings = self.compute_embeddings_from_mel(mel_batch)

            return avg_embedding, duration_sec, num_segments, segment_embeddings
        except Exception as e:
            logger.error(f"Failed to analyze {audio_path}: {e}")
            return None, 0, 0, None

    def compute_embeddings_from_mel(self, mel_segments: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Compute CLAP embeddings from already-computed mel-spectrogram segments.

        Supports both ONNX and PyTorch backends and processes `segment_batch_size`
        segments per forward pass to reduce overhead and memory pressure.

        Args:
            mel_segments: np.ndarray of shape (num_segments, 1, n_mels, time) or
                          (num_segments, n_mels, time). Values should be log-mel dB.

        Returns:
            Tuple of (averaged_embedding, list_of_segment_embeddings)
        """
        try:
            if mel_segments is None:
                return None, None

            # Normalize input shape to (num_segments, n_mels, time)
            ms = np.array(mel_segments)
            if ms.ndim == 4 and ms.shape[1] == 1:
                ms = ms[:, 0, :, :]  # (num_segments, n_mels, time)
            elif ms.ndim == 3:
                pass
            else:
                logger.error(f"Unsupported mel_segments shape: {ms.shape}")
                return None, None

            num_segments = ms.shape[0]
            segment_embeddings = []

            # Frequency resampling helper: linear interpolation across mel axis
            def resample_mel_frequency(mel, new_n_mels=N_MELS):
                old_n, T = mel.shape
                if old_n == new_n_mels:
                    return mel  # no-op when frequency count already matches
                old_pos = np.linspace(0.0, 1.0, old_n)
                new_pos = np.linspace(0.0, 1.0, new_n_mels)
                res = np.zeros((new_n_mels, T), dtype=mel.dtype)
                for t in range(T):
                    res[:, t] = np.interp(new_pos, old_pos, mel[:, t])
                return res

            # Prepare batched inputs for inference: (num_segments, 1, time, n_mels)
            batched_mels = []
            for seg in ms:
                seg_resampled = resample_mel_frequency(seg, new_n_mels=N_MELS)
                mel_input = seg_resampled.T.astype(np.float32)  # (time, n_mels)
                batched_mels.append(mel_input)

            batched = np.stack(batched_mels, axis=0)  # (num_segments, time, n_mels)
            batched = batched[:, np.newaxis, :, :]   # (num_segments, 1, time, n_mels)

            # Run inference in chunks according to self.segment_batch_size
            for start in range(0, num_segments, self.segment_batch_size):
                end = min(start + self.segment_batch_size, num_segments)
                batch_np = batched[start:end]

                if self._backend == 'onnx':
                    onnx_inputs = {'mel_spectrogram': batch_np}
                    outputs = self.session.run(None, onnx_inputs)
                    batch_embs = outputs[0]
                    for emb in batch_embs:
                        segment_embeddings.append(emb.astype(np.float32))

                else:
                    # PyTorch backend
                    import torch
                    device = getattr(self, '_device', torch.device('cpu'))
                    batch_tensor = torch.from_numpy(batch_np).float().to(device)
                    amp_enabled = self.use_amp and device.type == 'cuda'
                    amp_device = device.type if device.type == 'cuda' else 'cpu'
                    with torch.no_grad(), torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                        out = self.audio_wrapper(batch_tensor)
                    out_np = out.float().cpu().numpy()
                    for emb in out_np:
                        segment_embeddings.append(emb.astype(np.float32))

            # Average
            avg_emb = np.mean(segment_embeddings, axis=0).astype(np.float32)
            return avg_emb, segment_embeddings
        except Exception as e:
            logger.error(f"Failed to compute embeddings from mel segments: {e}")
            return None, None

    def compute_embeddings_from_audio(self, audio_segments: list) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Compute CLAP embeddings from raw audio waveform segments using the
        correct teacher mel-spectrogram parameters (n_mels=64, n_fft=1024,
        hop=320, fmin=50).

        Use this instead of compute_embeddings_from_mel when raw audio is
        available to avoid feeding student-format mel to the teacher.

        Args:
            audio_segments: list of np.ndarray, each shape (n_samples,) at 48kHz

        Returns:
            Tuple of (averaged_embedding, list_of_segment_embeddings)
        """
        try:
            if not audio_segments:
                return None, None
            mel_inputs = [self.compute_mel_spectrogram(seg)[0] for seg in audio_segments]
            mel_batch = np.stack(mel_inputs).astype(np.float32)
            mel_batch = mel_batch[:, np.newaxis, :, :]  # (N, 1, n_mels, time)
            return self.compute_embeddings_from_mel(mel_batch)
        except Exception as e:
            logger.error(f"Failed to compute embeddings from audio segments: {e}")
            return None, None
