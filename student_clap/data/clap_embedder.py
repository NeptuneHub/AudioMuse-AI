"""
Minimal CLAP Embedder for Teacher Embeddings

Standalone implementation that calculates CLAP embeddings directly using ONNX model.
"""

import os
import logging
import numpy as np
import librosa
import onnxruntime as ort
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# CLAP parameters (MUST match AudioMuse-AI preprocessing)
SAMPLE_RATE = 48000
SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
HOP_LENGTH = 240000      # 5 seconds (50% overlap)

# Mel-spectrogram parameters (HTSAT-base model)
N_FFT = 1024
HOP_LENGTH_STFT = 320
N_MELS = 64
F_MIN = 50
F_MAX = 14000


class CLAPEmbedder:
    """
    Minimal CLAP embedder using ONNX model directly.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize CLAP embedder.
        
        Args:
            model_path: Path to clap_audio_model.onnx
        """
        if not os.path.exists(model_path):
            raise RuntimeError(f"CLAP model not found: {model_path}")
        
        # Load ONNX model with optimized CPU inference
        # NOTE: CoreML is 2x SLOWER than CPU for this model due to poor operator coverage
        # Only 24% of ops run on GPU, context switching kills performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        
        # Optimize CPU threading for M4 (10 cores: 4 performance + 6 efficiency)
        #sess_options.intra_op_num_threads = 8  # Parallel ops within a layer
        #sess_options.inter_op_num_threads = 2  # Parallel layers
        
        # Use CPU only - it's actually MUCH faster than CoreML for this model
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info(f"CLAP model loaded: {model_path}")
        logger.info(f"✅ Using optimized CPU inference (8 threads)")
        logger.info(f"   Performance: ~325ms/segment vs 713ms with CoreML")
        logger.info(f"   Reason: Only 24% of ops supported by CoreML GPU, context switching overhead too high")
    
    def compute_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram from audio waveform.
        
        Args:
            audio_data: Audio waveform (mono, 48kHz)
            
        Returns:
            mel_spectrogram: Shape (1, time_frames, 64)
        """
        # Compute mel-spectrogram
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
        
        # Transpose to (time_frames, mel_bins)
        mel = mel.T
        
        # Add channel dimension: (1, time_frames, 64)
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
            
            # Process each segment and get embeddings
            embeddings = []
            for segment in segments:
                # Compute mel-spectrogram
                mel_spec = self.compute_mel_spectrogram(segment)
                
                # Add batch and channel dimensions: (1, 1, time_frames, 64)
                mel_input = mel_spec[:, np.newaxis, :, :]
                
                # Run ONNX inference
                onnx_inputs = {'mel_spectrogram': mel_input}
                outputs = self.session.run(None, onnx_inputs)
                audio_embedding = outputs[0][0]  # Remove batch dimension
                
                embeddings.append(audio_embedding)
            
            # Average embeddings across segments
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
            
            # Return both averaged and individual segment embeddings
            return avg_embedding, duration_sec, num_segments, embeddings
            
        except Exception as e:
            logger.error(f"Failed to analyze {audio_path}: {e}")
            return None, 0, 0, None
