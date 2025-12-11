"""
CLAP Audio Analyzer for Text-Based Music Search
Uses ONNX CLAP model to generate 512-dim embeddings for audio files
and text queries for natural language music search.

ONNX Runtime provides:
- ~2-3GB less RAM usage compared to PyTorch
- Faster inference
- Identical embeddings to the original .pt model
"""

import os
import sys
import logging
import numpy as np
from typing import Tuple, Optional

# Silence transformers warning about missing PyTorch/TensorFlow/Flax
# We only use transformers for tokenizer, not for model inference
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import config

logger = logging.getLogger(__name__)

# Global ONNX session (lazy loaded)
_onnx_session = None
_tokenizer = None


def _load_onnx_model():
    """Load CLAP ONNX model with optimized settings."""
    import onnxruntime as ort
    import gc
    
    logger.info(f"Loading CLAP ONNX model from {config.CLAP_MODEL_PATH}...")
    
    # Configure ONNX Runtime session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    
    # Threading configuration based on CLAP_PYTHON_MULTITHREADS:
    # - False (default): Use ONNX internal threading (auto-detects all CPU cores)
    # - True: Disable ONNX threading (set to 1), use Python ThreadPoolExecutor instead
    if not config.CLAP_PYTHON_MULTITHREADS:
        # ONNX handles threading internally - use all available cores
        sess_options.intra_op_num_threads = 0  # 0 = use all available cores
        sess_options.inter_op_num_threads = 0  # 0 = use all available cores
        logger.info("CLAP: Using ONNX internal threading (auto-detect all cores)")
    else:
        # Python ThreadPoolExecutor will handle threading - disable ONNX threading
        sess_options.intra_op_num_threads = 1  # Single-threaded ONNX operations
        sess_options.inter_op_num_threads = 1  # Single-threaded ONNX operations
        logger.info("CLAP: Using Python threading (auto-calculated threads), ONNX single-threaded")
    
    # GPU support: Try CUDA first, fallback to CPU
    # This matches the approach used in analysis.py for MusicNN models
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(
            config.CLAP_MODEL_PATH,
            sess_options=sess_options,
            providers=providers
        )
        
        # Log which provider is actually being used
        active_providers = session.get_providers()
        logger.info(f"✓ CLAP ONNX model loaded successfully")
        logger.info(f"  Active execution provider: {active_providers[0]}")
        
        # If CUDA provider is available but fails during initialization,
        # recreate session with CPU-only to avoid runtime CUDA errors
        if active_providers[0] == 'CUDAExecutionProvider':
            try:
                # Test CUDA initialization by running a dummy inference on audio input
                # CLAP model has 3 inputs: mel_spectrogram (audio), input_ids & attention_mask (text)
                input_names = [i.name for i in session.get_inputs()]
                test_inputs = {}
                for input_name in input_names:
                    if input_name == 'mel_spectrogram':
                        # Audio input: (batch, channels, time_frames, mel_bins) = (1, 1, 1000, 64)
                        test_inputs[input_name] = np.zeros((1, 1, 1000, 64), dtype=np.float32)
                    elif input_name == 'input_ids':
                        # Text token IDs: (batch, seq_len)
                        test_inputs[input_name] = np.zeros((1, 77), dtype=np.int64)
                    elif input_name == 'attention_mask':
                        # Text attention mask: (batch, seq_len)
                        test_inputs[input_name] = np.ones((1, 77), dtype=np.int64)
                
                # Run inference to test CUDA
                _ = session.run(None, test_inputs)
                logger.info("  ✓ CUDA initialized successfully")
            except Exception as cuda_error:
                logger.warning(f"  CUDA initialization failed: {cuda_error}")
                logger.warning("  Falling back to CPU-only execution")
                # Recreate session with CPU-only
                session = ort.InferenceSession(
                    config.CLAP_MODEL_PATH,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                active_providers = session.get_providers()
                logger.info(f"  Active execution provider: {active_providers[0]}")
        
        logger.info(f"  Inputs: {[i.name for i in session.get_inputs()]}")
        logger.info(f"  Outputs: {[o.name for o in session.get_outputs()]}")
        
        gc.collect()
        return session
        
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise


def _load_tokenizer():
    """Load RoBERTa tokenizer for text processing."""
    from transformers import AutoTokenizer
    
    logger.info("Loading RoBERTa tokenizer...")
    
    # Use the same tokenizer as the CLAP model (roberta-base)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    logger.info("✓ Tokenizer loaded successfully")
    return tokenizer


def initialize_clap_model():
    """Initialize CLAP ONNX model if enabled and not already loaded."""
    global _onnx_session, _tokenizer
    
    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping model initialization.")
        return False
    
    if _onnx_session is not None:
        logger.debug("CLAP ONNX model already initialized.")
        return True
    
    if not os.path.exists(config.CLAP_MODEL_PATH):
        logger.error(f"CLAP ONNX model not found at {config.CLAP_MODEL_PATH}")
        return False
    
    try:
        _onnx_session = _load_onnx_model()
        _tokenizer = _load_tokenizer()
        logger.info("CLAP ONNX model initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CLAP ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_clap_model():
    """Get the global CLAP ONNX session, initializing if needed."""
    if _onnx_session is None:
        if not initialize_clap_model():
            raise RuntimeError("CLAP ONNX model could not be initialized")
    return _onnx_session


def get_tokenizer():
    """Get the global tokenizer, initializing if needed."""
    if _tokenizer is None:
        if not initialize_clap_model():
            raise RuntimeError("CLAP tokenizer could not be initialized")
    return _tokenizer


def compute_mel_spectrogram(audio_data: np.ndarray, sr: int = 48000) -> np.ndarray:
    """
    Compute log mel-spectrogram from audio waveform.
    This preprocessing is required for the ONNX model which expects mel-spectrogram input.
    
    Args:
        audio_data: Audio waveform (mono, 48kHz)
        sr: Sample rate (should be 48000 for CLAP)
    
    Returns:
        mel_spectrogram: Log mel-spectrogram of shape (1, time_frames, 64)
    """
    import librosa
    
    # CLAP HTSAT-base model parameters (from torchlibrosa)
    # CRITICAL: These must match exactly or embeddings will be completely different!
    n_fft = 1024
    hop_length = 320  # HTSAT uses 320, NOT 480!
    n_mels = 64
    f_min = 50  # HTSAT uses 50, NOT 0!
    f_max = 14000
    
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann',
        center=True,
        pad_mode='reflect',  # HTSAT uses 'reflect', not 'constant'
        power=2.0,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    
    # Convert to log scale
    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    
    # Transpose to (time_frames, mel_bins)
    mel = mel.T
    
    # Add channel dimension: (1, time_frames, 64)
    mel = mel[np.newaxis, :, :]
    
    return mel.astype(np.float32)


def analyze_audio_file(audio_path: str) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Analyze an audio file and return CLAP embedding using ONNX Runtime.
    Uses ThreadPoolExecutor to process batches in parallel for CPU efficiency.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (embedding_vector, duration_seconds, num_segments)
        Returns (None, 0, 0) if CLAP is disabled or analysis fails
    """
    if not config.CLAP_ENABLED:
        return None, 0, 0
    
    try:
        import librosa
        
        session = get_clap_model()
        
        # Load audio at CLAP's expected sample rate (48kHz)
        SAMPLE_RATE = 48000
        SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
        HOP_LENGTH = 240000      # 5 seconds (50% overlap)
        
        audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # CRITICAL: Quantize audio to int16 and back (matching PyTorch CLAP preprocessing)
        # This simulates the precision loss that happens in real-world audio processing
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
        
        def process_segment(audio_segment):
            """Compute mel-spectrogram and get embedding for one segment."""
            # Compute mel-spectrogram
            mel_spec = compute_mel_spectrogram(audio_segment, SAMPLE_RATE)
            
            # Add batch and channel dimensions: (1, 1, time_frames, 64)
            mel_input = mel_spec[:, np.newaxis, :, :]
            
            # Create dummy inputs for text encoder (not used in audio mode)
            dummy_input_ids = np.zeros((1, 77), dtype=np.int64)
            dummy_attention_mask = np.zeros((1, 77), dtype=np.int64)
            
            # Run ONNX inference
            onnx_inputs = {
                'mel_spectrogram': mel_input,
                'input_ids': dummy_input_ids,
                'attention_mask': dummy_attention_mask
            }
            
            outputs = session.run(None, onnx_inputs)
            audio_embedding = outputs[0]  # First output is audio_embedding
            
            return audio_embedding[0]  # Remove batch dimension
        
        # Choose processing mode based on CLAP_PYTHON_MULTITHREADS
        if not config.CLAP_PYTHON_MULTITHREADS:
            # ONNX internal threading - process segments sequentially
            logger.info(f"CLAP: Processing {num_segments} segments sequentially (ONNX internal threading)")
            all_embeddings = []
            for seg in segments:
                try:
                    embedding = process_segment(seg)
                    all_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Segment processing failed: {e}")
                    raise
        else:
            # Python ThreadPoolExecutor - parallel processing with ONNX single-threaded
            # Auto-calculate thread count: (physical_cores - 1) + (logical_cores // 2)
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            num_threads = max(1, (physical_cores - 1) + ((logical_cores - physical_cores) // 2))
            logger.info(f"CLAP: Processing {num_segments} segments with {num_threads} Python threads (physical: {physical_cores}, logical: {logical_cores})")
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Pre-allocate result array
            all_embeddings = [None] * num_segments
            
            executor = None
            try:
                executor = ThreadPoolExecutor(max_workers=num_threads)
                
                # Submit all segments
                future_to_idx = {executor.submit(process_segment, seg): i 
                                for i, seg in enumerate(segments)}
                
                # Collect results as they complete (maintain order)
                for future in as_completed(future_to_idx):
                    try:
                        idx = future_to_idx[future]
                        embedding = future.result()
                        all_embeddings[idx] = embedding
                    except Exception as e:
                        logger.error(f"Segment processing failed: {e}")
                        raise
            finally:
                # Force immediate shutdown of all threads
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                    import time
                    time.sleep(0.1)
                
                # Cleanup thread-local storage
                import gc
                gc.collect()
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        avg_embedding = np.mean(all_embeddings, axis=0)
        
        # Normalize (should already be normalized, but ensure it)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Aggressive cleanup to prevent memory leaks
        del all_embeddings, segments, audio_data
        import gc
        gc.collect()
        
        return avg_embedding, duration_sec, num_segments
        
    except Exception as e:
        logger.error(f"CLAP analysis failed for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0
    finally:
        # Force cleanup even on error
        import gc
        gc.collect()


def get_text_embedding(query_text: str) -> Optional[np.ndarray]:
    """
    Get CLAP embedding for a text query using ONNX Runtime.
    
    Args:
        query_text: Natural language query
        
    Returns:
        512-dim normalized embedding vector or None if failed
    """
    if not config.CLAP_ENABLED:
        return None
    
    try:
        session = get_clap_model()
        tokenizer = get_tokenizer()
        
        # Tokenize text (max_length=77 for CLAP)
        encoded = tokenizer(
            query_text,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Create dummy mel-spectrogram input (not used in text mode)
        dummy_mel = np.zeros((1, 1, 1000, 64), dtype=np.float32)
        
        # Run ONNX inference
        onnx_inputs = {
            'mel_spectrogram': dummy_mel,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        outputs = session.run(None, onnx_inputs)
        text_embedding = outputs[1]  # Second output is text_embedding
        
        # Extract embedding (remove batch dimension)
        text_embedding = text_embedding[0]
        
        # Should already be normalized, but ensure it
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        return text_embedding
        
    except Exception as e:
        logger.error(f"Failed to get text embedding for '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return None


def is_clap_available() -> bool:
    """Check if CLAP is enabled and ONNX model can be loaded."""
    if not config.CLAP_ENABLED:
        return False
    
    if _onnx_session is not None:
        return True
    
    return initialize_clap_model()
