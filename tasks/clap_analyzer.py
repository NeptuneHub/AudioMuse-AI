"""
CLAP Audio Analyzer for Text-Based Music Search
Uses split ONNX CLAP models:
- Audio model: For analyzing music files (worker containers)
- Text model: For text search queries (Flask containers)

Split models allow loading only what's needed, saving memory:
- Audio analysis: ~268MB (audio model only)
- Text search: ~478MB (text model only)
- Combined (old): ~746MB (both models)

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
try:
    from config import AUDIO_LOAD_TIMEOUT
except Exception:
    AUDIO_LOAD_TIMEOUT = None
from tasks.memory_utils import cleanup_cuda_memory, handle_onnx_memory_error, comprehensive_memory_cleanup

logger = logging.getLogger(__name__)

# Global ONNX sessions (lazy loaded)
_audio_session = None  # For audio analysis (worker containers)
_text_session = None   # For text search (Flask containers)
_tokenizer = None
_cached_dummy_input_ids = None  # Reusable dummy input for audio-only inference


def _load_audio_model():
    """Load CLAP audio-only ONNX model for music analysis (worker containers)."""
    import onnxruntime as ort
    import gc
    
    model_path = config.CLAP_AUDIO_MODEL_PATH
    logger.info(f"Loading CLAP audio model from {model_path}...")
    
    # Configure ONNX Runtime session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    
    # Threading configuration based on CLAP_PYTHON_MULTITHREADS:
    # - False (default): Use half of logical cores to avoid starving other processes
    # - True: Disable ONNX threading (set to 1), use Python ThreadPoolExecutor instead
    if not config.CLAP_PYTHON_MULTITHREADS:
        # Use half of available cores (including hyperthreading) to leave room for other processes
        import psutil
        logical_cores = psutil.cpu_count(logical=True) or 4
        num_threads = max(1, logical_cores // 2)
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        logger.info(f"CLAP Audio: Using {num_threads} threads (half of {logical_cores} logical cores)")
    else:
        # Python ThreadPoolExecutor will handle threading - disable ONNX threading
        sess_options.intra_op_num_threads = 1  # Single-threaded ONNX operations
        sess_options.inter_op_num_threads = 1  # Single-threaded ONNX operations
        logger.info("CLAP Audio: Using Python threading (auto-calculated threads), ONNX single-threaded")
    
    # GPU support: ONNX Runtime handles CUDA availability internally
    session = None
    
    # Configure provider options with GPU memory management
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        gpu_device_id = 0
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            gpu_device_id = 0
        
        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'DEFAULT',
        }
        provider_options = [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]
        logger.info(f"CUDA provider available - will attempt to use GPU (device_id={gpu_device_id})")
    else:
        provider_options = [('CPUExecutionProvider', {})]
        logger.info("CUDA provider not available - using CPU only")
    
    # Create session
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[p[0] for p in provider_options],
            provider_options=[p[1] for p in provider_options]
        )
        
        active_provider = session.get_providers()[0]
        logger.info(f"✓ CLAP audio model loaded successfully (~268MB)")
        logger.info(f"  Active execution provider: {active_provider}")
            
    except Exception as e:
        logger.warning(f"Failed to load with preferred providers: {e}")
        logger.info("Attempting final CPU-only fallback...")
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            logger.info(f"✓ CLAP audio model loaded successfully (CPU fallback)")
        except Exception as cpu_error:
            logger.error(f"Failed to load ONNX audio model even with CPU: {cpu_error}")
            raise
    
    if session is None:
        raise RuntimeError("Failed to create audio ONNX session")
    
    logger.info(f"  Inputs: {[i.name for i in session.get_inputs()]}")
    logger.info(f"  Outputs: {[o.name for o in session.get_outputs()]}")
    
    gc.collect()
    return session


def _load_text_model():
    """Load CLAP text-only ONNX model for text search (Flask containers)."""
    import onnxruntime as ort
    import gc
    
    model_path = config.CLAP_TEXT_MODEL_PATH
    logger.info(f"Loading CLAP text model from {model_path}...")
    
    # Configure ONNX Runtime session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    
    if not config.CLAP_PYTHON_MULTITHREADS:
        logger.info("CLAP Text: Using ONNX Runtime automatic thread management")
    else:
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        logger.info("CLAP Text: Using Python threading, ONNX single-threaded")
    
    # Text model typically runs on CPU in Flask containers
    session = None
    available_providers = ort.get_available_providers()
    
    if 'CUDAExecutionProvider' in available_providers:
        gpu_device_id = 0
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            gpu_device_id = 0
        
        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'DEFAULT',
        }
        provider_options = [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]
        logger.info(f"CUDA provider available - will attempt to use GPU (device_id={gpu_device_id})")
    else:
        provider_options = [('CPUExecutionProvider', {})]
        logger.info("CUDA provider not available - using CPU only")
    
    # Create session
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[p[0] for p in provider_options],
            provider_options=[p[1] for p in provider_options]
        )
        
        active_provider = session.get_providers()[0]
        logger.info(f"✓ CLAP text model loaded successfully (~478MB)")
        logger.info(f"  Active execution provider: {active_provider}")
            
    except Exception as e:
        logger.warning(f"Failed to load with preferred providers: {e}")
        logger.info("Attempting final CPU-only fallback...")
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            logger.info(f"✓ CLAP text model loaded successfully (CPU fallback)")
        except Exception as cpu_error:
            logger.error(f"Failed to load ONNX text model even with CPU: {cpu_error}")
            raise
    
    if session is None:
        raise RuntimeError("Failed to create text ONNX session")
    
    logger.info(f"  Inputs: {[i.name for i in session.get_inputs()]}")
    logger.info(f"  Outputs: {[o.name for o in session.get_outputs()]}")
    
    gc.collect()
    return session


def _load_onnx_model():
    """DEPRECATED: Load combined CLAP ONNX model (for backward compatibility)."""
    import onnxruntime as ort
    import gc
    
    logger.warning("DEPRECATED: Using legacy combined CLAP model. Consider updating to split models.")
    logger.info(f"Loading CLAP ONNX model from {config.CLAP_MODEL_PATH}...")
    
    # Configure ONNX Runtime session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    
    # Threading configuration based on CLAP_PYTHON_MULTITHREADS:
    # - False (default): Let ONNX Runtime decide optimal thread count automatically
    # - True: Disable ONNX threading (set to 1), use Python ThreadPoolExecutor instead
    if not config.CLAP_PYTHON_MULTITHREADS:
        # Let ONNX Runtime handle threading automatically (optimal for most cases)
        # import psutil
        # logical_cores = psutil.cpu_count(logical=True) or 4
        # num_threads = max(1, logical_cores - 2)  # All cores minus 2, minimum 1
        # sess_options.intra_op_num_threads = num_threads
        # sess_options.inter_op_num_threads = num_threads
        # logger.info(f"CLAP: Using {num_threads} threads ({logical_cores} logical cores - 2)")
        logger.info("CLAP: Using ONNX Runtime automatic thread management")
    else:
        # Python ThreadPoolExecutor will handle threading - disable ONNX threading
        sess_options.intra_op_num_threads = 1  # Single-threaded ONNX operations
        sess_options.inter_op_num_threads = 1  # Single-threaded ONNX operations
        logger.info("CLAP: Using Python threading (auto-calculated threads), ONNX single-threaded")
    
    # GPU support: ONNX Runtime handles CUDA availability internally
    # If CUDA fails, it automatically falls back to CPU
    session = None
    
    # Configure provider options with GPU memory management
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        # Get GPU device ID from environment or default to 0
        # Docker sets NVIDIA_VISIBLE_DEVICES, CUDA runtime uses CUDA_VISIBLE_DEVICES
        gpu_device_id = 0
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            # If CUDA_VISIBLE_DEVICES is set, use first device (already mapped to 0)
            gpu_device_id = 0
        
        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',  # Prevent memory fragmentation
            'cudnn_conv_algo_search': 'DEFAULT',
        }
        provider_options = [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]
        logger.info(f"CUDA provider available - will attempt to use GPU (device_id={gpu_device_id})")
    else:
        provider_options = [('CPUExecutionProvider', {})]
        logger.info("CUDA provider not available - using CPU only")
    
    # Create session with determined providers
    try:
        session = ort.InferenceSession(
            config.CLAP_MODEL_PATH,
            sess_options=sess_options,
            providers=[p[0] for p in provider_options],
            provider_options=[p[1] for p in provider_options]
        )
        
        active_provider = session.get_providers()[0]
        logger.info(f"✓ CLAP ONNX model loaded successfully")
        logger.info(f"  Active execution provider: {active_provider}")
            
    except Exception as e:
        # Final fallback: force CPU-only
        logger.warning(f"Failed to load with preferred providers: {e}")
        logger.info("Attempting final CPU-only fallback...")
        try:
            session = ort.InferenceSession(
                config.CLAP_MODEL_PATH,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            logger.info(f"✓ CLAP ONNX model loaded successfully (CPU fallback)")
            logger.info(f"  Active execution provider: CPUExecutionProvider")
        except Exception as cpu_error:
            logger.error(f"Failed to load ONNX model even with CPU: {cpu_error}")
            raise
    
    if session is None:
        raise RuntimeError("Failed to create ONNX session")
    
    logger.info(f"  Inputs: {[i.name for i in session.get_inputs()]}")
    logger.info(f"  Outputs: {[o.name for o in session.get_outputs()]}")
    
    gc.collect()
    return session


def _load_tokenizer():
    """Load RoBERTa tokenizer for text processing."""
    from transformers import AutoTokenizer
    
    logger.info("Loading RoBERTa tokenizer...")
    
    # Use the same tokenizer as the CLAP model (roberta-base)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    logger.info("✓ Tokenizer loaded successfully")
    return tokenizer


def initialize_clap_audio_model():
    """Initialize CLAP audio model for music analysis (worker containers only)."""
    global _audio_session
    
    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping audio model initialization.")
        return False
    
    if _audio_session is not None:
        logger.debug("CLAP audio model already initialized.")
        return True
    
    if not os.path.exists(config.CLAP_AUDIO_MODEL_PATH):
        logger.error(f"CLAP audio model not found at {config.CLAP_AUDIO_MODEL_PATH}")
        return False
    
    try:
        _audio_session = _load_audio_model()
        logger.info("✓ CLAP audio model initialized successfully (for music analysis)")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CLAP audio model: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_clap_text_model():
    """Initialize CLAP text model for text search (Flask containers only)."""
    global _text_session, _tokenizer
    
    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping text model initialization.")
        return False
    
    if _text_session is not None:
        logger.debug("CLAP text model already initialized.")
        return True
    
    if not os.path.exists(config.CLAP_TEXT_MODEL_PATH):
        logger.error(f"CLAP text model not found at {config.CLAP_TEXT_MODEL_PATH}")
        return False
    
    try:
        _text_session = _load_text_model()
        _tokenizer = _load_tokenizer()
        logger.info("✓ CLAP text model initialized successfully (for text search)")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CLAP text model: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_clap_model():
    """Initialize CLAP ONNX model if enabled and not already loaded.
    
    DEPRECATED: For backward compatibility. Use initialize_clap_audio_model() or
    initialize_clap_text_model() instead to load only what you need.
    """
    global _audio_session, _text_session, _tokenizer
    
    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping model initialization.")
        return False
    
    # Try split models first
    if os.path.exists(config.CLAP_AUDIO_MODEL_PATH) and os.path.exists(config.CLAP_TEXT_MODEL_PATH):
        logger.warning("DEPRECATED: initialize_clap_model() called but split models available.")
        logger.warning("Consider using initialize_clap_audio_model() or initialize_clap_text_model() instead.")
        
        # Load both for backward compatibility
        audio_ok = initialize_clap_audio_model()
        text_ok = initialize_clap_text_model()
        return audio_ok and text_ok
    
    # Fall back to legacy combined model
    if _audio_session is not None or _text_session is not None:
        logger.debug("CLAP model already initialized.")
        return True
    
    if not os.path.exists(config.CLAP_MODEL_PATH):
        logger.error(f"CLAP model not found at {config.CLAP_MODEL_PATH}")
        return False
    
    try:
        # Load legacy combined model into both slots
        combined_session = _load_onnx_model()
        _audio_session = combined_session
        _text_session = combined_session
        _tokenizer = _load_tokenizer()
        logger.info("CLAP ONNX model initialized successfully (legacy combined model).")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CLAP ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return False


def unload_clap_model():
    """Unload CLAP model from memory to free RAM."""
    global _audio_session, _text_session, _tokenizer, _cached_dummy_input_ids
    
    if _audio_session is None and _text_session is None:
        return False
    
    try:
        # Clear ONNX sessions
        freed_mb = 0
        if _audio_session is not None:
            _audio_session = None
            freed_mb += 268
        if _text_session is not None:
            _text_session = None
            freed_mb += 478
        
        _tokenizer = None
        _cached_dummy_input_ids = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"✓ CLAP model(s) unloaded from memory (~{freed_mb}MB freed)")
        return True
    except Exception as e:
        logger.error(f"Error unloading CLAP model: {e}")
        return False


def is_clap_model_loaded():
    """Check if any CLAP model is currently loaded in memory."""
    return _audio_session is not None or _text_session is not None


def is_clap_audio_loaded():
    """Check if CLAP audio model is currently loaded."""
    return _audio_session is not None


def is_clap_text_loaded():
    """Check if CLAP text model is currently loaded."""
    return _text_session is not None


def get_clap_audio_model():
    """Get the CLAP audio session, initializing if needed (lazy loading)."""
    if _audio_session is None:
        logger.info("Lazy-loading CLAP audio model on first use...")
        if not initialize_clap_audio_model():
            raise RuntimeError("Failed to initialize CLAP audio model")
    return _audio_session


def get_clap_text_model():
    """Get the CLAP text session, initializing if needed (lazy loading)."""
    if _text_session is None:
        logger.info("Lazy-loading CLAP text model on first use...")
        if not initialize_clap_text_model():
            raise RuntimeError("Failed to initialize CLAP text model")
    return _text_session


def get_clap_model():
    """DEPRECATED: Get the global CLAP ONNX session. Use get_clap_audio_model() or get_clap_text_model() instead."""
    # For backward compatibility, return audio model if available
    if _audio_session is not None:
        return _audio_session
    if _text_session is not None:
        return _text_session
    
    logger.warning("DEPRECATED: get_clap_model() called. Use get_clap_audio_model() or get_clap_text_model()")
    logger.info("Lazy-loading CLAP model on first use (saves RAM at startup)...")
    if not initialize_clap_model():
        raise RuntimeError("Failed to initialize CLAP model")
    return _audio_session or _text_session


def get_tokenizer():
    """Get the global tokenizer, initializing if needed (lazy loading)."""
    global _tokenizer
    
    if _tokenizer is None:
        # Initialize tokenizer with text model
        if not initialize_clap_text_model():
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
        
        # Get audio-only model for music analysis
        session = get_clap_audio_model()
        
        # Load audio at CLAP's expected sample rate (48kHz)
        SAMPLE_RATE = 48000
        SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
        HOP_LENGTH = 240000      # 5 seconds (50% overlap)
        
        # Use configured AUDIO_LOAD_TIMEOUT if present to limit librosa loading duration
        try:
            if AUDIO_LOAD_TIMEOUT is not None:
                audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=AUDIO_LOAD_TIMEOUT)
            else:
                audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio for CLAP analysis ({audio_path}) with AUDIO_LOAD_TIMEOUT={AUDIO_LOAD_TIMEOUT}: {e}")
            import traceback
            traceback.print_exc()
            # Ensure CUDA memory is cleaned up on load failure
            cleanup_cuda_memory(force=True)
            return None, 0, 0
        
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
            
            # Run ONNX inference (audio model only needs mel_spectrogram)
            onnx_inputs = {
                'mel_spectrogram': mel_input
            }
            
            try:
                outputs = session.run(None, onnx_inputs)
                audio_embedding = outputs[0]  # Output is audio_embedding
            except Exception as e:
                # Handle memory allocation errors with cleanup and retry
                def cleanup_fn():
                    cleanup_cuda_memory(force=True)
                
                def retry_fn():
                    return session.run(None, onnx_inputs)
                
                result = handle_onnx_memory_error(
                    e,
                    f"CLAP segment inference",
                    cleanup_func=cleanup_fn,
                    retry_func=retry_fn
                )
                
                if result is not None:
                    audio_embedding = result[0]
                else:
                    raise
            
            return audio_embedding[0]  # Remove batch dimension
        
        # Simplified approach: always run batched ONNX inference for all segments (CPU or GPU).
        # This avoids complex branching and reduces Python overhead. We still keep OOM handling.
        if num_segments < 1:
            logger.error("No segments to process for CLAP analysis")
            raise RuntimeError("No segments generated for CLAP analysis")

        logger.info(f"CLAP: Running batched ONNX inference with {num_segments} segments")

        # Compute mel-spectrograms for all segments and stack into a single batch
        mel_list = [compute_mel_spectrogram(seg, SAMPLE_RATE) for seg in segments]
        # Each mel has shape (1, time_frames, 64); add channel dim -> (1,1,time,64), then stack -> (batch,1,time,64)
        try:
            mel_batch = np.concatenate([m[:, np.newaxis, :, :] for m in mel_list], axis=0)
        except Exception as e:
            logger.error(f"Failed to create mel batch: {e}")
            # Clean up mel_list before raising
            del mel_list
            import gc
            gc.collect()
            raise

        onnx_inputs = {'mel_spectrogram': mel_batch}
        try:
            outputs = session.run(None, onnx_inputs)
            all_embeddings = outputs[0]  # shape: (batch, dim)
        except Exception as e:
            # Handle memory allocation errors with cleanup and retry
            def cleanup_fn():
                cleanup_cuda_memory(force=True)

            def retry_fn():
                return session.run(None, onnx_inputs)

            result = handle_onnx_memory_error(
                e,
                f"CLAP batched segment inference",
                cleanup_func=cleanup_fn,
                retry_func=retry_fn
            )

            if result is not None:
                all_embeddings = result[0]
            else:
                raise
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        avg_embedding = np.mean(all_embeddings, axis=0)
        
        # Normalize (should already be normalized, but ensure it)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # CRITICAL: Delete all large memory allocations immediately
        del all_embeddings, mel_batch, mel_list, segments, audio_data
        import gc
        gc.collect()
        
        # Comprehensive CUDA and ONNX memory cleanup
        comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
        
        return avg_embedding, duration_sec, num_segments
        
    except Exception as e:
        logger.error(f"CLAP analysis failed for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        
        # Comprehensive cleanup on error - force cleanup including ONNX pool reset
        comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
        
        return None, 0, 0
    finally:
        # Final cleanup even on error
        import gc
        gc.collect()


def get_text_embedding(query_text: str) -> Optional[np.ndarray]:
    """
    Get CLAP embedding for a text query using ONNX Runtime (text model only).
    
    Args:
        query_text: Natural language query
        
    Returns:
        512-dim normalized embedding vector or None if failed
    """
    if not config.CLAP_ENABLED:
        return None
    
    try:
        # Get text-only model for text search
        session = get_clap_text_model()
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
        
        # Run ONNX inference (text model only needs input_ids and attention_mask)
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        outputs = session.run(None, onnx_inputs)
        text_embedding = outputs[0]  # Output is text_embedding
        
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


def get_text_embeddings_batch(query_texts: list) -> Optional[np.ndarray]:
    """
    Get CLAP embeddings for multiple text queries in a single batch.
    More efficient than calling get_text_embedding() repeatedly.
    
    Args:
        query_texts: List of natural language queries
        
    Returns:
        (N, 512) array of normalized embeddings or None if failed
    """
    
    if not config.CLAP_ENABLED:
        return None
    
    if not query_texts:
        return None
    
    try:
        # Get text-only model for text search
        session = get_clap_text_model()
        tokenizer = get_tokenizer()
        
        batch_size = len(query_texts)
        
        # Tokenize all texts at once (max_length=77 for CLAP)
        encoded = tokenizer(
            query_texts,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Run ONNX inference for batch (text model only needs input_ids and attention_mask)
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        outputs = session.run(None, onnx_inputs)
        text_embeddings = outputs[0]  # Output is text_embedding (batch_size, 512)
        
        # Normalize each embedding
        norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / norms
        
        return text_embeddings
        
    except Exception as e:
        logger.error(f"Failed to get batch text embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None


def is_clap_available() -> bool:
    """
    Check if CLAP is enabled and model files exist.
    Does NOT load the model - use get_clap_audio_model() or get_clap_text_model() for lazy loading.
    """
    if not config.CLAP_ENABLED:
        return False
    
    # Check if split models exist (preferred)
    if os.path.exists(config.CLAP_AUDIO_MODEL_PATH) and os.path.exists(config.CLAP_TEXT_MODEL_PATH):
        return True
    
    # Fall back to legacy combined model
    return os.path.exists(config.CLAP_MODEL_PATH)
