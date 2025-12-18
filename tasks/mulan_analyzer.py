"""
MuLan (MuQ) Audio Analyzer for Text-Based Music Search
Uses MuQ-MuLan model to generate embeddings for audio files
and text queries for natural language music search.

MuQ uses:
- MERT for audio encoding (music understanding transformer)
- T5 for text encoding (text-to-text transfer transformer)

Model automatically downloaded from HuggingFace on first use.
"""

import os
import gc
import logging
import traceback
import numpy as np
import librosa
import torch
import config
from typing import Tuple, Optional

# Silence transformers warning
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Configure PyTorch threading BEFORE any operations (must be at module import time)
num_threads = os.cpu_count() or 4
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

logger = logging.getLogger(__name__)

# Global MuQ model (lazy loaded)
_mulan_model = None
_device = None


def _get_device():
    """Determine device (GPU/CPU) for PyTorch inference."""
    global _device
    
    if _device is not None:
        return _device
    
    if torch.cuda.is_available():
        _device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        _device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return _device


def _load_mulan_model():
    """Load MuQ-MuLan PyTorch model from local HuggingFace cache."""
    import warnings
    
    # Suppress warnings during import
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from muq import MuQMuLan
    
    logger.info(f"Loading MuQ-MuLan model from local cache ({config.MULAN_MODEL_NAME})...")
    
    try:
        device = _get_device()
        
        # Force offline mode to prevent re-downloading/checking
        # We save the previous state to restore it later
        prev_offline = os.environ.get('HF_HUB_OFFLINE')
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            # Load from cached model
            # Suppress warnings that look like downloads or errors
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*resume_download.*")
                warnings.filterwarnings("ignore", message=".*weight_norm.*")
                warnings.filterwarnings("ignore", message=".*register_pytree_node.*")
                warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
                
                model = MuQMuLan.from_pretrained(config.MULAN_MODEL_NAME)
                logger.info("✓ Model loaded purely from local cache (Offline mode verified)")
        except Exception as e:
            # If offline load fails, try online (first run)
            logger.info(f"Model not found in cache ({e}), attempting download...")
            os.environ['HF_HUB_OFFLINE'] = '0'
            model = MuQMuLan.from_pretrained(config.MULAN_MODEL_NAME)
        finally:
            # Restore environment variable
            if prev_offline is None:
                del os.environ['HF_HUB_OFFLINE']
            else:
                os.environ['HF_HUB_OFFLINE'] = prev_offline
            
        model = model.to(device).eval()
        
        logger.info(f"✓ MuQ-MuLan model loaded successfully on {device}")
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load MuQ-MuLan model: {e}")
        import traceback
        traceback.print_exc()
        raise


def initialize_mulan_model():
    """Initialize MuLan PyTorch model if enabled and not already loaded."""
    global _mulan_model
    
    if not config.MULAN_ENABLED:
        logger.info("MuLan is disabled in config. Skipping model initialization.")
        return False
    
    if _mulan_model is not None:
        logger.debug("MuLan model already initialized.")
        return True
    
    try:
        _mulan_model = _load_mulan_model()
        logger.info("MuLan model initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MuLan model: {e}")
        import traceback
        traceback.print_exc()
        return False


def unload_mulan_model():
    """Unload MuLan model from memory to free RAM."""
    global _mulan_model, _device
    
    if _mulan_model is None:
        return False
    
    try:
        # Clear model
        _mulan_model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if _device is not None and _device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("✓ MuLan model unloaded from memory")
        return True
    except Exception as e:
        logger.error(f"Error unloading MuLan model: {e}")
        return False


def is_mulan_model_loaded():
    """Check if MuLan model is currently loaded in memory."""
    return _mulan_model is not None


def get_mulan_model():
    """Get the global MuLan model, initializing if needed (lazy loading)."""
    if _mulan_model is None:
        logger.info("Lazy-loading MuLan model on first use...")
        if not initialize_mulan_model():
            raise RuntimeError("MuLan model could not be initialized")
    return _mulan_model


def analyze_audio_file(audio_path: str) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Analyze an audio file and return MuLan embedding using MuQ PyTorch model.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (embedding_vector, duration_seconds, num_segments)
        Returns (None, 0, 0) if MuLan is disabled or analysis fails
    
    Note: MuQ strictly requires 24 kHz audio as input.
    """
    if not config.MULAN_ENABLED:
        return None, 0, 0
    
    try:
        model = get_mulan_model()
        device = _get_device()
        
        # Load audio at MuQ's required sample rate (24kHz)
        SAMPLE_RATE = 24000
        ANALYSIS_DURATION = 30.0  # Analyze 30 seconds for speed
        
        # Optimization: Load only a chunk (e.g. 30s) instead of full file
        # We try to load from the middle if possible to get representative audio
        try:
            # Fast duration check
            full_duration = librosa.get_duration(path=audio_path)
            
            if full_duration > ANALYSIS_DURATION:
                # Start from 20% into the track to skip intro, but ensure we have enough audio
                offset = min(full_duration * 0.2, full_duration - ANALYSIS_DURATION)
                load_duration = ANALYSIS_DURATION
            else:
                offset = 0.0
                load_duration = None # Load all
                
            audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, offset=offset, duration=load_duration)
        except Exception:
            # Fallback: just load the first 30s
            audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=ANALYSIS_DURATION)
        
        duration_sec = len(audio_data) / SAMPLE_RATE
        
        # Convert to tensor (fp32 recommended by MuQ to avoid NaN issues)
        wavs = torch.tensor(audio_data).unsqueeze(0).to(device)
        
        # Extract music embedding using MuQ-MuLan
        with torch.no_grad():
            audio_embeds = model(wavs=wavs)
        
        # Convert to numpy array
        audio_embedding = audio_embeds.cpu().numpy()
        
        # Flatten to 1D if needed
        if audio_embedding.ndim > 1:
            audio_embedding = audio_embedding.flatten()
        
        # Normalize embedding
        audio_embedding = audio_embedding / np.linalg.norm(audio_embedding)
        
        # Cleanup
        del wavs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return audio_embedding, duration_sec, 1  # Single pass, no segments
        
    except Exception as e:
        logger.error(f"MuLan analysis failed for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0
    finally:
        # Force cleanup
        gc.collect()
        if _device is not None and _device.type == 'cuda':
            torch.cuda.empty_cache()


def get_text_embedding(query_text: str) -> Optional[np.ndarray]:
    """
    Generate MuLan text embedding from natural language query using MuQ PyTorch model.
    
    Args:
        query_text: Natural language query (English or Chinese supported)
        
    Returns:
        Normalized embedding vector or None if failed
    """
    if not config.MULAN_ENABLED:
        return None
    
    try:
        model = get_mulan_model()
        device = _get_device()
        
        # Extract text embeddings using MuQ-MuLan
        with torch.no_grad():
            text_embeds = model(texts=[query_text])
        
        # Convert to numpy and flatten
        text_embedding = text_embeds.cpu().numpy()
        if text_embedding.ndim > 1:
            text_embedding = text_embedding.flatten()
        
        # Normalize embedding
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        return text_embedding
        
    except Exception as e:
        logger.error(f"MuLan text embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_text_embeddings_batch(query_texts: list) -> Optional[np.ndarray]:
    """
    Generate MuLan text embeddings for a batch of queries using MuQ PyTorch model.
    
    Args:
        query_texts: List of natural language queries (English or Chinese supported)
        
    Returns:
        Array of normalized embeddings (batch_size, embedding_dim) or None if failed
    """
    if not config.MULAN_ENABLED:
        return None
    
    try:
        model = get_mulan_model()
        device = _get_device()
        
        # Extract text embeddings using MuQ-MuLan (handles batch processing internally)
        with torch.no_grad():
            text_embeds = model(texts=query_texts)
            
            # Move to CPU and convert to numpy
            text_embeddings = text_embeds.cpu().numpy()
        
        # Normalize each embedding
        norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / norms
        
        return text_embeddings
        
    except Exception as e:
        logger.error(f"Failed to get batch text embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None
