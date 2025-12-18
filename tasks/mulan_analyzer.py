"""
MuLan (MuQ) Audio Analyzer for Text-Based Music Search
Uses MuQ-MuLan ONNX models to generate embeddings for audio files
and text queries for natural language music search.

MuQ uses:
- Audio encoder: Processes raw audio at 24kHz
- Text encoder: XLM-RoBERTa based text understanding

Models loaded from pre-converted ONNX files (no PyTorch dependency).
"""

import os
import gc
import logging
import traceback
import numpy as np
import librosa
import onnxruntime as ort
import config
from typing import Tuple, Optional
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Global MuLan ONNX sessions (lazy loaded)
_audio_session = None
_text_session = None
_tokenizer = None


def _load_mulan_models():
    """Load MuQ-MuLan ONNX models and tokenizer from local files."""
    global _audio_session, _text_session, _tokenizer
    
    logger.info("Loading MuQ-MuLan ONNX models...")
    
    try:
        # Check if model files exist
        if not os.path.exists(config.AUDIO_MODEL_PATH):
            raise FileNotFoundError(f"Audio model not found: {config.AUDIO_MODEL_PATH}")
        if not os.path.exists(config.TEXT_MODEL_PATH):
            raise FileNotFoundError(f"Text model not found: {config.TEXT_MODEL_PATH}")
        if not os.path.exists(config.TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found: {config.TOKENIZER_PATH}")
        
        # Configure ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.inter_op_num_threads = os.cpu_count() or 4
        
        # Select execution provider (CPU or CUDA)
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers() and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("CUDA available - using GPU acceleration")
        else:
            logger.info("Using CPU execution")
        
        # Load audio encoder (with external data file)
        logger.info(f"Loading audio encoder: {config.AUDIO_MODEL_PATH}")
        _audio_session = ort.InferenceSession(
            config.AUDIO_MODEL_PATH,
            sess_options=sess_options,
            providers=providers
        )
        
        # Load text encoder (with external data file)  
        logger.info(f"Loading text encoder: {config.TEXT_MODEL_PATH}")
        _text_session = ort.InferenceSession(
            config.TEXT_MODEL_PATH,
            sess_options=sess_options,
            providers=providers
        )
        
        # Load tokenizer (XLM-RoBERTa from MuLan training)
        logger.info(f"Loading tokenizer: {config.TOKENIZER_PATH}")
        _tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
        
        # Set padding/truncation for tokenizer
        _tokenizer.enable_padding(pad_id=1, pad_token="<pad>", length=128)
        _tokenizer.enable_truncation(max_length=128)
        
        logger.info("✓ MuQ-MuLan ONNX models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load MuQ-MuLan ONNX models: {e}")
        traceback.print_exc()
        raise


def initialize_mulan_model():
    """Initialize MuLan ONNX models if enabled and not already loaded."""
    global _audio_session, _text_session, _tokenizer
    
    if not config.MULAN_ENABLED:
        logger.info("MuLan is disabled in config. Skipping model initialization.")
        return False
    
    if _audio_session is not None and _text_session is not None and _tokenizer is not None:
        logger.debug("MuLan models already initialized.")
        return True
    
    try:
        _load_mulan_models()
        logger.info("MuLan ONNX models initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MuLan models: {e}")
        traceback.print_exc()
        return False


def unload_mulan_model():
    """Unload MuLan models from memory to free RAM."""
    global _audio_session, _text_session, _tokenizer
    
    if _audio_session is None and _text_session is None and _tokenizer is None:
        return False
    
    try:
        # Clear sessions
        _audio_session = None
        _text_session = None
        _tokenizer = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✓ MuLan models unloaded from memory")
        return True
    except Exception as e:
        logger.error(f"Error unloading MuLan models: {e}")
        return False


def is_mulan_model_loaded():
    """Check if MuLan models are currently loaded in memory."""
    return _audio_session is not None and _text_session is not None and _tokenizer is not None


def get_mulan_sessions():
    """Get the global MuLan sessions, initializing if needed (lazy loading)."""
    if _audio_session is None or _text_session is None or _tokenizer is None:
        logger.info("Lazy-loading MuLan models on first use...")
        if not initialize_mulan_model():
            raise RuntimeError("MuLan models could not be initialized")
    return _audio_session, _text_session, _tokenizer


def analyze_audio_file(audio_path: str) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Analyze an audio file and return MuLan embedding using ONNX model.
    
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
        audio_session, _, _ = get_mulan_sessions()
        
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
        
        # Prepare input: (batch_size, samples) -> (1, num_samples)
        # Note: MuLan audio encoder expects variable length input, but we ensure reasonable length
        audio_input = audio_data.astype(np.float32).reshape(1, -1)
        
        # Run ONNX inference
        audio_embedding = audio_session.run(
            ['audio_embedding'],
            {'wavs': audio_input}
        )[0]
        
        # Flatten to 1D if needed
        if audio_embedding.ndim > 1:
            audio_embedding = audio_embedding.flatten()
        
        # Normalize embedding (should already be normalized by model, but ensure it)
        norm = np.linalg.norm(audio_embedding)
        if norm > 0:
            audio_embedding = audio_embedding / norm
        
        return audio_embedding, duration_sec, 1  # Single pass, no segments
        
    except Exception as e:
        logger.error(f"MuLan analysis failed for {audio_path}: {e}")
        traceback.print_exc()
        return None, 0, 0
    finally:
        # Force cleanup
        gc.collect()


def get_text_embedding(query_text: str) -> Optional[np.ndarray]:
    """
    Generate MuLan text embedding from natural language query using ONNX model.
    
    Args:
        query_text: Natural language query (English or Chinese supported)
        
    Returns:
        Normalized embedding vector or None if failed
    """
    if not config.MULAN_ENABLED:
        return None
    
    try:
        _, text_session, tokenizer = get_mulan_sessions()
        
        # Tokenize input text (XLM-RoBERTa tokenizer from MuLan training)
        encoding = tokenizer.encode(query_text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        
        # Run ONNX inference
        text_embedding = text_session.run(
            ['text_embedding'],
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        )[0]
        
        # Flatten to 1D
        if text_embedding.ndim > 1:
            text_embedding = text_embedding.flatten()
        
        # Normalize embedding (should already be normalized by model, but ensure it)
        norm = np.linalg.norm(text_embedding)
        if norm > 0:
            text_embedding = text_embedding / norm
        
        return text_embedding
        
    except Exception as e:
        logger.error(f"MuLan text embedding generation failed: {e}")
        traceback.print_exc()
        return None


def get_text_embeddings_batch(query_texts: list) -> Optional[np.ndarray]:
    """
    Generate MuLan text embeddings for a batch of queries using ONNX model.
    
    Args:
        query_texts: List of natural language queries (English or Chinese supported)
        
    Returns:
        Array of normalized embeddings (batch_size, embedding_dim) or None if failed
    """
    if not config.MULAN_ENABLED:
        return None
    
    try:
        _, text_session, tokenizer = get_mulan_sessions()
        
        # Tokenize all texts
        encodings = [tokenizer.encode(text) for text in query_texts]
        
        # Prepare batch inputs (pad to same length)
        max_len = max(len(enc.ids) for enc in encodings)
        input_ids_batch = []
        attention_mask_batch = []
        
        for enc in encodings:
            ids = enc.ids + [1] * (max_len - len(enc.ids))  # Pad with pad_token_id=1
            mask = enc.attention_mask + [0] * (max_len - len(enc.attention_mask))
            input_ids_batch.append(ids)
            attention_mask_batch.append(mask)
        
        input_ids = np.array(input_ids_batch, dtype=np.int64)
        attention_mask = np.array(attention_mask_batch, dtype=np.int64)
        
        # Run ONNX inference
        text_embeddings = text_session.run(
            ['text_embedding'],
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        )[0]
        
        # Normalize each embedding
        norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        text_embeddings = text_embeddings / norms
        
        return text_embeddings
        
    except Exception as e:
        logger.error(f"Failed to get batch text embeddings: {e}")
        traceback.print_exc()
        return None
