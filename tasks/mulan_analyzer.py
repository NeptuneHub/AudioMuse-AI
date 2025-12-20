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
from transformers import AutoTokenizer

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
        if not os.path.exists(config.MULAN_MODEL_DIR):
            raise FileNotFoundError(f"Tokenizer directory not found: {config.MULAN_MODEL_DIR}")
        
        # Configure ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use half of physical CPU cores (no hyperthreading) to prevent resource exhaustion
        # This prevents system crashes while allowing reasonable parallelism
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or 4
        num_threads = max(1, physical_cores // 2)  # Integer division, minimum 1
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        logger.info(f"MuLan: Using {num_threads} threads (half of {physical_cores} physical cores)")
        
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
        
        # Load tokenizer from extracted directory (uses transformers for compatibility)
        logger.info(f"Loading tokenizer from: {config.MULAN_MODEL_DIR}")
        _tokenizer = AutoTokenizer.from_pretrained(config.MULAN_MODEL_DIR)
        
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
        SEGMENT_DURATION = 10.0  # FIXED 10 second segments (model requirement)
        SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)  # EXACTLY 240,000 samples
        HOP_DURATION = 5.0  # 50% overlap = 5 second hop
        HOP_SAMPLES = int(HOP_DURATION * SAMPLE_RATE)  # 120,000 samples
        ANALYSIS_WINDOW = 50.0  # Prefer central 50 seconds if available
        
        # Get full duration first
        full_duration = librosa.get_duration(path=audio_path)
        
        # Determine what to load based on song length
        if full_duration > ANALYSIS_WINDOW:
            # Long song: use central 50 seconds
            offset = (full_duration - ANALYSIS_WINDOW) / 2
            load_duration = ANALYSIS_WINDOW
        elif full_duration >= SEGMENT_DURATION:
            # Song is between 10s and 50s: use whole song
            offset = 0.0
            load_duration = full_duration
        else:
            # Song is shorter than 10s: load what we have, will pad to 10s
            offset = 0.0
            load_duration = full_duration
        
        # Load audio from the selected portion
        audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, offset=offset, duration=load_duration)
        
        # Calculate number of 10-second segments with 50% overlap
        if len(audio_data) < SEGMENT_SAMPLES:
            # Audio is shorter than 10s: create 1 segment with padding
            num_segments = 1
        else:
            # Calculate overlapping segments: floor((length - segment_size) / hop_size) + 1
            num_segments = int((len(audio_data) - SEGMENT_SAMPLES) / HOP_SAMPLES) + 1
        
        segment_embeddings = []
        
        # Process each FIXED 10-second segment
        for seg_idx in range(num_segments):
            start_sample = seg_idx * HOP_SAMPLES
            end_sample = start_sample + SEGMENT_SAMPLES
            
            if start_sample >= len(audio_data):
                # No more audio data
                break
            
            # Extract segment
            if end_sample <= len(audio_data):
                # Full segment available
                segment = audio_data[start_sample:end_sample]
            else:
                # Last segment: need padding
                segment = audio_data[start_sample:]
                padding = SEGMENT_SAMPLES - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant')
            
            # Ensure EXACTLY 240,000 samples (critical requirement)
            assert len(segment) == SEGMENT_SAMPLES, f"Segment must be exactly {SEGMENT_SAMPLES} samples, got {len(segment)}"
            
            # Prepare input: (batch_size, samples) -> (1, 240000)
            audio_input = segment.astype(np.float32).reshape(1, -1)
            
            # Run ONNX inference
            audio_embedding = audio_session.run(
                ['audio_embedding'],
                {'wavs': audio_input}
            )[0]
            
            # Flatten to 1D if needed
            if audio_embedding.ndim > 1:
                audio_embedding = audio_embedding.flatten()
            
            segment_embeddings.append(audio_embedding)
        
        # Average all segment embeddings
        final_embedding = np.mean(segment_embeddings, axis=0)
        
        # Normalize embedding
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm
        
        duration_sec = load_duration
        
        return final_embedding, duration_sec, len(segment_embeddings)
        
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
        # AutoTokenizer returns dict with input_ids and attention_mask
        encoding = tokenizer(
            query_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        input_ids = encoding['input_ids'].astype(np.int64)
        attention_mask = encoding['attention_mask'].astype(np.int64)
        
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
        
        # Tokenize all texts (AutoTokenizer handles batching automatically)
        encoding = tokenizer(
            query_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        input_ids = encoding['input_ids'].astype(np.int64)
        attention_mask = encoding['attention_mask'].astype(np.int64)
        
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
