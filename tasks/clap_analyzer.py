"""
CLAP Audio Analyzer for Text-Based Music Search
Uses LAION-CLAP model to generate 512-dim embeddings for audio files
and text queries for natural language music search.
"""

import os
import sys
import logging
import numpy as np
from typing import Tuple, Optional

# CRITICAL: Set thread limits BEFORE importing torch/transformers
# Prevents CPU oversubscription when multiple workers load model simultaneously
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import config

logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_clap_model = None


def _suppress_output(func):
    """Decorator to suppress stdout/stderr during model loading."""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return wrapper


def _load_clap_model():
    """Load CLAP model with aggressive memory optimization."""
    import laion_clap
    import torch
    import gc
    
    # Force CPU-only mode and clear cache
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    
    # Patch load_state_dict to ignore unexpected keys (like position_ids)
    original_load = model.model.load_state_dict
    model.model.load_state_dict = lambda *args, **kwargs: original_load(
        *args, **{**kwargs, 'strict': False}
    )
    
    model.load_ckpt(config.CLAP_MODEL_PATH)
    model.model.load_state_dict = original_load
    
    model.eval()
    
    # CRITICAL: Aggressive memory optimization
    # 1. Disable all gradients (saves ~50% memory)
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Convert to half precision (FP16) - saves 50% memory on model weights
    # Only if not using CPU (CPU doesn't support FP16 well)
    # model = model.half()  # Skip on CPU, causes issues
    
    # 3. Force model to CPU and clear any GPU memory
    model = model.cpu()
    
    # 4. Delete optimizer states and unnecessary buffers
    if hasattr(model, 'optimizer'):
        del model.optimizer
    
    # 5. Run garbage collection
    gc.collect()
    
    logger.info(f"✓ CLAP model loaded with memory optimization")
    
    return model


def initialize_clap_model():
    """Initialize CLAP model if enabled and not already loaded."""
    global _clap_model
    
    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping model initialization.")
        return False
    
    if _clap_model is not None:
        logger.debug("CLAP model already initialized.")
        return True
    
    if not os.path.exists(config.CLAP_MODEL_PATH):
        logger.error(f"CLAP model not found at {config.CLAP_MODEL_PATH}")
        return False
    
    try:
        logger.info(f"Loading CLAP model from {config.CLAP_MODEL_PATH}...")
        _clap_model = _load_clap_model()
        logger.info("CLAP model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load CLAP model: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_clap_model():
    """Get the global CLAP model instance, initializing if needed."""
    if _clap_model is None:
        if not initialize_clap_model():
            raise RuntimeError("CLAP model could not be initialized")
    return _clap_model


def analyze_audio_file(audio_path: str) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Analyze an audio file and return CLAP embedding.
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
        import torch
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        model = get_clap_model()
        
        # Load audio at CLAP's expected sample rate (48kHz)
        SAMPLE_RATE = 48000
        SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
        HOP_LENGTH = 240000      # 5 seconds (50% overlap)
        
        audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
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
        
        # Process batches in PARALLEL using threads (like reference implementation)
        # BATCH_SIZE=4: each batch has 4 segments processed together
        # NUM_THREADS: dynamic based on CPU cores (reference project uses max(1, cpu_count - 2))
        BATCH_SIZE = 4
        NUM_THREADS = max(1, os.cpu_count() - 2)  # Leave 2 cores for system, minimum 1 thread
        
        # Create batches
        segment_batches = []
        for i in range(0, num_segments, BATCH_SIZE):
            batch = segments[i:i + BATCH_SIZE]
            segment_batches.append(batch)
        
        # Process batches in parallel
        def process_batch(batch_segments):
            """Process one batch of segments through the model."""
            # CRITICAL: Set num_threads=1 inside each thread to prevent OpenMP conflicts
            torch.set_num_threads(1)
            batch_array = np.stack(batch_segments, axis=0)
            with torch.no_grad():
                embeddings = model.get_audio_embedding_from_data(
                    x=batch_array,
                    use_tensor=False
                )
            return embeddings
        
        all_embeddings = []
        
        # Use ThreadPoolExecutor to process batches in parallel
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(process_batch, batch): i 
                             for i, batch in enumerate(segment_batches)}
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    embeddings = future.result()
                    all_embeddings.append(embeddings)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    raise
        
        # Combine and average all embeddings
        all_embeddings = np.vstack(all_embeddings)
        avg_embedding = np.mean(all_embeddings, axis=0)
        
        # Normalize
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
    Get CLAP embedding for a text query.
    
    Args:
        query_text: Natural language query
        
    Returns:
        512-dim normalized embedding vector or None if failed
    """
    if not config.CLAP_ENABLED:
        return None
    
    try:
        import torch
        import transformers.models.roberta.modeling_roberta as roberta_module
        
        model = get_clap_model()
        
        # Patch RoBERTa forward to handle dimension issues
        if not hasattr(roberta_module, '_original_roberta_forward'):
            roberta_module._original_roberta_forward = roberta_module.RobertaModel.forward
            
            def patched_forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                              position_ids=None, head_mask=None, inputs_embeds=None,
                              encoder_hidden_states=None, encoder_attention_mask=None,
                              past_key_values=None, use_cache=None, output_attentions=None,
                              output_hidden_states=None, return_dict=None):
                if input_ids is not None and input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask is not None and attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                if token_type_ids is not None and token_type_ids.dim() == 1:
                    token_type_ids = token_type_ids.unsqueeze(0)
                if inputs_embeds is not None and inputs_embeds.dim() == 2:
                    inputs_embeds = inputs_embeds.unsqueeze(0)
                
                return roberta_module._original_roberta_forward(
                    self, input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids, position_ids=position_ids,
                    head_mask=head_mask, inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values, use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            
            roberta_module.RobertaModel.forward = patched_forward
        
        with torch.no_grad():
            text_embedding = model.get_text_embedding([query_text], use_tensor=False)
        
        # Ensure 1D numpy array
        if isinstance(text_embedding, np.ndarray):
            if text_embedding.ndim == 2:
                text_embedding = text_embedding[0]
        else:
            text_embedding = np.array(text_embedding).flatten()
        
        # Normalize
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        return text_embedding
        
    except Exception as e:
        logger.error(f"Failed to get text embedding for '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return None


def is_clap_available() -> bool:
    """Check if CLAP is enabled and model can be loaded."""
    if not config.CLAP_ENABLED:
        return False
    
    if _clap_model is not None:
        return True
    
    return initialize_clap_model()
