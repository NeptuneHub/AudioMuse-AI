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
# REQUIRED: OpenMP/MKL threading causes deadlocks in forked worker processes
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
    """Load CLAP model with GPU support and automatic CPU fallback."""
    import laion_clap
    import torch
    import gc
    
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    if use_gpu:
        logger.info("GPU detected - loading CLAP model on CUDA")
        torch.cuda.empty_cache()
    else:
        logger.info("No GPU detected - loading CLAP model on CPU")
        torch.set_num_threads(1)
    
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
    
    # CRITICAL: Memory optimization
    # 1. Disable all gradients (saves ~50% memory)
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Move model to appropriate device (GPU or CPU)
    model = model.to(device)
    
    # 3. Delete optimizer states and unnecessary buffers
    if hasattr(model, 'optimizer'):
        del model.optimizer
    
    # 4. Run garbage collection
    gc.collect()
    
    logger.info(f"✓ CLAP model loaded on {device} with memory optimization")
    
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
        # DISABLED: Testing PyTorch internal threading instead
        # from concurrent.futures import ThreadPoolExecutor, as_completed
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
        
        # Check if using GPU
        use_gpu = torch.cuda.is_available()
        
        if use_gpu:
            # GPU: Process in batches sequentially (GPU operations are not thread-safe)
            logger.info(f"CLAP: Processing {num_segments} segments on GPU in batches")
            
            # Process segments in batches for efficiency
            BATCH_SIZE = 4  # Process multiple segments at once on GPU
            all_embeddings = []
            
            for i in range(0, num_segments, BATCH_SIZE):
                batch = segments[i:i + BATCH_SIZE]
                batch_array = np.stack(batch, axis=0)
                
                with torch.no_grad():
                    embeddings = model.get_audio_embedding_from_data(
                        x=batch_array,
                        use_tensor=False
                    )
                all_embeddings.append(embeddings)
            
            # Combine all batch embeddings
            all_embeddings = np.vstack(all_embeddings)
            
        else:
            # CPU: Use multi-threading for parallel processing
            # Use physical CPU cores only (excluding hyperthreading) for optimal performance
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            NUM_THREADS = max(1, physical_cores - 1)
            BATCH_SIZE = 1  # Each batch = 1 segment, threads grab next segment when done
            
            logger.info(f"CLAP: Processing {num_segments} segments with {NUM_THREADS} CPU threads, batch_size={BATCH_SIZE}")
            
            # Create batches
            segment_batches = []
            for i in range(0, num_segments, BATCH_SIZE):
                batch = segments[i:i + BATCH_SIZE]
                segment_batches.append(batch)
            
            # Process batches in parallel
            def process_batch(batch_segments):
                """Process one batch of segments through the model."""
                import torch
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
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # CRITICAL: Explicit thread cleanup to prevent leaks in long-running workers
            executor = None
            try:
                executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
                
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
            finally:
                # CRITICAL: Force immediate shutdown of all threads
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                    # Give threads time to actually terminate
                    import time
                    time.sleep(0.1)
                
                # Force cleanup of any lingering thread-local storage
                import threading
                thread_count_before = threading.active_count()
                
                # Aggressive thread cleanup
                import gc
                gc.collect()
                
                thread_count_after = threading.active_count()
                if thread_count_after > thread_count_before:
                    logger.warning(f"Thread leak detected: {thread_count_before} -> {thread_count_after} active threads")
            
            # Combine all batch embeddings
            all_embeddings = np.vstack(all_embeddings)
        
        # all_embeddings is already the right shape from model
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
        device = next(model.parameters()).device  # Get device from model
        
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
        
        # If on GPU, ensure result is moved back to CPU
        if isinstance(text_embedding, torch.Tensor):
            text_embedding = text_embedding.cpu().numpy()
        
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
