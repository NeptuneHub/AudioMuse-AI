"""
Memory management and data sanitization utilities for AudioMuse AI.

This module provides utilities to address two critical issues:
1. PostgreSQL NUL byte errors from corrupted metadata
2. ONNX Runtime GPU memory allocation failures from fragmentation

Key functions:
- sanitize_string_for_db: Remove NULL bytes and control characters before DB writes
- cleanup_cuda_memory: Force CUDA cache clearing and garbage collection
- cleanup_onnx_session: Explicit session disposal with immediate GC
- handle_onnx_memory_error: Detect allocation errors, trigger cleanup, enable retry
- SessionRecycler: Recreate sessions every N tracks to prevent cumulative leaks
"""

import gc
import logging
import re
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


def sanitize_string_for_db(value: Optional[str]) -> Optional[str]:
    """
    Remove NULL bytes (0x00) and control characters from strings before database writes.
    
    PostgreSQL TEXT/VARCHAR columns reject strings containing NULL bytes (0x00), which
    can appear in corrupted metadata from music files. This function sanitizes strings
    to prevent database insertion errors.
    
    Args:
        value: String to sanitize (can be None)
        
    Returns:
        Sanitized string with NULL bytes and control characters removed, or None if input is None
        
    Examples:
        >>> sanitize_string_for_db("Tyler, The Creator\x00YoungBoy")
        "Tyler, The CreatorYoungBoy"
        >>> sanitize_string_for_db(None)
        None
        >>> sanitize_string_for_db("")
        ""
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        # Convert to string if not already
        value = str(value)
    
    # Remove NULL bytes (0x00)
    value = value.replace('\x00', '')
    
    # Remove other control characters (0x01-0x1F except newline, tab, carriage return)
    # Keep: \t (0x09), \n (0x0A), \r (0x0D)
    # Remove: 0x01-0x08, 0x0B-0x0C, 0x0E-0x1F
    value = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F]', '', value)
    
    return value


def cleanup_cuda_memory(force: bool = False) -> bool:
    """
    Force CUDA cache clearing and garbage collection to free GPU memory.
    
    ONNX Runtime with CUDA can accumulate memory fragmentation over many inferences,
    leading to BFCArena allocation failures. This function forces cleanup.
    
    Args:
        force: If True, performs aggressive cleanup including cache emptying
        
    Returns:
        True if CUDA cleanup was performed, False if CUDA not available
        
    Note:
        This is a heavy operation and should be used strategically:
        - After completing analysis of a batch of tracks
        - After an allocation error occurs
        - Between albums or at periodic intervals
    """
    cuda_cleanup_performed = False
    
    # Try PyTorch cleanup first (if available)
    try:
        import torch
        if torch.cuda.is_available():
            if force:
                # Aggressive cleanup: empty cache completely
                torch.cuda.empty_cache()
                logger.debug("PyTorch CUDA cache emptied")
            else:
                # Standard cleanup: synchronize and collect
                torch.cuda.synchronize()
                logger.debug("PyTorch CUDA synchronize completed")
            cuda_cleanup_performed = True
    except ImportError:
        # PyTorch not available, try alternative methods
        pass
    except Exception as e:
        logger.warning(f"Error during PyTorch CUDA cleanup: {e}")
    
    # Try CuPy cleanup if PyTorch failed/unavailable
    if not cuda_cleanup_performed:
        try:
            import cupy
            if force:
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
                logger.debug("CuPy memory pool cleared")
            cuda_cleanup_performed = True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error during CuPy CUDA cleanup: {e}")
    
    # Always run garbage collection
    gc.collect()
    
    if not cuda_cleanup_performed:
        logger.debug("No CUDA cleanup libraries available (PyTorch/CuPy)")
    
    return cuda_cleanup_performed


def cleanup_onnx_session(session, name: str = "session") -> None:
    """
    Explicit ONNX session disposal with immediate garbage collection.
    
    Properly disposing of ONNX Runtime sessions helps prevent memory leaks,
    especially with GPU providers where resources may not be immediately released.
    
    Args:
        session: ONNX Runtime InferenceSession to dispose
        name: Human-readable name for logging
        
    Example:
        >>> session = ort.InferenceSession(model_path)
        >>> # ... use session ...
        >>> cleanup_onnx_session(session, "embedding_model")
    """
    if session is None:
        return
    
    try:
        # ONNX Runtime InferenceSession doesn't have explicit close/dispose,
        # but deleting the reference and forcing GC helps
        del session
        gc.collect()
        logger.debug(f"Cleaned up ONNX session: {name}")
    except Exception as e:
        logger.warning(f"Error cleaning up ONNX session {name}: {e}")


def cleanup_tensors(*tensor_vars) -> None:
    """
    Explicit tensor cleanup with immediate garbage collection.
    
    Deletes multiple tensor variables and forces garbage collection.
    This is critical for large tensors like mel-spectrograms and embeddings.
    
    Args:
        *tensor_vars: Variable names to delete from caller's scope
        
    Example:
        >>> mel_list = [...]
        >>> mel_batch = np.array(...)
        >>> cleanup_tensors('mel_list', 'mel_batch')  # Cleans caller's variables
    """
    import inspect
    
    # Get caller's frame to delete variables in their scope
    frame = inspect.currentframe().f_back
    
    for var_name in tensor_vars:
        if var_name in frame.f_locals:
            try:
                del frame.f_locals[var_name]
                logger.debug(f"Deleted tensor variable: {var_name}")
            except Exception as e:
                logger.warning(f"Failed to delete tensor {var_name}: {e}")
        elif var_name in frame.f_globals:
            try:
                del frame.f_globals[var_name]
                logger.debug(f"Deleted global tensor variable: {var_name}")
            except Exception as e:
                logger.warning(f"Failed to delete global tensor {var_name}: {e}")
    
    # Force garbage collection after deletions
    gc.collect()


def reset_onnx_memory_pool() -> bool:
    """
    Reset ONNX Runtime CUDA memory pool to clear accumulated allocations.
    
    ONNX Runtime's BFCArena can accumulate memory fragmentation over many inferences.
    This function attempts to reset the memory pool by triggering internal cleanup.
    
    Returns:
        True if reset was attempted, False if not supported
        
    Note:
        This is an experimental function that uses internal ONNX Runtime mechanisms.
        Results may vary across ONNX Runtime versions.
    """
    try:
        import onnxruntime as ort
        
        # Try to access CUDA provider's memory management (if available)
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            # Force garbage collection first
            gc.collect()
            
            # Create and immediately delete a minimal session to trigger cleanup
            # This forces ONNX Runtime to cleanup its internal caches
            try:
                import tempfile
                import onnx
                from onnx import helper, TensorProto
                
                # Create minimal ONNX model
                input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
                output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
                identity_node = helper.make_node('Identity', ['input'], ['output'], name='identity')
                graph = helper.make_graph([identity_node], 'reset_graph', [input_tensor], [output_tensor])
                model = helper.make_model(graph, producer_name='memory_reset')
                
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as tmp_file:
                    onnx.save(model, tmp_file.name)
                    
                    # Create and immediately destroy session to force cleanup
                    temp_session = ort.InferenceSession(tmp_file.name, providers=['CUDAExecutionProvider'])
                    del temp_session
                    gc.collect()
                    
                    logger.debug("ONNX CUDA memory pool reset attempted")
                    return True
                    
            except Exception as e:
                logger.debug(f"Detailed ONNX memory reset failed: {e}")
                # Fallback to simple garbage collection
                gc.collect()
                return True
        else:
            logger.debug("CUDA provider not available for memory pool reset")
            return False
            
    except ImportError as e:
        logger.debug(f"ONNX Runtime not available for memory pool reset: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error resetting ONNX memory pool: {e}")
        return False


def comprehensive_memory_cleanup(force_cuda: bool = True, reset_onnx_pool: bool = True) -> Dict[str, bool]:
    """
    Perform comprehensive memory cleanup combining all available methods.
    
    Args:
        force_cuda: Whether to perform aggressive CUDA cache cleanup
        reset_onnx_pool: Whether to attempt ONNX memory pool reset
        
    Returns:
        Dict with cleanup results: {'cuda': bool, 'onnx_pool': bool, 'gc': bool}
        
    Example:
        >>> results = comprehensive_memory_cleanup()
        >>> if not results['cuda']:
        ...     logger.warning("CUDA cleanup failed")
    """
    results = {
        'cuda': False,
        'onnx_pool': False,
        'gc': True
    }
    
    # CUDA cleanup
    if force_cuda:
        results['cuda'] = cleanup_cuda_memory(force=True)
    
    # ONNX memory pool reset
    if reset_onnx_pool:
        results['onnx_pool'] = reset_onnx_memory_pool()
    
    # Final garbage collection
    gc.collect()
    
    successful_cleanups = sum(results.values())
    logger.info(f"Comprehensive cleanup completed: {successful_cleanups}/3 methods successful")
    
    return results


def handle_onnx_memory_error(
    error: Exception,
    context: str,
    cleanup_func: Optional[Callable] = None,
    retry_func: Optional[Callable] = None,
    fallback_to_cpu: bool = False,
    session_creator: Optional[Callable] = None
) -> Optional[Any]:
    """
    Detect ONNX memory allocation errors, trigger cleanup, and optionally retry or fallback to CPU.
    
    ONNX Runtime GPU memory allocation failures manifest as:
    - "Failed to allocate memory for requested buffer"
    - BFCArena allocation errors
    
    This function detects these errors, performs cleanup, and can retry the operation.
    If fallback_to_cpu is True and session_creator is provided, it will recreate the
    session with CPUExecutionProvider instead of retrying with the same session.
    
    Args:
        error: The exception that was raised
        context: Human-readable context (e.g., "embedding inference for track X")
        cleanup_func: Optional function to call for cleanup before retry
        retry_func: Optional function to call for retry (should return result)
        fallback_to_cpu: If True, recreate session with CPU provider on OOM
        session_creator: Callable that returns (new_session, provider) for CPU fallback
        
    Returns:
        Result from retry_func if retry successful, or tuple (result, provider) if fallback_to_cpu
        None if no retry or retry failed
        
    Raises:
        Original exception if it's not a memory error or retry is not configured
        
    Example:
        >>> try:
        ...     result = session.run(outputs, inputs)
        ... except Exception as e:
        ...     result = handle_onnx_memory_error(
        ...         e,
        ...         "embedding inference",
        ...         cleanup_func=lambda: cleanup_cuda_memory(force=True),
        ...         retry_func=lambda: session.run(outputs, inputs)
        ...     )
    """
    error_str = str(error)
    
    # Check if this is a memory allocation error
    is_memory_error = (
        "Failed to allocate memory" in error_str or
        "BFCArena" in error_str or
        "OOM" in error_str or
        "out of memory" in error_str.lower()
    )
    
    if not is_memory_error:
        # Not a memory error, re-raise
        raise error
    
    logger.warning(f"GPU memory allocation error detected in {context}: {error_str}")
    
    # Perform cleanup if provided
    if cleanup_func:
        try:
            logger.info(f"Performing cleanup for {context}...")
            cleanup_func()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed for {context}: {cleanup_error}")
    
    # Fallback to CPU if requested
    if fallback_to_cpu and session_creator:
        try:
            logger.info(f"Falling back to CPU for {context}...")
            new_session, provider = session_creator()
            logger.info(f"Successfully created CPU session for {context}")
            
            # Retry with new CPU session if retry_func provided
            if retry_func:
                result = retry_func()
                logger.info(f"CPU fallback successful for {context}")
                return result, new_session, provider
            else:
                return None, new_session, provider
        except Exception as fallback_error:
            logger.error(f"CPU fallback failed for {context}: {fallback_error}")
            raise fallback_error
    
    # Retry if retry function provided (without CPU fallback)
    if retry_func:
        try:
            logger.info(f"Retrying {context} after cleanup...")
            result = retry_func()
            logger.info(f"Retry successful for {context}")
            return result
        except Exception as retry_error:
            logger.error(f"Retry failed for {context}: {retry_error}")
            raise retry_error
    else:
        # No retry function, re-raise
        raise error


class SessionRecycler:
    """
    Recreate ONNX Runtime sessions every N tracks to prevent cumulative memory leaks.
    
    Even with proper cleanup, ONNX Runtime sessions can accumulate memory over many
    inferences due to internal caching and fragmentation. This class tracks usage
    and recreates sessions periodically.
    
    Usage:
        >>> recycler = SessionRecycler(recycle_interval=20)
        >>> 
        >>> # Initial session creation
        >>> session = ort.InferenceSession(model_path)
        >>> 
        >>> # In processing loop
        >>> for track in tracks:
        ...     if recycler.should_recycle():
        ...         cleanup_onnx_session(session, "embedding")
        ...         session = ort.InferenceSession(model_path)
        ...         recycler.mark_recycled()
        ...     
        ...     result = session.run(outputs, inputs)
        ...     recycler.increment()
    """
    
    def __init__(self, recycle_interval: int = 20):
        """
        Initialize session recycler.
        
        Args:
            recycle_interval: Number of uses before recycling (default: 20 tracks)
        """
        self.recycle_interval = recycle_interval
        self.use_count = 0
        logger.info(f"SessionRecycler initialized with interval={recycle_interval}")
    
    def increment(self) -> None:
        """Increment the usage counter (call after each use)."""
        self.use_count += 1
    
    def should_recycle(self) -> bool:
        """
        Check if session should be recycled based on usage count.
        
        Returns:
            True if use_count >= recycle_interval
        """
        return self.use_count >= self.recycle_interval
    
    def mark_recycled(self) -> None:
        """Reset the counter after recycling (call after creating new session)."""
        old_count = self.use_count
        self.use_count = 0
        logger.info(f"Session recycled after {old_count} uses")
    
    def get_use_count(self) -> int:
        """Get current usage count."""
        return self.use_count
    
    def reset(self) -> None:
        """Reset counter to zero (e.g., at start of new album)."""
        self.use_count = 0
