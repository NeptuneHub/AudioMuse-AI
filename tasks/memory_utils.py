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
    try:
        import torch
        if torch.cuda.is_available():
            if force:
                # Aggressive cleanup: empty cache completely
                torch.cuda.empty_cache()
                logger.debug("Forced CUDA cache empty")
            else:
                # Standard cleanup: synchronize and collect
                torch.cuda.synchronize()
                logger.debug("CUDA synchronize completed")
            
            # Always run garbage collection with CUDA cleanup
            gc.collect()
            return True
    except ImportError:
        # torch not available, skip CUDA cleanup
        pass
    except Exception as e:
        logger.warning(f"Error during CUDA cleanup: {e}")
    
    return False


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


def handle_onnx_memory_error(
    error: Exception,
    context: str,
    cleanup_func: Optional[Callable] = None,
    retry_func: Optional[Callable] = None
) -> Optional[Any]:
    """
    Detect ONNX memory allocation errors, trigger cleanup, and optionally retry.
    
    ONNX Runtime GPU memory allocation failures manifest as:
    - "Failed to allocate memory for requested buffer"
    - BFCArena allocation errors
    
    This function detects these errors, performs cleanup, and can retry the operation.
    
    Args:
        error: The exception that was raised
        context: Human-readable context (e.g., "embedding inference for track X")
        cleanup_func: Optional function to call for cleanup before retry
        retry_func: Optional function to call for retry (should return result)
        
    Returns:
        Result from retry_func if retry successful, None if no retry or retry failed
        
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
    
    # Retry if retry function provided
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
