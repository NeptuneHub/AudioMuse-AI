"""
Memory and data sanitization utilities for AudioMuse-AI.

This module provides utilities to:
1. Sanitize strings for PostgreSQL (remove NULL bytes)
2. Manage ONNX Runtime memory and CUDA cleanup
3. Detect and handle memory pressure
"""

import gc
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)


def sanitize_string_for_db(value: Any) -> Optional[str]:
    """
    Sanitize a string value for safe PostgreSQL insertion.
    
    Removes NULL bytes (0x00) which PostgreSQL TEXT/VARCHAR columns reject.
    Also handles None, bytes, and other edge cases.
    
    Args:
        value: Input value (str, bytes, None, or other)
        
    Returns:
        Sanitized string or None if input is None/empty
        
    Examples:
        >>> sanitize_string_for_db("Tyler, The Creator\\x00YoungBoy")
        'Tyler, The CreatorYoungBoy'
        >>> sanitize_string_for_db(None)
        None
        >>> sanitize_string_for_db("")
        None
    """
    if value is None:
        return None
    
    # Convert to string if needed
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8', errors='ignore')
        except Exception:
            logger.warning("Failed to decode bytes to string, returning None")
            return None
    elif not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            logger.warning(f"Failed to convert {type(value)} to string, returning None")
            return None
    
    # Remove NULL bytes (0x00) - PostgreSQL doesn't allow them
    sanitized = value.replace('\x00', '')
    
    # Also remove other control characters that might cause issues
    # Keep only printable characters and common whitespace
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
    
    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()
    
    # Return None for empty strings
    if not sanitized:
        return None
    
    return sanitized


def cleanup_cuda_memory(force: bool = False):
    """
    Explicitly cleanup CUDA memory to prevent fragmentation.
    
    This addresses ONNX Runtime BFCArena allocation failures by:
    1. Forcing Python garbage collection
    2. Clearing CUDA memory cache if PyTorch is available
    3. Synchronizing CUDA operations
    
    Args:
        force: If True, performs aggressive cleanup even if not strictly needed
    """
    # Always run garbage collection first
    gc.collect()
    
    try:
        # Try to clear PyTorch CUDA cache if available
        # PyTorch and ONNX Runtime can share CUDA memory allocator
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if force:
                # Aggressive cleanup: synchronize all devices
                torch.cuda.synchronize()
            logger.debug("✓ Cleared PyTorch CUDA cache")
    except ImportError:
        # PyTorch not available, that's fine - ONNX Runtime has its own allocator
        pass
    except Exception as e:
        logger.debug(f"Could not clear PyTorch CUDA cache: {e}")
    
    # Additional garbage collection after CUDA cleanup
    if force:
        gc.collect()
        gc.collect()  # Run twice for cyclic references


def cleanup_onnx_session(session, session_name: str = "session"):
    """
    Properly cleanup an ONNX Runtime session to free memory.
    
    Args:
        session: ONNX Runtime InferenceSession to cleanup
        session_name: Name for logging purposes
    """
    if session is not None:
        try:
            # Delete the session
            del session
            logger.debug(f"✓ Deleted {session_name}")
        except Exception as e:
            logger.warning(f"Error deleting {session_name}: {e}")
    
    # Force garbage collection
    gc.collect()


def handle_onnx_memory_error(error: Exception, context: str = "inference") -> bool:
    """
    Handle ONNX Runtime memory allocation errors.
    
    Logs the error and performs aggressive cleanup to recover.
    Returns True if cleanup was successful, False if error is unrecoverable.
    
    Args:
        error: The exception that occurred
        context: Description of where the error occurred
        
    Returns:
        True if recovery attempted, False if unrecoverable error
    """
    error_str = str(error).lower()
    
    # Check if this is a memory allocation error
    is_memory_error = any(keyword in error_str for keyword in [
        'allocate', 'memory', 'bfcarena', 'cuda', 'out of memory'
    ])
    
    if is_memory_error:
        logger.warning(f"⚠️ ONNX memory allocation error during {context}: {error}")
        logger.info("🔧 Performing aggressive memory cleanup...")
        
        # Aggressive cleanup
        cleanup_cuda_memory(force=True)
        
        logger.info("✓ Memory cleanup completed - continuing analysis")
        return True
    else:
        # Not a memory error - may be more serious
        logger.error(f"❌ ONNX error during {context}: {error}")
        return False


class SessionRecycler:
    """
    Manages ONNX session lifecycle to prevent memory accumulation.
    
    Recreates sessions periodically to avoid cumulative memory leaks.
    """
    
    def __init__(self, max_uses: int = 50):
        """
        Initialize session recycler.
        
        Args:
            max_uses: Number of inferences before recreating session
        """
        self.max_uses = max_uses
        self.use_count = 0
        self.sessions = {}
        
    def should_recycle(self) -> bool:
        """Check if sessions should be recycled."""
        return self.use_count >= self.max_uses
    
    def increment(self):
        """Increment usage counter."""
        self.use_count += 1
        
    def reset(self):
        """Reset usage counter after recycling."""
        self.use_count = 0
        
    def recycle_sessions(self, session_dict: dict, loader_func: callable) -> dict:
        """
        Recycle ONNX sessions by recreating them.
        
        Args:
            session_dict: Dictionary of current sessions
            loader_func: Function to reload sessions
            
        Returns:
            New session dictionary
        """
        logger.info(f"♻️ Recycling ONNX sessions after {self.use_count} uses")
        
        # Cleanup old sessions
        for name, session in session_dict.items():
            cleanup_onnx_session(session, name)
        
        # Force aggressive cleanup
        cleanup_cuda_memory(force=True)
        
        # Reload sessions
        new_sessions = loader_func()
        
        # Reset counter
        self.reset()
        
        logger.info("✓ Sessions recycled successfully")
        return new_sessions
