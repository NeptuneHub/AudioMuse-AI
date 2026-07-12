# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Memory-reclamation helpers for GPU, ONNX and process heap.

Shared cleanup utilities used across analysis and embedding tasks to release
memory between jobs and recover from allocator pressure. Reclaims through
legitimate frees only; it never sets MALLOC_ARENA_MAX or otherwise tampers with
glibc allocator defaults.

Main Features:
* comprehensive_memory_cleanup: one-shot gc + optional CUDA (PyTorch/CuPy) cache
  release + ONNX pool reset + Linux malloc_trim(0) to return freed heap to the OS.
* handle_onnx_memory_error: detects ONNX/GPU OOM strings and drives cleanup,
  retry, or CPU-session fallback.
* SessionRecycler: interval counter that signals when to rebuild a long-lived
  ONNX session to bound its memory growth.
"""

import gc
import logging
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


def cleanup_cuda_memory(force: bool = False) -> bool:
    cuda_cleanup_performed = False

    try:
        import torch

        if torch.cuda.is_available():
            if force:
                torch.cuda.empty_cache()
                logger.debug("PyTorch CUDA cache emptied")
            else:
                torch.cuda.synchronize()
                logger.debug("PyTorch CUDA synchronize completed")
            cuda_cleanup_performed = True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error during PyTorch CUDA cleanup: {e}")

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

    gc.collect()

    if not cuda_cleanup_performed:
        logger.debug("No CUDA cleanup libraries available (PyTorch/CuPy)")

    return cuda_cleanup_performed


def cleanup_onnx_session(session, name: str = "session") -> None:
    if session is None:
        return

    try:
        del session
        gc.collect()
        logger.debug(f"Cleaned up ONNX session: {name}")
    except Exception as e:
        logger.warning(f"Error cleaning up ONNX session {name}: {e}")


def reset_onnx_memory_pool() -> bool:
    try:
        import onnxruntime as ort

        gc.collect()

        providers = ort.get_available_providers()
        preferred_provider = None

        if 'CUDAExecutionProvider' in providers:
            preferred_provider = 'CUDAExecutionProvider'
            logger.debug("Using CUDA provider for ONNX memory pool reset")
        elif 'MIGraphXExecutionProvider' in providers:
            preferred_provider = 'MIGraphXExecutionProvider'
            logger.debug("Using MIGraphX provider for ONNX memory pool reset (AMD ROCm)")
        elif 'CPUExecutionProvider' in providers:
            preferred_provider = 'CPUExecutionProvider'
            logger.debug("Using CPU provider for ONNX memory pool reset")
        else:
            logger.debug("No suitable ONNX provider found for memory pool reset")
            return False

        try:
            import tempfile
            import onnx
            from onnx import helper, TensorProto

            input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
            output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
            identity_node = helper.make_node('Identity', ['input'], ['output'], name='identity')
            graph = helper.make_graph(
                [identity_node], 'reset_graph', [input_tensor], [output_tensor]
            )
            model = helper.make_model(graph, producer_name='memory_reset')

            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as tmp_file:
                onnx.save(model, tmp_file.name)

                temp_session = ort.InferenceSession(tmp_file.name, providers=[preferred_provider])
                del temp_session
                gc.collect()

                logger.debug(f"ONNX {preferred_provider} memory pool reset attempted")
                return True

        except Exception as e:
            logger.debug(f"Detailed ONNX memory reset failed: {e}")
            gc.collect()
            return True

    except ImportError as e:
        logger.debug(f"ONNX Runtime not available for memory pool reset: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error resetting ONNX memory pool: {e}")
        return False


def release_memory_to_os() -> bool:
    gc.collect()
    import platform

    if platform.system() != "Linux":
        return False
    try:
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return False
        ctypes.CDLL(libc_name).malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


def comprehensive_memory_cleanup(
    force_cuda: bool = True, reset_onnx_pool: bool = True
) -> Dict[str, bool]:
    results = {'cuda': False, 'onnx_pool': False, 'gc': True, 'malloc_trim': False}

    if force_cuda:
        results['cuda'] = cleanup_cuda_memory(force=True)

    if reset_onnx_pool:
        results['onnx_pool'] = reset_onnx_memory_pool()

    results['malloc_trim'] = release_memory_to_os()

    return results


def handle_onnx_memory_error(
    error: Exception,
    context: str,
    cleanup_func: Optional[Callable] = None,
    retry_func: Optional[Callable] = None,
    fallback_to_cpu: bool = False,
    session_creator: Optional[Callable] = None,
) -> Optional[Any]:
    error_str = str(error)

    is_memory_error = (
        "Failed to allocate memory" in error_str
        or "BFCArena" in error_str
        or "OOM" in error_str
        or "out of memory" in error_str.lower()
    )

    if not is_memory_error:
        raise error

    logger.warning(f"GPU memory allocation error detected in {context}: {error_str}")

    if cleanup_func:
        try:
            logger.info(f"Performing cleanup for {context}...")
            cleanup_func()
        except Exception:
            logger.exception(f"Cleanup failed for {context}")

    if fallback_to_cpu and session_creator:
        try:
            logger.info(f"Falling back to CPU for {context}...")
            new_session, provider = session_creator()
            logger.info(f"Successfully created CPU session for {context}")

            if retry_func:
                result = retry_func()
                logger.info(f"CPU fallback successful for {context}")
                return result, new_session, provider
            else:
                return None, new_session, provider
        except Exception as fallback_error:
            logger.exception(f"CPU fallback failed for {context}")
            raise fallback_error

    if retry_func:
        try:
            logger.info(f"Retrying {context} after cleanup...")
            result = retry_func()
            logger.info(f"Retry successful for {context}")
            return result
        except Exception as retry_error:
            logger.exception(f"Retry failed for {context}")
            raise retry_error
    else:
        raise error


class SessionRecycler:
    def __init__(self, recycle_interval: int = 20):
        self.recycle_interval = recycle_interval
        self.use_count = 0

    def increment(self) -> None:
        self.use_count += 1

    def should_recycle(self) -> bool:
        return self.use_count >= self.recycle_interval

    def mark_recycled(self) -> None:
        old_count = self.use_count
        self.use_count = 0
        logger.info(f"Session recycled after {old_count} uses")

    def get_use_count(self) -> int:
        return self.use_count

    def reset(self) -> None:
        self.use_count = 0
