"""
Idle warm-cache for the gte-multilingual-base lyrics-search model.

Mirrors the CLAP text-search warmup pattern (tasks/clap_text_search.py): the
Flask page warms the model on load, searches auto-warm and reset the timer, and
a background worker unloads the model after an idle period to free RAM.

The lock is a REENTRANT lock that doubles as the model-use mutex: a search holds
it across warmup + inference, and the unload worker must hold it to unload. That
prevents the background unload (which tears down the ONNX session) from running
while an in-flight ``session.run()`` is using the model.
"""

import logging
import threading
import time
from typing import Dict

import config

logger = logging.getLogger(__name__)

_WARM = {
    'expiry_time': None,
    'timer_thread': None,
    'lock': threading.RLock(),
}


def warm_lock() -> threading.RLock:
    """Return the reentrant model-use lock; hold it across warmup + inference."""
    return _WARM['lock']


def _unload_timer_worker():
    """Unload the gte model once the idle timer expires.

    The expiry re-check AND the unload happen while holding the lock, and a
    search holds the same lock across warmup + inference. So a search that just
    reset the timer cancels the unload (expiry is re-read under the lock), and
    the unload can never run concurrently with an in-flight embedding call.
    """
    global _WARM
    while True:
        with _WARM['lock']:
            expiry = _WARM['expiry_time']
            if expiry is None:
                break
            if expiry - time.time() <= 0:
                from lyrics import gte_onnx
                if gte_onnx.is_loaded():
                    logger.info("GTE warm cache expired - unloading lyrics embedding model")
                    gte_onnx.reset_session()
                    try:
                        from tasks.memory_utils import comprehensive_memory_cleanup
                        comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=True)
                    except Exception as e:
                        logger.debug("GTE unload cleanup failed: %s", e)
                _WARM['expiry_time'] = None
                _WARM['timer_thread'] = None
                break
            time_remaining = expiry - time.time()
        time.sleep(min(1.0, max(0.05, time_remaining)))


def warmup_gte_model() -> Dict:
    """Load the gte model if needed and (re)start the idle-unload timer."""
    global _WARM
    from lyrics import gte_onnx

    if not gte_onnx.is_loaded():
        logger.info("Warming up gte-multilingual-base lyrics-search model...")
        try:
            gte_onnx.load_gte_model()
        except Exception as e:
            logger.error("GTE warmup failed: %s", e)
            return {'loaded': False, 'expiry_seconds': 0}

    duration = config.LYRICS_GTE_WARMUP_DURATION
    with _WARM['lock']:
        _WARM['expiry_time'] = time.time() + duration
        if _WARM['timer_thread'] is None or not _WARM['timer_thread'].is_alive():
            thread = threading.Thread(target=_unload_timer_worker, daemon=True)
            thread.start()
            _WARM['timer_thread'] = thread
            logger.info("Started GTE warm cache timer (%ss)", duration)

    return {'loaded': True, 'expiry_seconds': duration}


def get_gte_warm_status() -> Dict:
    """Return whether the gte model is warm and seconds until idle-unload."""
    global _WARM
    from lyrics import gte_onnx

    with _WARM['lock']:
        expiry = _WARM['expiry_time']

    if expiry is None or not gte_onnx.is_loaded():
        return {'active': False, 'seconds_remaining': 0}
    return {'active': True, 'seconds_remaining': max(0, int(expiry - time.time()))}
