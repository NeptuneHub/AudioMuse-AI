# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Idle-based warm cache for the GTE lyrics-search embedding model.

Keeps the gte-multilingual-base ONNX session loaded for a bounded window after
a lyrics text search so back-to-back queries avoid repeated cold loads, then
unloads it once the window lapses. Complements tasks.lyrics_manager (which owns
the index caches) by managing only the ONNX model lifetime.

Main Features:
* warmup_gte_model: loads the model on demand and (re)arms the expiry timer.
* _unload_timer_worker: single background thread that unloads gte_onnx and runs
  comprehensive_memory_cleanup once the warm window expires; guarded by an RLock
  shared via warm_lock so warmups and the timer never race.
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
    return _WARM['lock']


def _unload_timer_worker():
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
    from lyrics import gte_onnx

    if not gte_onnx.is_loaded():
        logger.info("Warming up gte-multilingual-base lyrics-search model...")
        try:
            gte_onnx.load_gte_model()
        except Exception:
            logger.exception("GTE warmup failed")
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
    from lyrics import gte_onnx

    with _WARM['lock']:
        expiry = _WARM['expiry_time']

    if expiry is None or not gte_onnx.is_loaded():
        return {'active': False, 'seconds_remaining': 0}
    return {'active': True, 'seconds_remaining': max(0, int(expiry - time.time()))}
