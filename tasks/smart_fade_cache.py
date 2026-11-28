# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Bounded reuse of small smart-fade measurements, identified by audio content.

Main Features:
* Reuse measurements across temporary downloads, invalidating changed content.
  A file's digest is read once per path, size and modification time
  (DIGEST_MEMORY of them), so the several measurements of one transition
  hash each download once.
* Separate measurement profiles and keep at most SMART_FADE_LEVEL_CACHE_SIZE
  completed entries; zero disables reuse. Never store audio or open files.
* Share an in-flight calculation between threads, release waiters on failures,
  and let later calls retry. No lock is held during measurement or waiting.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future

import config

_cache = OrderedDict()
_digests = OrderedDict()
_lock = threading.Lock()
DIGEST_MEMORY = 32


def clear():
    with _lock:
        _cache.clear()
        _digests.clear()


def _digest(path):
    status = os.stat(path)
    identity = (os.path.abspath(path), status.st_size, status.st_mtime_ns)
    with _lock:
        known = _digests.get(identity)
        if known is not None:
            _digests.move_to_end(identity)
            return known
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    with _lock:
        _digests[identity] = digest.digest()
        while len(_digests) > DIGEST_MEMORY:
            _digests.popitem(last=False)
    return digest.digest()


def _key(path, profile):
    return _digest(path), profile


def remember(path, profile, measure):
    limit = max(0, config.SMART_FADE_LEVEL_CACHE_SIZE)
    if not limit:
        return measure()
    key = _key(path, profile)
    with _lock:
        future = _cache.get(key)
        owner = future is None
        if owner:
            future = Future()
            _cache[key] = future
        else:
            _cache.move_to_end(key)
    if owner:
        try:
            future.set_result(measure())
        except BaseException as exc:
            future.set_exception(exc)
            with _lock:
                if _cache.get(key) is future:
                    del _cache[key]
            raise
        with _lock:
            for previous in list(_cache):
                if len(_cache) <= limit:
                    break
                if _cache[previous].done():
                    del _cache[previous]
    return dict(future.result())
