# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content-keyed measurements remain safe with concurrent native requests.

Main Features:
* Measurement profiles stay independent, even with identical audio content.
* Non-ASCII paths work; cached results retain no file handles.
* Zero disables reuse; concurrent failures release waiters and allow retries.
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest

from tasks import smart_fade_cache as cache
from tasks import smart_fade_mix as mix


@pytest.fixture(autouse=True)
def empty_cache():
    cache.clear()
    yield
    cache.clear()


def test_profiles_are_independent_and_handles_close_for_unicode_paths(tmp_path):
    path = tmp_path / 'Beyonc\u00e9 & \u00e9tude.audio'
    path.write_bytes(b'same audio')
    assert cache.remember(path, ('a',), lambda: {'level': 1}) == {'level': 1}
    assert cache.remember(path, ('b',), lambda: {'level': 2}) == {'level': 2}
    renamed = path.with_name('renamed.audio')
    path.rename(renamed)
    assert cache.remember(renamed, ('a',), lambda: pytest.fail('must reuse the content')) == {'level': 1}
    renamed.unlink()


def test_zero_disables_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.config, 'SMART_FADE_LEVEL_CACHE_SIZE', 0)
    path = tmp_path / 'song.audio'
    path.write_bytes(b'audio')
    calls = []

    def measure():
        calls.append(1)
        return {'count': len(calls)}

    assert cache.remember(path, ('a',), measure) == {'count': 1}
    assert cache.remember(path, ('a',), measure) == {'count': 2}


def test_a_failed_shared_measurement_releases_waiters_and_allows_retry(tmp_path, monkeypatch):
    path = tmp_path / 'song.audio'
    path.write_bytes(b'audio')
    ready = threading.Barrier(2)
    waiting = threading.Event()
    key = cache._key
    calls = []

    def together(*args):
        result = key(*args)
        ready.wait(timeout=3)
        return result

    class ObservedFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                waiting.set()
            return super().result(timeout)

    def broken():
        calls.append(1)
        assert waiting.wait(3)
        raise ValueError('bad audio')

    monkeypatch.setattr(cache, '_key', together)
    monkeypatch.setattr(cache, 'Future', ObservedFuture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cache.remember, path, ('a',), broken) for _ in range(2)]
        for future in futures:
            with pytest.raises(ValueError, match='bad audio'):
                future.result(timeout=5)
    assert len(calls) == 1
    monkeypatch.setattr(cache, '_key', key)
    assert cache.remember(path, ('a',), lambda: {'level': 2}) == {'level': 2}


def test_song_energy_is_reused_and_cannot_be_mutated(tmp_path, monkeypatch):
    path = tmp_path / 'song.audio'
    path.write_bytes(b'audio')
    calls = []
    original = mix._measure_levels

    def measured(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(mix, '_measure_levels', measured)
    monkeypatch.setattr(mix, '_load', lambda path, rate: np.zeros(rate))
    first = mix._levels(path)
    second = mix._levels(path)
    assert len(calls) == 1 and first['loudness'] == second['loudness']
    with pytest.raises(ValueError):
        first['energy'][0] = 999.0
