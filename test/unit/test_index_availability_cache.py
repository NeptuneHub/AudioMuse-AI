# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""AvailabilityCache memo semantics: TTL, idle drop and invalidation.

Guards the per-server availability memo that the hyperbolic and neural
fingerprint indexes keep in front of the mask builder. The entries hold one id
string per catalogue track, so both halves matter: a hit must be served
without rebuilding, and an idle cache must let that memory go.

Main Features:
* TTL accounting is measured from when the value was stored, not from when the
  lookup began, so a slow build is not born already expired.
* The idle drop is armed for exactly one TTL, the only window that never
  discards a still-servable entry and never keeps an unservable one.
* invalidate clears per server and globally.
"""

import time

import pytest

from tasks.index_availability import AvailabilityCache


class TestAvailabilityCacheTtl:
    def test_second_lookup_inside_ttl_is_served_from_the_memo(self):
        cache = AvailabilityCache(ttl_seconds=30.0)
        calls = []

        def build():
            calls.append(1)
            return frozenset({'a'})

        assert cache.get('s1', 'scope', build) == frozenset({'a'})
        assert cache.get('s1', 'scope', build) == frozenset({'a'})
        assert len(calls) == 1

    def test_a_slow_build_is_not_born_expired(self):
        cache = AvailabilityCache(ttl_seconds=0.4)
        calls = []

        def slow_build():
            calls.append(1)
            time.sleep(0.5)
            return frozenset({'a'})

        cache.get('s1', 'scope', slow_build)
        assert cache.get('s1', 'scope', slow_build) == frozenset({'a'})
        assert len(calls) == 1, "build duration must not be charged against the TTL"

    def test_entry_past_its_ttl_is_rebuilt(self):
        cache = AvailabilityCache(ttl_seconds=0.2)
        calls = []

        def build():
            calls.append(1)
            return frozenset({'a'})

        cache.get('s1', 'scope', build)
        time.sleep(0.35)
        cache.get('s1', 'scope', build)
        assert len(calls) == 2

    def test_distinct_servers_and_scopes_do_not_share_an_entry(self):
        cache = AvailabilityCache(ttl_seconds=30.0)
        cache.get('s1', 'scope_a', lambda: frozenset({'a'}))
        assert cache.get('s2', 'scope_a', lambda: frozenset({'b'})) == frozenset({'b'})
        assert cache.get('s1', 'scope_b', lambda: frozenset({'c'})) == frozenset({'c'})
        assert len(cache) == 3


class TestAvailabilityCacheIdleDrop:
    def test_idle_cache_releases_its_entries(self):
        cache = AvailabilityCache(ttl_seconds=0.3)
        cache.get('s1', 'scope', lambda: frozenset({'a'}))
        assert len(cache) == 1

        deadline = time.time() + 10
        while len(cache) and time.time() < deadline:
            time.sleep(0.05)
        assert len(cache) == 0, "an idle memo must not stay resident"

    def test_the_drop_never_fires_while_an_entry_is_still_servable(self):
        cache = AvailabilityCache(ttl_seconds=1.0)
        calls = []

        def build():
            calls.append(1)
            return frozenset({'a'})

        cache.get('s1', 'scope', build)
        time.sleep(0.3)
        cache.get('s1', 'scope', build)
        assert len(calls) == 1, "a live entry was dropped before its TTL lapsed"

    def test_a_query_after_the_drop_rebuilds(self):
        cache = AvailabilityCache(ttl_seconds=0.3)
        calls = []

        def build():
            calls.append(1)
            return frozenset({'a'})

        cache.get('s1', 'scope', build)
        deadline = time.time() + 10
        while len(cache) and time.time() < deadline:
            time.sleep(0.05)
        assert cache.get('s1', 'scope', build) == frozenset({'a'})
        assert len(calls) == 2


class TestAvailabilityCacheInvalidate:
    def test_invalidate_one_server_leaves_the_others(self):
        cache = AvailabilityCache(ttl_seconds=30.0)
        cache.get('s1', 'scope', lambda: frozenset({'a'}))
        cache.get('s2', 'scope', lambda: frozenset({'b'}))
        cache.invalidate('s1')
        assert len(cache) == 1
        assert cache.get('s2', 'scope', lambda: pytest.fail("s2 was evicted")) == frozenset({'b'})

    def test_invalidate_all_clears_every_server(self):
        cache = AvailabilityCache(ttl_seconds=30.0)
        cache.get('s1', 'scope', lambda: frozenset({'a'}))
        cache.get('s2', 'scope', lambda: frozenset({'b'}))
        cache.invalidate()
        assert len(cache) == 0


class TestAvailabilityCacheHeapRelease:
    def test_the_idle_drop_asks_the_heap_back_from_the_os(self):
        import tasks.memory_utils as memory_utils

        calls = []
        original = memory_utils.release_memory_to_os
        memory_utils.release_memory_to_os = lambda: calls.append(1)
        try:
            cache = AvailabilityCache(ttl_seconds=0.3)
            cache.get('s1', 'scope', lambda: frozenset({'a'}))
            deadline = time.time() + 10
            while len(cache) and time.time() < deadline:
                time.sleep(0.05)
            assert len(cache) == 0
            assert calls, "clearing the dict alone may not lower resident memory"
        finally:
            memory_utils.release_memory_to_os = original

    def test_a_sweep_landing_early_keeps_a_still_servable_entry(self):
        cache = AvailabilityCache(ttl_seconds=30.0)
        cache.get('s1', 'scope', lambda: frozenset({'a'}))

        cache._idle_timer.arm(0.05, cache._drop_idle_entries)
        time.sleep(0.5)

        assert len(cache) == 1, "a still-servable entry must not be wiped by a racing sweep"

    def test_an_entry_kept_by_an_early_sweep_is_still_drained_later(self):
        cache = AvailabilityCache(ttl_seconds=0.4)
        cache.get('s1', 'scope', lambda: frozenset({'a'}))

        # Fire a sweep while the entry is still fresh. It is kept, so the sweep
        # must reschedule itself: without that, nothing else is coming and the
        # entry sits resident for the life of the process.
        cache._idle_timer.arm(0.05, cache._drop_idle_entries)

        deadline = time.time() + 10
        while len(cache) and time.time() < deadline:
            time.sleep(0.05)
        assert len(cache) == 0, "an entry kept by an early sweep was never rescheduled"
