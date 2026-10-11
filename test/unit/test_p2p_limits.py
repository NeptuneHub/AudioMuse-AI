# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Limits of the P2P node (p2p.limits) and the supply requests the runtime accepts.

The upload and download caps apply to every byte of the sharing protocol, so
the metered stream is what the node reads and writes through; these tests
check the buckets with a fake clock and the stream with trio's in-memory
streams and its virtual clock.

Main Features:
* a TokenBucket admits its burst, refills at its rate and is unlimited at 0
* MeteredStream moves data in 16 KiB chunks, charges every chunk to the
  right bucket and counts the bytes; a peer closing mid-frame is StreamClosed
* a capped transfer takes the time the KiB/s cap implies
* QueryRateLimiter is one budget shared by every caller (Tor hides who calls):
  allow takes no peer, the burst is two seconds of the rate and it refills
* Limits reads the four P2P_MAX_* settings (KiB/s and parallel slots) and
  builds the shared query budget
* the runtime accepts only well-formed supply requests with a bounded ttl
"""

import inspect

import pytest
import trio
import trio.testing

import config
from p2p import limits, runtime


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Pipe:
    def __init__(self, stream):
        self.stream = stream
        self.reads = []
        self.writes = []

    async def read(self, size):
        self.reads.append(size)
        return await self.stream.receive_some(size)

    async def write(self, data):
        self.writes.append(len(data))
        await self.stream.send_all(data)


class _Recording(limits.TokenBucket):
    def __init__(self):
        super().__init__(0)
        self.taken = []

    async def take(self, amount):
        self.taken.append(amount)


class TestTokenBucket:
    def test_the_burst_is_available_at_once(self):
        clock = Clock()
        bucket = limits.TokenBucket(100.0, burst=300.0, clock=clock)

        assert bucket.try_take(300)
        assert not bucket.try_take(1)

    def test_it_refills_at_its_rate(self):
        clock = Clock()
        bucket = limits.TokenBucket(100.0, burst=100.0, clock=clock)
        assert bucket.try_take(100)

        clock.now += 0.5
        assert bucket.try_take(50)
        assert not bucket.try_take(1)

    def test_it_never_holds_more_than_its_burst(self):
        clock = Clock()
        bucket = limits.TokenBucket(100.0, burst=100.0, clock=clock)
        bucket.try_take(0)
        clock.now += 1000

        assert bucket.try_take(100)
        assert not bucket.try_take(1)

    def test_zero_means_unlimited(self):
        bucket = limits.TokenBucket(0, clock=Clock())

        assert bucket.unlimited
        assert bucket.try_take(10 ** 12)
        assert bucket.wait_seconds(10 ** 12) == 0.0

    def test_the_default_burst_covers_one_chunk(self):
        assert limits.TokenBucket(1.0, clock=Clock()).capacity == limits.CHUNK_BYTES

    def test_a_capped_transfer_takes_the_implied_time(self):
        clock = trio.testing.MockClock(autojump_threshold=0)

        async def run():
            bucket = limits.TokenBucket(64 * limits.KIB)
            start = trio.current_time()
            await bucket.take(64 * limits.KIB)
            await bucket.take(128 * limits.KIB)
            return trio.current_time() - start

        elapsed = trio.run(run, clock=clock)

        assert elapsed == pytest.approx(2.0, abs=0.05)


class TestMeteredStream:
    def _pair(self):
        left, right = trio.testing.memory_stream_pair()
        return _Pipe(left), _Pipe(right)

    def test_writes_go_out_in_16_kib_chunks_and_are_charged(self):
        async def run():
            pipe, other = self._pair()
            upload, download = _Recording(), _Recording()
            counters = {}
            metered = limits.MeteredStream(pipe, upload, download, counters)
            reader = limits.MeteredStream(other, _Recording(), _Recording())
            data = bytes(range(256)) * 160
            async with trio.open_nursery() as nursery:
                nursery.start_soon(metered.write, data)
                received = await reader.read_exactly(len(data))
            return pipe, upload, download, counters, received == data

        pipe, upload, download, counters, same = trio.run(run)

        assert same
        assert pipe.writes == [16384, 16384, 8192]
        assert upload.taken == [16384, 16384, 8192]
        assert download.taken == []
        assert counters == {'bytes_out': 40960}

    def test_reads_are_charged_to_the_download_bucket(self):
        async def run():
            pipe, other = self._pair()
            download = _Recording()
            counters = {}
            metered = limits.MeteredStream(pipe, _Recording(), download, counters)
            await other.write(b'x' * 20000)
            first = await metered.read_exactly(3)
            rest = await metered.read_exactly(19997)
            return download, counters, first + rest

        download, counters, data = trio.run(run)

        assert data == b'x' * 20000
        assert sum(download.taken) >= 20000
        assert all(amount <= limits.CHUNK_BYTES for amount in download.taken)
        assert counters['bytes_in'] == 20000

    def test_a_close_mid_frame_is_stream_closed(self):
        async def run():
            left, right = trio.testing.memory_stream_pair()
            metered = limits.MeteredStream(_Pipe(left), _Recording(), _Recording())
            await right.send_all(b'abc')
            await right.aclose()
            await metered.read_exactly(10)

        with pytest.raises(limits.StreamClosed):
            trio.run(run)


class TestQueryRateLimiter:
    def test_allow_takes_no_caller_and_nothing_is_kept_per_caller(self):
        assert list(inspect.signature(limits.QueryRateLimiter.allow).parameters) == ['self']
        assert not hasattr(limits.QueryRateLimiter, 'forget')

    def test_one_budget_is_shared_by_every_caller(self):
        clock = Clock()
        limiter = limits.QueryRateLimiter(rate=2, clock=clock)

        allowed = [limiter.allow() for _ in range(10)]

        assert allowed.count(True) == 2 * limits.QUERY_BURST_SECONDS
        assert allowed[:4] == [True] * 4

    def test_the_budget_refills_with_time(self):
        clock = Clock()
        limiter = limits.QueryRateLimiter(rate=1, clock=clock)
        while limiter.allow():
            pass

        assert not limiter.allow()
        clock.now += 1.0

        assert limiter.allow()
        assert not limiter.allow()

    def test_the_default_rate_is_forty_queries_a_second_with_a_two_second_burst(self):
        clock = Clock()
        limiter = limits.QueryRateLimiter(clock=clock)

        allowed = sum(1 for _ in range(200) if limiter.allow())

        assert limits.QUERY_RATE_PER_SECOND == 40.0
        assert allowed == 80


class TestLimitsFromConfig:
    def test_the_settings_become_buckets_and_slots(self, monkeypatch):
        monkeypatch.setattr(config, 'P2P_MAX_UPLOAD_KIBPS', 256)
        monkeypatch.setattr(config, 'P2P_MAX_DOWNLOAD_KIBPS', 0)
        monkeypatch.setattr(config, 'P2P_MAX_PARALLEL_UPLOADS', 3)
        monkeypatch.setattr(config, 'P2P_MAX_PARALLEL_DOWNLOADS', 5)

        built = limits.Limits(clock=Clock())

        assert built.upload.rate == 256 * 1024
        assert built.download.unlimited
        assert built.uploads.total_tokens == 3
        assert built.downloads.total_tokens == 5

    def test_explicit_values_win_over_the_settings(self):
        built = limits.Limits(upload_kibps=1, download_kibps=2, parallel_uploads=1, parallel_downloads=1,
                              clock=Clock())

        assert (built.upload.rate, built.download.rate) == (1024, 2048)

    def test_parallel_slots_are_at_least_one(self):
        built = limits.Limits(upload_kibps=0, download_kibps=0, parallel_uploads=0, parallel_downloads=-3,
                              clock=Clock())

        assert (built.uploads.total_tokens, built.downloads.total_tokens) == (1, 1)

    def test_the_query_budget_is_one_shared_limiter_on_the_same_clock(self):
        clock = Clock()
        built = limits.Limits(upload_kibps=0, download_kibps=0, parallel_uploads=1, parallel_downloads=1,
                              clock=clock)

        assert isinstance(built.queries, limits.QueryRateLimiter)
        while built.queries.allow():
            pass
        clock.now += 1.0
        assert built.queries.allow()

    def test_meter_shares_the_counters(self):
        built = limits.Limits(upload_kibps=0, download_kibps=0, parallel_uploads=1, parallel_downloads=1)

        assert built.meter(object())._counters is built.counters


class TestSupplyRequests:
    def _message(self, **overrides):
        message = {'v': 1, 'op': 'supply', 'rid': 'r1', 'item': 'fp_item', 'server': 's', 'provider': 'p',
                   'wants': ['c', 'l', 'x'], 'reply': 'audiomuse_p2p_r_0123456789abcdef', 'ttl': 5.0}
        message.update(overrides)
        return message

    def test_a_good_request_keeps_only_known_outputs(self):
        request = runtime.valid_supply(self._message())

        assert request['wants'] == ['c', 'l']
        assert request['ttl'] == 5.0

    def test_an_integer_ttl_becomes_seconds(self):
        assert runtime.valid_supply(self._message(ttl=120))['ttl'] == 120.0

    @pytest.mark.parametrize('overrides', [
        {'rid': ''}, {'rid': 'r' * 65}, {'item': None}, {'reply': 'audiomuse_job'}, {'wants': 'c'},
        {'wants': [1]}, {'ttl': 0}, {'ttl': 121}, {'ttl': True}, {'ttl': '5'},
    ])
    def test_malformed_requests_are_refused(self, overrides):
        assert runtime.valid_supply(self._message(**overrides)) is None
