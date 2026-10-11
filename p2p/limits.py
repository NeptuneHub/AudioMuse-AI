# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Bandwidth, parallelism and request-rate limits of the P2P node.

Every byte of the /audiomuse-ai/1.0.0 protocol, in both directions, goes
through MeteredStream, which moves it in 16 KiB chunks and charges each chunk
to the upload or download token bucket first. Housekeeping (probing members,
the directory) runs outside the caps and is small by design.

Main Features:
* TokenBucket: a KiB/s budget with a one-second burst; 0 means unlimited
* MeteredStream: read_exactly / write over a Tor stream, byte-counted and
  paced by the buckets
* Limits: the buckets and the trio CapacityLimiters that cap parallel
  downloads and uploads (a full upload limiter answers BUSY), built from the
  P2P_MAX_* settings
* QueryRateLimiter: one budget of QUERY frames per second for every caller
  together (Tor hides who is calling, so there is no per-caller budget)
"""

import trio

import config

CHUNK_BYTES = 16 * 1024
KIB = 1024
QUERY_RATE_PER_SECOND = 40.0
QUERY_BURST_SECONDS = 2.0


class StreamClosed(Exception):
    pass


class TokenBucket:
    def __init__(self, rate_per_second, burst=None, clock=None):
        self.rate = max(0.0, float(rate_per_second))
        self.capacity = float(burst) if burst is not None else max(self.rate, float(CHUNK_BYTES))
        self._clock = clock or trio.current_time
        self._tokens = self.capacity
        self._stamp = None

    @property
    def unlimited(self):
        return self.rate <= 0.0

    def _refill(self):
        now = self._clock()
        if self._stamp is None:
            self._stamp = now
        self._tokens = min(self.capacity, self._tokens + (now - self._stamp) * self.rate)
        self._stamp = now

    def try_take(self, amount):
        if self.unlimited:
            return True
        self._refill()
        if self._tokens >= amount:
            self._tokens -= amount
            return True
        return False

    def wait_seconds(self, amount):
        if self.unlimited:
            return 0.0
        self._refill()
        missing = min(float(amount), self.capacity) - self._tokens
        return 0.0 if missing <= 0 else missing / self.rate

    async def take(self, amount):
        if self.unlimited:
            return
        remaining = float(amount)
        while remaining > 0:
            step = min(remaining, self.capacity)
            delay = self.wait_seconds(step)
            if delay > 0:
                await trio.sleep(delay)
                self._refill()
            self._tokens -= step
            remaining -= step


class MeteredStream:
    def __init__(self, stream, upload, download, counters=None):
        self.stream = stream
        self._upload = upload
        self._download = download
        self._buffer = bytearray()
        self._counters = counters if counters is not None else {}

    def _count(self, key, amount):
        self._counters[key] = self._counters.get(key, 0) + amount

    async def read_exactly(self, size):
        size = int(size)
        while len(self._buffer) < size:
            want = min(CHUNK_BYTES, size - len(self._buffer))
            await self._download.take(want)
            chunk = await self.stream.read(want)
            if not chunk:
                raise StreamClosed('the peer closed the stream mid-frame')
            self._buffer += chunk
            self._count('bytes_in', len(chunk))
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    async def write(self, data):
        view = memoryview(bytes(data))
        for start in range(0, len(view), CHUNK_BYTES):
            chunk = view[start:start + CHUNK_BYTES]
            await self._upload.take(len(chunk))
            await self.stream.write(chunk.tobytes())
            self._count('bytes_out', len(chunk))


class QueryRateLimiter:
    def __init__(self, rate=QUERY_RATE_PER_SECOND, clock=None):
        self._bucket = TokenBucket(rate, rate * QUERY_BURST_SECONDS, clock or trio.current_time)

    def allow(self):
        return self._bucket.try_take(1)


class Limits:
    def __init__(self, upload_kibps=None, download_kibps=None, parallel_uploads=None,
                 parallel_downloads=None, clock=None):
        upload_kibps = config.P2P_MAX_UPLOAD_KIBPS if upload_kibps is None else upload_kibps
        download_kibps = config.P2P_MAX_DOWNLOAD_KIBPS if download_kibps is None else download_kibps
        parallel_uploads = (
            config.P2P_MAX_PARALLEL_UPLOADS if parallel_uploads is None else parallel_uploads
        )
        parallel_downloads = (
            config.P2P_MAX_PARALLEL_DOWNLOADS if parallel_downloads is None else parallel_downloads
        )
        self.upload = TokenBucket(max(0, int(upload_kibps)) * KIB, clock=clock)
        self.download = TokenBucket(max(0, int(download_kibps)) * KIB, clock=clock)
        self.uploads = trio.CapacityLimiter(max(1, int(parallel_uploads)))
        self.downloads = trio.CapacityLimiter(max(1, int(parallel_downloads)))
        self.queries = QueryRateLimiter(clock=clock)
        self.counters = {}

    def meter(self, stream):
        return MeteredStream(stream, self.upload, self.download, self.counters)
