# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Songs the page's player plays, downloaded once from the media server and kept on disk a while.

The player asks for a song in pieces (the play route with start and seconds), so
each song is downloaded once into a temporary folder and every play request
reads that copy, even while the media server still delivers it. Transition
analysis also copies the completed download while holding a reference.

Main Features:
* acquire(key, item_id) returns the song's entry, starting its download in a
  thread (inside a copy of the request's server context) the first time; release
  marks it unused. A failed download is dropped, so the next request retries
* GrowingDownload.chunks() reads the file while the provider still writes it and
  ends when the download has ended and every byte was read; head() waits for the
  first bytes
* An entry unused for IDLE_SECONDS is removed, and the oldest unused ones go
  when the copies pass MAX_BYTES; a sweeper thread checks every SWEEP_SECONDS. A
  folder is removed only after its download has ended
"""

import contextlib
import contextvars
import logging
import os
import shutil
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

IDLE_SECONDS = 900
MAX_BYTES = 4 * 2 ** 30
SWEEP_SECONDS = 60
POLL_SECONDS = 0.02
CHUNK_BYTES = 1024 * 1024


class GrowingDownload:
    def __init__(self, item_id, folder):
        os.makedirs(folder)
        self.folder = folder
        self.done = threading.Event()
        self.succeeded = False
        self.thread = threading.Thread(
            target=self._run, args=(contextvars.copy_context(), item_id), name='play-download', daemon=True)
        self.thread.start()

    def _run(self, context, item_id):
        from tasks import mediaserver

        try:
            self.succeeded = bool(context.run(mediaserver.download_track, self.folder, {'Id': item_id, 'id': item_id}))
            if not self.succeeded:
                logger.error('The media server did not deliver %s for playback', item_id)
        except Exception:
            logger.exception('Playback download failed for %s', item_id)
        finally:
            self.done.set()

    def file(self):
        names = os.listdir(self.folder)
        return os.path.join(self.folder, names[0]) if names else None

    def size(self):
        name = self.file()
        try:
            return os.path.getsize(name) if name else 0
        except OSError:
            return 0

    def chunks(self):
        handle = None
        try:
            while True:
                finished = self.done.is_set()
                if handle is None:
                    name = self.file()
                    if name is None:
                        if finished:
                            return
                        time.sleep(POLL_SECONDS)
                        continue
                    handle = open(name, 'rb')
                chunk = handle.read(CHUNK_BYTES)
                if chunk:
                    yield chunk
                elif finished:
                    return
                else:
                    time.sleep(POLL_SECONDS)
        finally:
            if handle is not None:
                handle.close()

    def head(self, size):
        data = b''
        with contextlib.closing(self.chunks()) as chunks:
            for chunk in chunks:
                data += chunk
                if len(data) >= size:
                    break
        return data[:size]


class Entry:
    def __init__(self, item_id, folder):
        self.folder = folder
        self.download = GrowingDownload(item_id, os.path.join(folder, 'song'))
        self.users = 0
        self.last = time.monotonic()

    def failed(self):
        return self.download.done.is_set() and (not self.download.succeeded or not self.download.size())


_entries = {}
_lock = threading.Lock()
_sweeper = None


def _remove(entry):
    if entry.download.done.is_set():
        shutil.rmtree(entry.folder, ignore_errors=True)
        return
    threading.Thread(target=lambda: (entry.download.done.wait(), shutil.rmtree(entry.folder, ignore_errors=True)),
                     name='play-cleanup', daemon=True).start()


def _sweep_locked():
    now = time.monotonic()
    for key, entry in list(_entries.items()):
        if not entry.users and (entry.failed() or now - entry.last > IDLE_SECONDS):
            _remove(_entries.pop(key))
    idle = sorted((entry.last, key) for key, entry in _entries.items() if not entry.users)
    total = sum(entry.download.size() for entry in _entries.values())
    for _, key in idle:
        if total <= MAX_BYTES:
            break
        entry = _entries.pop(key)
        total -= entry.download.size()
        _remove(entry)


def sweep():
    with _lock:
        _sweep_locked()


def _sweep_forever():
    while True:
        time.sleep(SWEEP_SECONDS)
        try:
            sweep()
        except Exception:
            logger.exception('Could not remove the old playback copies')


def acquire(key, item_id):
    global _sweeper
    with _lock:
        _sweep_locked()
        entry = _entries.get(key)
        if entry is None or entry.failed():
            if entry is not None:
                _remove(entry)
            entry = Entry(item_id, tempfile.mkdtemp(prefix='audiomuse_play_'))
            _entries[key] = entry
        entry.users += 1
        entry.last = time.monotonic()
        if _sweeper is None:
            _sweeper = threading.Thread(target=_sweep_forever, name='play-cache-sweep', daemon=True)
            _sweeper.start()
    return entry


def release(entry):
    with _lock:
        entry.users -= 1
        entry.last = time.monotonic()


def clear():
    with _lock:
        for key in list(_entries):
            _remove(_entries.pop(key))
