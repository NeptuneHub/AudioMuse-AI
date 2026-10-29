# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""PyAV decode of one audio file, in this process or as a separate child process.

tasks.analysis.song falls back to PyAV when librosa cannot open a file and runs
that decode through run in a child process, so a native crash inside FFmpeg on a
damaged file ends only the child, never the worker or Flask that asked for the
audio. The container and source runs start the child as
``python -m tasks.pyav_decode``; the frozen native apps start their own
executable with ``--decode-audio``, which native_common.frozen_children routes to
run. It sits beside tasks.memory_utils, below the analysis layer that uses it.

Main Features:
* decode_with_pyav skips undecodable packets and every frame whose channel count
  differs from the majority of the first 8 frames (#968, PR #973: a damaged AAC
  decodes frames claiming 42 channels; HE-AAC v2 declares mono, decodes stereo).
  A file keeps min_fraction of its declared duration, or of its packets for raw
  ADTS, whose duration is only a bitrate guess
* Frames go through swresample to PACKED float before they are read: PyAV's planar
  to_ndarray walks plane pointers until a NULL and overruns at 8+ channels, which
  segfaulted on the #968 file and on any valid 7.1 file. A sample-rate change
  rebuilds the resampler at the first (or requested) rate, keeping the timeline
* It averages the channels itself: swresample downmixes power-preserving
  (1/sqrt(2)) while librosa is amplitude-preserving ((L+R)/2), and the two
  decoders must agree or one file yields two embeddings
* Imports nothing from the app but the logging setup: importing config opens a
  database connection, which no child may do per decoded file
* run takes the load cap and the minimum fraction from the command line (the
  parent's values may come from the database), ends itself past wall_seconds plus
  a grace (an orphan never spins) and leaves through os._exit after the reply, so a
  crash in interpreter teardown cannot spoil a complete answer
* The reply goes to the ORIGINAL stdout: one JSON line (status, sample rate,
  sample count) then raw float32 samples; stdout is pointed at stderr first, so a
  stray print cannot corrupt it. Logs go to stderr, a pipe the parent relays
* A Python error from PyAV is status unreadable (marked not analyzable); out of
  memory, a missing file and a permission error are transient (retried later)
* On Windows the crash dialog is off, so a crash ends the process at once
"""

import ctypes
import json
import logging
import os
import signal
import sys
import threading
from collections import Counter

import numpy as np

from app_logging import configure_logging

logger = logging.getLogger(__name__)

STATUS_OK = 'ok'
STATUS_UNREADABLE = 'unreadable'
STATUS_TRANSIENT = 'transient'
CHILD_FLAG = '--decode-audio'

_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002
_DURATION_GUESSED_FORMATS = frozenset({'aac'})
_REFERENCE_FRAMES = 8
_WALL_FLOOR_SECONDS = 300
_ORPHAN_GRACE_SECONDS = 60


def wall_seconds(load_timeout):
    return max(_WALL_FLOOR_SECONDS, int(load_timeout)) if load_timeout else None


def _frame_to_mono_mean(rframe):
    channels = len(rframe.layout.channels)
    arr = rframe.to_ndarray().reshape(-1, channels).mean(axis=1)
    return arr.astype(np.float32, copy=False)


def _declared_seconds(container, load_timeout):
    import av

    if not container.duration:
        return None
    declared = float(container.duration) / av.time_base
    if declared <= 0:
        return None
    return min(declared, load_timeout) if load_timeout else declared


def _enough_survived(audio, sr, declared, name, min_fraction):
    if not declared or not sr or not min_fraction:
        return True
    recovered = audio.size / float(sr)
    if recovered >= declared * min_fraction:
        return True
    logger.error(
        "Only %.1fs of the %.1fs %s declares survived the corrupt packets (%.0f%%, "
        "minimum %.0f%%); treating it as not decodable.",
        recovered, declared, name, 100.0 * recovered / declared,
        100.0 * min_fraction,
    )
    return False


def _enough_packets_survived(stats, name, min_fraction):
    if not stats['skipped'] or not min_fraction:
        return True
    kept = 1.0 - min(stats['skipped'], stats['packets']) / float(max(stats['packets'], 1))
    if kept >= min_fraction:
        return True
    logger.error(
        "Only %.0f%% of the %d packets of %s decoded (minimum %.0f%%); treating it as "
        "not decodable.", 100.0 * kept, stats['packets'], name, 100.0 * min_fraction,
    )
    return False


def _decoded_frames(container, stream, stats):
    import av

    for packet in container.demux(stream):
        stats['packets'] += 1
        try:
            frames = list(packet.decode())
        except av.FFmpegError:
            stats['skipped'] += 1
            continue
        yield from frames


def _matching(frames, channels, stats):
    for frame in frames:
        if len(frame.layout.channels) == channels:
            yield frame
        else:
            stats['skipped'] += 1


def _majority_channels(frames):
    return Counter(len(frame.layout.channels) for frame in frames).most_common(1)[0][0]


def _tolerant_frames(container, stream, stats):
    head = []
    channels = None
    for frame in _decoded_frames(container, stream, stats):
        if channels is not None:
            yield from _matching([frame], channels, stats)
            continue
        head.append(frame)
        if len(head) == _REFERENCE_FRAMES:
            channels = _majority_channels(head)
            yield from _matching(head, channels, stats)
            head = []
    if head:
        yield from _matching(head, _majority_channels(head), stats)


def _resample_into(resampler, frame, chunks):
    if resampler is None:
        return 0
    added = 0
    for rframe in resampler.resample(frame):
        arr = _frame_to_mono_mean(rframe)
        if arr.size:
            chunks.append(arr)
            added += arr.size
    return added


def decode_with_pyav(file_path, target_sr, load_timeout, min_fraction):
    import av

    max_samples = int(load_timeout * target_sr) if (load_timeout and target_sr) else None
    chunks = []
    total = 0
    actual_sr = target_sr
    stats = {'packets': 0, 'skipped': 0}
    declared = None
    resampler = None
    source = None
    with av.open(file_path) as container:
        if not container.streams.audio:
            return np.array([], dtype=np.float32), actual_sr
        declared = _declared_seconds(container, load_timeout)
        guessed = container.format.name in _DURATION_GUESSED_FORMATS
        stream = container.streams.audio[0]
        for frame in _tolerant_frames(container, stream, stats):
            if actual_sr is None:
                actual_sr = frame.sample_rate
                if load_timeout:
                    max_samples = int(load_timeout * actual_sr)
            shape = (frame.format.name, frame.layout.name, frame.sample_rate)
            if shape != source:
                total += _resample_into(resampler, None, chunks)
                resampler = av.audio.resampler.AudioResampler(
                    format="flt", layout=frame.layout, rate=actual_sr
                )
                source = shape
            total += _resample_into(resampler, frame, chunks)
            if max_samples and total >= max_samples:
                break
        _resample_into(resampler, None, chunks)
    name = os.path.basename(file_path)
    if stats['skipped']:
        logger.warning(
            "Skipped %d corrupt audio packet(s) while decoding %s; the recovered "
            "audio is shorter than the file claims.", stats['skipped'], name
        )
    if not chunks:
        return np.array([], dtype=np.float32), actual_sr
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if max_samples:
        audio = audio[:max_samples]
    if guessed:
        survived = _enough_packets_survived(stats, name, min_fraction)
    else:
        survived = _enough_survived(audio, actual_sr, declared, name, min_fraction)
    if not survived:
        return np.array([], dtype=np.float32), actual_sr
    return audio, actual_sr


def _disable_crash_dialog():
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetErrorMode(_SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX)


def _bound_lifetime(load_timeout):
    limit = wall_seconds(load_timeout)
    if not limit:
        return
    limit += _ORPHAN_GRACE_SECONDS
    if hasattr(signal, 'alarm'):
        signal.alarm(limit)
        return
    watchdog = threading.Timer(limit, os._exit, args=(1,))
    watchdog.daemon = True
    watchdog.start()


def _decode_for_reply(file_path, target_sr, load_timeout, min_fraction):
    name = os.path.basename(file_path)
    nothing = np.array([], dtype=np.float32)
    try:
        audio, sr = decode_with_pyav(file_path, target_sr, load_timeout, min_fraction)
    except (MemoryError, FileNotFoundError, PermissionError):
        logger.exception("Temporary error while PyAV read %s", name)
        return STATUS_TRANSIENT, nothing, None
    except Exception:
        logger.exception("PyAV could not read %s", name)
        return STATUS_UNREADABLE, nothing, None
    return STATUS_OK, np.ascontiguousarray(audio, dtype=np.float32), sr


def run(argv):
    reply = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    _disable_crash_dialog()
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    configure_logging()
    file_path, target_sr, load_timeout, min_fraction = argv
    _bound_lifetime(int(load_timeout))
    status, audio, sr = _decode_for_reply(
        file_path, int(target_sr) or None, int(load_timeout), float(min_fraction)
    )
    header = json.dumps({'status': status, 'sr': sr, 'samples': int(audio.size)})
    with reply:
        reply.write(header.encode('ascii') + b'\n')
        reply.write(audio.data)
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(run(sys.argv[1:]))
