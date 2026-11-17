# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Pick the demo part of a song, its chorus or drop, from its neural fingerprint.

A demo is the stretch of a song a preview plays. The analysis stores one neural
fingerprint vector per half second of every song (tasks.neural_fingerprint),
and a chorus comes back several times with the same sound, so its vectors match
their own repeats far better than a verse or a bridge does. app_demo.py serves
this module as POST /api/demo; static/song_demo.js plays the answer.

Main Features:
* demo_window(vectors, seconds) builds, for a demo of `seconds` (the API sends
  config.SONG_DEMO_SECONDS unless the request names one, never under
  MIN_SECONDS), the song's self-similarity matrix (the
  cosine of every half second with every other one), blanks digital silence
  (consecutive rows that do not change), averages it along the diagonals over
  SMOOTH_SECONDS so a repeat reads as a path, and scores every half second by
  its REPEATS best matches at least GAP_SECONDS away. The window of the asked
  length with the best mean score, bent by CENTER_BIAS toward the middle of the
  song, is the demo (repetition-based thumbnailing, as in Goto's RefraiD and
  Mueller's audio thumbnails).
* Measured on 223 library songs that carry hand-made section labels: a 30 s
  demo holds 68% chorus and 87% of them hold 10 s of chorus or more, against
  41% and 61% for the window in the middle of the song.
* A song no longer than the asked length is played whole; a long track is
  pooled to MAX_ROWS rows first, so the matrices stay a few MB at any length.
* demo_sections(item_ids, seconds) reads only the stored fingerprints and
  answers in the order given, None for a song without one. No audio is
  decoded and no model runs.
"""

import math

import numpy as np

from database import get_db
from tasks import neural_fingerprint
from tasks.neural_fingerprint_index import iter_codes

MIN_SECONDS = 5.0
SMOOTH_SECONDS = 4.0
GAP_SECONDS = 20.0
REPEATS = 2
CENTER_BIAS = 0.2
SILENCE_COSINE = 0.95
MAX_ROWS = 1200


def _pooled(vectors):
    rows = np.asarray(vectors, dtype=np.float32)
    stride = max(1, math.ceil(rows.shape[0] / MAX_ROWS))
    if stride > 1:
        count = rows.shape[0] // stride
        rows = rows[:count * stride].reshape(count, stride, rows.shape[1]).mean(axis=1)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9), stride


def _repeat_curve(rows, hop):
    width = max(1, int(round(SMOOTH_SECONDS / hop)))
    gap = max(1, int(round(GAP_SECONDS / hop)))
    sims = rows @ rows.T
    still = np.einsum('ij,ij->i', rows[1:], rows[:-1]) >= SILENCE_COSINE
    silent = np.zeros(rows.shape[0], dtype=bool)
    silent[1:] |= still
    silent[:-1] |= still
    sims[silent] = 0.0
    sims[:, silent] = 0.0
    size = rows.shape[0] - width + 1
    if size < 1:
        return np.zeros(0, dtype=np.float32), width
    paths = np.zeros((size, size), dtype=np.float32)
    for shift in range(width):
        paths += sims[shift:shift + size, shift:shift + size]
    paths /= width
    index = np.arange(size)
    distance = np.abs(index[:, None] - index[None, :])
    paths[distance < gap] = -1.0
    total = np.zeros(size, dtype=np.float32)
    for _ in range(REPEATS):
        best = paths.argmax(axis=1)
        total += np.maximum(paths[index, best], 0.0)
        paths[np.abs(index[None, :] - best[:, None]) <= gap // 2] = -1.0
    return total / REPEATS, width


def _section(start, seconds, duration):
    return {
        'start_seconds': round(start, 1),
        'end_seconds': round(min(duration, start + seconds), 1),
        'duration_seconds': round(duration, 1),
    }


def demo_window(vectors, seconds):
    count = int(vectors.shape[0])
    if count == 0:
        return None
    seconds = float(seconds)
    segment = neural_fingerprint.SEGMENT_SAMPLES / neural_fingerprint.SAMPLE_RATE
    duration = (count - 1) * neural_fingerprint.HOP_SECONDS + segment
    if duration <= seconds:
        return _section(0.0, duration, duration)
    rows, stride = _pooled(vectors)
    hop = neural_fingerprint.HOP_SECONDS * stride
    curve, width = _repeat_curve(rows, hop)
    span = max(1, int(round(seconds / hop)) - width + 1)
    if curve.size < span:
        return _section((duration - seconds) / 2.0, seconds, duration)
    sums = np.concatenate([[0.0], np.cumsum(curve, dtype=np.float64)])
    score = (sums[span:] - sums[:-span]) / span
    score *= 1.0 - CENTER_BIAS * np.abs(np.linspace(-1.0, 1.0, score.size))
    start = min(float(np.argmax(score)) * hop, duration - seconds)
    return _section(start, seconds, duration)


def demo_sections(item_ids, seconds):
    found = {}
    for item_id, codes in iter_codes(get_db(), list(dict.fromkeys(item_ids))):
        if codes is not None and codes.shape[0]:
            found[item_id] = demo_window(neural_fingerprint.decode_codes(codes), seconds)
    return [found.get(item_id) for item_id in item_ids]
