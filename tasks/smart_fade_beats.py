# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beats, downbeats and bar positions of one smart fade window.

tasks.smart_fade_mix asks it for each song's window (at most WINDOW_SECONDS)
in the transition API. It runs the Beat This! small tracker (ISMIR 2024, code
and weights under the MIT licence) from beat_this_small.onnx
(BEAT_THIS_MODEL_PATH, downloaded from the model release), exported by
scripts/onnx_export/export_beat_this_to_onnx.py.

Main Features:
* The log-mel front end matches upstream's torchaudio one: 22.05 kHz, a
  1024-point STFT divided by sqrt(1024), hop 441 (50 frames a second), 128
  slaney mel bands from 30 Hz to 11 kHz, log(1 + 1000 x). The window sits
  BORDER frames into the fixed 1500-frame input, padded with silence, as
  upstream's chunking does.
* Peaks are logits above 0 that are the largest within +-PEAK_FRAMES frames,
  adjacent frames merged; every downbeat moves to its nearest beat.
* rhythm fits one regular grid through the beats, numbered by whole steps of
  the period so a missed or an extra beat keeps its place, with Huber weights,
  over the span where the mix can happen (the whole window when the span
  holds under MIN_FIT_BEATS). The residual and coverage there say how steady
  the beat is (steady).
* The bar position (phase: the grid beat that starts a bar) comes from the
  downbeats in that span, so a bar of another length elsewhere in the window
  does not matter. It needs BAR_BEATS beats a bar on BAR_AGREEMENT of the
  downbeat gaps and of the phase votes, and a bar length within BAR_TOLERANCE
  of four periods; otherwise it is None.
* One CPU session capped at THREADS, created once and shared by the
  planner's two threads. A missing or failing model raises BeatModelError, a
  ValueError, so the caller plays the default crossfade, and it is logged
  every time.
"""

import logging
import threading

import numpy as np

import config

logger = logging.getLogger(__name__)

RATE = 22050
N_FFT = 1024
HOP = 441
FPS = RATE / HOP
MELS = 128
F_MIN = 30.0
F_MAX = 11000.0
LOG_MULTIPLIER = 1000.0
FRAMES = 1500
BORDER = 6
USABLE_FRAMES = FRAMES - 2 * BORDER
WINDOW_SECONDS = (USABLE_FRAMES - 1) * HOP / RATE
PEAK_FRAMES = 3
THREADS = 4
VERSION = 'beat_this_small0_v1'
MIN_BEATS = 4
MIN_FIT_BEATS = 12
STEADY_SECONDS = 0.04
MIN_COVERAGE = 0.8
BAR_BEATS = 4
MIN_DOWNBEATS = 3
BAR_AGREEMENT = 0.75
BAR_TOLERANCE = 0.02
HUBER = 1.345
FIT_PASSES = 3

_session = None
_lock = threading.Lock()
_front = None


class BeatModelError(ValueError):
    pass


def _model():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = THREADS
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            try:
                _session = ort.InferenceSession(config.BEAT_THIS_MODEL_PATH, sess_options=options,
                                                providers=['CPUExecutionProvider'])
            except Exception as exc:
                logger.error('Smart fade beat model %s could not be loaded; transitions play the default crossfade',
                             config.BEAT_THIS_MODEL_PATH, exc_info=True)
                raise BeatModelError('The beat model could not be loaded.') from exc
        return _session


def _filters():
    global _front
    if _front is None:
        import librosa
        from scipy.signal import get_window

        bank = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=MELS, fmin=F_MIN, fmax=F_MAX, htk=False, norm=None)
        _front = (bank.astype(np.float32), get_window('hann', N_FFT, fftbins=True))
    return _front


def log_mel(audio):
    import librosa

    bank, window = _filters()
    spectrum = np.abs(librosa.stft(np.asarray(audio, dtype=np.float32), n_fft=N_FFT, hop_length=HOP, window=window,
                                   center=True, pad_mode='reflect')) / np.sqrt(N_FFT)
    return np.log1p(LOG_MULTIPLIER * (bank @ spectrum)).T.astype(np.float32)


def logits(audio, rate):
    import soxr

    signal = np.asarray(audio, dtype=np.float32)
    if rate != RATE:
        signal = soxr.resample(signal, rate, RATE).astype(np.float32)
    spect = log_mel(signal)[:USABLE_FRAMES]
    window = np.zeros((1, FRAMES, MELS), dtype=np.float32)
    window[0, BORDER:BORDER + spect.shape[0]] = spect
    try:
        beat, downbeat = _model().run(None, {'spect': window})
    except BeatModelError:
        raise
    except Exception as exc:
        logger.error('Smart fade beat model failed on a window; this transition plays the default crossfade', exc_info=True)
        raise BeatModelError('The beat model failed.') from exc
    return beat[0, BORDER:BORDER + spect.shape[0]], downbeat[0, BORDER:BORDER + spect.shape[0]]


def peaks(logit):
    logit = np.asarray(logit, dtype=np.float64)
    if not logit.size:
        return np.zeros(0)
    padded = np.pad(logit, PEAK_FRAMES, constant_values=-np.inf)
    largest = np.lib.stride_tricks.sliding_window_view(padded, 2 * PEAK_FRAMES + 1).max(axis=1)
    frames = np.flatnonzero((logit == largest) & (logit > 0.0))
    merged, run = [], []
    for frame in frames:
        if run and frame - run[-1] > 1:
            merged.append(float(np.mean(run)))
            run = []
        run.append(frame)
    if run:
        merged.append(float(np.mean(run)))
    return np.asarray(merged)


def events(audio, rate):
    beat, downbeat = logits(audio, rate)
    beats = peaks(beat) / FPS
    downbeats = peaks(downbeat) / FPS
    if beats.size and downbeats.size:
        downbeats = np.unique(beats[np.abs(downbeats[:, None] - beats[None, :]).argmin(axis=1)])
    strength = float(np.mean(1.0 / (1.0 + np.exp(-beat[np.round(beats * FPS).astype(int)])))) if beats.size else 0.0
    return {'beats': beats, 'downbeats': downbeats, 'strength': strength}


def _numbered(beats, period):
    kept, index = [beats[0]], [0]
    for time in beats[1:]:
        steps = int(round((time - kept[-1]) / period))
        if steps >= 1:
            kept.append(time)
            index.append(index[-1] + steps)
    return np.asarray(kept), np.asarray(index, dtype=np.float64)


def _huber_line(index, times):
    weights = np.ones_like(times)
    period, first = np.polyfit(index, times, 1)
    for _ in range(FIT_PASSES):
        residual = times - (first + period * index)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-4
        limit = HUBER * scale
        weights = np.where(np.abs(residual) <= limit, 1.0, limit / np.maximum(np.abs(residual), 1e-12))
        period, first = np.polyfit(index, times, 1, w=np.sqrt(weights))
    return float(period), float(first), times - (first + period * index)


def rhythm(beats, downbeats, span=None):
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    if beats.size < MIN_BEATS:
        raise ValueError('No steady beat to match.')
    low, high = span if span is not None else (-np.inf, np.inf)
    period = float(np.median(np.diff(beats)))
    for _ in range(2):
        kept, index = _numbered(beats, period)
        inside = (kept >= low) & (kept <= high)
        if inside.sum() < MIN_FIT_BEATS:
            inside = np.ones_like(inside)
        if kept[inside].size < MIN_BEATS:
            raise ValueError('No steady beat to match.')
        period, first, residual = _huber_line(index[inside], kept[inside])
        if period <= 0:
            raise ValueError('No steady beat to match.')
    count = int(index[-1]) + 1
    grid = first + period * np.arange(count)
    seen = index[inside]
    spread = float(1.4826 * np.median(np.abs(residual)))
    coverage = float(np.unique(seen).size / (seen[-1] - seen[0] + 1))
    keep = grid >= 0.0
    downbeats = np.asarray(downbeats, dtype=np.float64)
    phase = _bar_phase(downbeats[(downbeats >= low) & (downbeats <= high)], period, first)
    offset = int(np.argmax(keep)) if keep.any() else 0
    return {
        'tempo': 60.0 / period, 'period': period, 'grid': grid[keep],
        'phase': None if phase is None else (phase - offset) % BAR_BEATS,
        'residual': spread, 'coverage': coverage,
        'steady': spread <= STEADY_SECONDS and coverage >= MIN_COVERAGE,
    }


def _bar_phase(downbeats, period, first):
    if downbeats.size < MIN_DOWNBEATS:
        return None
    index = np.round((downbeats - first) / period)
    on_grid = np.abs(downbeats - (first + index * period)) <= 0.25 * period
    index = index[on_grid].astype(int)
    if index.size < MIN_DOWNBEATS:
        return None
    gaps = np.diff(index)
    if not gaps.size or np.mean((gaps > 0) & (gaps % BAR_BEATS == 0)) < BAR_AGREEMENT:
        return None
    single = gaps == BAR_BEATS
    if single.any():
        bar = float(np.median(np.diff(downbeats[on_grid])[single]))
        if abs(bar / (BAR_BEATS * period) - 1.0) > BAR_TOLERANCE:
            return None
    votes = np.bincount(index % BAR_BEATS, minlength=BAR_BEATS)
    best = int(np.argmax(votes))
    return best if votes[best] >= BAR_AGREEMENT * index.size else None


def downbeat_times(found):
    if found['phase'] is None:
        return np.zeros(0)
    return found['grid'][found['phase']::BAR_BEATS]


def warm():
    logits(np.zeros(RATE, dtype=np.float32), RATE)
