# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Chromaprint algorithm 2 in numpy, with the soft information fpcalc throws away.

fpcalc produces the stored fingerprints, and this module reproduces its integers
for an in-memory clip (0.5 to 1.6 percent of bits differ, from the resampler)
while also keeping every classifier's continuous value. That value's distance
to the quantiser threshold is how sure each bit is, which is what a noisy
recording needs: bits that sit on a threshold flip under noise, bits far from
it do not, and only the query side can know this.

Main Features:
* chroma_image: 11025 Hz mono, 4096-sample Hamming frames scaled like
  Chromaprint's 16-bit path with a hop of 1366, power spectrum summed into 12
  pitch classes between 28 and 3520 Hz (A0 at 27.5 Hz), the 5-tap temporal
  filter, then per-frame Euclidean normalisation with the 0.01 silence floor.
* classifier_values: the 16 classifiers of algorithm 2 (filter type, chroma row,
  height, width) as log ratios of integral-image areas, one value per frame
  window of 16.
* quantize: the three thresholds per classifier, Gray coded, classifier 0 in
  the top two bits, matching fpcalc's sub-fingerprint layout.
* bit_reliability: per sub-fingerprint and per bit, the distance of the value to
  the threshold that would flip that bit, in units of the quantiser range.
* fingerprint: the pair (integers, reliabilities) for a clip.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 11025
FRAME = 4096
HOP = 4096 - 2730
MIN_FREQ = 28
MAX_FREQ = 3520
BASE_FREQ = 440.0 / 16.0
NUM_BANDS = 12
FILTER_COEFFICIENTS = np.array([0.25, 0.75, 1.0, 0.75, 0.25])
NORM_THRESHOLD = 0.01
MAX_FILTER_WIDTH = 16
GRAY_CODE = np.array([0, 1, 3, 2], dtype=np.uint32)
WINDOW_SCALE = 0.5
CLASSIFIERS = (
    ((0, 4, 3, 15), (1.98215, 2.35817, 2.63523)),
    ((4, 4, 6, 15), (-1.03809, -0.651211, -0.282167)),
    ((1, 0, 4, 16), (-0.298702, 0.119262, 0.558497)),
    ((3, 8, 2, 12), (-0.105439, 0.0153946, 0.135898)),
    ((3, 4, 4, 8), (-0.142891, 0.0258736, 0.200632)),
    ((4, 0, 3, 5), (-0.826319, -0.590612, -0.368214)),
    ((1, 2, 2, 9), (-0.557409, -0.233035, 0.0534525)),
    ((2, 7, 3, 4), (-0.0646826, 0.00620476, 0.0784847)),
    ((2, 6, 2, 16), (-0.192387, -0.029699, 0.215855)),
    ((2, 1, 3, 2), (-0.0397818, -0.00568076, 0.0292026)),
    ((5, 10, 1, 15), (-0.53823, -0.369934, -0.190235)),
    ((3, 6, 2, 10), (-0.124877, 0.0296483, 0.139239)),
    ((2, 1, 1, 14), (-0.101475, 0.0225617, 0.231971)),
    ((3, 5, 6, 4), (-0.0799915, -0.00729616, 0.063262)),
    ((1, 9, 2, 12), (-0.272556, 0.019424, 0.302559)),
    ((3, 4, 2, 14), (-0.164292, -0.0321188, 0.0846339)),
)


def _bin_notes():
    min_index = max(1, int(round(FRAME * MIN_FREQ / SAMPLE_RATE)))
    max_index = min(FRAME // 2, int(round(FRAME * MAX_FREQ / SAMPLE_RATE)))
    index = np.arange(min_index, max_index)
    octave = np.log(index * SAMPLE_RATE / FRAME / BASE_FREQ) / np.log(2.0)
    note = (NUM_BANDS * (octave - np.floor(octave))).astype(int) % NUM_BANDS
    return index, note


_BIN_INDEX, _BIN_NOTE = _bin_notes()
_WINDOW = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))) * WINDOW_SCALE


def chroma_image(audio, sample_rate):
    from tasks.analysis import resample_audio

    signal = np.asarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        signal = resample_audio(signal, sample_rate, SAMPLE_RATE)
    n_frames = 1 + (signal.size - FRAME) // HOP
    if n_frames < MAX_FILTER_WIDTH + FILTER_COEFFICIENTS.size:
        return None
    frames = sliding_window_view(signal, FRAME)[::HOP][:n_frames] * _WINDOW[None, :]
    spectrum = np.fft.rfft(frames, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    chroma = np.zeros((n_frames, NUM_BANDS))
    for note in range(NUM_BANDS):
        chroma[:, note] = power[:, _BIN_INDEX[_BIN_NOTE == note]].sum(axis=1)
    taps = FILTER_COEFFICIENTS.size
    filtered = np.zeros((n_frames - taps + 1, NUM_BANDS))
    for j, coefficient in enumerate(FILTER_COEFFICIENTS):
        filtered += coefficient * chroma[j: j + n_frames - taps + 1]
    norms = np.linalg.norm(filtered, axis=1)
    return np.where(norms[:, None] < NORM_THRESHOLD, 0.0, filtered / np.maximum(norms[:, None], 1e-12))


def _integral(image):
    padded = np.zeros((image.shape[0] + 1, image.shape[1] + 1))
    padded[1:, 1:] = image.cumsum(axis=0).cumsum(axis=1)
    return padded


def _area(ii, x1, y1, x2, y2):
    return ii[x2 + 1, y2 + 1] - ii[x1, y2 + 1] - ii[x2 + 1, y1] + ii[x1, y1]


def _filter_value(ii, kind, x, y, w, h):
    if kind == 0:
        a = _area(ii, x, y, x + w - 1, y + h - 1)
        b = np.zeros_like(a)
    elif kind == 1:
        h2 = h // 2
        a = _area(ii, x, y + h2, x + w - 1, y + h - 1)
        b = _area(ii, x, y, x + w - 1, y + h2 - 1)
    elif kind == 2:
        w2 = w // 2
        a = _area(ii, x + w2, y, x + w - 1, y + h - 1)
        b = _area(ii, x, y, x + w2 - 1, y + h - 1)
    elif kind == 3:
        w2, h2 = w // 2, h // 2
        a = _area(ii, x, y + h2, x + w2 - 1, y + h - 1) + _area(ii, x + w2, y, x + w - 1, y + h2 - 1)
        b = _area(ii, x, y, x + w2 - 1, y + h2 - 1) + _area(ii, x + w2, y + h2, x + w - 1, y + h - 1)
    elif kind == 4:
        h3 = h // 3
        a = _area(ii, x, y + h3, x + w - 1, y + 2 * h3 - 1)
        b = _area(ii, x, y, x + w - 1, y + h3 - 1) + _area(ii, x, y + 2 * h3, x + w - 1, y + h - 1)
    else:
        w3 = w // 3
        a = _area(ii, x + w3, y, x + 2 * w3 - 1, y + h - 1)
        b = _area(ii, x, y, x + w3 - 1, y + h - 1) + _area(ii, x + 2 * w3, y, x + w - 1, y + h - 1)
    return np.log((1.0 + a) / (1.0 + b))


def classifier_values(image):
    ii = _integral(image)
    x = np.arange(image.shape[0] - MAX_FILTER_WIDTH + 1)
    values = np.zeros((x.size, len(CLASSIFIERS)))
    for c, ((kind, y, h, w), _thresholds) in enumerate(CLASSIFIERS):
        values[:, c] = _filter_value(ii, kind, x, y, w, h)
    return values


def quantize(values):
    ints = np.zeros(values.shape[0], dtype=np.uint32)
    for c, (_filter, (t0, t1, t2)) in enumerate(CLASSIFIERS):
        v = values[:, c]
        level = (v >= t0).astype(np.uint32) + (v >= t1).astype(np.uint32) + (v >= t2).astype(np.uint32)
        ints = (ints << np.uint32(2)) | GRAY_CODE[level]
    return ints


def bit_reliability(values):
    reliability = np.zeros((values.shape[0], 32), dtype=np.float32)
    for c, (_filter, (t0, t1, t2)) in enumerate(CLASSIFIERS):
        v = values[:, c]
        span = t2 - t0
        reliability[:, 31 - 2 * c] = np.abs(v - t1) / span
        reliability[:, 30 - 2 * c] = np.minimum(np.abs(v - t0), np.abs(v - t2)) / span
    return reliability


def fingerprint(audio, sample_rate):
    image = chroma_image(audio, sample_rate)
    if image is None:
        return None, None
    values = classifier_values(image)
    return quantize(values), bit_reliability(values)
