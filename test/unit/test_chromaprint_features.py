# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The numpy Chromaprint reproduces fpcalc and exposes per-bit reliability.

Main Features:
* the sub-fingerprint count follows fpcalc's framing (hop 1366, 5 filter taps,
  16-frame classifier window)
* a clip too short for one classifier window yields no fingerprint
* quantisation is Gray coded with classifier 0 in the top two bits, and the
  reliability of a bit is zero exactly on its threshold and grows away from it
* on the repo's own test songs the integers match fpcalc's within two percent
  of bits at zero offset, when fpcalc is available
* a pure tone gives a chroma image concentrated on one pitch class
"""

import glob
import os
import shutil
import subprocess

import numpy as np
import pytest

from tasks import chromaprint_features as cf

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def test_sub_fingerprint_count_follows_the_fpcalc_framing():
    seconds = 30
    signal = np.random.default_rng(1).standard_normal(seconds * cf.SAMPLE_RATE).astype(np.float32) * 0.1
    ints, rel = cf.fingerprint(signal, cf.SAMPLE_RATE)
    n_frames = 1 + (signal.size - cf.FRAME) // cf.HOP
    assert ints.size == n_frames - cf.FILTER_COEFFICIENTS.size + 1 - cf.MAX_FILTER_WIDTH + 1
    assert rel.shape == (ints.size, 32)


def test_a_clip_too_short_for_one_window_yields_no_fingerprint():
    ints, rel = cf.fingerprint(np.zeros(cf.SAMPLE_RATE * 2, dtype=np.float32), cf.SAMPLE_RATE)
    assert ints is None and rel is None


def test_quantize_is_gray_coded_with_classifier_zero_on_top():
    values = np.zeros((1, 16))
    for c, (_filter, (t0, t1, t2)) in enumerate(cf.CLASSIFIERS):
        values[0, c] = t0 - 1.0
    values[0, 0] = cf.CLASSIFIERS[0][1][2] + 1.0
    ints = cf.quantize(values)
    assert (ints[0] >> np.uint32(30)) & 3 == 2
    assert ints[0] & 0x3FFFFFFF == 0
    values[0, 15] = cf.CLASSIFIERS[15][1][1] + 1e-6
    assert cf.quantize(values)[0] & 3 == 3


def test_bit_reliability_is_zero_on_the_threshold_and_grows_away_from_it():
    values = np.zeros((3, 16))
    t0, t1, t2 = cf.CLASSIFIERS[3][1]
    values[:, 3] = [t1, t1 + 0.5 * (t2 - t0), t0]
    rel = cf.bit_reliability(values)
    coarse, fine = 31 - 2 * 3, 30 - 2 * 3
    assert rel[0, coarse] == 0.0
    assert rel[1, coarse] == pytest.approx(0.5)
    assert rel[2, fine] == 0.0
    assert rel[1, fine] > 0


def test_a_pure_tone_lands_on_one_pitch_class():
    t = np.arange(cf.SAMPLE_RATE * 6) / cf.SAMPLE_RATE
    centre_of_class_zero = cf.BASE_FREQ * 2 ** (4 + 0.5 / cf.NUM_BANDS)
    image = cf.chroma_image((0.3 * np.sin(2 * np.pi * centre_of_class_zero * t)).astype(np.float32), cf.SAMPLE_RATE)
    profile = image.mean(axis=0)
    assert int(np.argmax(profile)) == 0
    assert profile[0] > 0.9


@pytest.mark.skipif(shutil.which('fpcalc') is None, reason='fpcalc not installed')
def test_matches_fpcalc_on_the_repo_test_songs():
    import librosa

    songs = sorted(glob.glob(os.path.join(REPO, 'test', 'songs', '*.mp3')))
    assert songs
    pop = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)
    for path in songs[:2]:
        out = subprocess.run(['fpcalc', '-raw', '-length', '30', path], capture_output=True, text=True, timeout=120)
        ref = None
        for line in out.stdout.splitlines():
            if line.startswith('FINGERPRINT='):
                ref = np.array([int(v) for v in line[12:].split(',') if v], dtype=np.int64).astype(np.uint32)
        assert ref is not None
        audio, sr = librosa.load(path, sr=None, mono=True, duration=30.0)
        mine, _rel = cf.fingerprint(audio, sr)
        n = min(ref.size, mine.size)
        ber = pop[(ref[:n] ^ mine[:n]).view(np.uint8)].sum() / (32.0 * n)
        assert abs(ref.size - mine.size) <= 1
        assert ber < 0.02
