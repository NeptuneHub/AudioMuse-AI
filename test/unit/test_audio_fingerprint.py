# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content-fingerprint SimHash id: determinism, robustness, and BIGINT range.

Verifies the audio -> tokens -> 64-bit SimHash -> signed BIGINT pipeline gives
the same id for near-identical input and a Postgres-safe signed value, without
requiring the optional Chromaprint native library.

Main Features:
* SimHash determinism, small-change robustness, and separation of different audio.
* Signed BIGINT round-trip stays inside the 64-bit signed range.
* End-to-end librosa chroma fallback yields a stable id for the same waveform.
"""

import numpy as np

from tasks import audio_fingerprint as afp


def _hamming(a, b):
    return bin(afp.from_signed_bigint(a) ^ afp.from_signed_bigint(b)).count('1')


class TestSimHash:
    def test_deterministic(self):
        tokens = [i * 2654435761 & 0xFFFFFFFF for i in range(500)]
        assert afp.simhash64(tokens) == afp.simhash64(list(tokens))

    def test_empty_is_none(self):
        assert afp.simhash64([]) is None
        assert afp.simhash64(None) is None

    def test_small_change_low_hamming(self):
        tokens = [(i * 1103515245 + 12345) & 0xFFFFFFFF for i in range(1000)]
        base = afp.simhash64(tokens)
        perturbed = list(tokens)
        perturbed[0] = 0
        perturbed[500] = 42
        changed = afp.simhash64(perturbed)
        assert bin(base ^ changed).count('1') <= 2

    def test_different_audio_differs(self):
        a = [(i * 2246822519) & 0xFFFFFFFF for i in range(1000)]
        b = [(i * 3266489917 + 7) & 0xFFFFFFFF for i in range(1000)]
        assert afp.simhash64(a) != afp.simhash64(b)


class TestBigintRange:
    def test_round_trip(self):
        for u in [0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]:
            signed = afp.to_signed_bigint(u)
            assert -(1 << 63) <= signed <= (1 << 63) - 1
            assert afp.from_signed_bigint(signed) == u

    def test_canonical_fits_bigint(self):
        signed = afp.to_signed_bigint(afp.simhash64([1, 2, 3, 4, 5, 6, 7, 8]))
        assert -(1 << 63) <= signed <= (1 << 63) - 1


class TestChromaFallback:
    def _tone(self, freq=220.0, sr=16000, seconds=4.0):
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_same_waveform_same_id(self):
        sr = 16000
        wave = self._tone(sr=sr)
        id1 = afp.canonical_fingerprint(wave, sr)
        id2 = afp.canonical_fingerprint(wave.copy(), sr)
        assert id1 is not None
        assert id1 == id2

    def test_amplitude_scale_stays_close(self):
        sr = 16000
        wave = self._tone(sr=sr)
        scaled = (wave * 0.9).astype(np.float32)
        assert _hamming(afp.canonical_fingerprint(scaled, sr), afp.canonical_fingerprint(wave, sr)) <= 8

    def test_too_short_returns_none(self):
        assert afp.canonical_fingerprint(np.zeros(10, dtype=np.float32), 16000) is None


class TestCanonicalIdStr:
    def test_round_trip_and_prefix(self):
        for signed in [0, 1, -5, (1 << 63) - 1, -(1 << 63)]:
            cid = afp.canonical_id_str(signed)
            assert cid.startswith('fp_')
            assert afp.is_fingerprint_id(cid)
        assert afp.canonical_id_str(None) is None

    def test_same_fingerprint_same_id(self):
        assert afp.canonical_id_str(123456789) == afp.canonical_id_str(123456789)

    def test_non_fingerprint_ids_rejected(self):
        assert not afp.is_fingerprint_id('abc123')
        assert not afp.is_fingerprint_id(None)
        assert not afp.is_fingerprint_id(42)
