# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Chromaprint catalogue-id behaviour.

Verifies that compute_chromaprint keeps fpcalc's raw fingerprint text, that
chromaprint_canonical_id hashes it into a deterministic fp_<hex> id, and that
is_fingerprint_id recognizes only canonical ids.

Main Features:
* Raw Chromaprint retention and deterministic canonical-id hashing.
* Graceful None on a missing fpcalc binary.
* fp_ prefix recognition and rejection of non-canonical ids.
"""

from types import SimpleNamespace

from tasks import audio_fingerprint as afp


class TestChromaprintIdentity:
    def test_keeps_raw_value_and_hashes_deterministically(self, monkeypatch):
        raw = 'AQAAE0mUaEkSRZEGAA'
        monkeypatch.setitem(afp._fpcalc_state, 'available', True)
        monkeypatch.setattr(
            afp.subprocess,
            'run',
            lambda *args, **kwargs: SimpleNamespace(
                stdout='{"duration": 123.0, "fingerprint": "' + raw + '"}'
            ),
        )

        assert afp.compute_chromaprint('/tmp/song.flac') == raw
        first = afp.chromaprint_canonical_id(raw)
        assert first == afp.chromaprint_canonical_id(raw)
        assert first.startswith('fp_') and len(first) == 35
        assert first != afp.chromaprint_canonical_id(raw + 'different')

    def test_missing_fpcalc_returns_none(self, monkeypatch):
        monkeypatch.setitem(afp._fpcalc_state, 'available', True)
        def missing(*args, **kwargs):
            raise FileNotFoundError('fpcalc')

        monkeypatch.setattr(afp.subprocess, 'run', missing)
        assert afp.compute_chromaprint('/tmp/song.flac') is None

    def test_empty_chromaprint_has_no_canonical_id(self):
        assert afp.chromaprint_canonical_id(None) is None
        assert afp.chromaprint_canonical_id('') is None


class TestIsFingerprintId:
    def test_canonical_ids_recognized(self):
        cid = afp.chromaprint_canonical_id('AQAAE0mUaEkSRZEGAA')
        assert afp.is_fingerprint_id(cid)
        assert afp.is_fingerprint_id('fp_deadbeef')

    def test_non_fingerprint_ids_rejected(self):
        assert not afp.is_fingerprint_id('abc123')
        assert not afp.is_fingerprint_id(None)
        assert not afp.is_fingerprint_id(42)
