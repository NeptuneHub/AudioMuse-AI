# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recording search manager: clip decoding, level normalisation and rank fusion.

Main Features:
* fuse_by_rank: two indexes agreeing on a mid rank beat one index's top hit, a
  track missing from a list gets no contribution there, a duplicate inside one
  list counts once, space-specific scores are dropped, the limit trims
* normalize_level brings the RMS to the target and rejects silence
* a clip below RECORDING_SEARCH_QUIET_LEVEL_DB is reported with its level and a
  warning that tells the user to record closer and louder
* decode_clip decodes a real WAV through the analysis loader from bytes or a
  stream, trims it to RECORDING_SEARCH_MAX_CLIP_SECONDS, rejects bytes that are
  not audio, and refuses a stream past RECORDING_SEARCH_MAX_UPLOAD_MB while
  copying it
* run_recording_search: combined mode records a skipped index as a warning and
  fuses the rest, fails only when every index is unavailable, single mode
  re-raises, an unknown mode is rejected before decoding
* combined mode pins a confident chromaprint identification first, copies its
  ber, z, offset and flag onto the fused row, and the identify list weighs
  double in the fusion
"""

import io
import wave

import numpy as np
import pytest

from tasks import recording_search_manager as rsm


def _rows(*ids):
    return [
        {'item_id': i, 'title': f't-{i}', 'author': f'a-{i}', 'album': '', 'similarity': 0.5}
        for i in ids
    ]


def test_two_indexes_agreeing_on_a_mid_rank_beat_one_index_top_hit():
    fused = rsm.fuse_by_rank({'musicnn': _rows('x', 'y', 'z'), 'dclap': _rows('w', 'y', 'v')}, k=60)
    assert fused[0]['item_id'] == 'y'
    assert fused[0]['agreement'] == 2
    assert fused[0]['ranks'] == {'musicnn': 2, 'dclap': 2}
    assert fused[0]['rrf_score'] == pytest.approx(2 / 62, abs=1e-6)


def test_track_missing_from_a_list_gets_no_contribution_there():
    fused = rsm.fuse_by_rank({'musicnn': _rows('x'), 'dclap': _rows('q')}, k=60)
    by_id = {entry['item_id']: entry for entry in fused}
    assert by_id['x']['ranks'] == {'musicnn': 1}
    assert by_id['x']['agreement'] == 1
    assert 'dclap' not in by_id['x']['similarity_by_source']


def test_duplicate_within_one_list_counts_once():
    fused = rsm.fuse_by_rank({'musicnn': _rows('x', 'x')}, k=60)
    assert len(fused) == 1
    assert fused[0]['rrf_score'] == pytest.approx(1 / 61, abs=1e-6)


def test_fused_rows_drop_space_specific_scores_and_keep_metadata():
    fused = rsm.fuse_by_rank({'musicnn': _rows('x')}, k=60)
    assert 'similarity' not in fused[0]
    assert 'distance' not in fused[0]
    assert fused[0]['title'] == 't-x'
    assert fused[0]['similarity_by_source'] == {'musicnn': 0.5}


def test_limit_trims_the_fused_output():
    fused = rsm.fuse_by_rank({'musicnn': _rows('a', 'b', 'c')}, k=60, limit=2)
    assert [entry['item_id'] for entry in fused] == ['a', 'b']


def test_rows_without_item_id_are_ignored():
    fused = rsm.fuse_by_rank({'musicnn': [{'title': 'no id'}] + _rows('a')}, k=60)
    assert [entry['item_id'] for entry in fused] == ['a']


def test_normalize_level_brings_rms_to_target():
    audio = (0.01 * np.sin(np.linspace(0, 200 * np.pi, 48000))).astype(np.float32)
    out = rsm.normalize_level(audio, target_db=-14.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(out ** 2)))
    assert rms_db == pytest.approx(-14.0, abs=0.1)
    assert out.dtype == np.float32


def test_normalize_level_rejects_silence():
    with pytest.raises(ValueError):
        rsm.normalize_level(np.zeros(1000, dtype=np.float32))


def test_quiet_clip_warning_fires_below_the_threshold_and_names_the_level(monkeypatch):
    monkeypatch.setattr(rsm.config, 'RECORDING_SEARCH_QUIET_LEVEL_DB', -30.0)
    assert rsm.quiet_clip_warning(-20.0) is None
    warning = rsm.quiet_clip_warning(-44.8)
    assert '-45 dBFS' in warning
    assert 'record again' in warning


def test_run_recording_search_reports_the_clip_level_and_warns_when_too_quiet(monkeypatch):
    monkeypatch.setattr(rsm.config, 'RECORDING_SEARCH_QUIET_LEVEL_DB', -30.0)
    quiet = (0.001 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    monkeypatch.setattr(rsm, 'decode_clip', lambda clip, filename: (quiet, 16000))
    monkeypatch.setattr(rsm, '_SEARCHERS', {'musicnn': lambda audio, sr, n: (_rows('a'), None)})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'musicnn', 10)
    assert payload['clip_level_db'] == pytest.approx(-63.0, abs=0.5)
    assert len(payload['warnings']) == 1
    assert 'very quiet' in payload['warnings'][0]
    loud = (0.2 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    monkeypatch.setattr(rsm, 'decode_clip', lambda clip, filename: (loud, 16000))
    payload = rsm.run_recording_search(b'x', 'c.wav', 'musicnn', 10)
    assert payload['warnings'] == []


def _wav_bytes(seconds, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_decode_clip_decodes_a_wav_and_trims_to_the_configured_seconds(monkeypatch):
    monkeypatch.setattr(rsm.config, 'RECORDING_SEARCH_MAX_CLIP_SECONDS', 1)
    audio, sr = rsm.decode_clip(_wav_bytes(3.0), 'clip.wav')
    assert sr == 16000
    assert audio.dtype == np.float32
    assert audio.shape[0] == 16000


def test_decode_clip_rejects_bytes_that_are_not_audio():
    with pytest.raises(ValueError):
        rsm.decode_clip(b'this is not audio at all', 'clip.mp3')


def test_decode_clip_rejects_an_empty_upload():
    with pytest.raises(ValueError):
        rsm.decode_clip(b'', 'clip.wav')


def test_decode_clip_accepts_a_stream_and_never_needs_the_bytes_in_memory(monkeypatch):
    monkeypatch.setattr(rsm, '_COPY_CHUNK_BYTES', 4096)
    audio, sr = rsm.decode_clip(io.BytesIO(_wav_bytes(1.0)), 'clip.wav')
    assert sr == 16000
    assert audio.shape[0] == 16000


def test_decode_clip_refuses_a_stream_past_the_upload_ceiling_while_copying(monkeypatch):
    monkeypatch.setattr(rsm.config, 'RECORDING_SEARCH_MAX_UPLOAD_MB', 0)
    monkeypatch.setattr(rsm, '_COPY_CHUNK_BYTES', 1024)
    with pytest.raises(ValueError, match='larger than 0 MB'):
        rsm.decode_clip(io.BytesIO(_wav_bytes(1.0)), 'clip.wav')


def test_safe_suffix_keeps_only_a_short_alphanumeric_extension():
    assert rsm._safe_suffix('rec.webm') == '.webm'
    assert rsm._safe_suffix('a b/c d.mp3!') == '.mp3'
    assert rsm._safe_suffix('noext') == '.bin'
    assert rsm._safe_suffix(None) == '.bin'


def _fake_pipeline(monkeypatch, searchers):
    monkeypatch.setattr(
        rsm, 'decode_clip', lambda file_bytes, filename: (np.ones(16000, dtype=np.float32), 16000)
    )
    monkeypatch.setattr(rsm, 'normalize_level', lambda audio, target_db=None: audio)
    monkeypatch.setattr(rsm, '_SEARCHERS', searchers)
    monkeypatch.setattr(rsm, 'SOURCES', tuple(searchers))


def test_a_confident_identification_is_pinned_first_in_combined_mode_and_annotated(monkeypatch):
    def identify(audio, sr, n):
        rows = _rows('song', 'other')
        rows[0].update({'ber': 0.34, 'z': 6.1, 'offset_seconds': 7.2, 'identified': True})
        rows[1].update({'ber': 0.41, 'z': 3.9, 'offset_seconds': 50.0, 'identified': False})
        return rows, None

    def musicnn(audio, sr, n):
        return _rows('a', 'b', 'song'), None

    def dclap(audio, sr, n):
        return _rows('a', 'b'), None

    _fake_pipeline(monkeypatch, {'identify': identify, 'musicnn': musicnn, 'dclap': dclap})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'combined', 10)
    first = payload['results'][0]
    assert first['item_id'] == 'song'
    assert first['identified'] is True
    assert first['offset_seconds'] == 7.2
    assert first['ranks'] == {'identify': 1, 'musicnn': 3}
    assert [row['item_id'] for row in payload['results'][1:3]] == ['a', 'b']


def test_the_identify_list_weighs_double_in_the_fusion():
    fused = rsm.fuse_by_rank({'identify': _rows('x'), 'musicnn': _rows('y')}, k=60, weights={'identify': 2.0})
    by_id = {entry['item_id']: entry for entry in fused}
    assert by_id['x']['rrf_score'] == pytest.approx(2 / 61, abs=1e-6)
    assert by_id['y']['rrf_score'] == pytest.approx(1 / 61, abs=1e-6)


def test_combined_mode_reports_an_unavailable_index_as_a_warning_and_fuses_the_rest(monkeypatch):
    def musicnn(audio, sr, n):
        return _rows('x', 'y'), None

    def dclap(audio, sr, n):
        raise RuntimeError('The DCLAP index is not loaded. Run analysis first.')

    def lyrics(audio, sr, n):
        return _rows('y'), 'some words'

    _fake_pipeline(monkeypatch, {'musicnn': musicnn, 'dclap': dclap, 'lyrics': lyrics})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'combined', 10)
    assert payload['sources'] == ['lyrics', 'musicnn']
    assert payload['warnings'] == [
        'DCLAP skipped: The DCLAP index is not loaded. Run analysis first.'
    ]
    assert payload['transcript'] == 'some words'
    assert payload['results'][0]['item_id'] == 'y'
    assert payload['count'] == 2


def test_single_mode_raises_when_its_index_is_unavailable(monkeypatch):
    def dclap(audio, sr, n):
        raise RuntimeError('down')

    _fake_pipeline(monkeypatch, {'dclap': dclap})
    with pytest.raises(RuntimeError):
        rsm.run_recording_search(b'x', 'c.wav', 'dclap', 10)


def test_combined_mode_fails_only_when_every_index_is_unavailable(monkeypatch):
    def down(audio, sr, n):
        raise RuntimeError('down')

    _fake_pipeline(monkeypatch, {'musicnn': down, 'dclap': down, 'lyrics': down})
    with pytest.raises(RuntimeError):
        rsm.run_recording_search(b'x', 'c.wav', 'combined', 10)


def test_unknown_mode_is_rejected_before_decoding(monkeypatch):
    def never(file_bytes, filename):
        raise AssertionError('decode_clip must not run for an unknown mode')

    monkeypatch.setattr(rsm, 'decode_clip', never)
    with pytest.raises(ValueError):
        rsm.run_recording_search(b'x', 'c.wav', 'bogus', 10)


def test_single_mode_returns_the_index_rows_untouched(monkeypatch):
    rows = _rows('a', 'b')
    _fake_pipeline(monkeypatch, {'musicnn': lambda audio, sr, n: (rows, None)})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'musicnn', 10)
    assert payload['results'] == rows
    assert payload['mode'] == 'musicnn'
    assert payload['clip_seconds'] == 1.0
    assert payload['sources'] == ['musicnn']
