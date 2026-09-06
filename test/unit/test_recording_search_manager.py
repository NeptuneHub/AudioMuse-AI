# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recording search manager: clip decoding, level normalisation and the three modes.

Main Features:
* normalize_level brings the RMS to the target and rejects silence
* a clip below RECORDING_SEARCH_QUIET_LEVEL_DB is reported with its level and a
  warning that tells the user to record closer and louder
* decode_clip decodes a real WAV through the analysis loader from bytes or a
  stream, trims it to RECORDING_SEARCH_MAX_CLIP_SECONDS, rejects bytes that are
  not audio, and refuses a stream past RECORDING_SEARCH_MAX_UPLOAD_MB while
  copying it
* run_recording_search runs exactly the mode asked for, returns its rows with
  titles attached, re-raises when that index is unavailable, rejects an unknown
  mode before decoding, and reports the transcript the lyrics mode heard
* the neural rows keep score, votes, offset, identified and lead, and the
  index status names each pack's state
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
    monkeypatch.setattr(rsm, '_SEARCHERS', {'identify': lambda audio, sr, n: (_rows('a'), None)})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'identify', 10)
    assert payload['clip_level_db'] == pytest.approx(-63.0, abs=0.5)
    assert len(payload['warnings']) == 1
    assert 'very quiet' in payload['warnings'][0]
    loud = (0.2 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    monkeypatch.setattr(rsm, 'decode_clip', lambda clip, filename: (loud, 16000))
    payload = rsm.run_recording_search(b'x', 'c.wav', 'identify', 10)
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


def test_each_mode_runs_only_its_own_searcher_and_returns_its_rows(monkeypatch):
    calls = []

    def identify(audio, sr, n):
        calls.append('identify')
        return _rows('a', 'b'), None

    def neural(audio, sr, n):
        calls.append('neural')
        return _rows('c'), None

    _fake_pipeline(monkeypatch, {'identify': identify, 'neural': neural})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'neural', 10)
    assert calls == ['neural']
    assert [row['item_id'] for row in payload['results']] == ['c']
    assert payload['mode'] == 'neural' and payload['count'] == 1
    assert payload['clip_seconds'] == 1.0 and payload['transcript'] is None
    assert 'sources' not in payload


def test_the_lyrics_mode_reports_what_whisper_heard(monkeypatch):
    _fake_pipeline(monkeypatch, {'lyrics': lambda audio, sr, n: (_rows('y'), 'some words')})
    payload = rsm.run_recording_search(b'x', 'c.wav', 'lyrics', 10)
    assert payload['transcript'] == 'some words'
    assert payload['results'][0]['item_id'] == 'y'


def test_a_mode_whose_index_is_unavailable_raises(monkeypatch):
    def down(audio, sr, n):
        raise RuntimeError('down')

    _fake_pipeline(monkeypatch, {'neural': down})
    with pytest.raises(RuntimeError):
        rsm.run_recording_search(b'x', 'c.wav', 'neural', 10)


def test_unknown_mode_is_rejected_before_decoding(monkeypatch):
    def never(file_bytes, filename):
        raise AssertionError('decode_clip must not run for an unknown mode')

    monkeypatch.setattr(rsm, 'decode_clip', never)
    with pytest.raises(ValueError):
        rsm.run_recording_search(b'x', 'c.wav', 'combined', 10)
    with pytest.raises(ValueError):
        rsm.run_recording_search(b'x', 'c.wav', 'musicnn', 10)


def test_neural_rows_get_titles_and_keep_their_alignment_fields(monkeypatch):
    from tasks import neural_fingerprint_index

    hits = [
        {'item_id': 'song', 'score': 0.81, 'votes': 30.5, 'offset_seconds': 95.5, 'identified': True, 'lead': 0.4},
        {'item_id': 'ghost', 'score': 0.41, 'votes': 3.0, 'offset_seconds': 1.0, 'identified': False, 'lead': None},
    ]
    monkeypatch.setattr(neural_fingerprint_index, 'identify', lambda audio, sr, n: hits)
    monkeypatch.setattr(
        rsm, '_fetch_track_metadata',
        lambda ids: {'song': {'title': 'Song', 'author': 'Band', 'album': 'LP'}},
    )
    rows, transcript = rsm.search_neural(np.ones(16000, dtype=np.float32), 16000, 5)
    assert transcript is None
    assert rows == [
        {
            'item_id': 'song', 'score': 0.81, 'votes': 30.5, 'offset_seconds': 95.5, 'identified': True,
            'lead': 0.4, 'title': 'Song', 'author': 'Band', 'album': 'LP',
        }
    ]


def test_index_status_names_each_pack_state(monkeypatch):
    from tasks import chromaprint_identify, neural_fingerprint_index
    import tasks.lyrics_manager as lyrics_manager

    monkeypatch.setattr(
        chromaprint_identify, 'get_status',
        lambda: {'available': True, 'loaded': False, 'building': True, 'tracks': 0, 'error': None},
    )
    monkeypatch.setattr(
        neural_fingerprint_index, 'get_status',
        lambda: {'available': False, 'loaded': False, 'building': False, 'tracks': 0, 'error': None},
    )
    monkeypatch.setattr(lyrics_manager, 'get_cache_stats', lambda: {'index_loaded': True})
    monkeypatch.setattr(rsm.config, 'LYRICS_ENABLED', True)
    assert rsm.get_index_status() == {'identify': 'building', 'neural': 'needs the model file', 'lyrics': True}
