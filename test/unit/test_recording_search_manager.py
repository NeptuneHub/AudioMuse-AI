# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recording search manager: clip decoding, level normalisation and the neural search.

Main Features:
* normalize_level brings the RMS to the target and rejects silence, and the
  level before the gain travels with the results
* decode_clip decodes a real WAV through the analysis loader from bytes or a
  stream, trims it to RECORDING_SEARCH_MAX_CLIP_SECONDS, rejects bytes that are
  not audio, and refuses a stream past RECORDING_SEARCH_MAX_UPLOAD_MB while
  copying it
* run_recording_search returns the neural rows with titles attached and the
  clip's length and level, and re-raises when the index is unavailable
* the neural rows keep score, votes, offset, identified and lead, warmup
  starts the pack loading, and the index status names the pack's state
"""

import io
import wave

import numpy as np
import pytest

from tasks import recording_search_manager as rsm


def _rows(*ids):
    return [
        {'item_id': i, 'title': f't-{i}', 'author': f'a-{i}', 'album': '', 'score': 0.5}
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


def test_run_recording_search_reports_the_clip_level_before_the_gain(monkeypatch):
    quiet = (0.001 * np.sin(np.linspace(0, 400 * np.pi, 16000))).astype(np.float32)
    monkeypatch.setattr(rsm, 'decode_clip', lambda clip, filename: (quiet, 16000))
    monkeypatch.setattr(rsm, 'search_neural', lambda audio, sr, n: _rows('a'))
    payload = rsm.run_recording_search(b'x', 'c.wav', 10)
    assert payload['clip_level_db'] == pytest.approx(-63.0, abs=0.5)
    assert 'warnings' not in payload


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
    stream = io.BytesIO(_wav_bytes(1.0))
    with pytest.raises(ValueError, match='larger than 0 MB'):
        rsm.decode_clip(stream, 'clip.wav')


def test_safe_suffix_keeps_only_a_known_alphanumeric_extension():
    assert rsm._safe_suffix('rec.webm') == '.webm'
    assert rsm._safe_suffix('a b/c d.mp3!') == '.mp3'
    assert rsm._safe_suffix('IMG_0001.MOV') == '.mov'
    assert rsm._safe_suffix('noext') == '.bin'
    assert rsm._safe_suffix(None) == '.bin'


def _fake_pipeline(monkeypatch, searcher):
    monkeypatch.setattr(
        rsm, 'decode_clip', lambda file_bytes, filename: (np.ones(16000, dtype=np.float32), 16000)
    )
    monkeypatch.setattr(rsm, 'normalize_level', lambda audio, target_db=None: audio)
    monkeypatch.setattr(rsm, 'search_neural', searcher)


def test_run_recording_search_returns_the_neural_rows_with_the_clip_length(monkeypatch):
    seen = {}

    def neural(audio, sr, n):
        seen.update(sr=sr, n=n, samples=int(audio.shape[0]))
        return _rows('c', 'd')

    _fake_pipeline(monkeypatch, neural)
    payload = rsm.run_recording_search(b'x', 'c.wav', 10)
    assert seen == {'sr': 16000, 'n': 10, 'samples': 16000}
    assert [row['item_id'] for row in payload['results']] == ['c', 'd']
    assert payload['count'] == 2
    assert payload['clip_seconds'] == 1.0
    assert 'mode' not in payload
    assert 'transcript' not in payload


def test_the_count_defaults_to_the_config_value_and_never_drops_below_one(monkeypatch):
    seen = []
    _fake_pipeline(monkeypatch, lambda audio, sr, n: seen.append(n) or [])
    monkeypatch.setattr(rsm.config, 'RECORDING_SEARCH_DEFAULT_N_RESULTS', 42)
    rsm.run_recording_search(b'x', 'c.wav')
    rsm.run_recording_search(b'x', 'c.wav', 0)
    assert seen == [42, 1]


def test_an_unavailable_index_raises(monkeypatch):
    def down(audio, sr, n):
        raise RuntimeError('down')

    _fake_pipeline(monkeypatch, down)
    with pytest.raises(RuntimeError):
        rsm.run_recording_search(b'x', 'c.wav', 10)


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
    rows = rsm.search_neural(np.ones(16000, dtype=np.float32), 16000, 5)
    assert rows == [
        {
            'item_id': 'song', 'score': 0.81, 'votes': 30.5, 'offset_seconds': 95.5, 'identified': True,
            'lead': 0.4, 'title': 'Song', 'author': 'Band', 'album': 'LP',
        }
    ]


def test_warmup_starts_the_pack_when_it_is_available_but_not_loaded(monkeypatch):
    from tasks import neural_fingerprint_index

    started = []
    monkeypatch.setattr(
        neural_fingerprint_index, 'get_status',
        lambda: {'available': True, 'loaded': False, 'building': False, 'tracks': 0, 'error': None},
    )
    monkeypatch.setattr(neural_fingerprint_index, 'start_background_load', lambda: started.append(True))
    monkeypatch.setattr(rsm._TIMER, 'arm', lambda duration, callback: True)
    status = rsm.warmup_recording_models()
    assert started == [True]
    assert status['loaded'] is False
    assert status['models'] == {'neural': False}
    assert status['expiry_seconds'] == rsm.config.RECORDING_SEARCH_WARMUP_DURATION


def test_index_status_names_the_pack_state(monkeypatch):
    from tasks import neural_fingerprint_index

    states = [
        ({'available': False, 'loaded': False, 'building': False}, 'needs the model file'),
        ({'available': True, 'loaded': False, 'building': True}, 'loading'),
        ({'available': True, 'loaded': False, 'building': False}, 'not loaded yet'),
        ({'available': True, 'loaded': False, 'building': False, 'error': 'No index yet'}, 'No index yet'),
        ({'available': True, 'loaded': False, 'building': False, 'synced': True}, 'ready'),
        ({'available': True, 'loaded': True, 'building': False}, 'ready'),
    ]
    for status, expected in states:
        monkeypatch.setattr(neural_fingerprint_index, 'get_status', lambda status=status: status)
        assert rsm.get_index_status() == {'neural': expected}
