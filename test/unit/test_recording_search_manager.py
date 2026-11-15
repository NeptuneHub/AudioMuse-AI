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
* search_by_track aligns up to three windows of a song's stored codes on the
  index with the song itself excluded, keeps the best score per song, and
  refuses a song without a fingerprint; it asks each window for a margin and
  leaves out the same title and author as the song or a kept result, a
  fingerprint twin of either (a same-average song that does not align stays)
  and results past MAX_SONGS_PER_ARTIST (the song does not count), keeps a
  result without a fingerprint, and 0 turns the fingerprint check and the cap off
* fade_edges keeps only the first and last 30 s (half of a shorter song);
  smart_fade_order keeps the first song first, chains each song into the one
  whose start matches its end, never starts later than 30 s nor ends earlier
  than 30 s before the end, never fades through silent rows (consecutive rows
  that are nearly identical), lets the last song play to its end and puts songs
  without a fingerprint last; a close fingerprint match goes to the song whose
  stored tempo can be stretched to the current one (half or double time too),
  a clearly better match still wins and an unknown tempo is never penalised;
  smart_fade_playlist reads only the stored edges and tempos of the given
  songs and orders them from the first one
"""

import io
import wave

import numpy as np
import pytest

from tasks import recording_search_manager as rsm


@pytest.fixture(autouse=True)
def neural_fingerprint_on(monkeypatch):
    monkeypatch.setattr(rsm.config, 'NEURAL_FINGERPRINT_ENABLED', True)


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


def test_a_disabled_feature_reports_it_and_warms_nothing(monkeypatch):
    monkeypatch.setattr(rsm.config, 'NEURAL_FINGERPRINT_ENABLED', False)
    assert rsm.get_index_status() == rsm._index_summary(state='disabled (NEURAL_FINGERPRINT_ENABLED=false)')
    status = rsm.warmup_recording_models()
    assert status['loaded'] is False
    assert status['models'] == {'neural': False, 'encoder': False}


def test_query_windows_cover_a_long_song_three_times_and_a_short_one_once():
    short = np.zeros((30, 128), dtype=np.float32)
    assert [w.shape[0] for w in rsm.query_windows(short)] == [30]
    long = np.arange(400, dtype=np.float32)[:, None].repeat(128, axis=1)
    windows = rsm.query_windows(long)
    assert [w.shape[0] for w in windows] == [40, 40, 40]
    assert [int(w[0, 0]) for w in windows] == [60, 180, 300]
    tight = np.arange(50, dtype=np.float32)[:, None].repeat(128, axis=1)
    assert [int(w[0, 0]) for w in rsm.query_windows(tight)] == [0, 5, 10]


def test_search_by_track_aligns_windows_of_the_stored_fingerprint_and_leaves_the_song_out(monkeypatch, tmp_path):
    from tasks import neural_fingerprint as nf
    from tasks import neural_fingerprint_index

    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((600, nf.DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.savez(tmp_path / 'pq.npz', codebook=nf.train_codebook(vectors, iterations=4))
    monkeypatch.setattr(rsm.config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'pq.npz'))
    for key in ('codebook', 'codebook_id', 'codebook_bias'):
        monkeypatch.setitem(nf._STATE, key, None)
    monkeypatch.setattr(rsm, '_stored_fingerprint', lambda item_id: nf.encode_blob(vectors) if item_id == 'song' else None)
    calls = []

    def identify_vectors(window, n_results, exclude_ids=()):
        calls.append((window.shape, n_results, tuple(exclude_ids)))
        score = 0.5 + 0.1 * len(calls)
        return [
            {'item_id': 'copy', 'score': score, 'votes': 9.0, 'offset_seconds': 1.0, 'identified': True, 'lead': 0.2},
            {'item_id': 'other', 'score': 0.3, 'votes': 2.0, 'offset_seconds': 5.0, 'identified': False, 'lead': None},
        ]

    monkeypatch.setattr(neural_fingerprint_index, 'identify_vectors', identify_vectors)
    monkeypatch.setattr(
        rsm, '_fetch_track_metadata',
        lambda ids: {i: {'title': i.title(), 'author': 'Band', 'album': ''} for i in ids},
    )
    monkeypatch.setattr(rsm, '_stored_twins', lambda ids: {})
    monkeypatch.setattr(rsm.config, 'NEURAL_FINGERPRINT_MIN_SCORE', 0.4)
    monkeypatch.setattr(rsm.config, 'NEURAL_FINGERPRINT_MIN_LEAD', 0.15)
    payload = rsm.search_by_track('song', 5)
    assert [c[0] for c in calls] == [(40, nf.DIM)] * 3
    assert all(c[1] == 25 and c[2] == ('song',) for c in calls)
    assert payload['item_id'] == 'song'
    assert [row['item_id'] for row in payload['results']] == ['copy', 'other']
    assert payload['results'][0]['score'] == pytest.approx(0.8)
    assert payload['results'][0]['identified'] is True
    assert payload['results'][0]['lead'] == pytest.approx(0.5)
    assert payload['results'][1]['identified'] is False
    assert payload['results'][1]['lead'] is None
    assert payload['results'][0]['title'] == 'Copy'
    assert payload['count'] == 2
    with pytest.raises(ValueError, match='no neural fingerprint'):
        rsm.search_by_track('missing', 5)


def _motif(rng):
    vector = rng.standard_normal(128).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _song(rng, rows, intro=None, outro=None):
    vectors = rng.standard_normal((rows, 128)).astype(np.float32)
    zone = rows * 3 // 10
    if intro is not None:
        vectors[:zone] = intro + 0.05 * rng.standard_normal((zone, 128))
    if outro is not None:
        vectors[-zone:] = outro + 0.05 * rng.standard_normal((zone, 128))
    return vectors


def _duration(rows):
    return (rows - 1) * 0.5 + 1.0


def _edges(*songs):
    return [None if song is None else rsm.fade_edges(song) for song in songs]


def test_fade_edges_keep_only_the_first_and_last_30_seconds_or_half_a_short_song():
    long_song = np.arange(400, dtype=np.float32)[:, None].repeat(4, axis=1)
    head, tail, count = rsm.fade_edges(long_song)
    assert count == 400 and head.shape[0] == tail.shape[0] == 60
    assert head[0, 0] == 0 and tail[0, 0] == 340 and tail[-1, 0] == 399
    head, tail, count = rsm.fade_edges(long_song[:50], decode=lambda part: part * 2)
    assert count == 50 and head.shape[0] == tail.shape[0] == 25 and tail[0, 0] == 50


def test_smart_fade_chains_each_song_into_the_one_whose_start_matches_its_end():
    rng = np.random.default_rng(7)
    first, second, third = _motif(rng), _motif(rng), _motif(rng)
    songs = _edges(
        _song(rng, 200, outro=first),
        _song(rng, 160, intro=second, outro=third),
        _song(rng, 240, intro=first, outro=second),
    )
    ordered = rsm.smart_fade_order(songs)
    assert [index for index, _fade in ordered] == [0, 2, 1]
    fades = dict(ordered)
    assert fades[0]['fade_in_seconds'] == 0.0
    for index, rows in ((0, 200), (1, 160), (2, 240)):
        assert fades[index]['duration_seconds'] == _duration(rows)
    for index in (2, 1):
        assert 0.0 <= fades[index]['fade_in_seconds'] <= 30.0
    for index in (0, 2):
        assert 0.0 <= fades[index]['duration_seconds'] - fades[index]['fade_out_seconds'] <= 30.0
    assert fades[1]['fade_out_seconds'] == _duration(160)


def test_smart_fade_never_fades_through_silence_at_the_edges():
    rng = np.random.default_rng(10)
    silence = _motif(rng)
    ending = _song(rng, 200)
    ending[-10:] = silence
    opening = _song(rng, 200)
    opening[:6] = silence
    ordered = rsm.smart_fade_order(_edges(ending, opening))
    fades = dict(ordered)
    assert [index for index, _fade in ordered] == [0, 1]
    assert fades[0]['fade_out_seconds'] <= _duration(200) - 5.0
    assert fades[1]['fade_in_seconds'] >= 3.0
    still = np.repeat(silence[None, :], 120, axis=0)
    lone = rsm.smart_fade_order(_edges(ending, still))
    assert [index for index, _fade in lone] == [0, 1]
    assert dict(lone)[1]['fade_in_seconds'] >= 0.0


def test_smart_fade_puts_songs_without_a_fingerprint_last_and_a_lone_song_plays_whole():
    rng = np.random.default_rng(8)
    ordered = rsm.smart_fade_order(_edges(None, _song(rng, 50), None, _song(rng, 5)))
    assert [index for index, _fade in ordered][2:] == [0, 2]
    assert ordered[0][0] == 1
    assert ordered[0][1]['fade_in_seconds'] == 0.0
    assert ordered[2][1] is None and ordered[3][1] is None
    assert rsm.smart_fade_order([None, None]) == [(0, None), (1, None)]
    lone = rsm.smart_fade_order(_edges(_song(rng, 3)))
    assert lone == [(0, {'fade_in_seconds': 0.0, 'fade_out_seconds': 2.0, 'duration_seconds': 2.0})]


def _blend(joint, other, share):
    mixed = share * joint + np.sqrt(1.0 - share ** 2) * other
    return mixed / np.linalg.norm(mixed)


def test_smart_fade_breaks_a_close_fingerprint_match_on_the_stored_tempo():
    rng = np.random.default_rng(11)
    joint, other = _motif(rng), _motif(rng)
    other -= (other @ joint) * joint
    other /= np.linalg.norm(other)
    current = _song(rng, 200, outro=joint)
    best = _song(rng, 200, intro=joint)
    close = _song(rng, 200, intro=_blend(joint, other, 0.97))
    distant = _song(rng, 200, intro=_blend(joint, other, 0.6))

    def order(songs, tempos=None):
        return [index for index, _fade in rsm.smart_fade_order(_edges(*songs), tempos)]

    assert order((current, best, close)) == [0, 1, 2]
    assert order((current, best, close), [120.0, 150.0, 123.0]) == [0, 2, 1]
    assert order((current, best, close), [120.0, 150.0, 246.0]) == [0, 2, 1]
    assert order((current, best, close), [120.0, None, 123.0]) == [0, 1, 2]
    assert order((current, best, close), [120.0, 0.0, 123.0]) == [0, 1, 2]
    assert order((current, best, distant), [120.0, 150.0, 123.0]) == [0, 1, 2]


def test_smart_fade_playlist_reads_the_stored_edges_and_orders_from_the_first_song(monkeypatch):
    rng = np.random.default_rng(9)
    joint, other_motif = _motif(rng), _motif(rng)
    stored = {
        'song': _song(rng, 200, outro=joint),
        'copy': _song(rng, 200, intro=other_motif),
        'other': _song(rng, 200, intro=joint),
    }
    asked = []

    def stored_edges(ids):
        asked.append(list(ids))
        return {i: rsm.fade_edges(stored[i]) for i in ids if i in stored}

    def stored_tempos(ids):
        asked.append(('tempo', list(ids)))
        return {'song': 120.0, 'other': 121.0}

    monkeypatch.setattr(rsm, '_stored_edges', stored_edges)
    monkeypatch.setattr(rsm, '_stored_tempos', stored_tempos)
    ordered = rsm.smart_fade_playlist(['song', 'copy', 'ghost', 'other'])
    assert asked == [['song', 'copy', 'ghost', 'other'], ('tempo', ['song', 'copy', 'ghost', 'other'])]
    assert [index for index, _fade in ordered] == [0, 3, 1, 2]
    fades = dict(ordered)
    assert fades[0]['fade_in_seconds'] == 0.0
    assert 0.0 <= _duration(200) - fades[0]['fade_out_seconds'] <= 30.0
    assert 0.0 <= fades[3]['fade_in_seconds'] <= 30.0
    assert fades[1]['fade_out_seconds'] == _duration(200)
    assert fades[2] is None


def _unit_rows(rows):
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_search_by_track_leaves_out_repeats_fingerprint_twins_and_extra_songs_by_one_artist(monkeypatch):
    from tasks import neural_fingerprint, neural_fingerprint_index

    rng = np.random.default_rng(12)

    def random_rows():
        return _unit_rows(rng.standard_normal((200, 128)))

    seed, hit = random_rows(), random_rows()
    rows = {
        'seed': seed, 'hit': hit, 'repeat': random_rows(),
        'remix': _unit_rows(hit + 0.05 * rng.standard_normal((200, 128))),
        'lookalike': hit[rng.permutation(200)],
        'cover': _unit_rows(seed + 0.05 * rng.standard_normal((200, 128))),
        'named': random_rows(), 'b1': random_rows(), 'b2': random_rows(), 'b3': random_rows(), 'b4': random_rows(),
        'extra1': random_rows(), 'extra2': random_rows(),
    }
    titles = {
        'seed': ('Seed', 'Band'), 'hit': ('Hit', 'Other'), 'repeat': (' hit ', 'OTHER'),
        'remix': ('Hit (extended version)', 'Other'), 'lookalike': ('Lookalike', 'Third'),
        'cover': ('Seed (remastered)', 'Band feat. Guest'), 'named': ('SEED', 'band'),
        'b1': ('One', 'Band'), 'b2': ('Two', 'Band'), 'b3': ('Three', 'Band'), 'b4': ('Four', ' band'),
        'none': ('No fingerprint', 'Fourth'), 'extra1': ('Extra one', 'Fifth'), 'extra2': ('Extra two', 'Sixth'),
    }
    ranked = ['hit', 'repeat', 'remix', 'lookalike', 'cover', 'named', 'b1', 'b2', 'b3', 'b4', 'none', 'nometa', 'extra1', 'extra2']
    asked = []

    def identify_vectors(window, n_results, exclude_ids=()):
        asked.append(n_results)
        return [{'item_id': i, 'score': 1.0 - 0.01 * n, 'votes': 1.0, 'offset_seconds': 0.0, 'identified': False, 'lead': None}
                for n, i in enumerate(ranked)]

    def twin(item_rows):
        mean = item_rows.mean(axis=0)
        return {'mean': mean / np.linalg.norm(mean), 'codes': item_rows}

    monkeypatch.setattr(rsm, '_stored_fingerprint', lambda item_id: b'blob')
    monkeypatch.setattr(neural_fingerprint, 'decode_blob', lambda blob: seed)
    monkeypatch.setattr(neural_fingerprint, 'decode_codes', lambda codes: codes)
    monkeypatch.setattr(neural_fingerprint_index, 'identify_vectors', identify_vectors)
    monkeypatch.setattr(rsm, '_stored_twins', lambda ids: {i: twin(rows[i]) for i in ids if i in rows})
    monkeypatch.setattr(rsm, '_fetch_track_metadata', lambda ids: {i: {'title': titles[i][0], 'author': titles[i][1], 'album': ''} for i in ids if i in titles})
    monkeypatch.setattr(rsm.config, 'DUPLICATE_DISTANCE_THRESHOLD_NEURAL_FINGERPRINT', 0.25)
    monkeypatch.setattr(rsm.config, 'MAX_SONGS_PER_ARTIST', 3)
    payload = rsm.search_by_track('seed', 7)
    assert asked == [27, 27, 27]
    assert [row['item_id'] for row in payload['results']] == ['hit', 'lookalike', 'b1', 'b2', 'b3', 'none', 'extra1']
    assert payload['count'] == 7 and payload['results'][0]['lead'] == pytest.approx(0.03)
    monkeypatch.setattr(rsm.config, 'DUPLICATE_DISTANCE_THRESHOLD_NEURAL_FINGERPRINT', 0.0)
    monkeypatch.setattr(rsm.config, 'MAX_SONGS_PER_ARTIST', 0)
    assert [row['item_id'] for row in rsm.search_by_track('seed', 7)['results']] == ['hit', 'remix', 'lookalike', 'cover', 'b1', 'b2', 'b3']


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


def test_warmup_starts_the_pack_when_it_is_available_but_not_loaded_and_preloads_the_encoder(monkeypatch):
    from tasks import neural_fingerprint, neural_fingerprint_index

    started = []
    monkeypatch.setattr(
        neural_fingerprint_index, 'get_status',
        lambda: {'available': True, 'loaded': False, 'building': False, 'tracks': 0, 'error': None},
    )
    monkeypatch.setattr(neural_fingerprint_index, 'start_background_load', lambda: started.append(True))
    monkeypatch.setattr(neural_fingerprint, 'warm_session', lambda: started.append('encoder') or True)
    monkeypatch.setattr(rsm._TIMER, 'arm', lambda duration, callback: True)
    status = rsm.warmup_recording_models()
    assert started == [True, 'encoder']
    assert status['loaded'] is False
    assert status['models'] == {'neural': False, 'encoder': True}
    assert status['expiry_seconds'] == rsm.config.RECORDING_SEARCH_WARMUP_DURATION


def test_the_idle_timer_releases_the_encoder_and_the_cell_cache_but_not_the_index(monkeypatch):
    from tasks import neural_fingerprint, neural_fingerprint_index

    released = []
    monkeypatch.setattr(neural_fingerprint, 'unload_session', lambda: released.append('encoder') or True)
    monkeypatch.setattr(neural_fingerprint_index, 'drop_cell_cache', lambda: released.append('cells'))
    monkeypatch.setattr(neural_fingerprint_index, 'unload', lambda: released.append('index') or True)
    rsm._unload_expired()
    assert released == ['encoder', 'cells']


def test_index_status_carries_the_counts_and_names_the_state(monkeypatch):
    from tasks import neural_fingerprint_index

    base = {'tracks': 0, 'cells': 0, 'cache_mb': 0.0}
    states = [
        ({'available': False, 'loaded': False, 'building': False}, 'needs the model file'),
        ({'available': True, 'loaded': False, 'building': True}, 'loading'),
        ({'available': True, 'loaded': False, 'building': False}, 'not loaded yet'),
        ({'available': True, 'loaded': False, 'building': False, 'error': 'No index yet'}, 'No index yet'),
    ]
    for status, expected in states:
        monkeypatch.setattr(neural_fingerprint_index, 'get_status', lambda status=status: {**base, **status})
        summary = rsm.get_index_status()
        assert summary['state'] == expected
        assert summary['loaded'] is False
        assert summary['song_count'] == 0
    loaded = {'available': True, 'loaded': True, 'building': False, 'tracks': 12345, 'cells': 111, 'cache_mb': 42.5}
    monkeypatch.setattr(neural_fingerprint_index, 'get_status', lambda: loaded)
    assert rsm.get_index_status() == {'loaded': True, 'song_count': 12345, 'cells': 111, 'cache_mb': 42.5, 'state': 'ready'}
