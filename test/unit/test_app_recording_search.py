# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recording search blueprint: request validation and error mapping.

Main Features:
* No clip or a bad count answers 400, a clip over
  RECORDING_SEARCH_MAX_UPLOAD_MB 413
* The manager's ValueError answers 400 and IndexUnavailable 503 with its
  message, any other exception (a plain RuntimeError too) a generic 500
  without internal detail
* A search streams the clip to the manager (never read into memory here),
  passes the count (config default when missing) and scopes the results to it
* by_track answers 400 without a song, with a bad count or an unknown song,
  passes the canonical id through, scopes the results and maps the errors
* No response carries an internal canonical id: results use the selected
  server's ids and the echoed id goes through provider_echo_id
* by_track ignores a smart_fade field; static/song_player.js reorders through
  the smart_fade endpoint and plays the decoded songs on one device-rate clock
  started in the Play click (Safari); the playing row seeks up to its fade;
  sources start on whole samples, then a fresh source plays on as decoded; a
  song the browser cannot load is announced and skipped
* With Smart fade every song plays at the gain the transition gives; without
  a beat lock it is the one "Default crossfade"; only the playing song's
  transition is planned, the next row's icon says how its song comes in; the
  page only executes the answer (gains, speeds, volumes, filters),
  computing none of it, and uses none of the old names
* The smart_fade endpoint answers 400 without a list, resolves the ids, returns
  the manager's order (keep_order only on JSON true) under the ids as sent,
  null times without a fingerprint, never a canonical id
* play (fingerprint off too) streams a song while it downloads, a WAV as
  bit-exact FLAC, an unreadable file as it is, then removes the copy; outside
  the library 404, a failed download 502
* transition answers 400 for bad from_id, to_id, out or in (other names
  refused) or an unknown kind in the optional transitions (the kinds on for
  that call only), 404 outside the library, plans from both downloads,
  answers the JSON in its order (type ... info), removes them; a pair
  without a steady beat gets the crossfade, 400 only without room
* warmup relays the manager status; with NEURAL_FINGERPRINT_ENABLED false
  every route but play answers 503 naming the flag and the page says so
* The page gets the HTTPS port, zero when it is off, and why it did not start
"""

import importlib.util
import io
import json
import os
import sys
import threading

import numpy as np
import pytest


def _load_bp_module():
    mod_name = 'app_recording_search'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'app_recording_search.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bp_mod():
    return _load_bp_module()


@pytest.fixture
def client(bp_mod):
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp_mod.recording_search_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture(autouse=True)
def empty_play_cache():
    import app_play_cache

    app_play_cache.clear()
    yield
    app_play_cache.clear()


@pytest.fixture(autouse=True)
def neural_fingerprint_on(monkeypatch):
    import config

    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', True)


@pytest.fixture(autouse=True)
def neutral_server_scope(monkeypatch):
    import app_helper
    import app_server_context

    monkeypatch.setattr(app_server_context, 'resolve_request_server_id', lambda data=None: None)
    monkeypatch.setattr(
        app_server_context,
        'scope_results',
        lambda rows, requested_n=None, id_key='item_id', translate=True: (
            rows[:requested_n] if requested_n else rows
        ),
    )
    monkeypatch.setattr(app_helper, 'attach_song_features', lambda rows, id_key='item_id': rows)


def _post(client, data):
    return client.post('/api/recording_search/search', data=data, content_type='multipart/form-data')


def _clip(name='a.wav'):
    return (io.BytesIO(b'abc'), name)


def _patch_manager(monkeypatch, func):
    import tasks.recording_search_manager as rsm

    monkeypatch.setattr(rsm, 'run_recording_search', func)


_INDEX_SUMMARY = {'loaded': True, 'song_count': 12, 'cells': 3, 'cache_mb': 0.5, 'state': 'ready'}


def _render_page(bp_mod, monkeypatch, status):
    import tls_listener
    import tasks.recording_search_manager as rsm

    captured = {}
    monkeypatch.setattr(tls_listener, 'https_status', lambda: status)
    monkeypatch.setattr(rsm, 'get_index_status', lambda: _INDEX_SUMMARY)
    monkeypatch.setattr(bp_mod, 'render_template', lambda name, **context: captured.update(context) or 'page')
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp_mod.recording_search_bp)
    assert app.test_client().get('/recording_search').status_code == 200
    return captured


def test_the_page_carries_the_built_in_https_port_or_zero_and_the_start_error(bp_mod, monkeypatch):
    running = _render_page(bp_mod, monkeypatch, {'enabled': True, 'running': True, 'port': 8443, 'error': None})
    assert running['https_port'] == 8443
    assert running['https_error'] == ''
    assert running['index_status'] == _INDEX_SUMMARY
    disabled = _render_page(bp_mod, monkeypatch, {'enabled': False, 'running': False, 'port': 0, 'error': 'disabled by FLASK_HTTPS_PORT'})
    assert disabled['https_port'] == 0
    assert disabled['https_error'] == 'disabled by FLASK_HTTPS_PORT'
    failed = _render_page(bp_mod, monkeypatch, {'enabled': True, 'running': False, 'port': 8443, 'error': 'address already in use'})
    assert failed['https_port'] == 8443
    assert failed['https_error'] == 'address already in use'


def test_missing_clip_answers_400(client):
    response = _post(client, {'n_results': '5'})
    assert response.status_code == 400
    assert 'clip' in response.get_json()['error']


def test_bad_count_answers_400(client):
    response = _post(client, {'clip': _clip(), 'n_results': 'ten'})
    assert response.status_code == 400


def test_oversized_clip_answers_413(client, monkeypatch):
    import config

    monkeypatch.setattr(config, 'RECORDING_SEARCH_MAX_UPLOAD_MB', 0)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 413


def test_manager_value_error_answers_400_with_the_registry_message_only(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError('The codebook /app/model/nmfp_pq.npy has shape (8, 16)')

    _patch_manager(monkeypatch, boom)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 400
    from error.error_dictionary import ERR_RECORDING_REJECTED, get_default_message
    assert response.get_json()['error_code'] == ERR_RECORDING_REJECTED
    assert response.get_json()['error'] == get_default_message(ERR_RECORDING_REJECTED)
    assert '/app/model' not in response.get_data(as_text=True), (
        'a broad except ValueError also catches library errors; the body carries only '
        'prewritten text and the log carries the exception'
    )


def test_index_unavailable_answers_503_with_its_code_and_a_plain_runtime_error_stays_generic(client, monkeypatch):
    from tasks.neural_fingerprint_index import IndexUnavailable

    def not_ready(*args, **kwargs):
        raise IndexUnavailable('No neural fingerprints are stored yet. Run analysis first.')

    _patch_manager(monkeypatch, not_ready)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 503
    assert response.get_json()['error_code'] == 3005
    assert 'Run analysis first' not in response.get_data(as_text=True)

    def internal(*args, **kwargs):
        raise RuntimeError('The fingerprint of fp_secret no longer matches the index')

    _patch_manager(monkeypatch, internal)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 500
    assert 'fp_secret' not in response.get_json()['error']
    assert 'container logs' in response.get_json()['error']


def test_unexpected_error_answers_generic_500_without_detail(client, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError('secret-internal-detail')

    _patch_manager(monkeypatch, boom)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 500
    assert 'secret' not in response.get_json()['error']


def test_success_passes_the_clip_through_scopes_results_and_reports_count(client, monkeypatch):
    seen = {}

    def fake(clip, filename, n_results):
        seen.update(file_bytes=clip.read(), filename=filename, n_results=n_results)
        return {
            'clip_seconds': 20.0,
            'clip_level_db': -14.0,
            'results': [
                {'item_id': 'a', 'title': 'A', 'author': 'x', 'score': 0.9},
                {'item_id': 'b', 'title': 'B', 'author': 'y', 'score': 0.8},
            ],
            'count': 2,
        }

    _patch_manager(monkeypatch, fake)
    response = _post(client, {'clip': _clip('rec.webm'), 'n_results': '1'})
    body = response.get_json()
    assert response.status_code == 200
    assert seen == {'file_bytes': b'abc', 'filename': 'rec.webm', 'n_results': 1}
    assert body['count'] == 1
    assert [row['item_id'] for row in body['results']] == ['a']


def test_count_defaults_to_the_config_default(client, monkeypatch):
    import config

    seen = {}

    def fake(clip, filename, n_results):
        seen.update(n_results=n_results)
        return {'clip_seconds': 1.0, 'clip_level_db': -14.0, 'results': [], 'count': 0}

    _patch_manager(monkeypatch, fake)
    response = _post(client, {'clip': _clip('rec.webm')})
    assert response.status_code == 200
    assert seen == {'n_results': config.RECORDING_SEARCH_DEFAULT_N_RESULTS}


def _post_track(client, body):
    return client.post('/api/recording_search/by_track', json=body)


def test_by_track_answers_400_without_a_song_or_with_an_unknown_one(client, monkeypatch):
    import app_server_context

    assert _post_track(client, {}).status_code == 400
    assert _post_track(client, {'item_id': 'x', 'n_results': 'ten'}).status_code == 400

    def unknown(raw_id, data=None):
        raise ValueError('unknown song')

    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', unknown)
    response = _post_track(client, {'item_id': 'x'})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'The request is invalid.'


def test_by_track_passes_the_canonical_id_and_scopes_the_results(client, monkeypatch):
    import app_server_context
    import tasks.recording_search_manager as rsm

    seen = {}
    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: f'fp_{raw_id}')

    def fake(item_id, n_results):
        seen.update(item_id=item_id, n_results=n_results)
        return {
            'item_id': item_id,
            'results': [
                {'item_id': 'a', 'title': 'A', 'author': 'x', 'score': 0.9},
                {'item_id': 'b', 'title': 'B', 'author': 'y', 'score': 0.8},
            ],
            'count': 2,
        }

    monkeypatch.setattr(rsm, 'search_by_track', fake)
    response = _post_track(client, {'item_id': '42', 'n_results': 1})
    body = response.get_json()
    assert response.status_code == 200
    assert seen == {'item_id': 'fp_42', 'n_results': 1}
    assert body['item_id'] == '42'
    assert body['count'] == 1
    assert [row['item_id'] for row in body['results']] == ['a']


def _translating_scope(monkeypatch):
    import app_server_context

    def scope(rows, requested_n=None, id_key='item_id', translate=True):
        kept = []
        for row in rows:
            if str(row[id_key]).startswith('fp_'):
                row = dict(row, **{id_key: row[id_key][3:].upper()})
            kept.append(row)
        return kept[:requested_n] if requested_n else kept

    monkeypatch.setattr(app_server_context, 'scope_results', scope)
    monkeypatch.setattr(app_server_context, 'provider_echo_id', lambda raw_id: raw_id[3:].upper() if str(raw_id).startswith('fp_') else raw_id)


def test_no_response_of_this_api_ever_carries_a_canonical_id(client, monkeypatch):
    import app_server_context
    import tasks.recording_search_manager as rsm

    _translating_scope(monkeypatch)
    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: raw_id if str(raw_id).startswith('fp_') else f'fp_{raw_id}')
    rows = [
        {'item_id': 'fp_a', 'title': 'A', 'author': 'x', 'score': 0.9, 'identified': True, 'lead': 0.5, 'offset_seconds': 1.0},
        {'item_id': 'fp_b', 'title': 'B', 'author': 'y', 'score': 0.8, 'identified': False, 'lead': None, 'offset_seconds': 0.0},
    ]
    monkeypatch.setattr(rsm, 'search_by_track', lambda item_id, n_results: {'item_id': item_id, 'results': [dict(r) for r in rows], 'count': 2})
    _patch_manager(monkeypatch, lambda clip, filename, n_results: {'clip_seconds': 20.0, 'clip_level_db': -14.0, 'results': [dict(r) for r in rows], 'count': 2})

    for sent in ('42', 'fp_42'):
        response = _post_track(client, {'item_id': sent})
        assert response.status_code == 200
        assert 'fp_' not in response.get_data(as_text=True)
        assert response.get_json()['item_id'] == '42'
        assert [row['item_id'] for row in response.get_json()['results']] == ['A', 'B']
    response = _post(client, {'clip': _clip('rec.webm')})
    assert response.status_code == 200
    assert 'fp_' not in response.get_data(as_text=True)
    assert [row['item_id'] for row in response.get_json()['results']] == ['A', 'B']


def test_by_track_maps_the_manager_errors_like_the_clip_search(client, monkeypatch):
    import tasks.recording_search_manager as rsm

    def rejected(item_id, n_results):
        raise ValueError('This song has no neural fingerprint yet.')

    monkeypatch.setattr(rsm, 'search_by_track', rejected)
    response = _post_track(client, {'item_id': 'x'})
    assert response.status_code == 400
    assert response.get_json()['error_code'] == 1014
    assert 'neural fingerprint the analysis computes' in response.get_json()['error']

    from tasks.neural_fingerprint_index import IndexUnavailable

    def down(item_id, n_results):
        raise IndexUnavailable('No neural fingerprint index is built yet.')

    monkeypatch.setattr(rsm, 'search_by_track', down)
    assert _post_track(client, {'item_id': 'x'}).status_code == 503

    def boom(item_id, n_results):
        raise KeyError('secret')

    monkeypatch.setattr(rsm, 'search_by_track', boom)
    response = _post_track(client, {'item_id': 'x'})
    assert response.status_code == 500
    assert 'secret' not in response.get_json()['error']


def test_by_track_knows_nothing_of_smart_fade(client, monkeypatch):
    import inspect

    import tasks.recording_search_manager as rsm

    assert list(inspect.signature(rsm.search_by_track).parameters) == ['item_id', 'n_results']
    calls = []

    def fake(item_id, n_results):
        calls.append((item_id, n_results))
        return {'item_id': item_id, 'results': [{'item_id': 'a', 'title': 'A', 'author': 'x', 'score': 0.9}], 'count': 1}

    monkeypatch.setattr(rsm, 'search_by_track', fake)
    response = _post_track(client, {'item_id': 'x', 'n_results': 5, 'smart_fade': True})
    assert response.status_code == 200
    assert calls == [('x', 5)]
    assert sorted(response.get_json()) == ['count', 'item_id', 'results']
    assert 'fade_in_seconds' not in response.get_json()['results'][0]


def _repo_file(*parts):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', *parts), encoding='utf-8') as handle:
        return handle.read()


def test_the_page_reorders_by_track_results_through_the_smart_fade_endpoint():
    source = _repo_file('templates', 'recording_search.html')
    player = _repo_file('static', 'song_player.js')
    app_source = _repo_file('app.py')
    assert 'body: JSON.stringify({ item_id: selectedSong.item_id, n_results: limit })' in source
    assert 'smart_fade: true' not in source and 'request.smart_fade' not in source
    assert "currentResults = await AudioMusePlayer.arrange(resultsList, data.results || [], { seed });" in source
    assert "'smart_fade': url_for('recording_search_bp.recording_search_smart_fade_api')," in app_source
    assert "const body = { item_ids: [...(seed ? [seed.item_id] : []), ...rows.map(row => String(row.item_id))] };" in player
    assert 'if (!nextChoice.smart || !rows.length) return rows;' in player


def test_the_player_cuts_and_fades_decoded_songs_itself_on_an_audio_clock_started_in_the_play_click():
    page = _repo_file('templates', 'recording_search.html')
    source = _repo_file('static', 'song_player.js')
    start = source.index('async function startPlayback(row) {')
    assert source.index('unlockAudio();', start) < source.index('stopAll();', start)
    assert "navigator.audioSession.type = 'playback'" in source
    assert 'audioCtx.decodeAudioData(data)' in source and 'source.start(when, onGrid(Math.max(0, second - piece.start)));' in source
    for code in (source, page):
        assert 'new Audio(' not in code and '.currentTime =' not in code and '.volume =' not in code
        assert 'X-Switch-At' not in code
    assert 'state.mix = planned ? mix : null;' in source
    assert 'separatedStretch' not in source and 'stretchSegment' not in source
    assert 'follow(node.gain.gain, side.volume, state.handoverAt, span);' in source
    assert 'if (values) param.setValueCurveAtTime(Float32Array.from(values), from, length);' in source
    assert 'const outgoing = state.node;' in source and 'stopVoice(outgoing, resume);' in source
    assert 'AudioMusePlayer.show(resultsList);' in page and 'function startSource(' not in page
    assert "if (choice && choice.smart) return { songs: songs.map(withFade), beatMatch: true, stopAtEnd: false, label: 'Smart fade' };" in source


def test_the_player_mixes_with_the_filter_and_fader_curves_the_api_gives():
    source = _repo_file('static', 'song_player.js')
    assert 'source.connect(fader);' in source and 'fader.connect(input);' in source and 'air.connect(tone);' in source and 'tone.connect(mid);' in source
    assert 'mid.connect(high);' in source and 'high.connect(gain);' in source and 'mid.connect(gain);' not in source
    assert "air.type = 'lowpass';" in source and 'air.frequency.value = audioCtx.sampleRate / 2;' in source
    assert "mid.type = 'peaking';" in source and 'mid.gain.value = 0;' in source and "tone.type = filter || 'lowshelf';" in source
    assert "high.type = 'highshelf';" in source and 'high.gain.value = 0;' in source
    mix = source[source.index('} else if (mix && state.pieces && at(mix.start) > now + 0.1) {'):source.index('swapView = {\n')]
    for line in ('[[outgoing, mix.from, outgoingSource], [blend, mix.to, blend.single.source]].forEach(([node, side, source]) => {',
                 'node.tone.frequency.value = mix.filters.bass_hz;', 'node.mid.frequency.value = mix.filters.mid_hz;', 'node.mid.Q.value = mix.filters.mid_q;',
                 'node.high.frequency.value = mix.filters.high_hz;',
                 'follow(node.high.gain, side.high_db, state.handoverAt, span);',
                 'follow(node.tone.gain, side.bass_db, state.handoverAt, span);',
                 'follow(node.mid.gain, side.mid_db, state.handoverAt, span);',
                 'follow(blend.tone.frequency, mix.to.highpass_hz, state.handoverAt, span);'):
        assert line in mix, line
    assert "const blend = startPieceVoice(state.next, state.pieces.to, state.handoverAt, mix.to.start, mix.to.highpass_hz ? 'highpass' : 'lowshelf');" in mix
    assert 'if (mix.to.highpass_hz) blend.tone.Q.value = -10 * Math.log10(2);' in mix
    assert 'lowpass_hz' not in source and 'mix.to.delay' not in source and 'incomingAt' not in source
    assert 'follow(source.playbackRate, side.speed, state.handoverAt, span);' in mix
    assert 'const outgoingSource = switchToPiece(outgoing, state.pieces.from, state.handoverAt);' in mix
    fallback = source[source.index('const span = Math.min(crossfadeSeconds(),'):]
    fallback = fallback[:fallback.index('state.scheduled = [entry];')]
    assert 'fallbackCurves(state, sourceAt, offset, span)' in fallback
    assert 'state.node.gain.gain.setValueCurveAtTime(fall, state.handoverAt, span);' in fallback
    assert 'startSource(state.next, state.handoverAt, offset, fading ? span : 0, rise)' in fallback
    assert 'function startSource(track, when, offset, fadeIn, curve = FADE_IN) {' in source
    assert 'if (fadeIn) voice.gain.gain.setValueCurveAtTime(curve, voice.when, fadeIn);' in source
    assert 'const FADE_OUT = FADE_IN.slice().reverse();' in source and 'linearRampToValueAtTime' not in source
    assert 'return playMode.beatMatch ? SMART_CROSSFADE_SECONDS : CROSSFADE_SECONDS;' in source
    assert 'const SMART_CROSSFADE_SECONDS = 7.5;' in source
    assert 'MUFFLE_TO_HZ' not in source and 'Filtered fade' not in source and 'createDelay(' not in source


def test_every_source_starts_on_a_whole_sample_and_each_mix_hands_over_to_a_fresh_source():
    source = _repo_file('static', 'song_player.js')
    assert 'return Math.round(seconds * audioCtx.sampleRate) / audioCtx.sampleRate;' in source
    assert 'track, air, chainGain, tone, mid, high, gain, level, when: onGrid(when), offset: onGrid(Math.max(0, offset)),' in source
    assert 'const joinAt = onGrid(when + piece.until - second);' in source
    assert 'const at = second => onGrid(state.node.when + second - state.node.offset);' in source
    mix = source[source.index('} else if (mix && state.pieces && at(mix.start) > now + 0.1) {'):source.index('swapView = {\n')]
    assert 'const resume = onGrid(state.handoverAt + mix.duration);' in mix
    assert 'const span = resume - state.handoverAt;' in mix
    assert 'stopVoice(blend, resume + JOIN_SECONDS);' in mix and 'const JOIN_SECONDS = 0.023;' in source
    assert 'outgoing.gain.gain.setValueAtTime(0, resume + 1 / audioCtx.sampleRate);' in mix
    assert 'blend.gain.gain.setValueCurveAtTime(FADE_OUT, resume + 1 / audioCtx.sampleRate, JOIN_SECONDS);' in mix
    assert 'const entry = startSource(state.next, resume, mix.to.continue_from, JOIN_SECONDS);' in mix
    assert 'state.scheduled = [blend, entry];' in mix


def test_the_page_takes_every_number_of_a_transition_from_the_api_and_computes_none():
    source = _repo_file('static', 'song_player.js')
    plan = source[source.index('function runPlans(batch) {'):source.index('function showTransition(')]
    assert 'new URLSearchParams({ from_id: from.item_id, to_id: to.item_id, out: from.end, in: to.start });' in plan
    assert 'return validPlan(plan) ? plan : null;' in plan
    play = source[source.index('async function playSong(index) {'):source.index('async function startPlayback(')]
    assert 'const plan = await Promise.race([planned, new Promise(resolve => setTimeout(resolve, left * 1000))]);' in play
    assert 'if (plan) levels.set(track, plan.from.gain);' in play
    assert 'if (plan === undefined) planned.then(late => { if (late && token === playToken) glide(state, late.from.gain); });' in play
    assert 'if (planned) levels.set(next, mix.to.gain);' in source
    load = source[source.index('function loadPiece(track, start, seconds) {'):source.index('function pieceAt(')]
    assert '?start=${start}&seconds=${seconds}`;' in load
    assert "const length = Number(response.headers.get('X-Song-Seconds'));" in load
    assert 'const buffer = await decode(await response.arrayBuffer());' in load
    voice = source[source.index('function makeVoice('):source.index('function endFeed(')]
    assert 'level.gain.value = levelOf(track);' in voice and 'gain.connect(level);' in voice and 'level.connect(audioCtx.destination);' in voice
    assert 'state.mix.lead' not in source and 'state.mix.buffer' not in source
    for gone in ('loudnessGain', 'kWeighting', 'safeCurves', 'shapeCurves', 'mixLevels', 'tempoGlide', 'swapBass', 'fadeMids', 'powerCurve',
                 'TARGET_LUFS', 'PEAK_CEILING', 'MIX_HEADROOM', 'BASS_CUT_DB', 'MID_CUT_DB', 'RISE_FROM_HZ', 'SHAPE_LIMIT_DB', 'ANCHOR_SECONDS',
                 'levels: ', 'mix_at', 'current_id', 'in_at', 'mix.current', 'mix.next', 'tempo_current'):
        assert gone not in source, gone


def test_a_plan_without_a_beat_lock_plays_the_labelled_default_crossfade_on_its_beat_and_section_points():
    source = _repo_file('static', 'song_player.js')
    mix = source[source.index('} else if (mix && state.pieces && at(mix.start) > now + 0.1) {'):source.index('swapView = {\n')]
    assert 'mix.from.lowpass_hz' not in mix
    assert "crossfade: { kind: 'crossfade', label: 'Default crossfade'," in source
    assert 'showTransition(playQueue[index + 1], mix)' in source
    assert 'showTransition(playQueue[index + 1], null)' in source
    assert 'no transition plan' not in source and "'plain'" not in source
    fallback = source[source.index('const span = Math.min(crossfadeSeconds(),'):]
    assert 'const fading = fadeEnd !== null || playMode.beatMatch;' in source
    assert 'stopVoice(state.node, state.handoverAt + span);' in fallback
    assert 'handsOver' not in source and 'handover_at' not in source and 'Natural hand-over' not in source


def test_only_the_next_ten_pairs_are_requested_and_only_confirmed_plans_get_badges():
    source = _repo_file('static', 'song_player.js')
    css = _repo_file('static', 'style.css')
    marks = source[source.index('const MARKS = {'):source.index('function markHtml(')]
    assert all(kind in marks for kind in ('crossfade:', 'blend:', 'rise:', 'percussion:', 'eqblend:', 'sweep:', 'vocals:'))
    assert all(f'[data-mark="{kind}"]' in css for kind in ('percussion', 'eqblend', 'sweep', 'vocals'))
    assert '[data-mark="blend"], [data-mark="rise"]' in css
    for gone in ('drop:', 'tail:', 'drums:', 'filter:', 'bridge:', 'pickup:', 'FILTER_TRANSITIONS'):
        assert gone not in source, gone
    transitions = source[source.index('const TRANSITIONS = {'):source.index('const TRANSITION_MARKS')]
    assert [line.split(':')[0].strip() for line in transitions.splitlines() if ': {' in line] == [
        'beat_mix', 'beat_rise', 'percussion_mix', 'eq_blend', 'filter_rise', 'vocal_handoff', 'crossfade']
    assert 'pending:' not in marks
    assert 'TRANSITION_LOOKAHEAD = 10;' in source
    assert 'batch.active < TRANSITION_REQUESTS' in source and 'batch.requests.get(key)' in source
    assert 'batch.requests.delete(job.key);' in source
    assert 'if (plan && planning === batch) paintPlan(batch, key, plan);' in source
    assert 'markHtml({ ...mark, seconds: plan.duration }, true)' in source
    assert 'planAhead(playQueue, index + 1)' in source
    assert 'audio_flac' not in source


def test_a_beat_matched_mix_plays_both_songs_at_the_api_speeds_without_time_stretching():
    source = _repo_file('static', 'song_player.js')
    mix = source[source.index('} else if (mix && state.pieces && at(mix.start) > now + 0.1) {'):source.index('showTransition(playQueue[index + 1], mix)')]
    assert 'follow(source.playbackRate, side.speed, state.handoverAt, span);' in mix
    assert 'const played = mix.to.continue_from - mix.to.start;' in mix
    assert 'at: t => mix.to.start + played * Math.min(1, Math.max(0, (t - state.handoverAt) / span))' in mix
    for gone in ('stretchSegment', 'similarity', 'WSOLA', 'mix.lead', 'mix.buffer', 'mix.follow', 'position', 'function along('):
        assert gone not in source, gone


def test_the_highlight_jumps_to_the_next_song_in_the_middle_of_a_transition():
    source = _repo_file('static', 'song_player.js')
    tick = source[source.index('function tick() {'):source.index('function glide(')]
    assert 'const view = swapView && now >= swapView.from ? swapView : null;' in tick
    assert "if (view) list.querySelectorAll('.result-item').forEach((item, n) => item.classList.toggle('playing', n === view.row));" in tick
    assert 'const second = view ? view.at(now) :' in tick
    mix = source[source.index('swapView = {'):source.index('showTransition(playQueue[index + 1], mix)')]
    assert 'from: state.handoverAt + mix.middle,' in mix and 'row: playQueue[index + 1].row,' in mix
    assert 'at: t => mix.to.start + played * Math.min(1, Math.max(0, (t - state.handoverAt) / span))' in mix
    assert 'after(state.handoverAt + (fading ? span / 2 : 0), () => begin(index + 1, state.next, entry));' in source
    assert source.count('swapView = null;') == 4 and 'mix-in' not in source
    assert 'playCurrent = state;\n            node.onMove = state.plan;\n            track.onLength = state.plan;\n            markPlaying(index);\n            tick();' in source


def test_the_playing_row_shows_the_progress_and_seeks_only_up_to_the_fade():
    css = _repo_file('static', 'style.css')
    source = _repo_file('static', 'song_player.js')
    style = css[css.index('.playable-list .result-item.playing {'):]
    style = style[:style.index('}')]
    assert 'var(--play-fill) var(--play-progress, 0%)' in style and 'var(--play-limit-color)' in style
    assert '--play-fill: #DBEAFE;' in css and 'touch-action: pan-y;' in style
    assert 'if (state.mix) return state.mix.start;' in source
    assert 'Math.min(share * lengthOf(scrub.state), seekLimit(scrub.state) - 1)' in source
    assert "if (!item || item.classList.contains('playing') || bar.classList.contains('hidden')) return;" in source
    assert "skipBtn.addEventListener('click', () => skipToFade(playCurrent));" in source
    skip = source[source.index('function skipToFade(state) {'):]
    skip = skip[:skip.index('\n        }\n')]
    assert 'state.skipWhenReady = true;' in skip and 'seekTo(state, target)' in skip
    assert source.index('if (!state.ready) {', source.index('function skipToFade(state) {')) < source.index('const target = seekLimit(state) - 6;')
    assert source.count('state.plan();\n                        if (state.skipWhenReady) skipToFade(state);') == 2


def _post_fade(client, body):
    return client.post('/api/recording_search/smart_fade', json=body)


def test_smart_fade_returns_the_songs_in_order_under_the_ids_as_sent(client, monkeypatch):
    import app_server_context
    import tasks.recording_search_manager as rsm

    for body in ({}, {'item_ids': []}, {'item_ids': 'a,b'}, {'item_ids': [None, ' ']}):
        assert _post_fade(client, body).status_code == 400

    _translating_scope(monkeypatch)
    monkeypatch.setattr(
        app_server_context, 'resolve_input_item_ids',
        lambda ids, data=None: {i: i if i.startswith('fp_') else f'fp_{i}' for i in ids},
    )
    seen = []

    def playlist(item_ids, keep_order=False):
        seen.append((list(item_ids), keep_order))
        if len(item_ids) != 3:
            return [(index, None) for index in range(len(item_ids))]
        return [
            (2, {'fade_in_seconds': 0.0, 'fade_out_seconds': 180.5, 'duration_seconds': 200.0}),
            (0, {'fade_in_seconds': 12.0, 'fade_out_seconds': 150.0, 'duration_seconds': 150.0}),
            (1, None),
        ]

    monkeypatch.setattr(rsm, 'smart_fade_playlist', playlist)
    response = _post_fade(client, {'item_ids': ['a', 'b', 'fp_c']})
    body = response.get_json()
    assert response.status_code == 200
    assert seen == [(['fp_a', 'fp_b', 'fp_c'], False)]
    assert _post_fade(client, {'item_ids': ['a'], 'keep_order': True}).status_code == 200
    assert _post_fade(client, {'item_ids': ['a'], 'keep_order': 'true'}).status_code == 200
    assert seen[1:] == [(['fp_a'], True), (['fp_a'], False)]
    assert 'fp_' not in response.get_data(as_text=True)
    assert [row['item_id'] for row in body['results']] == ['C', 'a', 'b']
    assert body['results'][1] == {'item_id': 'a', 'fade_in_seconds': 12.0, 'fade_out_seconds': 150.0, 'duration_seconds': 150.0}
    assert body['results'][2] == {'item_id': 'b', 'fade_in_seconds': None, 'fade_out_seconds': None, 'duration_seconds': None}
    assert body['count'] == 3


def _fake_media(monkeypatch, tmp_path, payload, release=None, name='42.mp3'):
    import contextlib

    import app_server_context
    import tasks.commons
    import tasks.mediaserver

    seen = {'downloads': 0}
    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: f'fp_{raw_id}')
    monkeypatch.setattr(app_server_context, 'use_request_server', lambda data=None: contextlib.nullcontext())
    monkeypatch.setattr(tasks.commons, 'fetch_track_metadata_map', lambda ids: {i: {'title': 'Song 1'} for i in ids if i == 'fp_42'})

    def download(folder, item):
        seen.update(folder=folder, item=item, downloads=seen['downloads'] + 1)
        if payload is None:
            return None
        parts = payload if isinstance(payload, list) else [payload]
        path = os.path.join(folder, name)
        with open(path, 'wb') as handle:
            for n, part in enumerate(parts):
                handle.write(part)
                handle.flush()
                if n == 0 and release is not None:
                    assert release.wait(10)
        return path

    monkeypatch.setattr(tasks.mediaserver, 'download_track', download)
    return seen


def _wav(seconds=1.0, rate=44100):
    import soundfile

    t = np.arange(int(seconds * rate)) / rate
    samples = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=1)
    samples = (samples * 20000).astype(np.int16)
    out = io.BytesIO()
    soundfile.write(out, samples, rate, format='WAV', subtype='PCM_16')
    return out.getvalue(), samples


def _flac_samples(data):
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        frames = [frame.to_ndarray() for frame in container.decode(stream)]
    return np.concatenate([f.reshape(-1, 2) if f.shape[0] == 1 else f.T for f in frames])


def test_play_streams_the_song_downloads_it_once_and_the_cache_removes_it(client, monkeypatch, tmp_path):
    import app_play_cache

    seen = _fake_media(monkeypatch, tmp_path, b'ID3audio')
    response = client.get('/api/recording_search/play/42')
    assert response.status_code == 200
    assert response.data == b'ID3audio'
    assert response.mimetype == 'audio/mpeg'
    assert 'Content-Length' not in response.headers
    assert seen['item'] == {'Id': '42', 'id': '42'}
    assert client.get('/api/recording_search/play/42').data == b'ID3audio'
    assert seen['downloads'] == 1 and os.path.exists(seen['folder'])
    monkeypatch.setattr(app_play_cache, 'IDLE_SECONDS', -1)
    app_play_cache.sweep()
    assert not os.path.exists(seen['folder'])


def test_play_sends_the_first_bytes_while_the_media_server_still_delivers_the_rest(client, monkeypatch, tmp_path):
    import threading

    release = threading.Event()
    first, rest = b'fLaC' + b'\x00' * 4092, b'\x01' * 8192
    seen = _fake_media(monkeypatch, tmp_path, [first, rest], release)
    response = client.get('/api/recording_search/play/42', buffered=False)
    assert response.status_code == 200 and response.mimetype == 'audio/flac'
    chunks = iter(response.response)
    assert next(chunks) == first
    release.set()
    assert b''.join(chunks) == rest
    response.close()
    assert seen['downloads'] == 1


def test_play_converts_a_wav_to_flac_with_every_sample_unchanged(client, monkeypatch, tmp_path):
    payload, samples = _wav()
    _fake_media(monkeypatch, tmp_path, payload)
    response = client.get('/api/recording_search/play/42')
    assert response.status_code == 200 and response.mimetype == 'audio/flac'
    assert response.data[:4] == b'fLaC' and len(response.data) < len(payload)
    assert np.array_equal(_flac_samples(response.data), samples)


def test_play_sends_a_file_the_converter_cannot_read_as_it_is(client, monkeypatch, tmp_path):
    payload = b'RIFF' + b'\x07' * 600
    _fake_media(monkeypatch, tmp_path, payload)
    response = client.get('/api/recording_search/play/42')
    assert response.status_code == 200
    assert response.data == payload and response.mimetype == 'application/octet-stream'


def test_play_refuses_a_song_outside_the_library_and_reports_a_failed_download(client, monkeypatch, tmp_path):
    seen = _fake_media(monkeypatch, tmp_path, None)
    response = client.get('/api/recording_search/play/7')
    assert response.status_code == 404
    assert 'item' not in seen
    response = client.get('/api/recording_search/play/42')
    assert response.status_code == 502
    assert client.get('/api/recording_search/play/42').status_code == 502 and seen['downloads'] == 2


def test_transition_answers_what_to_play_in_order_and_a_pair_without_a_beat_gets_the_crossfade(client, monkeypatch, tmp_path):
    import tasks.commons
    import tasks.smart_fade_levels
    import tasks.smart_fade_mix
    import tasks.smart_fade_voice

    seen = _fake_media(monkeypatch, tmp_path, b'audio')
    monkeypatch.setattr(tasks.commons, 'fetch_track_metadata_map', lambda ids: {i: {'title': 'Song'} for i in ids if i in ('fp_42', 'fp_43')})
    plan = {'mix_at': 191.5, 'style': 'fade'}
    answer = {'type': 'crossfade', 'start': 191.5, 'duration': 10.0, 'middle': 5.0, 'step': 0.1, 'filters': {'bass_hz': 180.0},
              'from': {'gain': 0.5}, 'to': {'gain': 0.7}, 'info': {'tempo_from': None}}
    calls = []
    folders = []

    def make_plan(path_out, out_seconds, path_in, in_seconds):
        folders.append(os.path.dirname(path_out))
        calls.append(('plan', os.path.basename(path_out), out_seconds, in_seconds, os.path.isdir(seen['folder'])))
        return dict(plan)

    def make_crossfade(path_out, out_seconds, path_in, in_seconds, seconds=None):
        calls.append(('crossfade', out_seconds, in_seconds))
        return dict(plan, beatless=True)

    def make_answer(path_out, path_in, planned, prepared=None):
        assert prepared == [path_out, path_in]
        calls.append(('answer', planned.get('beatless', False)))
        return answer

    def check_voice(path_out, path_in, planned):
        assert os.path.isfile(path_out) and os.path.isfile(path_in)
        assert planned['requested_end'] == 200.0
        calls.append(('voice',))
        return planned

    monkeypatch.setattr(tasks.smart_fade_mix, 'plan_transition', make_plan)
    monkeypatch.setattr(tasks.smart_fade_levels, 'crossfade_plan', make_crossfade)
    monkeypatch.setattr(tasks.smart_fade_levels, 'transition', make_answer)
    monkeypatch.setattr(tasks.smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(tasks.smart_fade_voice, 'clear_voice', check_voice)
    url = '/api/recording_search/transition'
    for query in ('', '?from_id=42&to_id=43&out=x&in=1', '?from_id=42&to_id=43&out=-1&in=1',
                  '?from_id=42&out=1&in=1', '?current_id=42&next_id=43&current_end=200&next_start=20'):
        response = client.get(url + query)
        assert response.status_code == 400 and 'from_id, to_id, out and in' in response.get_json()['error']
    assert client.get(url + '?from_id=42&to_id=7&out=200&in=20').status_code == 404
    assert calls == []

    response = client.get(url + '?from_id=42&to_id=43&out=200&in=20.5')
    assert response.status_code == 200 and response.mimetype == 'application/json'
    assert list(json.loads(response.data)) == ['type', 'start', 'duration', 'middle', 'step', 'filters', 'from', 'to', 'info']
    assert response.get_json() == answer
    timings = dict(part.strip().split(';dur=') for part in response.headers['Server-Timing'].split(','))
    assert set(timings) == {'setup', 'fetch', 'plan', 'voice', 'level_wait', 'mix', 'total'}
    assert all(float(value) >= 0.0 for value in timings.values())
    assert calls == [('plan', '42.mp3', 200.0, 20.5, True), ('voice',), ('answer', False)]
    assert all(not os.path.exists(folder) for folder in folders)
    calls.clear()

    def no_beat(*args):
        calls.append(('plan',))
        raise ValueError('No steady beat to match.')

    monkeypatch.setattr(tasks.smart_fade_mix, 'plan_transition', no_beat)
    response = client.get(url + '?from_id=42&to_id=43&out=200&in=20')
    assert response.status_code == 200 and response.get_json() == answer
    assert calls == [('plan',), ('crossfade', 200.0, 20.0), ('voice',), ('answer', True)]

    def no_room(*args):
        raise ValueError('No room for a crossfade at the fade points.')

    monkeypatch.setattr(tasks.smart_fade_levels, 'crossfade_plan', no_room)
    response = client.get(url + '?from_id=42&to_id=43&out=200&in=20')
    assert response.status_code == 400 and 'no audible room' in response.get_json()['error']


def test_transition_plans_the_pair_again_without_a_lock_when_the_voice_check_drops_it(client, monkeypatch, tmp_path):
    from tasks import commons, smart_fade_levels, smart_fade_mix, smart_fade_voice

    _fake_media(monkeypatch, tmp_path, b'audio')
    monkeypatch.setattr(commons, 'fetch_track_metadata_map', lambda ids: {i: {'title': 'Song'} for i in ids})
    calls = []
    outcome = {'voice': 'drops', 'unlocked': 'plans'}

    def plan(path_out, out_seconds, path_in, in_seconds, lock=True):
        calls.append(('plan', lock))
        if not lock and outcome['unlocked'] == 'fails':
            raise ValueError('No room for a crossfade at the fade points.')
        return {'style': 'blend' if lock else 'fade', 'mix_at': 180.0 if lock else 184.0}

    def voice(path_out, path_in, planned):
        calls.append(('voice', planned['style'], planned['requested_end']))
        if planned['style'] == 'blend' and outcome['voice'] == 'drops':
            return dict(planned, style='fade', beat_rejection='audible_cue')
        return dict(planned, beat_rejection=planned.get('beat_rejection'))

    def answer(path_out, path_in, planned, prepared=None):
        return {'type': planned['style'], 'start': planned['mix_at'], 'info': {'beat_rejection': planned['beat_rejection']}}

    monkeypatch.setattr(smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', plan)
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', voice)
    monkeypatch.setattr(smart_fade_levels, 'transition', answer)
    url = '/api/recording_search/transition?from_id=42&to_id=43&out=200&in=20'

    assert client.get(url).get_json() == {'type': 'fade', 'start': 184.0, 'info': {'beat_rejection': 'audible_cue'}}
    assert calls == [('plan', True), ('voice', 'blend', 200.0), ('plan', False), ('voice', 'fade', 200.0)]
    calls.clear()
    outcome['unlocked'] = 'fails'
    assert client.get(url).get_json() == {'type': 'fade', 'start': 180.0, 'info': {'beat_rejection': 'audible_cue'}}
    assert calls == [('plan', True), ('voice', 'blend', 200.0), ('plan', False)]
    calls.clear()
    outcome['voice'] = 'keeps'
    assert client.get(url).get_json() == {'type': 'blend', 'start': 180.0, 'info': {'beat_rejection': None}}
    assert calls == [('plan', True), ('voice', 'blend', 200.0)]


def test_transition_prepares_both_song_levels_while_planning(client, monkeypatch, tmp_path):
    from tasks import commons, smart_fade_levels, smart_fade_mix, smart_fade_voice

    _fake_media(monkeypatch, tmp_path, b'audio')
    monkeypatch.setattr(commons, 'fetch_track_metadata_map', lambda ids: {i: {'title': 'Song'} for i in ids})
    ready = threading.Barrier(3)

    def prepare(path):
        ready.wait(timeout=3)
        return path

    def plan(*args):
        ready.wait(timeout=3)
        return {'mix_at': 180.0}

    def answer(path_out, path_in, planned, prepared=None):
        assert prepared == [path_out, path_in]
        return {'start': planned['mix_at']}

    monkeypatch.setattr(smart_fade_levels, 'prepare_song', prepare, raising=False)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', plan)
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', answer)
    response = client.get('/api/recording_search/transition?from_id=42&to_id=43&out=200&in=20')
    assert response.status_code == 200 and response.get_json() == {'start': 180.0}


def test_transition_downloads_in_parallel_with_separate_request_contexts(client, bp_mod, monkeypatch, tmp_path):
    from flask import g, request
    from tasks import smart_fade_levels, smart_fade_mix, smart_fade_voice

    ready = threading.Barrier(2)
    seen = []

    def fetch(item_id, folder):
        g.download_id = item_id
        ready.wait(timeout=3)
        assert g.download_id == item_id
        assert request.headers['X-Media-Server'] == 'selected-server'
        path = os.path.join(folder, 'same-provider-name.audio')
        with open(path, 'wb') as handle:
            handle.write(b'audio')
        seen.append((item_id, path))
        return path, None

    monkeypatch.setattr(bp_mod, '_fetch_song', fetch)
    monkeypatch.setattr(smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', lambda *args: {})
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', lambda a, b, plan, prepared: {'separate': a != b})
    response = client.get('/api/recording_search/transition?from_id=42&to_id=43&out=200&in=20',
                          headers={'X-Media-Server': 'selected-server'})
    assert response.status_code == 200 and response.get_json() == {'separate': True}
    assert sorted(item for item, path in seen) == ['42', '43']
    assert all(not os.path.exists(path) for item, path in seen)


def test_transition_downloads_an_identical_song_once(client, bp_mod, monkeypatch):
    from tasks import smart_fade_levels, smart_fade_mix, smart_fade_voice

    seen = []

    def fetch(item_id, folder):
        seen.append(item_id)
        return os.path.join(folder, item_id), None

    monkeypatch.setattr(bp_mod, '_fetch_song', fetch)
    monkeypatch.setattr(smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', lambda *args: {})
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', lambda a, b, plan, prepared: {'same': a == b})
    response = client.get('/api/recording_search/transition?from_id=42&to_id=42&out=200&in=20')
    assert response.status_code == 200 and response.get_json() == {'same': True}
    assert seen == ['42']


def test_transition_plans_one_request_with_the_switches_it_sends_and_refuses_an_unknown_kind(client, bp_mod, monkeypatch):
    import config
    from tasks import smart_fade_levels, smart_fade_mix, smart_fade_voice

    for flag in smart_fade_mix.SWITCHES.values():
        monkeypatch.setattr(config, flag, True)
    seen = []

    def plan(*args):
        seen.append({style for style in smart_fade_mix.SWITCHES if smart_fade_mix.enabled(style)})
        return {}

    def fetch(item_id, folder):
        seen.append(item_id)
        return os.path.join(folder, item_id), None

    monkeypatch.setattr(bp_mod, '_fetch_song', fetch)
    monkeypatch.setattr(smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', plan)
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', lambda a, b, plan, prepared: {'eq_blend': smart_fade_mix.enabled('eq_blend')})
    url = '/api/recording_search/transition?from_id=42&to_id=42&out=200&in=20'

    response = client.get(url + '&transitions=eq_blend,nope')
    assert response.status_code == 400 and 'beat_mix' in response.get_json()['error'] and seen == []
    assert client.get(url + '&transitions=beat_mix,%20crossfade').get_json() == {'eq_blend': False}
    assert client.get(url + '&transitions=').get_json() == {'eq_blend': False}
    assert client.get(url + '&transitions=eq_blend').get_json() == {'eq_blend': True}
    assert client.get(url).get_json() == {'eq_blend': True}
    assert seen == ['42', {'blend'}, '42', set(), '42', {'eq_blend'}, '42', set(smart_fade_mix.SWITCHES)]
    assert all(smart_fade_mix.enabled(style) for style in smart_fade_mix.SWITCHES)


def test_transition_reuses_completed_playback_downloads(client, monkeypatch, tmp_path):
    import app_play_cache
    from tasks import commons, smart_fade_levels, smart_fade_mix, smart_fade_voice

    seen = _fake_media(monkeypatch, tmp_path, b'ID3audio')
    monkeypatch.setattr(commons, 'fetch_track_metadata_map', lambda ids: {item: {} for item in ids})
    monkeypatch.setattr(smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', lambda *args: {})
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', lambda a, b, plan, prepared: {'ready': True})
    response = client.get('/api/recording_search/play/42')
    assert response.status_code == 200 and response.data == b'ID3audio'
    response.close()
    for _ in range(2):
        response = client.get('/api/recording_search/transition?from_id=42&to_id=43&out=200&in=20')
        assert response.status_code == 200
    assert seen['downloads'] == 2
    assert all(entry.users == 0 for entry in app_play_cache._entries.values())


def test_transition_retries_a_download_that_failed_after_writing_partial_audio(client, monkeypatch, tmp_path):
    from tasks import mediaserver, smart_fade_levels, smart_fade_mix, smart_fade_voice

    _fake_media(monkeypatch, tmp_path, b'audio')
    calls = []

    def download(folder, item):
        calls.append(item)
        path = os.path.join(folder, 'song.audio')
        with open(path, 'wb') as handle:
            handle.write(b'audio')
        return None if len(calls) == 1 else path

    monkeypatch.setattr(mediaserver, 'download_track', download)
    monkeypatch.setattr(smart_fade_levels, 'prepare_song', lambda path: path)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', lambda *args: {})
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', lambda a, b, plan, prepared: {'ready': True})
    url = '/api/recording_search/transition?from_id=42&to_id=42&out=200&in=20'
    assert client.get(url).status_code == 502
    assert client.get(url).status_code == 200
    assert len(calls) == 2


def test_transition_download_cache_keeps_selected_servers_separate(client, monkeypatch, tmp_path):
    import contextlib
    from flask import request
    import app_server_context
    from tasks import mediaserver, smart_fade_levels, smart_fade_mix, smart_fade_voice
    from tasks.mediaserver import context

    _fake_media(monkeypatch, tmp_path, b'audio')
    downloaded = []

    @contextlib.contextmanager
    def selected():
        with context.use_server({'server_id': request.args['server']}):
            yield

    def download(folder, item):
        server = context.active_server_id()
        downloaded.append(server)
        path = os.path.join(folder, 'song.audio')
        with open(path, 'w') as handle:
            handle.write(server)
        return path

    def prepare(path):
        with open(path) as handle:
            return handle.read()

    monkeypatch.setattr(app_server_context, 'resolve_request_server_id', lambda data=None: request.args['server'])
    monkeypatch.setattr(app_server_context, 'use_request_server', selected)
    monkeypatch.setattr(mediaserver, 'download_track', download)
    monkeypatch.setattr(smart_fade_levels, 'prepare_song', prepare)
    monkeypatch.setattr(smart_fade_mix, 'plan_transition', lambda *args: {})
    monkeypatch.setattr(smart_fade_voice, 'clear_voice', lambda a, b, plan: plan)
    monkeypatch.setattr(smart_fade_levels, 'transition', lambda a, b, plan, prepared: {'server': prepared[0]})
    for server in ('first', 'second', 'first'):
        response = client.get('/api/recording_search/transition?from_id=42&to_id=42&out=200&in=20&server=' + server)
        assert response.status_code == 200 and response.get_json() == {'server': server}
    assert downloaded == ['first', 'second']


def test_a_disabled_feature_answers_503_on_every_route_but_play_and_the_page_says_so(client, bp_mod, monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', False)
    for response in (
        _post(client, {'clip': _clip()}),
        _post_track(client, {'item_id': 'x'}),
        _post_fade(client, {'item_ids': ['x']}),
        client.get('/api/recording_search/transition?from_id=x&to_id=y&out=1&in=1'),
        client.post('/api/recording_search/warmup'),
    ):
        assert response.status_code == 503
        assert 'NEURAL_FINGERPRINT_ENABLED' in response.get_json()['error']
    _fake_media(monkeypatch, tmp_path, b'ID3audio')
    response = client.get('/api/recording_search/play/42')
    assert response.status_code == 200 and response.data == b'ID3audio'
    page = _render_page(bp_mod, monkeypatch, {'enabled': True, 'running': True, 'port': 8443, 'error': None})
    assert page['neural_enabled'] is False


def test_an_insecure_page_shows_a_short_https_notice_with_the_same_port_and_never_redirects():
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'templates', 'recording_search.html')
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    secure = source[source.index('const secureUrl'):source.index(';', source.index('const secureUrl'))]
    assert "const samePort = window.location.port || '80';" in source
    assert 'window.location.hostname' in secure
    assert '${samePort}' in secure
    assert 'HTTPS_PORT}' not in secure
    assert 'location.assign(' not in source
    assert 'Most browsers allow recording only on HTTPS' in source
    assert 'self-signed certificate, accept the browser warning once' in source
    assert 'chrome://flags' not in source


def test_warmup_relays_the_manager_status(client, monkeypatch):
    import tasks.recording_search_manager as rsm

    monkeypatch.setattr(
        rsm,
        'warmup_recording_models',
        lambda: {'loaded': True, 'models': {'neural': True}, 'expiry_seconds': 300},
    )
    response = client.post('/api/recording_search/warmup')
    assert response.status_code == 200
    assert response.get_json()['models']['neural'] is True


def test_an_unsupported_song_is_announced_and_skipped_and_the_clock_runs_at_the_device_rate():
    source = _repo_file('static', 'song_player.js')
    assert 'audioCtx = new AudioContext();' in source
    assert 'sampleRate:' not in source and 'CONTEXT_RATE' not in source
    assert "throw new Error('Song not supported in the frontend.');" in source
    nxt = source[source.index('async function prepareNext(state, token) {'):source.index('async function playSong(index) {')]
    assert 'pieceAt(next, entryOffset(state.index + 1)).catch(error => { notice = skipNotice(song, error); return null; }),' in nxt
    assert 'playQueue.splice(state.index + 1, 1);' in nxt
    play = source[source.index('async function playSong(index) {'):source.index('async function startPlayback(')]
    assert 'while (index < playQueue.length) {' in play and 'playQueue.splice(index, 1);' in play
    assert 'prepareNext(state, token)' in source
    for gone in ('state.failed', 'failed: null', 'This browser cannot play this song.'):
        assert gone not in source, gone


def test_play_cuts_an_exact_piece_with_the_song_length_and_refuses_a_bad_range(client, monkeypatch, tmp_path):
    payload, samples = _wav(seconds=3.0)
    seen = _fake_media(monkeypatch, tmp_path, payload)
    url = '/api/recording_search/play/42'
    for query in ('?start=1', '?seconds=1', '?start=-1&seconds=1', '?start=1&seconds=0', '?start=x&seconds=1', '?start=0&seconds=601'):
        assert client.get(url + query).status_code == 400, query
    response = client.get(url + '?start=1&seconds=1.5')
    assert response.status_code == 200 and response.mimetype == 'audio/flac'
    assert np.array_equal(_flac_samples(response.data), samples[44100:44100 + 66150])
    again = client.get(url + '?start=0&seconds=1')
    assert again.status_code == 200 and seen['downloads'] == 1
    assert float(again.headers['X-Song-Seconds']) == pytest.approx(3.0)
    assert np.array_equal(_flac_samples(again.data), samples[:44100])


def test_play_answers_422_for_a_piece_of_a_file_the_server_cannot_read(client, monkeypatch, tmp_path):
    _fake_media(monkeypatch, tmp_path, b'RIFF' + b'\x07' * 600)
    response = client.get('/api/recording_search/play/42?start=0&seconds=5')
    assert response.status_code == 422 and 'could not read' in response.get_json()['error']
