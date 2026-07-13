# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the tasks.analysis audio-analysis internals.

Covers the ONNX helper functions and the analyze_track pipeline with mocked
models and audio loading, plus the media-server reachability probe.

Main Features:
* ONNX output-name resolution, run_inference, and numerically stable sigmoid.
* Robust audio load with fallback and analyze_track key/tempo/energy output.
* OOM-to-CPU inference fallback and media-server auth/unreachable detection.
* Chromaprint fail-soft: a failed fingerprint keeps the track's analysis alive
  under its provider id and records the empty-string retry-stop sentinel.
* run_analysis_task scope handling: empty enabled-server list skips instead of
  falling back to the config default server.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from tasks.analysis import (
    sigmoid,
    robust_load_audio_with_fallback,
    analyze_track,
)
from tasks.analysis_helper import run_inference, _find_onnx_name


def test_union_analysis_runs_each_server_once_with_no_sweeps(monkeypatch):
    import tasks.analysis as analysis
    import tasks.multiserver_sync as sync

    servers = [
        {'server_id': 'a', 'name': 'A', 'is_default': True},
        {'server_id': 'b', 'name': 'B', 'is_default': False},
        {'server_id': 'c', 'name': 'C', 'is_default': False},
    ]
    events = []
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: servers)
    monkeypatch.setattr(analysis, 'get_current_job', lambda connection=None: None)
    monkeypatch.setattr(analysis, 'get_task_info_from_db', lambda task_id: None)
    monkeypatch.setattr(analysis, 'save_task_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, '_run_all_index_builds', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        analysis,
        'run_analysis_server_task',
        lambda *args, server_id=None, **kwargs: events.append(('analyze', server_id))
        or {'status': 'SUCCESS'},
    )

    def forbidden_sweep(*args, **kwargs):
        raise AssertionError('analysis must never run an alignment sweep')

    monkeypatch.setattr(sync, 'sweep_all_secondary_servers', forbidden_sweep)

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == 'SUCCESS'
    assert events == [
        ('analyze', 'a'),
        ('analyze', 'b'),
        ('analyze', 'c'),
    ]


def test_run_analysis_task_skips_when_no_enabled_server_matches_scope(monkeypatch):
    import tasks.analysis as analysis

    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: [])
    monkeypatch.setattr(analysis, 'get_current_job', lambda connection=None: None)
    statuses = []
    monkeypatch.setattr(
        analysis,
        'save_task_status',
        lambda task_id, task_type, status, **kwargs: statuses.append(status),
    )
    server_runs = []
    monkeypatch.setattr(
        analysis,
        'run_analysis_server_task',
        lambda *args, **kwargs: server_runs.append((args, kwargs)),
    )

    result = analysis.run_analysis_task(0, 5, server_scope='default')

    assert result['status'] == 'SKIPPED'
    assert 'default' in result['message']
    assert not server_runs
    assert statuses == ['SUCCESS']


def test_enabled_analysis_servers_registry_failure_keeps_config_default(monkeypatch):
    import importlib
    import tasks.analysis as analysis

    registry = importlib.import_module('tasks.mediaserver.registry')

    def broken_scope(scope):
        raise RuntimeError('registry down')

    monkeypatch.setattr(registry, 'servers_for_scope', broken_scope)

    assert analysis._enabled_analysis_servers('all') == [None]


_FAKE_EMBEDDING = np.sin(np.arange(1, 201, dtype=np.float32))


def _run_album_impl(monkeypatch, tmp_path, item, known_index, persisted_ids, map_upserts,
                    analyzed_embedding=None, existing_ids_fn=None, persist_calls=None,
                    tracks=None, job=None):
    import importlib
    import tasks.analysis as analysis
    import tasks.analysis_helper as helper
    import tasks.clap_analyzer as clap

    registry = importlib.import_module('tasks.mediaserver.registry')
    album_tracks = tracks if tracks is not None else [item]

    monkeypatch.setattr(analysis, 'get_current_job', lambda connection=None: job)
    monkeypatch.setattr(analysis, 'save_task_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, 'get_tracks_from_album', lambda album_id: album_tracks)
    monkeypatch.setattr(
        analysis, 'download_track', lambda temp_dir, track: str(tmp_path / 'gone.flac')
    )
    monkeypatch.setattr(analysis, 'load_musicnn_sessions', lambda model_paths: {})
    monkeypatch.setattr(analysis, 'cleanup_musicnn_sessions', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, 'cleanup_optional_models', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        analysis, 'comprehensive_memory_cleanup', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(analysis, 'cleanup_cuda_memory', lambda *args, **kwargs: None)
    fake_embedding = (
        _FAKE_EMBEDDING if analyzed_embedding is None else analyzed_embedding
    )
    monkeypatch.setattr(
        analysis,
        'analyze_track',
        lambda *args, **kwargs: (
            {
                'tempo': 120.0,
                'energy': 0.5,
                'key': 'C',
                'scale': 'major',
                'moods': {'happy': 0.9},
            },
            fake_embedding,
        ),
    )

    monkeypatch.setattr(clap, 'is_clap_available', lambda: False)
    monkeypatch.setattr(registry, 'get_default_server_id', lambda conn=None: 'srv-def')
    monkeypatch.setattr(
        registry,
        'upsert_track_maps',
        lambda server_id, mapping, conn=None: map_upserts.append((server_id, mapping)),
    )

    monkeypatch.setattr(
        helper,
        'attach_catalog_item_ids',
        lambda tracks, server_id=None, conn=None: tracks,
    )
    monkeypatch.setattr(
        helper,
        'get_existing_track_ids',
        existing_ids_fn or (lambda ids: {i for i in ids if str(i).startswith('fp_')}),
    )
    monkeypatch.setattr(helper, 'get_missing_ids_in_table', lambda table, ids: set())
    monkeypatch.setattr(helper, 'load_fingerprint_index', lambda: known_index)
    monkeypatch.setattr(
        helper, 'upsert_artist_mappings_for_tracks', lambda tracks, album_name=None: None
    )
    monkeypatch.setattr(helper, 'run_song_analyzed_hook', lambda *args, **kwargs: None)

    def fake_persist(track, *args, **kwargs):
        persisted_ids.append(helper.catalog_item_id(track))
        if persist_calls is not None:
            persist_calls.append(kwargs)

    monkeypatch.setattr(helper, 'persist_musicnn_results', fake_persist)
    monkeypatch.setattr(
        helper, 'persist_clap_embedding', lambda *args, **kwargs: False
    )

    return analysis._analyze_album_task_impl('album1', 'Album One', 5, 'parent1')


def test_new_track_persists_under_signature_id_and_maps_it(monkeypatch, tmp_path):
    from tasks import simhash

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, simhash.CatalogResolver(), persisted_ids, map_upserts
    )

    expected_id = simhash.canonical_id_str(simhash.embedding_signature(_FAKE_EMBEDDING))
    assert result['status'] == 'SUCCESS'
    assert result['tracks_analyzed'] == 1
    assert persisted_ids == [expected_id]
    assert item['_catalog_item_id'] == expected_id
    assert map_upserts == [('srv-def', {expected_id: ('prov1', 'fingerprint')})]


def test_same_audio_skips_persist_and_just_maps_the_server(monkeypatch, tmp_path):
    from tasks import simhash

    known_id = simhash.canonical_id_str(simhash.embedding_signature(_FAKE_EMBEDDING))
    catalog = simhash.CatalogResolver()
    catalog.register(known_id, embedding=_FAKE_EMBEDDING)

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, catalog, persisted_ids, map_upserts
    )

    assert result['status'] == 'SUCCESS'
    assert result['tracks_analyzed'] == 1
    assert persisted_ids == []
    assert map_upserts == [('srv-def', {known_id: ('prov1', 'fingerprint')})]


def test_same_signature_different_audio_gets_its_own_id(monkeypatch, tmp_path):
    from tasks import simhash

    half = simhash.SIGNATURE_BITS // 2
    first = np.concatenate([np.full(half, 1.0), np.full(half, -1.0)]).astype(np.float32)
    second = first.copy()
    second[0:half:2] = 2.0
    second[1:half:2] = 0.1
    second[half::2] = -2.0
    second[half + 1::2] = -0.1
    assert simhash.embedding_signature(first) == simhash.embedding_signature(second)
    assert simhash.cosine_distance(first, second) > 0.01

    taken_id = simhash.canonical_id_str(simhash.embedding_signature(first))
    catalog = simhash.CatalogResolver()
    catalog.register(taken_id, embedding=first)

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, catalog, persisted_ids, map_upserts,
        analyzed_embedding=second,
    )

    assert result['status'] == 'SUCCESS'
    assert len(persisted_ids) == 1
    assert persisted_ids[0] != taken_id
    assert persisted_ids[0].startswith('fp_2')
    assert map_upserts == [('srv-def', {persisted_ids[0]: ('prov1', 'fingerprint')})]


def test_degenerate_embedding_is_still_mapped_so_it_is_not_re_analyzed_forever(
    monkeypatch, tmp_path
):
    """A constant/non-finite embedding has no signature, so resolve() returns no
    canonical id. The track must STILL get a track_server_map row: without one
    nothing records it as done for this server and every later run re-downloads
    and re-runs MusiCNN on it, forever."""
    from tasks import simhash

    item = {'Id': 'prov-degenerate', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch,
        tmp_path,
        item,
        simhash.CatalogResolver(),
        persisted_ids,
        map_upserts,
        analyzed_embedding=np.zeros(simhash.SIGNATURE_BITS, dtype=np.float32),
        existing_ids_fn=lambda ids: {i for i in ids if i in persisted_ids},
    )

    assert result['status'] == 'SUCCESS'
    assert persisted_ids == ['prov-degenerate']
    assert map_upserts == [
        ('srv-def', {'prov-degenerate': ('prov-degenerate', 'analysis')})
    ]


def test_default_server_writes_file_path_but_a_secondary_never_does(
    monkeypatch, tmp_path
):
    """score.file_path is the matcher's top-priority tier and the row is SHARED,
    so only the default server may stamp its own path layout onto it."""
    from tasks import simhash
    from tasks.mediaserver import context as ms_context

    item = {
        'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist',
        'FilePath': '/music/song.flac',
    }

    default_calls = []
    _run_album_impl(
        monkeypatch, tmp_path, dict(item), simhash.CatalogResolver(), [], [],
        persist_calls=default_calls,
    )
    assert default_calls[0]['is_default_server'] is True

    secondary_calls = []
    with ms_context.use_server({'server_id': 'srv-b', 'server_type': 'plex'}):
        _run_album_impl(
            monkeypatch, tmp_path, dict(item), simhash.CatalogResolver(), [], [],
            persist_calls=secondary_calls,
        )
    assert secondary_calls[0]['is_default_server'] is False


def test_persist_musicnn_results_drops_file_path_for_a_secondary_server(monkeypatch):
    from tasks import analysis_helper as helper

    saved = {}
    monkeypatch.setattr(
        helper,
        'save_track_analysis_and_embedding',
        lambda *args, **kwargs: saved.update(kwargs),
    )
    item = {
        'Id': 'p1', 'Name': 'Song', 'AlbumArtist': 'Artist',
        'FilePath': '/music/song.flac', '_catalog_item_id': 'fp_2abc',
    }
    analysis = {'tempo': 120.0, 'energy': 0.5, 'key': 'C', 'scale': 'major'}

    helper.persist_musicnn_results(item, analysis, {}, b'', '', is_default_server=True)
    assert saved['file_path'] == '/music/song.flac'

    saved.clear()
    helper.persist_musicnn_results(item, analysis, {}, b'', '', is_default_server=False)
    assert saved['file_path'] is None


def test_revocation_is_checked_once_per_album_not_once_per_track(monkeypatch, tmp_path):
    """The per-track loop used to run TWO get_task_info_from_db queries plus a
    status write for every track, including skipped ones."""
    from unittest.mock import MagicMock
    from tasks import simhash
    import tasks.analysis as analysis

    tracks = [
        {'Id': f'prov{i}', 'Name': f'Song {i}', 'AlbumArtist': 'Artist'}
        for i in range(4)
    ]
    job = MagicMock()
    job.id = 'job-1'
    job.meta = {}

    status_calls = []
    monkeypatch.setattr(
        analysis,
        'get_task_statuses',
        lambda ids: status_calls.append(list(ids)) or {},
    )

    def forbidden(task_id):
        raise AssertionError('the per-track loop must not query task info per track')

    monkeypatch.setattr(analysis, 'get_task_info_from_db', forbidden)

    result = _run_album_impl(
        monkeypatch, tmp_path, tracks[0], simhash.CatalogResolver(), [], [],
        tracks=tracks, job=job,
    )

    assert result['status'] == 'SUCCESS'
    assert len(status_calls) == 1
    assert status_calls[0] == ['job-1', 'parent1']


def _run_parent_phase(monkeypatch, albums, tracks_by_album, work_map):
    import importlib
    import tasks.analysis as analysis
    import tasks.analysis_helper as helper
    import tasks.clap_analyzer as clap

    registry = importlib.import_module('tasks.mediaserver.registry')

    monkeypatch.setattr(analysis, 'get_current_job', lambda connection=None: None)
    monkeypatch.setattr(analysis, 'get_task_info_from_db', lambda task_id: None)
    monkeypatch.setattr(analysis, 'save_task_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, 'clean_temp', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, 'get_recent_albums', lambda limit: albums)
    monkeypatch.setattr(
        analysis, 'get_tracks_from_album', lambda album_id: tracks_by_album[album_id]
    )
    monkeypatch.setattr(
        analysis, 'get_failed_child_summary', lambda task_id: (0, [])
    )
    monkeypatch.setattr(analysis, '_run_all_index_builds', lambda *a, **k: None)
    monkeypatch.setattr(analysis, 'LYRICS_ENABLED', False)
    monkeypatch.setattr(clap, 'is_clap_available', lambda: False)
    monkeypatch.setattr(registry, 'get_default_server_id', lambda conn=None: 'srv-def')
    monkeypatch.setattr(
        helper, 'load_server_work_map', lambda *args, **kwargs: work_map
    )

    def forbidden(*args, **kwargs):
        raise AssertionError('the album loop must not query the DB per album')

    for name in ('get_existing_track_ids', 'get_missing_ids_in_table',
                 'attach_catalog_item_ids'):
        monkeypatch.setattr(helper, name, forbidden)

    enqueued = []
    jobs = {}

    def _finished_job(job_id):
        job = Mock()
        job.id = job_id
        job.is_finished = True
        job.is_failed = False
        job.is_canceled = False
        return job

    class FakeQueue:
        @staticmethod
        def enqueue(func, args=None, **kwargs):
            job = _finished_job(f'job-{len(enqueued)}')
            jobs[job.id] = job
            enqueued.append(args)
            return job

    class FakeJob:
        @staticmethod
        def fetch(job_id, connection=None):
            return jobs[job_id]

    monkeypatch.setattr(analysis, 'rq_queue_default', FakeQueue)
    monkeypatch.setattr(analysis, 'Job', FakeJob)
    monkeypatch.setattr(analysis, 'get_child_tasks_from_db', lambda task_id: [])

    result = analysis._run_analysis_server_task_impl(
        0, 5, server_id='srv-def', task_id='parent-1'
    )
    return result, enqueued


def test_phase_status_message_reports_songs_not_only_albums():
    """An album counts as skipped only when EVERY track in it is done, so on a
    mostly-analyzed library the album counter reads as if nothing was skipped
    while the album jobs skip thousands of songs one by one."""
    import tasks.analysis as analysis

    message = analysis._phase_status_message(
        albums_launched=35,
        albums_skipped=3,
        albums_completed=14,
        active_jobs=21,
        total_albums=38,
        songs_seen=10000,
        songs_done=9800,
        songs_to_analyze=200,
        finalizing=True,
    )

    assert '200 to analyze' in message
    assert '9800 already analyzed' in message
    assert 'of 10000 seen' in message
    assert 'launched 35, skipped 3' in message
    assert message.endswith('(Finalizing)')


def test_settled_library_enqueues_nothing_and_never_queries_per_album(monkeypatch):
    """The whole point of the work map: a run with nothing to do costs ONE query,
    not a handful per album."""
    import tasks.analysis_helper as helper

    albums = [{'Id': f'al{i}', 'Name': f'Album {i}'} for i in range(3)]
    tracks_by_album = {
        f'al{i}': [{'Id': f'p{i}-{t}', 'Name': 't'} for t in range(2)]
        for i in range(3)
    }
    work_map = {
        f'p{i}-{t}': helper.WORK_MUSICNN for i in range(3) for t in range(2)
    }

    result, enqueued = _run_parent_phase(monkeypatch, albums, tracks_by_album, work_map)

    assert result['status'] == 'SUCCESS'
    assert enqueued == []
    assert '3 skipped of 3' in result['message']
    assert '0 sent for analysis' in result['message']
    assert '6 already analyzed of 6' in result['message']


def test_album_with_one_unanalyzed_track_is_still_enqueued(monkeypatch):
    import tasks.analysis_helper as helper

    albums = [{'Id': 'al0', 'Name': 'Album 0'}, {'Id': 'al1', 'Name': 'Album 1'}]
    tracks_by_album = {
        'al0': [{'Id': 'done-1', 'Name': 't'}, {'Id': 'done-2', 'Name': 't'}],
        'al1': [{'Id': 'done-3', 'Name': 't'}, {'Id': 'missing', 'Name': 't'}],
    }
    work_map = {
        'done-1': helper.WORK_MUSICNN,
        'done-2': helper.WORK_MUSICNN,
        'done-3': helper.WORK_MUSICNN,
    }

    result, enqueued = _run_parent_phase(monkeypatch, albums, tracks_by_album, work_map)

    assert result['status'] == 'SUCCESS'
    assert [args[0] for args in enqueued] == ['al1']
    assert '1 sent for analysis' in result['message']
    assert '3 already analyzed of 4' in result['message']


def test_unknown_catalogue_track_requires_real_musicnn_analysis():
    from tasks.analysis_helper import decide_track_needs

    assert decide_track_needs(
        'provider-new',
        existing={'fp_existing'},
        missing_clap={'provider-new'},
        missing_lyrics={'provider-new'},
        lyrics_enabled=True,
    ) == (True, True, True)


class TestFindOnnxName:
    def test_direct_match(self):
        names = ['model/Placeholder', 'model/dense/BiasAdd']
        result = _find_onnx_name('model/Placeholder', names)
        assert result == 'model/Placeholder'

    def test_strip_colon_suffix(self):
        names = ['model/Placeholder', 'model/dense/BiasAdd']
        result = _find_onnx_name('model/Placeholder:0', names)
        assert result == 'model/Placeholder'

    def test_extract_last_part_after_slash(self):
        names = ['Placeholder', 'BiasAdd']
        result = _find_onnx_name('model/dense/Placeholder:0', names)
        assert result == 'Placeholder'

    def test_replace_slash_with_underscore(self):
        names = ['model_Placeholder', 'model_dense_BiasAdd']
        result = _find_onnx_name('model/Placeholder:0', names)
        assert result == 'model_Placeholder'

    def test_fallback_to_first_name(self):
        names = ['first_input', 'second_input']
        result = _find_onnx_name('completely_unknown_name', names)
        assert result == 'first_input'

    def test_empty_names_list(self):
        names = []
        result = _find_onnx_name('any_name', names)
        assert result is None

    def test_complex_tensorflow_name(self):
        names = ['serving_default_model_Placeholder']
        result = _find_onnx_name('serving_default_model_Placeholder:0', names)
        assert result == 'serving_default_model_Placeholder'

    def test_nested_path_extraction(self):
        names = ['BiasAdd']
        result = _find_onnx_name('model/layer1/layer2/BiasAdd:0', names)
        assert result == 'BiasAdd'


class TestRunInference:
    def test_successful_inference_direct_match(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'model/Placeholder'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = Mock()
        mock_output.name = 'model/dense/BiasAdd'
        mock_session.get_outputs.return_value = [mock_output]

        expected_result = np.array([[0.1, 0.2, 0.3]])
        mock_session.run.return_value = [expected_result]

        feed_dict = {'model/Placeholder': np.random.rand(1, 10)}
        result = run_inference(mock_session, feed_dict, 'model/dense/BiasAdd')

        assert result is not None
        np.testing.assert_array_equal(result, expected_result)
        mock_session.run.assert_called_once()

    def test_inference_with_tensorflow_style_names(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'model_Placeholder'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]

        expected_result = np.array([[0.5]])
        mock_session.run.return_value = [expected_result]

        feed_dict = {'model/Placeholder:0': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)

        assert result is not None
        np.testing.assert_array_equal(result, expected_result)

    def test_inference_without_output_tensor_name(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output1 = Mock()
        mock_output1.name = 'first_output'
        mock_output2 = Mock()
        mock_output2.name = 'second_output'
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]

        expected_result = np.array([[1.0, 2.0]])
        mock_session.run.return_value = [expected_result]

        feed_dict = {'input': np.random.rand(1, 3)}
        result = run_inference(mock_session, feed_dict, output_tensor_name=None)

        assert result is not None
        mock_session.run.assert_called_with(['first_output'], {'input': feed_dict['input']})

    def test_inference_with_multiple_inputs(self):
        mock_session = Mock()

        mock_input1 = Mock()
        mock_input1.name = 'input1'
        mock_input2 = Mock()
        mock_input2.name = 'input2'
        mock_session.get_inputs.return_value = [mock_input1, mock_input2]

        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]

        expected_result = np.array([[0.7]])
        mock_session.run.return_value = [expected_result]

        rng = np.random.default_rng(0)
        feed_dict = {'input1': rng.random((1, 5)), 'input2': rng.random((1, 3))}
        result = run_inference(mock_session, feed_dict)

        assert result is not None
        call_args = mock_session.run.call_args
        assert 'input1' in call_args[0][1]
        assert 'input2' in call_args[0][1]

    def test_inference_returns_none_when_input_mapping_fails(self):
        mock_session = Mock()

        mock_session.get_inputs.return_value = []

        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]

        feed_dict = {'unknown_input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)

        assert result is None

    def test_inference_returns_none_when_no_outputs(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]

        mock_session.get_outputs.return_value = []

        feed_dict = {'input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)

        assert result is None

    def test_inference_with_path_based_name_mapping(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'Placeholder'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = Mock()
        mock_output.name = 'BiasAdd'
        mock_session.get_outputs.return_value = [mock_output]

        expected_result = np.array([[0.3, 0.4]])
        mock_session.run.return_value = [expected_result]

        feed_dict = {'model/dense/Placeholder:0': np.random.rand(1, 8)}
        result = run_inference(mock_session, feed_dict, 'model/dense/BiasAdd:0')

        assert result is not None
        np.testing.assert_array_equal(result, expected_result)

    def test_inference_with_underscore_conversion(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'model_Placeholder'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = Mock()
        mock_output.name = 'model_output'
        mock_session.get_outputs.return_value = [mock_output]

        expected_result = np.array([[0.6]])
        mock_session.run.return_value = [expected_result]

        feed_dict = {'model/Placeholder': np.random.rand(1, 4)}
        result = run_inference(mock_session, feed_dict, 'model/output')

        assert result is not None
        np.testing.assert_array_equal(result, expected_result)

    def test_inference_result_unwrapping(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]

        expected_array = np.array([[1.0, 2.0, 3.0]])
        mock_session.run.return_value = [expected_array]

        feed_dict = {'input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected_array)

    def test_inference_with_empty_result_list(self):
        mock_session = Mock()

        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]

        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_outputs.return_value = [mock_output]

        mock_session.run.return_value = []

        feed_dict = {'input': np.random.rand(1, 5)}
        result = run_inference(mock_session, feed_dict)

        assert result == []


class TestSigmoid:
    def test_sigmoid_basic(self):
        result = sigmoid(0)
        assert np.isclose(result, 0.5)

    def test_sigmoid_positive(self):
        result = sigmoid(2.0)
        assert result > 0.5
        assert result < 1.0

    def test_sigmoid_negative(self):
        result = sigmoid(-2.0)
        assert result > 0.0
        assert result < 0.5

    def test_sigmoid_array(self):
        x = np.array([0, 1, -1, 2, -2])
        result = sigmoid(x)

        assert len(result) == 5
        assert np.all(result > 0)
        assert np.all(result < 1)
        assert np.isclose(result[0], 0.5)

    def test_sigmoid_numerical_stability_large_positive(self):
        result = sigmoid(100)
        assert np.isfinite(result)
        assert np.isclose(result, 1.0)

    def test_sigmoid_numerical_stability_large_negative(self):
        result = sigmoid(-100)
        assert np.isfinite(result)
        assert np.isclose(result, 0.0)

    def test_sigmoid_symmetry(self):
        x = 1.5
        assert np.isclose(sigmoid(x) + sigmoid(-x), 1.0)


class TestRobustLoadAudioWithFallback:
    @patch('tasks.analysis.librosa.load')
    def test_successful_direct_load(self, mock_librosa_load):
        expected_audio = np.random.rand(16000)
        expected_sr = 16000
        mock_librosa_load.return_value = (expected_audio, expected_sr)

        audio, sr = robust_load_audio_with_fallback('test.mp3', target_sr=16000)

        assert audio is not None
        assert sr == expected_sr
        np.testing.assert_array_equal(audio, expected_audio)
        mock_librosa_load.assert_called_once()

    @patch('tasks.analysis.librosa.load')
    def test_direct_load_with_custom_sample_rate(self, mock_librosa_load):
        expected_audio = np.random.rand(22050)
        expected_sr = 22050
        mock_librosa_load.return_value = (expected_audio, expected_sr)

        audio, sr = robust_load_audio_with_fallback('test.wav', target_sr=22050)

        assert sr == 22050
        mock_librosa_load.assert_called_once_with('test.wav', sr=22050, mono=True, duration=600)

    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis._decode_audio_with_pyav')
    def test_fallback_on_librosa_failure(self, mock_pyav_decode, mock_librosa_load):
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_pyav_decode.return_value = np.random.rand(16000).astype(np.float32)

        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')

        assert audio is not None
        assert sr == 16000
        mock_pyav_decode.assert_called_once_with('corrupted.mp3', 16000)

    @patch('tasks.analysis.librosa.load')
    def test_returns_none_on_empty_audio(self, mock_librosa_load):
        mock_librosa_load.return_value = (np.array([]), 16000)

        audio, sr = robust_load_audio_with_fallback('empty.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.librosa.load')
    def test_returns_none_on_none_audio(self, mock_librosa_load):
        mock_librosa_load.return_value = (None, 16000)

        audio, sr = robust_load_audio_with_fallback('invalid.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis._decode_audio_with_pyav')
    def test_fallback_handles_silent_audio(self, mock_pyav_decode, mock_librosa_load):
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_pyav_decode.return_value = np.zeros(16000, dtype=np.float32)

        audio, sr = robust_load_audio_with_fallback('silent.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.librosa.load')
    @patch('tasks.analysis._decode_audio_with_pyav')
    def test_fallback_handles_decode_failure(self, mock_pyav_decode, mock_librosa_load):
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_pyav_decode.side_effect = Exception("PyAV failed")

        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.librosa.load')
    def test_uses_audio_load_timeout_config(self, mock_librosa_load):
        mock_librosa_load.return_value = (np.random.rand(16000), 16000)

        robust_load_audio_with_fallback('test.mp3', target_sr=16000)

        call_args = mock_librosa_load.call_args
        assert 'duration' in call_args.kwargs
        assert call_args.kwargs['duration'] == 600


class TestAnalyzeTrack:
    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_successful_track_analysis(
        self, mock_audio_load, mock_mel, mock_beat, mock_rms, mock_chroma, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100, 200]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session

        mood_labels = ['happy', 'sad', 'energetic', 'calm', 'aggressive']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert embeddings is not None
        assert 'tempo' in result
        assert 'key' in result
        assert 'scale' in result
        assert 'moods' in result
        assert 'energy' in result
        assert isinstance(result['moods'], dict)
        assert len(result['moods']) == len(mood_labels)

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_audio_load_failure(self, mock_audio_load):
        mock_audio_load.return_value = (None, None)

        mood_labels = ['happy', 'sad']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('bad_file.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_empty_audio(self, mock_audio_load):
        mock_audio_load.return_value = (np.array([]), 16000)

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('empty.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_silent_audio(self, mock_audio_load):
        mock_audio_load.return_value = (np.zeros(16000), 16000)

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('silent.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_returns_none_on_short_audio(
        self, mock_audio_load, mock_beat, mock_rms, mock_chroma, mock_mel
    ):
        mock_audio = np.random.rand(100)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 10)
        mock_mel.return_value = np.random.rand(96, 10)

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('short.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_spectrogram_dtype_conversion(
        self, mock_audio_load, mock_mel, mock_beat, mock_rms, mock_chroma, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000).astype(np.float64)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000).astype(np.float64)

        captured_input = None
        call_count = [0]

        def capture_run(output_names, feed_dict):
            nonlocal captured_input
            call_count[0] += 1
            if call_count[0] == 1:
                for key, val in feed_dict.items():
                    captured_input = val
            return [np.random.rand(5, 200).astype(np.float32)]

        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.side_effect = capture_run
        mock_onnx_session.return_value = mock_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        analyze_track('test.mp3', mood_labels, model_paths)

        assert captured_input is not None
        assert captured_input.dtype == np.dtype('float32')

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_key_detection_logic(
        self, mock_audio_load, mock_mel, mock_beat, mock_rms, mock_chroma, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])

        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, _ = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert 'key' in result
        assert 'scale' in result
        assert result['key'] in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        assert result['scale'] in ['major', 'minor']

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_model_inference_failure_handling(
        self, mock_audio_load, mock_mel, mock_beat, mock_rms, mock_chroma, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        mock_onnx_session.side_effect = Exception("Model loading failed")

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/embedding.onnx'}

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_tempo_extraction(
        self, mock_audio_load, mock_mel, mock_beat, mock_rms, mock_chroma, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        expected_tempo = 128.5
        mock_beat.return_value = (expected_tempo, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, _ = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert result['tempo'] == expected_tempo
        assert isinstance(result['tempo'], float)

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    def test_energy_calculation(
        self, mock_audio_load, mock_mel, mock_beat, mock_rms, mock_chroma, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))

        rms_values = np.array([[0.1, 0.2, 0.3, 0.4]])
        expected_energy = np.mean(rms_values)
        mock_rms.return_value = rms_values
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.rand(5, 200)]
        mock_onnx_session.return_value = mock_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, _ = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert np.isclose(result['energy'], expected_energy)
        assert isinstance(result['energy'], float)


class TestOOMFallback:
    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis_helper.ort.get_available_providers')
    def test_embedding_oom_fallback_to_cpu(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_beat,
        mock_rms,
        mock_chroma,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        gpu_session_call_count = [0]
        cpu_session_call_count = [0]

        def gpu_run(output_names, feed_dict):
            gpu_session_call_count[0] += 1
            if gpu_session_call_count[0] == 1:
                import onnxruntime as ort

                raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                    "Failed to allocate memory for requested buffer of size 765249024"
                )
            return [np.random.rand(5, 200)]

        def cpu_run(output_names, feed_dict):
            cpu_session_call_count[0] += 1
            return [np.random.rand(5, 200)]

        sessions_created = []

        def create_session(model_path, providers=None, provider_options=None, **kwargs):
            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            if (
                isinstance(providers, list)
                and 'CPUExecutionProvider' in providers
                and len(providers) == 1
            ):
                mock_session.run.side_effect = cpu_run
                sessions_created.append('CPU')
            else:
                mock_session.run.side_effect = gpu_run
                sessions_created.append('GPU')

            return mock_session

        mock_onnx_session.side_effect = create_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert embeddings is not None
        assert 'CPU' in sessions_created
        assert cpu_session_call_count[0] > 0

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis_helper.ort.get_available_providers')
    def test_prediction_oom_fallback_to_cpu(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_beat,
        mock_rms,
        mock_chroma,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        gpu_session_call_count = [0]
        cpu_session_call_count = [0]

        def gpu_run(output_names, feed_dict):
            gpu_session_call_count[0] += 1
            if gpu_session_call_count[0] == 2:
                import onnxruntime as ort

                raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                    "Failed to allocate memory for requested buffer"
                )
            return [np.random.rand(5, 200)]

        def cpu_run(output_names, feed_dict):
            cpu_session_call_count[0] += 1
            return [np.random.rand(5, 200)]

        sessions_created = []

        def create_session(model_path, providers=None, provider_options=None, **kwargs):
            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            if (
                isinstance(providers, list)
                and 'CPUExecutionProvider' in providers
                and len(providers) == 1
            ):
                mock_session.run.side_effect = cpu_run
                sessions_created.append('CPU')
            else:
                mock_session.run.side_effect = gpu_run
                sessions_created.append('GPU')

            return mock_session

        mock_onnx_session.side_effect = create_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert embeddings is not None
        assert 'CPU' in sessions_created
        assert cpu_session_call_count[0] > 0

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis_helper.ort.get_available_providers')
    def test_non_oom_exception_is_reraised(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_beat,
        mock_rms,
        mock_chroma,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        def gpu_run(output_names, feed_dict):
            import onnxruntime as ort

            raise ort.capi.onnxruntime_pybind11_state.RuntimeException(
                "Model execution error: Invalid input shape"
            )

        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.side_effect = gpu_run
        mock_onnx_session.return_value = mock_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis_helper.ort.InferenceSession')
    @patch('tasks.analysis.librosa.feature.chroma_stft')
    @patch('tasks.analysis.librosa.feature.rms')
    @patch('tasks.analysis.librosa.beat.beat_track')
    @patch('tasks.analysis.librosa.feature.melspectrogram')
    @patch('tasks.analysis.robust_load_audio_with_fallback')
    @patch('tasks.analysis_helper.ort.get_available_providers')
    def test_successful_gpu_inference_no_fallback(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_beat,
        mock_rms,
        mock_chroma,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_beat.return_value = (120.0, np.array([0, 100]))
        mock_rms.return_value = np.array([[0.5]])
        mock_chroma.return_value = np.random.rand(12, 100)
        mock_mel.return_value = np.random.rand(96, 1000)

        cpu_fallback_used = [False]

        def create_session(model_path, providers=None, provider_options=None, **kwargs):
            if (
                isinstance(providers, list)
                and 'CPUExecutionProvider' in providers
                and len(providers) == 1
            ):
                cpu_fallback_used[0] = True

            mock_session = Mock()
            mock_input = Mock()
            mock_input.name = 'input'
            mock_output = Mock()
            mock_output.name = 'output'
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            call_count = [0]

            def successful_run(output_names, feed_dict):
                call_count[0] += 1
                if call_count[0] <= 2:
                    return [np.random.rand(5, 200)]
                else:
                    return [np.random.rand(5, 2)]

            mock_session.run.side_effect = successful_run
            return mock_session

        mock_onnx_session.side_effect = create_session

        mood_labels = ['happy']
        model_paths = {
            'embedding': '/path/to/embedding.onnx',
            'prediction': '/path/to/prediction.onnx',
            'danceable': '/path/to/danceable.onnx',
            'aggressive': '/path/to/aggressive.onnx',
            'happy': '/path/to/happy.onnx',
            'party': '/path/to/party.onnx',
            'relaxed': '/path/to/relaxed.onnx',
            'sad': '/path/to/sad.onnx',
        }

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is not None
        assert embeddings is not None
        assert cpu_fallback_used[0] is False


class TestMediaServerProbe:
    def test_probe_detects_auth_failure_from_flag(self):
        from tasks.analysis import _probe_looks_like_auth_failure

        assert _probe_looks_like_auth_failure({'ok': False, 'auth_failed': True}) is True

    def test_probe_detects_auth_failure_from_message(self):
        from tasks.analysis import _probe_looks_like_auth_failure

        assert (
            _probe_looks_like_auth_failure({'ok': False, 'error': 'HTTP 401 Unauthorized'}) is True
        )

    def test_probe_ignores_generic_failure(self):
        from tasks.analysis import _probe_looks_like_auth_failure

        assert (
            _probe_looks_like_auth_failure({'ok': False, 'error': 'connection timed out'}) is False
        )

    def test_verify_returns_silently_when_reachable(self):
        from tasks.analysis import _verify_media_server_reachable

        with patch('tasks.analysis.mediaserver_test_connection', return_value={'ok': True}):
            _verify_media_server_reachable()

    def test_verify_raises_auth_error_on_bad_credentials(self):
        from tasks.analysis import _verify_media_server_reachable
        from error.error_manager import AudioMuseError
        from error.error_dictionary import ERR_MEDIASERVER_AUTH

        with patch(
            'tasks.analysis.mediaserver_test_connection',
            return_value={'ok': False, 'auth_failed': True, 'error': 'Wrong username or password'},
        ):
            with pytest.raises(AudioMuseError) as exc_info:
                _verify_media_server_reachable()
        assert exc_info.value.code == ERR_MEDIASERVER_AUTH

    def test_verify_raises_unreachable_on_generic_failure(self):
        from tasks.analysis import _verify_media_server_reachable
        from error.error_manager import AudioMuseError
        from error.error_dictionary import ERR_MEDIASERVER_UNREACHABLE

        with patch(
            'tasks.analysis.mediaserver_test_connection',
            return_value={'ok': False, 'error': 'connection refused'},
        ):
            with pytest.raises(AudioMuseError) as exc_info:
                _verify_media_server_reachable()
        assert exc_info.value.code == ERR_MEDIASERVER_UNREACHABLE
