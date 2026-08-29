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
* Chromaprint backfill liveness: the per-track loop writes throttled progress so
  the task row never looks hung, and it honours revocation mid-loop instead of
  fingerprinting thousands of tracks after a Cancel.
"""

import logging
import sys

import librosa
import numpy as np
import pytest
import config
import taskqueue
from unittest.mock import Mock, patch
from tasks.analysis import (
    sigmoid,
    robust_load_audio_with_fallback,
    analyze_track,
)
from tasks.onnx_utils import run_inference, _find_onnx_name
from tasks.analysis.song import (
    _decode_audio_with_pyav,
    _estimate_energy,
    _estimate_key_scale,
    _estimate_tempo,
)


def test_union_analysis_runs_each_server_once_with_no_sweeps(monkeypatch):
    import tasks.analysis.main as analysis
    import tasks.multiserver_sync as sync

    servers = [
        {'server_id': 'a', 'name': 'A', 'is_default': True},
        {'server_id': 'b', 'name': 'B', 'is_default': False},
        {'server_id': 'c', 'name': 'C', 'is_default': False},
    ]
    events = []
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: servers)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr('tasks.task_run.get_task_info_from_db', lambda task_id: None)
    monkeypatch.setattr(
        analysis, 'get_task_statuses', lambda ids: {i: 'RUNNING' for i in ids}
    )
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


def _union_harness(monkeypatch, phase_results):
    import tasks.analysis.main as analysis

    servers = [
        {'server_id': f's{i}', 'name': name, 'is_default': i == 0}
        for i, (name, _) in enumerate(phase_results)
    ]
    saved = []
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: servers)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr('tasks.task_run.get_task_info_from_db', lambda task_id: None)
    monkeypatch.setattr(
        analysis, 'get_task_statuses', lambda ids: {i: 'RUNNING' for i in ids}
    )
    monkeypatch.setattr(analysis, '_run_all_index_builds', lambda *a, **k: None)
    monkeypatch.setattr(
        analysis, '_albums_per_server', lambda servers, n: [[] for _ in servers]
    )
    monkeypatch.setattr(
        analysis,
        'save_task_status',
        lambda task_id, task_type, status, **kwargs: saved.append(
            (status, kwargs.get('details') or {})
        ),
    )
    by_id = {f's{i}': status for i, (_, status) in enumerate(phase_results)}
    monkeypatch.setattr(
        analysis,
        'run_analysis_server_task',
        lambda *a, server_id=None, **k: {'status': by_id[server_id]},
    )
    return analysis.run_analysis_task(0, 5), saved


def test_union_analysis_succeeds_when_only_some_servers_fail(monkeypatch):
    result, saved = _union_harness(
        monkeypatch, [('Jellyfin', 'FAIL'), ('Plex', 'SUCCESS')]
    )

    assert result['status'] == 'SUCCESS'
    assert result['failed_servers'] == ['Jellyfin']
    status, details = saved[-1]
    assert status == 'SUCCESS'
    assert 'error' not in details
    assert 'Jellyfin' in details['message']


def test_union_analysis_fails_only_when_every_server_fails(monkeypatch):
    from error.error_dictionary import ERR_ANALYSIS_SERVER_FAILED

    result, saved = _union_harness(
        monkeypatch, [('Jellyfin', 'FAIL'), ('Plex', 'FAIL')]
    )

    assert result['status'] == 'FAIL'
    status, details = saved[-1]
    assert status == 'FAIL'
    assert details['error']['error_code'] == ERR_ANALYSIS_SERVER_FAILED
    assert details['error']['error_code'] != 9999


def test_union_analysis_treats_a_wiped_parent_row_as_revoked(monkeypatch):
    import tasks.analysis.main as analysis

    servers = [
        {'server_id': 's0', 'name': 'A', 'is_default': True},
        {'server_id': 's1', 'name': 'B', 'is_default': False},
    ]
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: servers)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr(analysis, 'get_task_statuses', lambda ids: {})
    monkeypatch.setattr(
        analysis, '_albums_per_server', lambda servers, n: [[] for _ in servers]
    )
    ran = []
    monkeypatch.setattr(
        analysis,
        'run_analysis_server_task',
        lambda *a, server_id=None, **k: ran.append(server_id) or {'status': 'SUCCESS'},
    )

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == 'REVOKED'
    assert ran == [], "no phase may run after the cancel wiped the row"


def test_dequeued_analysis_with_wiped_claim_stops_before_listing_servers(monkeypatch):
    import tasks.analysis.main as analysis

    job = Mock(id='analysis-cancelled')
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: job.id)
    monkeypatch.setattr(analysis, 'get_task_statuses', lambda ids: {})
    list_servers = Mock(side_effect=AssertionError('cancelled job must not do work'))
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', list_servers)
    save = Mock(side_effect=AssertionError('cancelled job must not recreate its row'))
    monkeypatch.setattr(analysis, 'save_task_status', save)

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == 'REVOKED'
    list_servers.assert_not_called()
    save.assert_not_called()


def test_dequeued_album_with_wiped_parent_stops_before_creating_child(monkeypatch):
    import tasks.analysis.album as album

    job = Mock(id='album-cancelled')
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: job.id)
    monkeypatch.setattr(album, 'get_task_statuses', lambda ids: {})
    tracks = Mock(side_effect=AssertionError('cancelled album must not fetch tracks'))
    monkeypatch.setattr(album, 'get_tracks_from_album', tracks)
    save = Mock(side_effect=AssertionError('cancelled album must not create a row'))
    monkeypatch.setattr(album, 'save_task_status', save)

    result = album._analyze_album_task_impl('a1', 'Cancelled', 5, 'parent-1')

    assert result['status'] == 'REVOKED'
    tracks.assert_not_called()
    save.assert_not_called()


def test_dequeued_index_rebuild_with_wiped_parent_does_no_build(monkeypatch):
    import tasks.analysis.index as index

    job = Mock(id='index-cancelled')
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: job.id)
    monkeypatch.setattr('database.get_task_statuses', lambda ids: {})
    build = Mock(side_effect=AssertionError('cancelled rebuild must not run'))
    monkeypatch.setattr(index, '_run_all_index_builds', build)

    result = index.rebuild_all_indexes_task('parent-1')

    assert result['status'] == 'REVOKED'
    build.assert_not_called()


def test_union_analysis_stops_when_a_phase_is_revoked(monkeypatch):
    result, _ = _union_harness(monkeypatch, [('Jellyfin', 'REVOKED'), ('Plex', 'SUCCESS')])

    assert result['status'] == 'REVOKED'
    assert result['servers_completed'] == 1


def test_run_analysis_task_skips_when_no_enabled_server_matches_scope(monkeypatch):
    import tasks.analysis.main as analysis

    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: [])
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
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
    import tasks.analysis.main as analysis

    registry = importlib.import_module('tasks.mediaserver.registry')

    def broken_scope(scope):
        raise RuntimeError('registry down')

    monkeypatch.setattr(registry, 'servers_for_scope', broken_scope)

    assert analysis._enabled_analysis_servers('all') == [None]


def test_enabled_analysis_servers_lost_connection_fails_the_batch_not_shrinks_it(monkeypatch):
    import importlib
    import pytest
    import tasks.analysis.main as analysis
    from psycopg2 import OperationalError

    registry = importlib.import_module('tasks.mediaserver.registry')

    def dropped(scope):
        raise OperationalError('connection lost')

    monkeypatch.setattr(registry, 'servers_for_scope', dropped)

    with pytest.raises(OperationalError):
        analysis._enabled_analysis_servers('all')


_FAKE_EMBEDDING = np.sin(np.arange(1, 201, dtype=np.float32))


def _run_album_impl(monkeypatch, tmp_path, item, known_index, persisted_ids, map_upserts,
                    analyzed_embedding=None, existing_ids_fn=None, persist_calls=None,
                    tracks=None, job=None, clap_broken=False, lyrics_enabled=False,
                    download_fn=None):
    import importlib
    import tasks.analysis.album as analysis
    import tasks.analysis.helper as helper
    import tasks.analysis.song as song
    import tasks.clap_analyzer as clap

    registry = importlib.import_module('tasks.mediaserver.registry')
    album_tracks = tracks if tracks is not None else [item]

    monkeypatch.setattr(
        taskqueue, 'current_task_id', lambda: job.id if job is not None else None
    )
    monkeypatch.setattr(analysis, 'save_task_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, 'save_task_status', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, 'get_tracks_from_album', lambda album_id: album_tracks)
    monkeypatch.setattr(
        analysis, 'download_track',
        download_fn or (lambda temp_dir, track: str(tmp_path / 'gone.flac')),
    )
    monkeypatch.setattr(song, 'load_musicnn_sessions', lambda model_paths: {})
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
        analysis, 'decode_audio_once',
        lambda path: (np.ones(16000, dtype=np.float32), 16000),
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
                'duration_seconds': 200.0,
            },
            fake_embedding,
        ),
    )
    monkeypatch.setattr(helper, '_fetch_row_duration', lambda item_id: 200.0)

    monkeypatch.setattr(analysis, 'LYRICS_ENABLED', lyrics_enabled)
    monkeypatch.setattr(clap, 'is_clap_available', lambda: True if clap_broken else False)
    if clap_broken:
        monkeypatch.setattr(helper, 'run_clap_for_track', lambda *a, **k: None)
        monkeypatch.setattr(
            clap, 'get_or_cache_other_feature_text_embeddings', lambda: None
        )
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
    seeded_catalogue = set(getattr(known_index, '_taken', set()))

    def _default_existing_ids(ids):
        have = seeded_catalogue | set(persisted_ids)
        return {i for i in ids if i in have}

    monkeypatch.setattr(
        helper, 'get_existing_track_ids', existing_ids_fn or _default_existing_ids
    )
    monkeypatch.setattr(
        helper,
        'get_missing_ids_in_table',
        lambda table, ids: (
            set(ids)
            if (clap_broken and table == 'clap_embedding')
            or (lyrics_enabled and table == 'lyrics_embedding')
            else set()
        ),
    )
    monkeypatch.setattr(helper, 'get_missing_base_ids', lambda ids: set())
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
    assert map_upserts == [('srv-def', {'prov1': (expected_id, 'fingerprint', None)})]


def test_missing_source_file_is_skipped_and_album_still_succeeds(monkeypatch, tmp_path):
    from tasks import simhash

    ok = {'Id': 'prov_ok', 'Name': 'Present', 'AlbumArtist': 'Artist'}
    gone = {'Id': 'prov_gone', 'Name': 'Deleted', 'AlbumArtist': 'Artist'}

    def _download(temp_dir, track):
        return None if track['Id'] == 'prov_gone' else str(tmp_path / 'present.flac')

    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, ok, simhash.CatalogResolver(), persisted_ids, map_upserts,
        tracks=[ok, gone], download_fn=_download,
    )

    assert result['status'] == 'SUCCESS'
    assert result['tracks_analyzed'] == 1
    assert result['tracks_unavailable'] == 1


def test_album_fails_when_every_track_source_is_unavailable(monkeypatch, tmp_path):
    import pytest
    from tasks import simhash

    gone = {'Id': 'prov_gone', 'Name': 'Deleted', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    with pytest.raises(RuntimeError):
        _run_album_impl(
            monkeypatch, tmp_path, gone, simhash.CatalogResolver(), persisted_ids,
            map_upserts, tracks=[gone], download_fn=lambda temp_dir, track: None,
        )


def test_same_audio_skips_persist_and_just_maps_the_server(monkeypatch, tmp_path):
    from tasks import simhash

    known_id = simhash.canonical_id_str(simhash.embedding_signature(_FAKE_EMBEDDING))
    catalog = simhash.CatalogResolver()
    catalog.register(known_id, embedding=_FAKE_EMBEDDING, duration=201.0)

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, catalog, persisted_ids, map_upserts
    )

    assert result['status'] == 'SUCCESS'
    assert result['tracks_analyzed'] == 1
    assert persisted_ids == []
    assert map_upserts == [('srv-def', {'prov1': (known_id, 'fingerprint', None)})]


def test_same_audio_with_different_duration_gets_its_own_id(monkeypatch, tmp_path):
    from tasks import simhash

    known_id = simhash.canonical_id_str(simhash.embedding_signature(_FAKE_EMBEDDING))
    catalog = simhash.CatalogResolver()
    catalog.register(known_id, embedding=_FAKE_EMBEDDING, duration=300.0)

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, catalog, persisted_ids, map_upserts
    )

    assert result['status'] == 'SUCCESS'
    assert len(persisted_ids) == 1
    assert persisted_ids[0] != known_id
    assert persisted_ids[0].startswith(simhash.CURRENT_ID_HEAD)


def test_same_audio_with_unknown_catalogue_duration_gets_its_own_id(
    monkeypatch, tmp_path
):
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
    assert len(persisted_ids) == 1
    assert persisted_ids[0] != known_id
    assert persisted_ids[0].startswith(simhash.CURRENT_ID_HEAD)


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
    assert persisted_ids[0].startswith(simhash.CURRENT_ID_HEAD)
    assert map_upserts == [('srv-def', {'prov1': (persisted_ids[0], 'fingerprint', None)})]


def test_degenerate_embedding_is_still_mapped_so_it_is_not_re_analyzed_forever(
    monkeypatch, tmp_path
):
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

    expected_id = simhash.unsignable_canonical_id('srv-def', 'prov-degenerate')
    assert result['status'] == 'SUCCESS'
    assert expected_id.startswith('fp_0'), "never a raw provider id in the catalogue"
    assert not simhash.signature_from_canonical_id(expected_id), (
        "and never mistakable for a signature id"
    )
    assert persisted_ids == [expected_id]
    assert map_upserts == [
        ('srv-def', {'prov-degenerate': (expected_id, 'analysis', None)})
    ]
    assert simhash.unsignable_canonical_id('srv-def', 'prov-degenerate') == expected_id, (
        "deterministic, or the track is re-analyzed on every run"
    )
    assert simhash.unsignable_canonical_id('srv-b', 'prov-degenerate') != expected_id, (
        "server-scoped, or two servers' provider ids collide in one namespace"
    )


def test_two_duplicate_files_on_one_server_both_get_a_map_row(monkeypatch, tmp_path):
    from tasks import simhash

    expected_id = simhash.canonical_id_str(simhash.embedding_signature(_FAKE_EMBEDDING))
    track_a = {'Id': 'provA', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    track_b = {'Id': 'provB', 'Name': 'Song (copy)', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, track_a, simhash.CatalogResolver(),
        persisted_ids, map_upserts, tracks=[track_a, track_b],
    )

    assert result['status'] == 'SUCCESS'
    assert result['tracks_analyzed'] == 2
    assert persisted_ids == [expected_id]

    assert len(map_upserts) == 2
    assert {sid for sid, _ in map_upserts} == {'srv-def'}
    merged = {}
    for _sid, mapping in map_upserts:
        merged.update(mapping)
    assert merged == {
        'provA': (expected_id, 'fingerprint', None),
        'provB': (expected_id, 'fingerprint', None),
    }


def test_a_signature_collision_is_refuted_against_the_catalogue_not_the_cache(
    monkeypatch,
):
    import tasks.analysis.helper as helper
    from tasks import simhash

    mine = _FAKE_EMBEDDING
    theirs = np.cos(np.arange(1, 201, dtype=np.float32))
    assert simhash.cosine_distance(mine, theirs) > 0.01, "must be different audio"

    minted = simhash.canonical_id_str(simhash.embedding_signature(mine))
    resolver = simhash.CatalogResolver()
    resolver.register(minted, embedding=mine)

    monkeypatch.setattr(
        helper, 'get_existing_track_ids', lambda ids: {i for i in ids if i == minted}
    )
    monkeypatch.setattr(
        helper, 'catalogue_embedding', lambda item_id: theirs if item_id == minted else None
    )

    kind, settled = helper.claim_new_canonical_id(resolver, minted, mine)

    assert kind == 'new', "a collision with DIFFERENT audio must never be adopted"
    assert settled != minted, "it must step to the next free id, not clobber the row"
    cached = resolver._embedding_for(minted)
    assert simhash.cosine_distance(theirs, cached) <= 0.01, (
        "the refused id must now cache the CATALOGUE's embedding, or the next "
        "copy of this audio resolves straight back onto the row we just refused"
    )


def test_a_failing_stage_never_blocks_a_later_one(monkeypatch, tmp_path):
    import tasks.analysis.helper as helper
    from tasks import simhash

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    lyrics_ran = []
    monkeypatch.setattr(
        helper,
        'run_lyrics_for_track',
        lambda *a, **k: lyrics_ran.append(True) or True,
    )

    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, simhash.CatalogResolver(),
        persisted_ids, map_upserts, clap_broken=True, lyrics_enabled=True,
        existing_ids_fn=lambda ids: set(ids),
    )

    assert result['status'] == 'SUCCESS'
    assert lyrics_ran, "a CLAP failure must not stop lyrics from being analyzed"
    assert result['tracks_not_analyzable'] == 0


def test_a_clap_failure_never_throws_away_a_completed_musicnn_analysis(
    monkeypatch, tmp_path
):
    from tasks import simhash

    item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
    persisted_ids, map_upserts = [], []
    result = _run_album_impl(
        monkeypatch, tmp_path, item, simhash.CatalogResolver(),
        persisted_ids, map_upserts, clap_broken=True,
    )

    expected_id = simhash.canonical_id_str(simhash.embedding_signature(_FAKE_EMBEDDING))
    assert result['status'] == 'SUCCESS'
    assert persisted_ids == [expected_id], (
        "the MusiCNN analysis must be persisted even though CLAP produced nothing"
    )
    assert map_upserts == [('srv-def', {'prov1': (expected_id, 'fingerprint', None)})], (
        "and the map row must be written, or the track is re-analyzed forever"
    )
    assert result['tracks_not_analyzable'] == 0, (
        "a CLAP miss on an otherwise-analyzed track is not an unanalyzable track"
    )


def test_every_server_records_its_own_path_on_its_own_map_row(monkeypatch, tmp_path):
    from tasks import simhash
    from tasks.mediaserver import context as ms_context

    item = {
        'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist',
        'FilePath': '/music/song.flac',
    }

    default_maps = []
    _run_album_impl(
        monkeypatch, tmp_path, dict(item), simhash.CatalogResolver(), [], default_maps,
    )
    assert default_maps[0][1]['prov1'][2] == '/music/song.flac'

    secondary_item = dict(item, FilePath='/plex-media/song.flac')
    secondary_maps = []
    with ms_context.use_server({'server_id': 'srv-b', 'server_type': 'plex'}):
        _run_album_impl(
            monkeypatch, tmp_path, secondary_item, simhash.CatalogResolver(),
            [], secondary_maps,
        )
    assert secondary_maps[0][0] == 'srv-b'
    assert secondary_maps[0][1]['prov1'][2] == '/plex-media/song.flac'


def test_persist_musicnn_results_never_writes_a_path_to_the_shared_row(monkeypatch):
    import tasks.analysis.helper as helper
    import tasks.analysis.song as song

    calls = []

    def _capture(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(song, 'save_track_analysis_and_embedding', _capture)
    item = {
        'Id': 'p1', 'Name': 'Song', 'AlbumArtist': 'Artist',
        'FilePath': '/music/song.flac', '_catalog_item_id': 'fp_2abc',
    }
    analysis = {
        'tempo': 120.0, 'energy': 0.5, 'key': 'C', 'scale': 'major',
        'duration_seconds': 231.4,
    }
    top_moods = {'happy': 0.9}

    helper.persist_musicnn_results(item, analysis, top_moods, b'emb', 'happy:0.90')

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        'fp_2abc', 'Song', 'Artist', 120.0, 'C', 'major', top_moods, b'emb',
    )
    assert kwargs['energy'] == 0.5
    assert kwargs['other_features'] == 'happy:0.90'
    assert kwargs['duration'] == analysis['duration_seconds']
    assert 'file_path' not in kwargs
    assert item['FilePath'] not in args
    assert item['FilePath'] not in kwargs.values()


def test_revocation_is_checked_once_per_album_not_once_per_track(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from tasks import simhash
    import tasks.analysis.album as analysis

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
        lambda ids: status_calls.append(list(ids)) or {i: 'RUNNING' for i in ids if i},
    )
    monkeypatch.setattr(analysis, 'ANALYSIS_MONITOR_DB_INTERVAL', 10_000_000)

    def forbidden(task_id):
        raise AssertionError('the per-track loop must not query task info per track')

    monkeypatch.setattr('tasks.task_run.get_task_info_from_db', forbidden, raising=False)

    result = _run_album_impl(
        monkeypatch, tmp_path, tracks[0], simhash.CatalogResolver(), [], [],
        tracks=tracks, job=job,
    )

    assert result['status'] == 'SUCCESS'
    assert len(status_calls) == 2
    assert status_calls[0] == ['parent1']
    assert status_calls[1] == ['job-1', 'parent1']


def _run_parent_phase(monkeypatch, albums, tracks_by_album, work_map,
                      baseline_read_error=None, status_calls=None,
                      expired_but_db_terminal=False, child_rows=None,
                      extra_jobs=None):
    import importlib
    import tasks.analysis.main as analysis
    import tasks.analysis.helper as helper
    import tasks.clap_analyzer as clap

    registry = importlib.import_module('tasks.mediaserver.registry')

    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr('tasks.task_run.get_task_info_from_db', lambda task_id: None)
    def _record_status(*args, **kwargs):
        if status_calls is not None:
            status_calls.append(kwargs.get('details') or {})

    monkeypatch.setattr(analysis, 'save_task_status', _record_status)
    monkeypatch.setattr(helper, 'save_task_status', _record_status)
    monkeypatch.setattr(analysis, 'clean_temp', lambda *args, **kwargs: None)
    monkeypatch.setattr(analysis, 'get_recent_albums', lambda limit: albums)
    monkeypatch.setattr(
        analysis, 'get_tracks_from_album', lambda album_id: tracks_by_album[album_id]
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
                 'get_missing_base_ids', 'attach_catalog_item_ids'):
        monkeypatch.setattr(helper, name, forbidden)

    enqueued = []
    queued_ids = []

    def _fake_enqueue(func, args=None, **kwargs):
        task_id = kwargs.get('task_id') or f'job-{len(enqueued)}'
        enqueued.append(args)
        queued_ids.append(task_id)
        return task_id

    pending_ids = list(
        row['task_id'] for row in (child_rows or [])
        if row['status'] in (config.TASK_STATUS_NEW, config.TASK_STATUS_RUNNING)
    )
    already_terminal_rows = [
        {
            'task_id': row['task_id'],
            'status': row['status'],
            'sub_type_identifier': row['sub_type_identifier'],
            'details': row.get('details') or {},
        }
        for row in (child_rows or [])
        if row['status'] not in (config.TASK_STATUS_NEW, config.TASK_STATUS_RUNNING)
    ]
    carried_over_read_done = []

    def _fake_reap(parent_task_id, conn=None):
        if not carried_over_read_done:
            carried_over_read_done.append(True)
            if baseline_read_error is not None:
                raise baseline_read_error
            return already_terminal_rows
        pending_ids.extend(queued_ids)
        queued_ids.clear()
        reaped = [
            {
                'task_id': task_id,
                'status': config.TASK_STATUS_SUCCESS,
                'sub_type_identifier': f'album-for-{task_id}',
                'details': {'tracks_analyzed': 1},
            }
            for task_id in pending_ids
        ]
        pending_ids.clear()
        return reaped

    monkeypatch.setattr(taskqueue, 'enqueue', _fake_enqueue)
    monkeypatch.setattr(taskqueue, 'reap_finished_children', _fake_reap)
    monkeypatch.setattr(
        taskqueue, 'live_children',
        lambda parent_task_id, conn=None: [
            {'task_id': row['task_id'], 'sub_type_identifier': row['sub_type_identifier']}
            for row in (child_rows or [])
            if row['status'] in (config.TASK_STATUS_NEW, config.TASK_STATUS_RUNNING)
        ],
    )
    if child_rows:
        monkeypatch.setattr(analysis.time, 'sleep', lambda *a, **k: None)

    def _statuses(ids):
        return {
            i: ('SUCCESS' if expired_but_db_terminal and str(i).startswith('job-')
                else 'RUNNING')
            for i in ids if i
        }

    monkeypatch.setattr(analysis, 'get_task_statuses', _statuses)
    if expired_but_db_terminal:
        monkeypatch.setattr(analysis, 'ANALYSIS_MONITOR_DB_INTERVAL', 0)
        monkeypatch.setattr(analysis.time, 'sleep', lambda *a, **k: None)

    result = analysis._run_analysis_server_task_impl(
        0, 5, server_id='srv-def', task_id='parent-1'
    )
    return result, enqueued


def test_union_run_counts_albums_across_every_server(monkeypatch):
    import tasks.analysis.main as analysis

    servers = [
        {'server_id': 'a', 'name': 'A', 'is_default': True},
        {'server_id': 'b', 'name': 'B', 'is_default': False},
    ]
    albums_by_server = {'a': [{'Id': 'a1'}, {'Id': 'a2'}], 'b': [{'Id': 'b1'}]}
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: servers)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr('tasks.task_run.get_task_info_from_db', lambda task_id: None)
    monkeypatch.setattr(
        analysis, 'get_task_statuses', lambda ids: {i: 'RUNNING' for i in ids}
    )
    monkeypatch.setattr(analysis, 'save_task_status', lambda *a, **k: None)
    monkeypatch.setattr(analysis, '_run_all_index_builds', lambda *a, **k: None)
    monkeypatch.setattr(
        analysis,
        '_albums_per_server',
        lambda servers_, limit: [albums_by_server[s['server_id']] for s in servers_],
    )

    calls = []

    def fake_phase(*args, server_id=None, albums=None, albums_offset=0,
                   albums_total=None, **kwargs):
        calls.append((server_id, len(albums), albums_offset, albums_total))
        return {'status': 'SUCCESS'}

    monkeypatch.setattr(analysis, 'run_analysis_server_task', fake_phase)

    analysis.run_analysis_task(0, 5)

    assert calls == [('a', 2, 0, 3), ('b', 1, 2, 3)]


def test_settled_library_enqueues_nothing_and_never_queries_per_album(monkeypatch):
    import tasks.analysis.helper as helper

    albums = [{'Id': f'al{i}', 'Name': f'Album {i}'} for i in range(3)]
    tracks_by_album = {
        f'al{i}': [{'Id': f'p{i}-{t}', 'Name': 't'} for t in range(2)]
        for i in range(3)
    }
    work_map = {
        f'p{i}-{t}': helper.WORK_MUSICNN | helper.WORK_BASE
        for i in range(3) for t in range(2)
    }

    result, enqueued = _run_parent_phase(monkeypatch, albums, tracks_by_album, work_map)

    assert result['status'] == 'SUCCESS'
    assert enqueued == []
    assert result['message'] == 'Albums 3/3'


def test_album_with_one_unanalyzed_track_is_still_enqueued(monkeypatch):
    import tasks.analysis.helper as helper

    albums = [{'Id': 'al0', 'Name': 'Album 0'}, {'Id': 'al1', 'Name': 'Album 1'}]
    tracks_by_album = {
        'al0': [{'Id': 'done-1', 'Name': 't'}, {'Id': 'done-2', 'Name': 't'}],
        'al1': [{'Id': 'done-3', 'Name': 't'}, {'Id': 'missing', 'Name': 't'}],
    }
    work_map = {
        'done-1': helper.WORK_MUSICNN | helper.WORK_BASE,
        'done-2': helper.WORK_MUSICNN | helper.WORK_BASE,
        'done-3': helper.WORK_MUSICNN | helper.WORK_BASE,
    }

    result, enqueued = _run_parent_phase(monkeypatch, albums, tracks_by_album, work_map)

    assert result['status'] == 'SUCCESS'
    assert [args[0] for args in enqueued] == ['al1']
    assert result['message'] == 'Albums 2/2'


def test_phase_outcome_never_reports_more_albums_than_the_total():
    import tasks.analysis.main as analysis

    message, status, _ = analysis._phase_outcome(
        7523, 6949, albums_launched=10, failed_count=0,
        failed_errors=[], albums_work_check_failed=0,
    )

    assert message == 'Albums 6949/6949'
    assert status == 'SUCCESS'


def test_retry_with_stale_child_rows_counts_each_album_once(monkeypatch):
    albums = [{'Id': f'al{i}', 'Name': f'Album {i}'} for i in range(3)]
    tracks_by_album = {f'al{i}': [{'Id': f'p{i}', 'Name': 't'}] for i in range(3)}
    work_map = {}
    child_rows = [
        {
            'task_id': 'stale-done-1', 'status': 'SUCCESS',
            'sub_type_identifier': 'al0', 'details': {'tracks_analyzed': 4},
        }
    ]

    status_calls = []
    result, enqueued = _run_parent_phase(
        monkeypatch, albums, tracks_by_album, work_map,
        child_rows=child_rows, status_calls=status_calls,
    )

    assert result['status'] == 'SUCCESS'
    assert result['message'] == 'Albums 3/3'
    assert result['albums_completed'] == 3
    assert [args[0] for args in enqueued] == ['al0', 'al1', 'al2']
    assert result['tracks_analyzed'] == 7
    reported = [d['albums_completed'] for d in status_calls if 'albums_completed' in d]
    assert reported
    assert max(reported) <= 3


def test_baseline_read_failure_is_absorbed_and_does_not_inflate_the_track_tally(
    monkeypatch, caplog
):
    import logging

    albums = [{'Id': f'al{i}', 'Name': f'Album {i}'} for i in range(3)]
    tracks_by_album = {f'al{i}': [{'Id': f'p{i}', 'Name': 't'}] for i in range(3)}
    work_map = {}

    status_calls = []
    with caplog.at_level(logging.ERROR, logger='tasks.analysis.main'):
        result, enqueued = _run_parent_phase(
            monkeypatch, albums, tracks_by_album, work_map,
            baseline_read_error=RuntimeError('db blip while reading the baseline'),
            status_calls=status_calls,
        )

    assert any(
        'Could not clear the finished album jobs' in record.getMessage()
        for record in caplog.records
    )
    assert result['status'] == 'SUCCESS'
    assert result['message'] == 'Albums 3/3'
    assert result['albums_completed'] == 3
    assert [args[0] for args in enqueued] == ['al0', 'al1', 'al2']
    assert result['tracks_analyzed'] == 3
    reported = [d['albums_completed'] for d in status_calls if 'albums_completed' in d]
    assert reported
    assert max(reported) <= 3


def test_retry_adopts_previous_attempts_running_album_instead_of_duplicating(monkeypatch):
    albums = [{'Id': 'al0', 'Name': 'Album 0'}]
    tracks_by_album = {'al0': [{'Id': 'p0', 'Name': 't'}]}
    work_map = {}
    child_rows = [
        {'task_id': 'stale-1', 'status': 'RUNNING', 'sub_type_identifier': 'al0'}
    ]
    result, enqueued = _run_parent_phase(
        monkeypatch, albums, tracks_by_album, work_map, child_rows=child_rows,
    )

    assert result['status'] == 'SUCCESS'
    assert enqueued == []
    assert result['message'] == 'Albums 1/1'


def test_retry_reenqueues_an_album_whose_child_row_is_gone(monkeypatch):
    albums = [{'Id': 'al0', 'Name': 'Album 0'}]
    tracks_by_album = {'al0': [{'Id': 'p0', 'Name': 't'}]}
    work_map = {}

    result, enqueued = _run_parent_phase(
        monkeypatch, albums, tracks_by_album, work_map, child_rows=[],
    )

    assert result['status'] == 'SUCCESS'
    assert [args[0] for args in enqueued] == ['al0']
    assert result['message'] == 'Albums 1/1'


def test_unknown_catalogue_track_requires_real_musicnn_analysis():
    from tasks.analysis.helper import plan_track_stages

    assert plan_track_stages(
        'provider-new',
        existing_ids={'fp_existing'},
        missing_clap_ids={'provider-new'},
        missing_lyrics_ids={'provider-new'},
        missing_base_ids=set(),
        lyrics_enabled=True,
    ) == (True, True, True, False)


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
    @patch('tasks.analysis.song.librosa.load')
    def test_successful_direct_load(self, mock_librosa_load):
        expected_audio = np.random.rand(16000)
        expected_sr = 16000
        mock_librosa_load.return_value = (expected_audio, expected_sr)

        audio, sr = robust_load_audio_with_fallback('test.mp3', target_sr=16000)

        assert audio is not None
        assert sr == expected_sr
        np.testing.assert_array_equal(audio, expected_audio)
        mock_librosa_load.assert_called_once()

    @patch('tasks.analysis.song.librosa.load')
    def test_direct_load_with_custom_sample_rate(self, mock_librosa_load):
        expected_audio = np.random.rand(22050)
        expected_sr = 22050
        mock_librosa_load.return_value = (expected_audio, expected_sr)

        audio, sr = robust_load_audio_with_fallback('test.wav', target_sr=22050)

        assert sr == 22050
        mock_librosa_load.assert_called_once_with('test.wav', sr=22050, mono=True, duration=600)

    @patch('tasks.analysis.song.librosa.load')
    @patch('tasks.analysis.song._decode_audio_with_pyav')
    def test_fallback_on_librosa_failure(self, mock_pyav_decode, mock_librosa_load):
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_pyav_decode.return_value = (np.random.rand(16000).astype(np.float32), 16000)

        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')

        assert audio is not None
        assert sr == 16000
        mock_pyav_decode.assert_called_once_with('corrupted.mp3', 16000)

    @patch('tasks.analysis.song.librosa.load')
    def test_returns_none_on_empty_audio(self, mock_librosa_load):
        mock_librosa_load.return_value = (np.array([]), 16000)

        audio, sr = robust_load_audio_with_fallback('empty.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.song.librosa.load')
    def test_returns_none_on_none_audio(self, mock_librosa_load):
        mock_librosa_load.return_value = (None, 16000)

        audio, sr = robust_load_audio_with_fallback('invalid.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.song.librosa.load')
    @patch('tasks.analysis.song._decode_audio_with_pyav')
    def test_fallback_handles_silent_audio(self, mock_pyav_decode, mock_librosa_load):
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_pyav_decode.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        audio, sr = robust_load_audio_with_fallback('silent.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.song.librosa.load')
    @patch('tasks.analysis.song._decode_audio_with_pyav')
    def test_fallback_handles_decode_failure(self, mock_pyav_decode, mock_librosa_load):
        mock_librosa_load.side_effect = Exception("Librosa failed")
        mock_pyav_decode.side_effect = Exception("PyAV failed")

        audio, sr = robust_load_audio_with_fallback('corrupted.mp3')

        assert audio is None
        assert sr is None

    @patch('tasks.analysis.song.librosa.load')
    def test_uses_audio_load_timeout_config(self, mock_librosa_load):
        mock_librosa_load.return_value = (np.random.rand(16000), 16000)

        robust_load_audio_with_fallback('test.mp3', target_sr=16000)

        call_args = mock_librosa_load.call_args
        assert 'duration' in call_args.kwargs
        assert call_args.kwargs['duration'] == 600


class TestPyAVRejectsWhatIsMostlyLost:
    RATE = 44100

    def _write_mp3(self, path, seconds):
        import av

        with av.open(str(path), 'w') as container:
            stream = container.add_stream('mp3', rate=self.RATE)
            stream.layout = 'stereo'
            n = int(self.RATE * seconds)
            t = np.arange(n, dtype=np.float32) / self.RATE
            data = np.vstack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])
            data = (data * 0.5 * 32767).astype(np.int16)
            frame = av.AudioFrame.from_ndarray(
                data.T.reshape(1, -1), format='s16', layout='stereo'
            )
            frame.sample_rate = self.RATE
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def test_a_lightly_damaged_file_stays_above_the_floor(self, tmp_path):
        good = tmp_path / 'good.mp3'
        bad = tmp_path / 'nicked.mp3'
        self._write_mp3(good, 10.0)
        raw = bytearray(good.read_bytes())
        offset = len(raw) // 2
        raw[offset:offset + 383] = (bytes([0x0b, 0x4e, 0x88, 0xf9]) * 383)[:383]
        bad.write_bytes(bytes(raw))

        audio, sr = _decode_audio_with_pyav(str(bad), None)

        assert audio.size / sr > 9.0

    def test_losing_most_of_the_declared_duration_is_rejected(self, tmp_path, caplog):
        good = tmp_path / 'good.mp3'
        stump = tmp_path / 'stump.mp3'
        self._write_mp3(good, 10.0)
        raw = good.read_bytes()
        stump.write_bytes(raw[: len(raw) // 4])

        with caplog.at_level(logging.ERROR, logger='tasks.analysis.song'):
            audio, _sr = _decode_audio_with_pyav(str(stump), None)

        assert audio.size == 0
        assert 'not decodable' in caplog.text

    def test_the_floor_only_applies_once_librosa_has_given_up(self, tmp_path):
        good = tmp_path / 'good.mp3'
        stump = tmp_path / 'stump.mp3'
        self._write_mp3(good, 10.0)
        raw = good.read_bytes()
        stump.write_bytes(raw[: len(raw) // 4])

        kept, _sr = robust_load_audio_with_fallback(str(stump), target_sr=None)
        assert kept is not None and kept.size > 0

        with patch('tasks.analysis.song.librosa.load', side_effect=RuntimeError('boom')):
            audio, sr = robust_load_audio_with_fallback(str(stump), target_sr=None)

        assert audio is None
        assert sr is None

    def test_lowering_the_floor_to_zero_accepts_anything_that_decodes(self, tmp_path):
        from tasks.analysis import song as song_mod

        good = tmp_path / 'good.mp3'
        stump = tmp_path / 'stump.mp3'
        self._write_mp3(good, 10.0)
        raw = good.read_bytes()
        stump.write_bytes(raw[: len(raw) // 4])

        with patch.object(song_mod, 'AUDIO_MIN_DECODED_FRACTION', 0.0):
            audio, _sr = _decode_audio_with_pyav(str(stump), None)

        assert audio.size > 0

    def test_the_load_timeout_cap_is_not_mistaken_for_a_loss(self, tmp_path):
        from tasks.analysis import song as song_mod

        path = tmp_path / 'long.mp3'
        self._write_mp3(path, 4.0)
        with patch.object(song_mod, 'AUDIO_LOAD_TIMEOUT', 2):
            audio, sr = _decode_audio_with_pyav(str(path), None)

        assert audio.size > 0
        assert audio.size / sr <= 2.5

    def test_an_unknown_declared_duration_never_rejects(self):
        from tasks.analysis import song as song_mod

        assert song_mod._enough_survived(np.ones(10, dtype=np.float32), 10, None, 'x')


class TestPyAVSurvivesCorruptPackets:
    RATE = 44100

    def _write_mp3(self, path, seconds):
        import av

        with av.open(str(path), 'w') as container:
            stream = container.add_stream('mp3', rate=self.RATE)
            stream.layout = 'stereo'
            n = int(self.RATE * seconds)
            t = np.arange(n, dtype=np.float32) / self.RATE
            data = np.vstack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])
            data = (data * 0.5 * 32767).astype(np.int16)
            frame = av.AudioFrame.from_ndarray(
                data.T.reshape(1, -1), format='s16', layout='stereo'
            )
            frame.sample_rate = self.RATE
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def _corrupt_middle(self, src, dst, nbytes=383):
        raw = bytearray(src.read_bytes())
        offset = len(raw) // 2
        pattern = bytes([0x0b, 0x4e, 0x88, 0xf9])
        raw[offset:offset + nbytes] = (pattern * nbytes)[:nbytes]
        dst.write_bytes(bytes(raw))

    def test_a_corrupt_packet_mid_file_keeps_the_audio_that_decoded(self, tmp_path, caplog):
        good = tmp_path / 'good.mp3'
        bad = tmp_path / 'bad.mp3'
        self._write_mp3(good, 10.0)
        self._corrupt_middle(good, bad)

        with caplog.at_level(logging.WARNING, logger='tasks.analysis.song'):
            audio, sr = _decode_audio_with_pyav(str(bad), None)

        assert 'corrupt audio packet' in caplog.text
        assert audio.size > 0
        assert sr == self.RATE
        assert audio.size / sr > 5.0

    def test_a_clean_file_matches_a_plain_librosa_decode_and_skips_nothing(self, tmp_path, caplog):
        good = tmp_path / 'clean.mp3'
        self._write_mp3(good, 2.0)

        with caplog.at_level(logging.WARNING, logger='tasks.analysis.song'):
            audio, sr = _decode_audio_with_pyav(str(good), None)
        reference, ref_sr = librosa.load(str(good), sr=None, mono=True)

        assert 'corrupt audio packet' not in caplog.text
        assert sr == ref_sr
        assert audio.size == reference.size
        np.testing.assert_allclose(audio, reference, atol=1e-4)


class TestPyAVDecodeDownmix:
    RATE = 16000

    def _write_wav(self, path, channels, layout):
        import av

        with av.open(str(path), 'w') as container:
            stream = container.add_stream('pcm_f32le', rate=self.RATE, layout=layout)
            frame = av.AudioFrame.from_ndarray(
                np.stack(channels).astype(np.float32), format='fltp', layout=layout
            )
            frame.sample_rate = self.RATE
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def test_stereo_downmix_averages_channels_instead_of_scaling_by_sqrt2(self, tmp_path):
        path = tmp_path / 'stereo.wav'
        left = np.full(self.RATE, 0.5, dtype=np.float32)
        right = np.full(self.RATE, 0.25, dtype=np.float32)
        self._write_wav(path, [left, right], 'stereo')

        audio, sr = _decode_audio_with_pyav(str(path), self.RATE)

        assert sr == self.RATE
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, 0.375, atol=1e-5)
        assert not np.allclose(audio, 0.75 / np.sqrt(2), atol=1e-3)

    def test_stereo_downmix_matches_librosa_mono_load(self, tmp_path):
        import librosa

        path = tmp_path / 'stereo_music.wav'
        t = np.arange(self.RATE * 2, dtype=np.float32) / self.RATE
        left = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = (0.3 * np.sin(2 * np.pi * 660 * t + 0.7)).astype(np.float32)
        self._write_wav(path, [left, right], 'stereo')

        pyav_audio, _ = _decode_audio_with_pyav(str(path), self.RATE)
        librosa_audio, _ = librosa.load(str(path), sr=self.RATE, mono=True)

        assert len(pyav_audio) == len(librosa_audio)
        np.testing.assert_allclose(pyav_audio, librosa_audio, atol=1e-6)

    def test_mono_source_passes_through_unscaled(self, tmp_path):
        path = tmp_path / 'mono.wav'
        signal = np.full(self.RATE, 0.4, dtype=np.float32)
        self._write_wav(path, [signal], 'mono')

        audio, sr = _decode_audio_with_pyav(str(path), self.RATE)

        assert sr == self.RATE
        assert audio.ndim == 1
        np.testing.assert_allclose(audio, 0.4, atol=1e-5)

    def test_multichannel_source_averages_every_channel(self, tmp_path):
        path = tmp_path / 'surround.wav'
        channels = [np.full(self.RATE, 0.1 * (i + 1), dtype=np.float32) for i in range(6)]
        self._write_wav(path, channels, '5.1')

        audio, sr = _decode_audio_with_pyav(str(path), self.RATE)

        assert sr == self.RATE
        assert audio.ndim == 1
        np.testing.assert_allclose(audio, 0.35, atol=1e-5)

    def test_native_rate_decode_reports_source_sample_rate(self, tmp_path):
        path = tmp_path / 'native.wav'
        left = np.full(self.RATE, 0.5, dtype=np.float32)
        right = np.full(self.RATE, 0.25, dtype=np.float32)
        self._write_wav(path, [left, right], 'stereo')

        audio, sr = _decode_audio_with_pyav(str(path), None)

        assert sr == self.RATE
        assert audio.ndim == 1
        np.testing.assert_allclose(audio, 0.375, atol=1e-5)


class TestAnalyzeTrack:
    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_successful_track_analysis(
        self, mock_audio_load, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_returns_none_on_audio_load_failure(self, mock_audio_load):
        mock_audio_load.return_value = (None, None)

        mood_labels = ['happy', 'sad']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('bad_file.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_returns_none_on_empty_audio(self, mock_audio_load):
        mock_audio_load.return_value = (np.array([]), 16000)

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('empty.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_returns_none_on_silent_audio(self, mock_audio_load):
        mock_audio_load.return_value = (np.zeros(16000), 16000)

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('silent.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_returns_none_on_short_audio(
        self, mock_audio_load, mock_tempo, mock_energy, mock_key_scale, mock_mel
    ):
        mock_audio = np.random.rand(100)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
        mock_mel.return_value = np.random.rand(96, 10)

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/model.onnx'}

        result, embeddings = analyze_track('short.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_spectrogram_dtype_conversion(
        self, mock_audio_load, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000).astype(np.float64)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_key_detection_logic(
        self, mock_audio_load, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5

        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_model_inference_failure_handling(
        self, mock_audio_load, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
        mock_mel.return_value = np.random.rand(96, 1000)

        mock_onnx_session.side_effect = Exception("Model loading failed")

        mood_labels = ['happy']
        model_paths = {'embedding': '/path/to/embedding.onnx'}

        result, embeddings = analyze_track('test.mp3', mood_labels, model_paths)

        assert result is None
        assert embeddings is None

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_tempo_extraction(
        self, mock_audio_load, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        expected_tempo = 128.5
        mock_tempo.return_value = expected_tempo
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    def test_energy_calculation(
        self, mock_audio_load, mock_mel, mock_tempo, mock_energy, mock_key_scale, mock_onnx_session
    ):
        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0

        expected_energy = 0.75
        mock_energy.return_value = expected_energy
        mock_key_scale.return_value = ('C', 'major')
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
    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    @patch('tasks.analysis.song.ort.get_available_providers')
    def test_embedding_oom_fallback_to_cpu(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_tempo,
        mock_energy,
        mock_key_scale,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    @patch('tasks.analysis.song.ort.get_available_providers')
    def test_prediction_oom_fallback_to_cpu(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_tempo,
        mock_energy,
        mock_key_scale,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    @patch('tasks.analysis.song.ort.get_available_providers')
    def test_non_oom_exception_is_reraised(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_tempo,
        mock_energy,
        mock_key_scale,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    @patch('tasks.analysis.song.ort.InferenceSession')
    @patch('tasks.analysis.song._estimate_key_scale')
    @patch('tasks.analysis.song._estimate_energy')
    @patch('tasks.analysis.song._estimate_tempo')
    @patch('tasks.analysis.song.librosa.feature.melspectrogram')
    @patch('tasks.analysis.song.robust_load_audio_with_fallback')
    @patch('tasks.analysis.song.ort.get_available_providers')
    def test_successful_gpu_inference_no_fallback(
        self,
        mock_providers,
        mock_audio_load,
        mock_mel,
        mock_tempo,
        mock_energy,
        mock_key_scale,
        mock_onnx_session,
    ):
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        mock_audio = np.random.rand(16000)
        mock_audio_load.return_value = (mock_audio, 16000)

        mock_tempo.return_value = 120.0
        mock_energy.return_value = 0.5
        mock_key_scale.return_value = ('C', 'major')
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

    def test_verify_consults_the_media_server_and_returns_silently_when_reachable(self):
        from tasks.analysis import _verify_media_server_reachable

        with patch(
            'tasks.analysis.main.mediaserver_test_connection', return_value={'ok': True}
        ) as probe:
            assert _verify_media_server_reachable() is None

        probe.assert_called_once_with()

    def test_verify_raises_auth_error_on_bad_credentials(self):
        from tasks.analysis import _verify_media_server_reachable
        from error.error_manager import AudioMuseError
        from error.error_dictionary import ERR_MEDIASERVER_AUTH

        with patch(
            'tasks.analysis.main.mediaserver_test_connection',
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
            'tasks.analysis.main.mediaserver_test_connection',
            return_value={'ok': False, 'error': 'connection refused'},
        ):
            with pytest.raises(AudioMuseError) as exc_info:
                _verify_media_server_reachable()
        assert exc_info.value.code == ERR_MEDIASERVER_UNREACHABLE


@pytest.mark.parametrize('terminal_status', ['REVOKED', 'FAIL', 'SUCCESS'])
def test_a_requeued_job_refuses_to_rerun_a_terminal_task(monkeypatch, terminal_status):
    import tasks.analysis.main as analysis

    monkeypatch.setattr(
        analysis, 'get_task_statuses', lambda ids: {i: terminal_status for i in ids}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            'a cancelled, failed or completed task must never run again'
        )

    monkeypatch.setattr(analysis, '_enabled_analysis_servers', forbidden)
    monkeypatch.setattr(analysis, '_run_all_index_builds', forbidden)
    monkeypatch.setattr(analysis, '_run_chromaprint_backfill', forbidden)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == terminal_status


def test_a_run_cancelled_during_the_album_phases_never_reaches_chromaprint(monkeypatch):
    import tasks.analysis.main as analysis

    servers = [
        {'server_id': 'a', 'name': 'A', 'is_default': True},
        {'server_id': 'b', 'name': 'B', 'is_default': False},
    ]
    statuses = {'value': 'RUNNING'}
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: servers)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr(analysis, 'save_task_status', lambda *a, **k: None)
    monkeypatch.setattr(
        analysis, 'get_task_statuses', lambda ids: {i: statuses['value'] for i in ids}
    )
    monkeypatch.setattr(
        analysis, '_albums_per_server', lambda servers_, limit: [[], []]
    )

    def finish_phase(*args, **kwargs):
        statuses['value'] = 'REVOKED'
        return {'status': 'SUCCESS'}

    monkeypatch.setattr(analysis, 'run_analysis_server_task', finish_phase)

    def forbidden(*args, **kwargs):
        raise AssertionError('the tail phases must not run after a cancel')

    monkeypatch.setattr(analysis, '_run_all_index_builds', forbidden)
    monkeypatch.setattr(analysis, '_run_chromaprint_backfill', forbidden)

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == 'REVOKED'


def test_a_live_run_is_not_blocked_by_the_terminal_guard(monkeypatch):
    import tasks.analysis.main as analysis

    monkeypatch.setattr(
        analysis, 'get_task_statuses', lambda ids: {i: 'RUNNING' for i in ids}
    )
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr(analysis, 'save_task_status', lambda *a, **k: None)
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: [])

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == 'SKIPPED'


def test_an_unreadable_status_lets_the_run_proceed_rather_than_stalling(monkeypatch):
    import tasks.analysis.main as analysis

    def boom(ids):
        raise RuntimeError('db down')

    monkeypatch.setattr(analysis, 'get_task_statuses', boom)
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr(analysis, 'save_task_status', lambda *a, **k: None)
    monkeypatch.setattr(analysis, '_enabled_analysis_servers', lambda scope: [])

    result = analysis.run_analysis_task(0, 5)

    assert result['status'] == 'SKIPPED'


def _chromaprint_backfill_harness(monkeypatch, targets_by_server, report_seconds=0):
    import contextlib
    import tasks.analysis.main as analysis
    from tasks.mediaserver import context as server_context

    monkeypatch.setattr(analysis, 'CHROMAPRINT_COLLECTION_ENABLED', True)
    monkeypatch.setattr(
        analysis, 'CHROMAPRINT_BACKFILL_REPORT_SECONDS', report_seconds
    )
    monkeypatch.setattr(analysis.chromaprint, 'is_available', lambda: True)
    monkeypatch.setattr(analysis, '_bind_server_context', lambda server_id: server_id)
    monkeypatch.setattr(
        server_context, 'use_server', lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        analysis,
        '_chromaprint_backfill_targets',
        lambda server_id, limit: list(targets_by_server.get(server_id, [])),
    )
    processed = []
    monkeypatch.setattr(
        analysis,
        '_backfill_one_track',
        lambda server_id, track_id, path: processed.append((server_id, track_id)) or True,
    )
    return analysis, processed


def _fake_targets(count, prefix='t'):
    return [(f'{prefix}{i}', f'/music/{prefix}{i}.flac') for i in range(count)]


def test_chromaprint_backfill_writes_progress_during_the_loop_not_only_before_it(
    monkeypatch,
):
    analysis, processed = _chromaprint_backfill_harness(
        monkeypatch, {'srv': _fake_targets(5)}
    )
    reports = []

    analysis._run_chromaprint_backfill(
        ['srv'], log_fn=lambda message, progress=99: reports.append(message)
    )

    assert len(processed) == 5
    assert len(reports) == 6
    assert '5/5 track(s)' in reports[-1]


def test_chromaprint_backfill_throttles_progress_writes_to_the_configured_interval(
    monkeypatch,
):
    analysis, processed = _chromaprint_backfill_harness(
        monkeypatch, {'srv': _fake_targets(5)}, report_seconds=3600
    )
    reports = []

    analysis._run_chromaprint_backfill(
        ['srv'], log_fn=lambda message, progress=99: reports.append(message)
    )

    assert len(processed) == 5
    assert len(reports) == 1


def test_chromaprint_backfill_stops_mid_loop_once_the_run_is_revoked(monkeypatch):
    analysis, processed = _chromaprint_backfill_harness(
        monkeypatch, {'srv': _fake_targets(10)}
    )

    analysis._run_chromaprint_backfill(
        ['srv'],
        log_fn=lambda message, progress=99: None,
        should_stop=lambda: len(processed) >= 3,
    )

    assert len(processed) == 3


def test_chromaprint_backfill_leaves_remaining_servers_untouched_once_revoked(
    monkeypatch,
):
    analysis, processed = _chromaprint_backfill_harness(
        monkeypatch,
        {'srv-a': _fake_targets(2, 'a'), 'srv-b': _fake_targets(2, 'b'),
         'srv-c': _fake_targets(2, 'c')},
    )

    analysis._run_chromaprint_backfill(
        ['srv-a', 'srv-b', 'srv-c'],
        log_fn=lambda message, progress=99: None,
        should_stop=lambda: len(processed) >= 2,
    )

    assert {server_id for server_id, _ in processed} == {'srv-a'}


def test_chromaprint_backfill_covers_every_server_when_nothing_is_revoked(monkeypatch):
    analysis, processed = _chromaprint_backfill_harness(
        monkeypatch,
        {'srv-a': _fake_targets(2, 'a'), 'srv-b': [], 'srv-c': _fake_targets(1, 'c')},
    )

    analysis._run_chromaprint_backfill(
        ['srv-a', 'srv-b', 'srv-c'], log_fn=lambda message, progress=99: None
    )

    assert {server_id for server_id, _ in processed} == {'srv-a', 'srv-c'}


def test_index_rebuild_reports_as_a_child_of_the_analysis_that_spawned_it(monkeypatch):
    import tasks.analysis.index as index

    job = Mock(id='index-1')
    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: job.id)
    monkeypatch.setattr('database.get_task_statuses', lambda ids: {ids[0]: 'RUNNING'})
    monkeypatch.setattr(index, '_run_all_index_builds', lambda **kwargs: None)

    captured = {}

    def fake_reporter(task_id, task_type, message, **kwargs):
        captured['task_type'] = task_type
        captured['parent_task_id'] = kwargs.get('parent_task_id')
        return lambda *a, **k: None

    monkeypatch.setattr(index, 'make_task_reporter', fake_reporter)

    index.rebuild_all_indexes_task('parent-1')

    assert captured['task_type'] == 'index_rebuild'
    assert captured['parent_task_id'] == 'parent-1'


class TestARetryKeepsTheSongsItAlreadyAnalysed:
    @staticmethod
    def _child(task_id, status, tracks=None, album='Album A'):
        details = {}
        if tracks is not None:
            details = {'final_summary_details': {'tracks_analyzed': tracks}}
        return {
            'task_id': task_id, 'status': status,
            'sub_type_identifier': album, 'details': details,
        }

    def test_successes_are_carried_over_and_failures_are_dropped(self, monkeypatch):
        from tasks.analysis import main as analysis_main

        monkeypatch.setattr(
            analysis_main.taskqueue, 'reap_finished_children',
            lambda _pid: [
                self._child('a', config.TASK_STATUS_SUCCESS, tracks=12),
                self._child('b', config.TASK_STATUS_SUCCESS, tracks=7),
                self._child('c', config.TASK_STATUS_FAILURE),
            ],
        )

        assert analysis_main._carried_over_tracks('parent-1') == 19, (
            'the songs a previous attempt analysed are real work and stay in the '
            'recap; only its failures are dropped, or failed_count >= albums_launched '
            'reports a fully successful retry as a failed phase'
        )

    def test_a_first_run_has_nothing_to_carry(self, monkeypatch):
        from tasks.analysis import main as analysis_main

        monkeypatch.setattr(
            analysis_main.taskqueue, 'reap_finished_children', lambda _pid: []
        )

        assert analysis_main._carried_over_tracks('parent-1') == 0

    def test_an_unreadable_child_list_carries_nothing_instead_of_raising(
        self, monkeypatch
    ):
        from tasks.analysis import main as analysis_main

        def boom(_pid):
            raise RuntimeError('database went away')

        monkeypatch.setattr(analysis_main.taskqueue, 'reap_finished_children', boom)

        assert analysis_main._carried_over_tracks('parent-1') == 0

    def test_the_index_rebuild_child_carries_no_album_count(self, monkeypatch):
        from tasks.analysis import main as analysis_main

        rebuild = self._child('r', config.TASK_STATUS_SUCCESS, tracks=99)
        rebuild['sub_type_identifier'] = None
        monkeypatch.setattr(
            analysis_main.taskqueue, 'reap_finished_children', lambda _pid: [rebuild]
        )

        assert analysis_main._carried_over_tracks('parent-1') == 0, (
            'an index rebuild is a child of the same parent but is not an album'
        )


CHROMA_C_MAJOR = [1.0, 0.05, 0.5, 0.05, 0.8, 0.5, 0.05, 0.9, 0.05, 0.5, 0.05, 0.4]
CHROMA_A_MINOR = [0.8, 0.05, 0.4, 0.05, 0.9, 0.4, 0.05, 0.5, 0.05, 1.0, 0.05, 0.5]


def _chroma_frames(weights):
    return np.tile(np.array(weights, dtype=float).reshape(12, 1), (1, 50))


def _patched_chroma(weights):
    return patch(
        'tasks.analysis.song.librosa.feature.chroma_cqt',
        return_value=_chroma_frames(weights),
    )


class TestEstimateKeyScale:
    def test_tonic_weighted_white_key_chroma_returns_c_major(self):
        with _patched_chroma(CHROMA_C_MAJOR):
            assert _estimate_key_scale(np.ones(1000, dtype=np.float32), 16000) == ('C', 'major')

    def test_tonic_weighted_white_key_chroma_returns_a_minor(self):
        with _patched_chroma(CHROMA_A_MINOR):
            assert _estimate_key_scale(np.ones(1000, dtype=np.float32), 16000) == ('A', 'minor')

    def test_relative_major_and_minor_share_pitch_classes_but_are_distinguished(self):
        major_classes = sorted(np.nonzero(np.array(CHROMA_C_MAJOR) > 0.1)[0].tolist())
        minor_classes = sorted(np.nonzero(np.array(CHROMA_A_MINOR) > 0.1)[0].tolist())
        assert major_classes == minor_classes
        audio = np.ones(1000, dtype=np.float32)
        with _patched_chroma(CHROMA_C_MAJOR):
            major = _estimate_key_scale(audio, 16000)
        with _patched_chroma(CHROMA_A_MINOR):
            minor = _estimate_key_scale(audio, 16000)
        assert major == ('C', 'major')
        assert minor == ('A', 'minor')

    def test_major_scale_is_reachable_at_all(self):
        with _patched_chroma(CHROMA_C_MAJOR):
            _, scale = _estimate_key_scale(np.ones(1000, dtype=np.float32), 16000)
        assert scale == 'major'

    @pytest.mark.parametrize('shift,expected', [(0, 'C'), (3, 'D#'), (5, 'F'), (7, 'G')])
    def test_transposing_the_chroma_transposes_the_detected_key(self, shift, expected):
        with _patched_chroma(np.roll(CHROMA_C_MAJOR, shift)):
            key, scale = _estimate_key_scale(np.ones(1000, dtype=np.float32), 16000)
        assert (key, scale) == (expected, 'major')

    def test_all_zero_chroma_falls_back_to_c_major(self):
        with patch(
            'tasks.analysis.song.librosa.feature.chroma_cqt',
            return_value=np.zeros((12, 50)),
        ):
            assert _estimate_key_scale(np.ones(1000, dtype=np.float32), 16000) == ('C', 'major')

    def test_empty_audio_falls_back_to_c_major(self):
        assert _estimate_key_scale(np.array([], dtype=np.float32), 16000) == ('C', 'major')


class TestEstimateEnergy:
    def test_digital_silence_maps_to_zero(self):
        assert _estimate_energy(np.zeros(32000, dtype=np.float32)) == 0.0

    def test_near_full_scale_signal_maps_close_to_one(self):
        rng = np.random.default_rng(0)
        signal = (rng.standard_normal(32000) * 0.9).astype(np.float32)
        assert _estimate_energy(signal) > 0.95

    def test_energy_increases_monotonically_with_level(self):
        rng = np.random.default_rng(0)
        values = [
            _estimate_energy(
                (rng.standard_normal(32000) * 10 ** (db / 20.0)).astype(np.float32)
            )
            for db in (-40, -30, -20, -10)
        ]
        assert values == sorted(values)

    def test_silent_frames_are_not_lifted_by_a_loud_peak(self):
        rng = np.random.default_rng(0)
        loud_then_silent = np.concatenate([
            (rng.standard_normal(16000) * 10.0).astype(np.float32),
            np.zeros(16000, dtype=np.float32),
        ])
        assert _estimate_energy(loud_then_silent) < 0.6

    def test_typical_music_levels_land_inside_the_configured_range(self):
        rng = np.random.default_rng(0)
        for db in (-30, -20, -12, -8):
            value = _estimate_energy(
                (rng.standard_normal(32000) * 10 ** (db / 20.0)).astype(np.float32)
            )
            assert config.ENERGY_MIN <= value <= config.ENERGY_MAX

    def test_empty_audio_maps_to_zero(self):
        assert _estimate_energy(np.array([], dtype=np.float32)) == 0.0


def _patched_raw_tempo(raw):
    return patch(
        'tasks.analysis.song.librosa.beat.beat_track',
        return_value=(np.array([raw]), None),
    )


class TestEstimateTempo:
    @pytest.mark.parametrize('raw', [60.0, 110.0, 174.0, 180.0])
    def test_in_range_tempo_is_returned_unchanged(self, raw):
        with _patched_raw_tempo(raw):
            assert _estimate_tempo(np.ones(1000, dtype=np.float32), 16000) == raw

    @pytest.mark.parametrize('raw', [40.0, 200.0])
    def test_tempo_exactly_on_a_configured_bound_is_returned_unchanged(self, raw):
        with _patched_raw_tempo(raw):
            assert _estimate_tempo(np.ones(1000, dtype=np.float32), 16000) == raw

    @pytest.mark.parametrize('raw,expected', [(35.0, 70.0), (20.0, 40.0)])
    def test_tempo_below_the_configured_minimum_is_doubled_into_range(self, raw, expected):
        with _patched_raw_tempo(raw):
            assert _estimate_tempo(np.ones(1000, dtype=np.float32), 16000) == expected

    @pytest.mark.parametrize('raw,expected', [(260.0, 130.0), (410.0, 102.5)])
    def test_tempo_above_the_configured_maximum_is_halved_into_range(self, raw, expected):
        with _patched_raw_tempo(raw):
            assert _estimate_tempo(np.ones(1000, dtype=np.float32), 16000) == expected

    @pytest.mark.parametrize('raw', [12.0, 33.0, 95.0, 210.0, 333.0, 640.0])
    def test_folding_never_leaves_the_configured_range(self, raw):
        with _patched_raw_tempo(raw):
            value = _estimate_tempo(np.ones(1000, dtype=np.float32), 16000)
        assert config.TEMPO_MIN_BPM <= value <= config.TEMPO_MAX_BPM

    def test_non_positive_tempo_maps_to_zero(self):
        with _patched_raw_tempo(0.0):
            assert _estimate_tempo(np.ones(1000, dtype=np.float32), 16000) == 0.0

    def test_empty_audio_maps_to_zero(self):
        assert _estimate_tempo(np.array([], dtype=np.float32), 16000) == 0.0


class _StageCalls:
    def __init__(self):
        self.downloads = 0
        self.decodes = 0
        self.native_seen = {}

    def download(self, temp_dir, item):
        self.downloads += 1
        return f'/nonexistent/track-{self.downloads}.mp3'

    def decode(self, path):
        self.decodes += 1
        return ('NATIVE_AUDIO', 44100)


def _run_single_track(plan, calls, monkeypatch, overrides=None):
    from tasks.analysis import album as album_mod
    from tasks.analysis.helper import TrackPlan

    monkeypatch.setattr(album_mod, 'download_track', calls.download)
    monkeypatch.setattr(album_mod, 'decode_audio_once', calls.decode)
    monkeypatch.setattr(album_mod._ah, 'catalog_item_id', lambda item: 'track-1')
    monkeypatch.setattr(album_mod._ah, 'run_song_analyzed_hook', lambda *a, **k: None)
    monkeypatch.setattr(album_mod._ah, 'top_moods_from', lambda *a, **k: {'rock': 0.5})
    monkeypatch.setattr(album_mod, '_stage_collect_chromaprint', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, '_stage_persist_musicnn', lambda *a, **k: None)

    def fake_musicnn(path, name, plan_, *a, native_audio=None, native_sr=None, **k):
        calls.native_seen['musicnn'] = (native_audio, native_sr)
        track_audio, track_sr = ('MUSICNN_AUDIO', 16000) if plan_.lyrics else (None, None)
        return (None, {'duration_seconds': 1.0}, 'EMB', track_audio, track_sr)

    def fake_identity(item, plan_, name, emb, index, pending, track_duration=None):
        return index, plan_, 'track-1', True

    def fake_base(path, tid, name, native_audio=None, native_sr=None, precomputed=None):
        calls.native_seen['base'] = (native_audio, native_sr)
        return True

    def fake_clap(path, tid, name, labels, native_audio=None, native_sr=None):
        calls.native_seen['clap'] = (native_audio, native_sr)
        return 'CLAP_EMB', True

    def fake_lyrics(item, path, audio, sr, name, moods, ensure_download):
        calls.native_seen['lyrics_path'] = ensure_download()
        return True

    monkeypatch.setattr(album_mod, '_stage_musicnn', fake_musicnn)
    monkeypatch.setattr(album_mod, '_stage_identity', fake_identity)
    monkeypatch.setattr(album_mod, '_stage_base', fake_base)
    monkeypatch.setattr(album_mod, '_stage_clap', fake_clap)
    monkeypatch.setattr(album_mod, '_stage_lyrics', fake_lyrics)

    for _name, _fake in (overrides or {}).items():
        monkeypatch.setattr(album_mod, _name, _fake)

    assert isinstance(plan, TrackPlan)
    album_mod._analyze_single_track(
        {'Id': 'x', 'Name': 'x'}, plan, 'Artist - Track', 'album-1', 'Album',
        'parent-1', 3, {}, Mock(), None, None, None, {}, {},
    )
    return calls


class TestAudioIsFetchedOncePerTrack:
    def test_a_brand_new_song_downloads_once_and_decodes_once(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        calls = _run_single_track(
            TrackPlan(musicnn=True, clap=True, lyrics=True, base=False),
            _StageCalls(), monkeypatch,
        )
        assert calls.downloads == 1
        assert calls.decodes == 1

    def test_a_base_only_reanalysis_downloads_once_and_decodes_once(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        calls = _run_single_track(
            TrackPlan(musicnn=False, clap=False, lyrics=False, base=True),
            _StageCalls(), monkeypatch,
        )
        assert calls.downloads == 1
        assert calls.decodes == 1

    def test_base_plus_clap_reanalysis_downloads_once_and_decodes_once(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        calls = _run_single_track(
            TrackPlan(musicnn=False, clap=True, lyrics=False, base=True),
            _StageCalls(), monkeypatch,
        )
        assert calls.downloads == 1
        assert calls.decodes == 1

    def test_every_stage_receives_the_same_decoded_audio(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        calls = _run_single_track(
            TrackPlan(musicnn=True, clap=True, lyrics=False, base=True),
            _StageCalls(), monkeypatch,
        )
        assert calls.decodes == 1
        assert calls.native_seen['musicnn'] == ('NATIVE_AUDIO', 44100)
        assert calls.native_seen['clap'] == ('NATIVE_AUDIO', 44100)
        assert calls.native_seen['base'] == ('NATIVE_AUDIO', 44100)

    def test_a_lyrics_only_plan_still_downloads_exactly_once(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        calls = _run_single_track(
            TrackPlan(musicnn=False, clap=False, lyrics=True, base=False),
            _StageCalls(), monkeypatch,
        )
        assert calls.downloads == 1
        assert calls.decodes == 0


class _UndecodableCalls(_StageCalls):
    def decode(self, path):
        self.decodes += 1
        return (None, None)


class TestEverySingleStageRerunHandlesUndecodableAudio:
    PLANS = {
        'musicnn only': dict(musicnn=True, clap=False, lyrics=False, base=False),
        'base only': dict(musicnn=False, clap=False, lyrics=False, base=True),
        'clap only': dict(musicnn=False, clap=True, lyrics=False, base=False),
        'lyrics only': dict(musicnn=False, clap=False, lyrics=True, base=False),
    }

    def _plan(self, name):
        from tasks.analysis.helper import TrackPlan

        return TrackPlan(**self.PLANS[name])

    @pytest.mark.parametrize('plan_name', ['base only', 'clap only'])
    def test_an_undecodable_file_is_marked_cacheable_on_the_audio_only_stages(
        self, plan_name, monkeypatch
    ):
        from tasks.analysis import album as album_mod

        overrides = {
            '_stage_base': lambda *a, **k: False,
            '_stage_clap': lambda *a, **k: (None, False),
        }
        monkeypatch.setattr(
            album_mod, 'robust_load_audio_with_fallback', lambda *a, **k: (None, None)
        )
        with pytest.raises(album_mod.TrackNotAnalyzable) as excinfo:
            _run_single_track(
                self._plan(plan_name), _UndecodableCalls(), monkeypatch, overrides
            )

        assert excinfo.value.cacheable is True, f"{plan_name} would loop forever"

    def test_the_musicnn_stage_turns_an_undecodable_file_into_a_cacheable_mark(self, monkeypatch):
        from tasks.analysis import album as album_mod
        from tasks.analysis.helper import TrackPlan
        from tasks.analysis.song import AudioNotDecodableError

        def dead_analyze(*a, **k):
            raise AudioNotDecodableError('no decodable audio')

        monkeypatch.setattr(album_mod._ah, 'ensure_musicnn_sessions', lambda *a, **k: {})
        monkeypatch.setattr(album_mod, 'analyze_track', dead_analyze)
        with pytest.raises(album_mod.TrackNotAnalyzable) as excinfo:
            album_mod._stage_musicnn(
                '/nonexistent.mp3', 'Artist - Track',
                TrackPlan(musicnn=True, clap=False, lyrics=False, base=False),
                {}, Mock(), None, 'Album',
            )

        assert excinfo.value.cacheable is True

    def test_the_lyrics_stage_turns_an_undecodable_file_into_a_cacheable_mark(self, monkeypatch):
        from tasks.analysis import album as album_mod
        from tasks.analysis.song import AudioNotDecodableError

        def dead_lyrics(*a, **k):
            raise AudioNotDecodableError('no decodable audio for lyrics ASR')

        monkeypatch.setattr(album_mod._ah, 'run_lyrics_for_track', dead_lyrics)
        with pytest.raises(album_mod.TrackNotAnalyzable) as excinfo:
            album_mod._stage_lyrics(
                {'Id': 'x'}, None, None, None, 'Artist - Track', None, lambda: None
            )

        assert excinfo.value.cacheable is True

    def test_lyrics_that_simply_are_not_found_is_not_a_cacheable_mark(self, monkeypatch):
        from tasks.analysis import album as album_mod

        monkeypatch.setattr(album_mod._ah, 'run_lyrics_for_track', lambda *a, **k: False)
        assert album_mod._stage_lyrics(
            {'Id': 'x'}, None, None, None, 'Artist - Track', None, lambda: None
        ) is False

    @pytest.mark.parametrize('plan_name', list(PLANS))
    def test_a_single_stage_rerun_reads_the_file_at_most_once(self, plan_name, monkeypatch):
        from tasks.analysis import album as album_mod

        reloads = []
        monkeypatch.setattr(
            album_mod, 'robust_load_audio_with_fallback',
            lambda *a, **k: reloads.append(a[0] if a else None) or (
                np.ones(160, dtype=np.float32), 16000
            ),
        )
        calls = _run_single_track(self._plan(plan_name), _StageCalls(), monkeypatch)

        assert calls.decodes + len(reloads) <= 1, (
            f"{plan_name} read the audio {calls.decodes + len(reloads)} times"
        )


class TestAudioIsReadFromDiskOnlyOnce:
    def test_a_failed_decode_reaches_no_stage_at_all(self, monkeypatch):
        from tasks.analysis import album as album_mod
        from tasks.analysis.helper import TrackPlan

        def boom(name):
            def _fail(*a, **k):
                pytest.fail(f"{name} ran on undecodable audio")
            return _fail

        calls = _UndecodableCalls()
        with pytest.raises(album_mod.TrackNotAnalyzable) as excinfo:
            _run_single_track(
                TrackPlan(musicnn=True, clap=True, lyrics=True, base=False),
                calls, monkeypatch,
                {
                    '_stage_musicnn': boom('musicnn'),
                    '_stage_base': boom('base'),
                    '_stage_clap': boom('clap'),
                    '_stage_lyrics': boom('lyrics'),
                },
            )

        assert calls.decodes == 1
        assert excinfo.value.cacheable is True

    @pytest.mark.parametrize('plan_kwargs', [
        dict(musicnn=False, clap=False, lyrics=False, base=True),
        dict(musicnn=False, clap=True, lyrics=False, base=False),
        dict(musicnn=False, clap=True, lyrics=True, base=True),
    ])
    def test_no_stage_runs_after_a_failed_decode_whatever_the_plan(self, plan_kwargs, monkeypatch):
        from tasks.analysis import album as album_mod
        from tasks.analysis.helper import TrackPlan

        def boom(*a, **k):
            pytest.fail('a stage ran on undecodable audio')

        calls = _UndecodableCalls()
        with pytest.raises(album_mod.TrackNotAnalyzable):
            _run_single_track(
                TrackPlan(**plan_kwargs), calls, monkeypatch,
                {'_stage_base': boom, '_stage_clap': boom, '_stage_lyrics': boom},
            )
        assert calls.decodes == 1

    def test_lyrics_reuses_the_single_decode_instead_of_reading_the_file_again(self, monkeypatch):
        from tasks.analysis import album as album_mod
        from tasks.analysis.helper import TrackPlan

        monkeypatch.setattr(album_mod, 'resample_audio', lambda a, o, t: 'RESAMPLED')
        seen = {}

        def fake_lyrics(item, path, audio, sr, name, moods, ensure_download):
            seen['audio'] = audio
            seen['sr'] = sr
            return True

        calls = _run_single_track(
            TrackPlan(musicnn=False, clap=False, lyrics=True, base=True),
            _StageCalls(), monkeypatch, {'_stage_lyrics': fake_lyrics},
        )

        assert calls.decodes == 1
        assert seen['audio'] == 'RESAMPLED'
        assert seen['sr'] == 16000

    def test_the_native_buffer_is_released_before_the_lyrics_stage_runs(self, monkeypatch):
        from tasks.analysis import album as album_mod
        from tasks.analysis.helper import TrackPlan

        monkeypatch.setattr(album_mod, 'resample_audio', lambda a, o, t: 'RESAMPLED')
        seen = {}

        def fake_lyrics(item, path, audio, sr, name, moods, ensure_download):
            frame = sys._getframe(1)
            seen['native_audio'] = frame.f_locals.get('native_audio')
            return True

        _run_single_track(
            TrackPlan(musicnn=False, clap=False, lyrics=True, base=True),
            _StageCalls(), monkeypatch, {'_stage_lyrics': fake_lyrics},
        )

        assert seen['native_audio'] is None


class TestUndecodableAudioOnAReanalysisIsMarkedNotAnalyzable:
    _BARREN = {
        '_stage_base': lambda *a, **k: False,
        '_stage_clap': lambda *a, **k: (None, False),
        '_stage_lyrics': lambda *a, **k: False,
    }

    def _run(self, plan, calls, monkeypatch, overrides=None):
        from tasks.analysis import album as album_mod

        with pytest.raises(album_mod.TrackNotAnalyzable) as excinfo:
            _run_single_track(plan, calls, monkeypatch, overrides or self._BARREN)
        return excinfo.value

    def test_a_base_only_reanalysis_that_cannot_decode_is_cacheable(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        error = self._run(
            TrackPlan(musicnn=False, clap=False, lyrics=False, base=True),
            _UndecodableCalls(), monkeypatch,
        )
        assert error.cacheable is True

    def test_a_clap_only_reanalysis_that_cannot_decode_is_cacheable(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        error = self._run(
            TrackPlan(musicnn=False, clap=True, lyrics=False, base=False),
            _UndecodableCalls(), monkeypatch,
        )
        assert error.cacheable is True

    def test_a_barren_stage_on_decodable_audio_stays_uncacheable(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        error = self._run(
            TrackPlan(musicnn=False, clap=False, lyrics=False, base=True),
            _StageCalls(), monkeypatch,
        )
        assert error.cacheable is False

class _BaseCalls:
    def __init__(self):
        self.downloads = 0
        self.decodes = 0
        self.musicnn_runs = 0
        self.feature_computations = 0
        self.written = []

    def download(self, temp_dir, item):
        self.downloads += 1
        return f'/nonexistent/track-{self.downloads}.mp3'

    def decode(self, path):
        self.decodes += 1
        return ('NATIVE_AUDIO', 44100)


def _run_track(plan, calls, monkeypatch, catalogued_elsewhere=False):
    from tasks.analysis import album as album_mod

    monkeypatch.setattr(album_mod, 'download_track', calls.download)
    monkeypatch.setattr(album_mod, 'decode_audio_once', calls.decode)
    monkeypatch.setattr(album_mod._ah, 'catalog_item_id', lambda item: 'track-1')
    monkeypatch.setattr(album_mod._ah, 'run_song_analyzed_hook', lambda *a, **k: None)
    monkeypatch.setattr(album_mod._ah, 'top_moods_from', lambda *a, **k: {'rock': 0.5})
    monkeypatch.setattr(album_mod, '_stage_collect_chromaprint', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, '_stage_persist_musicnn', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, '_stage_clap', lambda *a, **k: (None, True))
    monkeypatch.setattr(album_mod, '_stage_lyrics', lambda *a, **k: True)

    analysis = {
        'tempo': 128.0, 'energy': 0.75, 'key': 'C', 'scale': 'major',
        'duration_seconds': 200.0,
    }

    def fake_musicnn(path, name, plan_, *a, native_audio=None, native_sr=None, **k):
        calls.musicnn_runs += 1
        calls.feature_computations += 1
        return (None, dict(analysis), 'EMB', None, None)

    def fake_identity(item, plan_, name, emb, index, pending, track_duration=None):
        if catalogued_elsewhere:
            return index, plan_._replace(musicnn=False, base=True), 'canonical-9', False
        return index, plan_, 'track-1', True

    def fake_extract(audio, sr):
        calls.feature_computations += 1
        return (128.0, 0.75, 'C', 'major')

    def fake_refresh(tid, tempo, energy, key, scale):
        calls.written.append((tid, tempo, energy, key, scale))
        return True

    monkeypatch.setattr(album_mod, '_stage_musicnn', fake_musicnn)
    monkeypatch.setattr(album_mod, '_stage_identity', fake_identity)
    monkeypatch.setattr(album_mod, 'extract_basic_features', fake_extract)
    monkeypatch.setattr(album_mod, 'resample_audio', lambda a, o, t: np.ones(16000, dtype=np.float32))
    monkeypatch.setattr(album_mod._ah, 'refresh_base_features', fake_refresh)

    album_mod._analyze_single_track(
        {'Id': 'x', 'Name': 'x'}, plan, 'Artist - Track', 'album-1', 'Album',
        'parent-1', 3, {}, Mock(), None, None, None, {}, {},
    )
    return calls


class TestBaseFeaturesAreComputedExactlyOnce:
    def test_an_already_analyzed_track_never_runs_musicnn_for_a_base_refresh(
        self, monkeypatch
    ):
        from tasks.analysis.helper import TrackPlan

        calls = _run_track(
            TrackPlan(musicnn=False, clap=False, lyrics=False, base=True),
            _BaseCalls(), monkeypatch,
        )
        assert calls.musicnn_runs == 0
        assert calls.feature_computations == 1
        assert calls.downloads == 1
        assert calls.decodes == 1
        assert calls.written == [('track-1', 128.0, 0.75, 'C', 'major')]

    def test_a_brand_new_track_computes_base_features_only_once(self, monkeypatch):
        from tasks.analysis.helper import TrackPlan

        calls = _run_track(
            TrackPlan(musicnn=True, clap=True, lyrics=True, base=False),
            _BaseCalls(), monkeypatch,
        )
        assert calls.musicnn_runs == 1
        assert calls.feature_computations == 1
        assert calls.downloads == 1
        assert calls.decodes == 1

    def test_a_duplicate_resolved_to_a_catalogue_row_reuses_the_features_already_computed(
        self, monkeypatch
    ):
        from tasks.analysis.helper import TrackPlan

        calls = _run_track(
            TrackPlan(musicnn=True, clap=False, lyrics=False, base=False),
            _BaseCalls(), monkeypatch, catalogued_elsewhere=True,
        )
        assert calls.musicnn_runs == 1
        assert calls.feature_computations == 1
        assert calls.decodes == 1
        assert calls.written == [('canonical-9', 128.0, 0.75, 'C', 'major')]

    def test_a_base_refresh_falls_back_to_computing_when_nothing_was_precomputed(self):
        from tasks.analysis.album import _base_features_from

        assert _base_features_from(None) is None
        assert _base_features_from({}) is None
        assert _base_features_from({'tempo': 1.0, 'energy': 0.5, 'key': 'C'}) is None
        assert _base_features_from(
            {'tempo': 1.0, 'energy': 0.5, 'key': 'C', 'scale': 'major'}
        ) == (1.0, 0.5, 'C', 'major')


def test_index_builds_recycle_the_db_connection_between_steps(monkeypatch):
    """The REAL _run_all_index_builds must close g.db after every step.

    A Postgres backend never hands memory back to the OS while its session
    lives, so one connection shared across all nine builds accumulates the
    union of their peaks.
    """
    import tasks.analysis.index as index

    order = []

    def stub(label):
        def _build(*args, **kwargs):
            order.append(f"build:{label}")
        return _build

    monkeypatch.setattr("tasks.ivf_manager.build_and_store_ivf_index", stub("ivf"))
    monkeypatch.setattr("tasks.clap_text_search.build_and_store_clap_index", stub("clap"))
    monkeypatch.setattr("tasks.lyrics_manager.build_and_store_lyrics_index", stub("lyrics"))
    monkeypatch.setattr(
        "tasks.lyrics_manager.build_and_store_lyrics_axes_index", stub("lyrics_axes")
    )
    monkeypatch.setattr(
        "tasks.sem_grove_manager.build_and_store_sem_grove_index", stub("semgrove")
    )
    monkeypatch.setattr(
        "tasks.artist_gmm_manager.build_and_store_artist_index", stub("artist")
    )
    monkeypatch.setattr(index, "build_and_store_map_projection", stub("map"))
    monkeypatch.setattr(index, "build_and_store_artist_projection", stub("artist_map"))
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.backfill_hyperbolic_columns", stub("hyper_backfill")
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.build_hyperbolic_tree_cache", stub("hyper_tree")
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_index.build_and_store_hyperbolic_index", stub("hyper_index")
    )

    monkeypatch.setattr(index, "get_db", lambda: object())
    monkeypatch.setattr(index, "close_db", lambda: order.append("close_db"))
    monkeypatch.setattr(index, "_checkpoint_postgres", lambda: None)
    monkeypatch.setattr(index.taskqueue, "publish_event", lambda *a, **k: None)
    monkeypatch.setattr(index, "release_memory_to_os", lambda: None)

    index._run_all_index_builds()

    builds = [entry for entry in order if entry.startswith("build:")]
    closes = [entry for entry in order if entry == "close_db"]
    assert builds, "no build step ran"
    # one recycle per STEP (the hyperbolic step runs three builds itself)
    assert len(closes) == 9
    # and every step is followed by a recycle, never two builds back to back
    assert order[-1] == "close_db"


def test_a_failing_index_build_still_recycles_the_connection(monkeypatch):
    import tasks.analysis.index as index

    closed = []
    monkeypatch.setattr(index, "close_db", lambda: closed.append(1))

    try:
        try:
            raise RuntimeError("build blew up")
        finally:
            index._recycle_db_connection()
    except RuntimeError:
        pass

    assert closed == [1]


def test_connection_recycle_never_propagates_a_close_failure(monkeypatch):
    import tasks.analysis.index as index

    def boom():
        raise RuntimeError("connection already gone")

    monkeypatch.setattr(index, "close_db", boom)

    index._recycle_db_connection()


def test_checkpoint_postgres_runs_checkpoint_commit_and_close(monkeypatch):
    """The post-build CHECKPOINT runs on its own connection and closes it."""
    import tasks.analysis.index as index

    seen = []

    class FakeCursor:
        def execute(self, sql):
            seen.append(sql)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            seen.append("commit")

    monkeypatch.setattr(index, "get_db", lambda: FakeConn())
    monkeypatch.setattr(index, "close_db", lambda: seen.append("close"))

    index._checkpoint_postgres()

    assert seen == ["CHECKPOINT", "commit", "close"]


def test_checkpoint_postgres_swallows_a_database_failure(monkeypatch):
    import tasks.analysis.index as index

    def boom():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(index, "get_db", boom)

    index._checkpoint_postgres()


def test_index_builds_end_with_a_database_checkpoint(monkeypatch):
    """The full rebuild issues CHECKPOINT after the last step's recycle."""
    import tasks.analysis.index as index

    order = []

    def stub(label):
        def _build(*args, **kwargs):
            order.append(f"build:{label}")
        return _build

    monkeypatch.setattr("tasks.ivf_manager.build_and_store_ivf_index", stub("ivf"))
    monkeypatch.setattr("tasks.clap_text_search.build_and_store_clap_index", stub("clap"))
    monkeypatch.setattr("tasks.lyrics_manager.build_and_store_lyrics_index", stub("lyrics"))
    monkeypatch.setattr(
        "tasks.lyrics_manager.build_and_store_lyrics_axes_index", stub("lyrics_axes")
    )
    monkeypatch.setattr(
        "tasks.sem_grove_manager.build_and_store_sem_grove_index", stub("semgrove")
    )
    monkeypatch.setattr(
        "tasks.artist_gmm_manager.build_and_store_artist_index", stub("artist")
    )
    monkeypatch.setattr(index, "build_and_store_map_projection", stub("map"))
    monkeypatch.setattr(index, "build_and_store_artist_projection", stub("artist_map"))
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.backfill_hyperbolic_columns", stub("hyper_backfill")
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.build_hyperbolic_tree_cache", stub("hyper_tree")
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_index.build_and_store_hyperbolic_index", stub("hyper_index")
    )
    monkeypatch.setattr(index, "get_db", lambda: object())
    monkeypatch.setattr(index, "close_db", lambda: order.append("close_db"))
    monkeypatch.setattr(index, "_checkpoint_postgres", lambda: order.append("checkpoint"))
    monkeypatch.setattr(index.taskqueue, "publish_event", lambda *a, **k: None)
    monkeypatch.setattr(index, "release_memory_to_os", lambda: None)

    index._run_all_index_builds()

    assert order[-1] == "checkpoint"
    # 8 single builds + the hyperbolic step's three internal builds
    assert sum(1 for e in order if e.startswith("build:")) == 11
    assert order.count("close_db") == 9
