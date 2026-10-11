# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the P2P hooks of the per-album analysis job.

Every stage and every database write of tasks.analysis.album is replaced by a
recorder, and a fake P2P session scripts what the peers supply, so the tests
read the exact call sequence of a track with sharing off, with the node down,
and with peers supplying.

Main Features:
* sharing off: the call sequence is the one of a plain analysis and the album
  job never opens a session
* the session opens for an album of brand-new songs: a track with no rows yet
  is in every missing set of the real album plan
* node not live or skipping: a 7002 warning for every track, local compute
* a new song, a catalogued backfill and a song matched to an existing row each
  ask once; every supplied output is saved through the same save function with
  the peer's version and id as its origin, other_features is refreshed and
  those stages are skipped; a fully supplied backfill downloads nothing and
  still counts; a supply is always taken, never recomputed to compare
* fp_0 / legacy ids and an unknown duration never ask; a Stop raised while
  waiting for peers stops the album instead of failing the track
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from psycopg2 import InterfaceError, OperationalError

import database
import taskqueue
from p2p import client as p2p_client
from p2p import ipc
from tasks import clap_analyzer, simhash
from tasks import neural_fingerprint as nfp
from tasks.analysis import album as album_mod
from tasks.analysis import helper as helper_mod
from tasks.analysis.helper import TrackPlan

PEER = 'peeralphatestonly' + 'a' * 39
SIG_ID = simhash.canonical_id_str(0xABCDEF)
OTHER_SIG_ID = simhash.canonical_id_str(0x123456)
UNSIGNABLE_ID = simhash.unsignable_canonical_id('srv-1', 'prov-1')
LABELS = {'label': 'embedding'}
VERSIONS = {'m': 'v3.6.0', 'c': 'v3.6.1', 'l': 'v3.5.2', 'n': 'v3.6.0'}
ALL = TrackPlan(True, True, True, False, True)
BACKFILL = TrackPlan(False, True, True, False, True)


def _unit(dim, seed):
    vec = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


PEER_CLAP = _unit(512, 1)
PEER_LYRICS = _unit(768, 2)
PEER_AXIS = np.full(27, 0.2, dtype=np.float32)
PEER_BLOB = b'NFP2-peer-blob'


def _supply(keys=('c', 'l', 'n')):
    outputs = {}
    if 'c' in keys:
        outputs['c'] = PEER_CLAP
    if 'l' in keys:
        outputs['l'] = PEER_LYRICS
        outputs['a'] = PEER_AXIS
    if 'n' in keys:
        outputs['n'] = PEER_BLOB
    versions = {key: VERSIONS[key] for key in ('m',) + tuple(keys)}
    return p2p_client.Supply(PEER, versions, outputs)


def _origin(key):
    return (VERSIONS[key], PEER)


class FakeSession:
    def __init__(self, supply=None, live=True, skipping=False):
        self.supply = supply
        self.live = live
        self.skipping = skipping
        self.reason = 'the P2P node is not running'
        self.asks = []
        self.closed = False

    def ask(self, item_id, server_id, provider_id, wants):
        self.asks.append({'item': item_id, 'server': server_id, 'provider': provider_id, 'wants': list(wants)})
        return self.supply

    def close(self):
        self.closed = True


class Rig:
    def __init__(self):
        self.events = []
        self.row_duration = 200.0
        self.musicnn_duration = 200.0
        self.identity = None
        self.clap_result = (_unit(512, 1), True)
        self.neural_result = True
        self.lyrics_result = True
        self.signature_checks = 0
        self.save_clap_error = None
        self.save_neural_ok = True

    def names(self):
        return [event[0] for event in self.events]

    def find(self, name):
        return [event for event in self.events if event[0] == name]


@pytest.fixture
def rig(monkeypatch):
    rig = Rig()
    record = rig.events.append

    def download(temp_dir, item):
        record(('download',))
        return '/nonexistent/track.flac'

    def decode(path):
        record(('decode',))
        return np.ones(16000, dtype=np.float32), 16000

    def musicnn(path, name, plan, *args, **kwargs):
        record(('musicnn',))
        analysis = {'tempo': 120.0, 'energy': 0.5, 'key': 'C', 'scale': 'major',
                    'duration_seconds': rig.musicnn_duration}
        return None, analysis, 'EMB', None, None

    def identity(item, plan, name, embedding, index, pending, track_duration=None):
        record(('identity', track_duration))
        if rig.identity is None:
            return index, plan, SIG_ID, True
        new_plan, item_id, keep = rig.identity
        return index, new_plan, item_id, keep

    def is_signature_id(item_id, _real=simhash.is_signature_id):
        rig.signature_checks += 1
        return _real(item_id)

    def fetch_row_duration(item_id):
        record(('row_duration', item_id))
        return rig.row_duration

    def save_clap(item_id, embedding, origin=None):
        record(('save_clap', item_id, embedding, origin))
        if rig.save_clap_error is not None:
            raise rig.save_clap_error

    def save_lyrics(item_id, embedding, axis=None, origin=None):
        record(('save_lyrics', item_id, embedding, axis, origin))

    def save_neural(item_id, blob, origin=None):
        record(('save_neural', item_id, blob, origin))
        return rig.save_neural_ok

    def other_features(embedding, labels, names):
        record(('other_features_computed', embedding, labels))
        return 'label:0.50'

    monkeypatch.setattr(album_mod, 'download_track', download)
    monkeypatch.setattr(album_mod, 'decode_audio_once', decode)
    monkeypatch.setattr(album_mod, '_stage_collect_chromaprint', lambda *a, **k: record(('chromaprint',)))
    monkeypatch.setattr(album_mod, '_stage_musicnn', musicnn)
    monkeypatch.setattr(album_mod, '_stage_identity', identity)
    monkeypatch.setattr(album_mod, '_stage_persist_musicnn', lambda *a, **k: record(('persist',)))
    monkeypatch.setattr(album_mod, '_stage_base', lambda *a, **k: record(('base',)) or True)
    monkeypatch.setattr(album_mod, '_stage_clap', lambda *a, **k: record(('clap',)) or rig.clap_result)
    monkeypatch.setattr(
        album_mod, '_stage_neural_fingerprint', lambda *a, **k: record(('neural',)) or rig.neural_result
    )
    monkeypatch.setattr(album_mod, '_stage_lyrics', lambda *a, **k: record(('lyrics',)) or rig.lyrics_result)
    monkeypatch.setattr(album_mod._ah, 'catalog_item_id', lambda item: item.get('_catalog_item_id', SIG_ID))
    monkeypatch.setattr(album_mod._ah, 'provider_item_id', lambda item: 'prov-1')
    monkeypatch.setattr(album_mod._ah, 'top_moods_from', lambda *a, **k: {'rock': 0.5})
    monkeypatch.setattr(
        album_mod._ah, 'run_song_analyzed_hook',
        lambda item, path, analysis, embedding, clap, *a: record(('hook', path, clap)),
    )
    monkeypatch.setattr(album_mod._ah, '_fetch_row_duration', fetch_row_duration)
    monkeypatch.setattr(album_mod._ah, 'analysis_server_identity', lambda: ('srv-1', 'Server'))
    monkeypatch.setattr(album_mod._ah, 'compute_other_features_str', other_features)
    monkeypatch.setattr(
        album_mod._ah, 'refresh_other_features',
        lambda item_id, value: record(('other_features', item_id, value)) or True,
    )
    monkeypatch.setattr(album_mod, 'save_clap_embedding', save_clap)
    monkeypatch.setattr(album_mod, 'save_lyrics_embedding', save_lyrics)
    monkeypatch.setattr(album_mod, 'save_neural_fingerprint', save_neural)
    monkeypatch.setattr(album_mod.simhash, 'is_signature_id', is_signature_id)
    return rig


_OMIT = object()


def _run_track(plan, p2p=_OMIT, item_id=SIG_ID):
    item = {'Id': 'prov-1', 'Name': 'Track', '_catalog_item_id': item_id}
    args = (
        item, plan, "'Track' by Artist", 'album-1', 'Album', 'parent-1', 3, {}, Mock(),
        None, None, LABELS, {}, {},
    )
    if p2p is _OMIT:
        return album_mod._analyze_single_track(*args)
    return album_mod._analyze_single_track(*args, p2p=p2p)


def _codes(caplog, code):
    return [r for r in caplog.records if f'[{code}]' in r.getMessage()]


PLAIN_NEW_SONG = ['download', 'chromaprint', 'decode', 'musicnn', 'identity', 'persist',
                  'clap', 'neural', 'lyrics', 'hook']
PLAIN_BACKFILL = ['download', 'decode', 'clap', 'neural', 'lyrics', 'hook']
SUPPLIED_SAVES = ['save_clap', 'other_features_computed', 'other_features', 'save_lyrics', 'save_neural']


class TestSharingOff:
    @pytest.mark.parametrize('plan, expected', [(ALL, PLAIN_NEW_SONG), (BACKFILL, PLAIN_BACKFILL)])
    def test_the_call_sequence_is_the_plain_analysis(self, rig, plan, expected):
        _run_track(plan)
        assert rig.names() == expected
        rig.events.clear()
        _run_track(plan, p2p=None)
        assert rig.names() == expected
        assert rig.signature_checks == 0

    def test_the_album_job_never_opens_a_session(self, monkeypatch, tmp_path):
        from test.unit.test_analysis import _run_album_impl

        def forbidden(cancel):
            raise AssertionError('sharing is off: no P2P session may be opened')

        monkeypatch.setattr(album_mod, 'P2P_ENABLED', False)
        monkeypatch.setattr(p2p_client.Session, 'open', forbidden)
        item = {'Id': 'prov1', 'Name': 'Song', 'AlbumArtist': 'Artist'}
        persisted, maps = [], []
        result = _run_album_impl(
            monkeypatch, tmp_path, item, simhash.CatalogResolver(), persisted, maps,
            clap_broken=True,
        )
        assert result['status'] == 'SUCCESS'
        assert result['tracks_analyzed'] == 1
        assert len(persisted) == 1


def _album_harness(monkeypatch, tracks, session, missing, existing=()):
    reporter = SimpleNamespace(state={'progress': 0})
    opened = []

    def report(*args, **kwargs):
        return None

    report.state = reporter.state

    def open_session(cancel):
        opened.append(cancel)
        return session

    monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)
    monkeypatch.setattr(album_mod, 'make_cancel_check', lambda *a, **k: (lambda force=False: None, lambda: None))
    monkeypatch.setattr(album_mod, 'make_task_reporter', lambda *a, **k: report)
    monkeypatch.setattr(album_mod, 'get_tracks_from_album', lambda album_id: tracks)
    ids = {track['_catalog_item_id'] for track in tracks}
    monkeypatch.setattr(
        album_mod._ah, 'build_album_plan',
        lambda *a, **k: (set(existing), set(ids) if 'c' in missing else set(),
                         set(ids) if 'l' in missing else set(), set(), LABELS, {},
                         set(ids) if 'n' in missing else set()),
    )
    monkeypatch.setattr(album_mod._ah, 'load_album_analysis_exclusions', lambda tracks: {})
    monkeypatch.setattr(album_mod._ah, 'upsert_artist_mappings_for_tracks', lambda *a, **k: None)
    monkeypatch.setattr(album_mod._ah, 'flush_pending_track_maps', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, 'cleanup_musicnn_sessions', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, 'cleanup_optional_models', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, 'comprehensive_memory_cleanup', lambda *a, **k: None)
    monkeypatch.setattr(album_mod, 'LYRICS_ENABLED', True)
    monkeypatch.setattr(album_mod, 'P2P_ENABLED', True)
    monkeypatch.setattr(p2p_client.Session, 'open', open_session)
    result = album_mod._analyze_album_task_impl('album-1', 'Album', 3, 'parent-1')
    return result, opened


def _tracks(*item_ids):
    return [
        {'Id': f'prov-{n}', 'Name': f'Track {n}', 'AlbumArtist': 'Artist', '_catalog_item_id': item_id}
        for n, item_id in enumerate(item_ids, 1)
    ]


def _empty_catalogue_db():
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value = cursor
    return conn, cursor


class TestSessionOpens:
    def test_a_track_with_no_rows_yet_is_in_every_missing_set(self, monkeypatch):
        conn, cursor = _empty_catalogue_db()
        monkeypatch.setattr(helper_mod, 'get_db', lambda: conn)
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(helper_mod, 'attach_catalog_item_ids', lambda tracks, server_id=None: tracks)
        monkeypatch.setattr(clap_analyzer, 'is_clap_available', lambda: True)
        monkeypatch.setattr(clap_analyzer, 'get_or_cache_other_feature_text_embeddings', lambda: LABELS)
        monkeypatch.setattr(nfp, 'is_available', lambda: True)
        tracks = [{'Id': 'prov-new', 'Name': 'Track', '_catalog_item_id': 'prov-new'}]

        existing, clap, lyrics, _, labels, _, neural = helper_mod.build_album_plan('Album', tracks, 3, True)

        assert existing == set()
        assert clap == lyrics == neural == {'prov-new'}
        assert labels is LABELS
        assert cursor.execute.call_count >= 4

    def test_an_album_of_new_songs_opens_the_session_and_asks_after_musicnn(self, rig, monkeypatch):
        session = FakeSession(supply=None)
        result, opened = _album_harness(monkeypatch, _tracks('prov-new'), session, missing='cln')
        assert len(opened) == 1
        assert result['tracks_analyzed'] == 1
        assert session.asks == [{'item': SIG_ID, 'server': 'srv-1', 'provider': 'prov-1', 'wants': ['c', 'l', 'n']}]
        assert rig.names().index('persist') < rig.names().index('clap')
        assert session.closed


class TestNodeNotLive:
    def test_every_track_logs_7002_and_is_computed_locally(self, rig, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        session = p2p_client.Session(lambda: None, state=ipc.STATE_NOT_RUNNING)
        tracks = _tracks(SIG_ID, OTHER_SIG_ID)
        result, opened = _album_harness(
            monkeypatch, tracks, session, missing='cln', existing={SIG_ID, OTHER_SIG_ID}
        )
        assert len(opened) == 1
        assert result['tracks_analyzed'] == 2
        warnings = [r for r in _codes(caplog, 7002) if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert all('Not running' in r.getMessage() for r in warnings)
        assert rig.names().count('clap') == 2
        assert rig.names().count('lyrics') == 2
        assert not set(SUPPLIED_SAVES) & set(rig.names())

    def test_a_skipping_session_logs_7002_per_track_and_never_asks(self, rig, caplog):
        caplog.set_level(logging.WARNING)
        session = FakeSession(supply=_supply(), skipping=True)
        _run_track(BACKFILL, p2p=session)
        assert session.asks == []
        assert len(_codes(caplog, 7002)) == 1
        assert rig.names() == ['row_duration'] + PLAIN_BACKFILL

    def test_an_album_without_missing_peer_outputs_opens_no_session(self, rig, monkeypatch):
        session = FakeSession()
        result, opened = _album_harness(
            monkeypatch, _tracks(SIG_ID), session, missing='', existing={SIG_ID}
        )
        assert opened == []
        assert result['tracks_skipped'] == 1

    def test_the_session_is_closed_when_the_album_ends(self, rig, monkeypatch):
        session = FakeSession(supply=None)
        _album_harness(monkeypatch, _tracks(SIG_ID), session, missing='c', existing={SIG_ID})
        assert session.closed


class TestCancelWhileWaiting:
    def test_a_stop_during_the_peer_wait_stops_the_album_instead_of_failing_the_track(
        self, rig, monkeypatch
    ):
        class CancelledSession(FakeSession):
            def ask(self, *args, **kwargs):
                raise taskqueue.TaskCancelled('stopped while waiting for peers')

        session = CancelledSession()
        with pytest.raises(taskqueue.TaskCancelled):
            _album_harness(
                monkeypatch, _tracks(SIG_ID, OTHER_SIG_ID), session, missing='c',
                existing={SIG_ID, OTHER_SIG_ID},
            )
        assert 'clap' not in rig.names()
        assert session.closed


class TestSupplied:
    def test_a_new_song_fully_supplied_skips_the_slow_stages(self, rig):
        session = FakeSession(supply=_supply())
        _run_track(ALL, p2p=session)

        assert rig.names() == [
            'download', 'chromaprint', 'decode', 'musicnn', 'identity', 'persist',
        ] + SUPPLIED_SAVES + ['hook']
        assert session.asks == [{'item': SIG_ID, 'server': 'srv-1', 'provider': 'prov-1', 'wants': ['c', 'l', 'n']}]
        _, embedding, labels = rig.find('other_features_computed')[0]
        assert embedding is PEER_CLAP and labels is LABELS
        assert rig.find('other_features')[0][1:] == (SIG_ID, 'label:0.50')
        assert rig.find('save_clap')[0][1:] == (SIG_ID, PEER_CLAP, _origin('c'))
        _, item_id, emb, axis, origin = rig.find('save_lyrics')[0]
        assert item_id == SIG_ID and emb is PEER_LYRICS and axis is PEER_AXIS
        assert origin == _origin('l')
        assert rig.find('save_neural')[0][1:] == (SIG_ID, PEER_BLOB, _origin('n'))
        assert rig.find('hook')[0][2] is PEER_CLAP

    def test_a_backfill_fully_supplied_downloads_nothing_and_counts_as_analyzed(self, rig, monkeypatch):
        session = FakeSession(supply=_supply())
        result, _ = _album_harness(
            monkeypatch, _tracks(SIG_ID), session, missing='cln', existing={SIG_ID}
        )
        assert result['tracks_analyzed'] == 1
        assert result['tracks_not_analyzable'] == 0
        assert 'download' not in rig.names() and 'decode' not in rig.names()
        assert not {'clap', 'neural', 'lyrics'} & set(rig.names())
        assert rig.find('row_duration') == [('row_duration', SIG_ID)]
        assert session.asks[0]['item'] == SIG_ID
        hook = rig.find('hook')[0]
        assert hook[1] is None and hook[2] is PEER_CLAP
        assert [event[-1] for event in rig.find('save_clap') + rig.find('save_neural')] == [
            _origin('c'), _origin('n'),
        ]

    def test_every_supply_is_taken_as_it_is_never_recomputed_to_compare(self, rig):
        for _ in range(200):
            _run_track(BACKFILL, p2p=FakeSession(supply=_supply()))
        assert not {'download', 'clap', 'neural', 'lyrics'} & set(rig.names())
        assert len(rig.find('save_clap')) == len(rig.find('save_neural')) == 200

    def test_a_partial_supply_runs_only_the_missing_stages(self, rig):
        session = FakeSession(supply=_supply(keys=('c',)))
        _run_track(BACKFILL, p2p=session)
        assert 'clap' not in rig.names()
        assert 'neural' in rig.names() and 'lyrics' in rig.names() and 'download' in rig.names()
        assert [event[-1] for event in rig.find('save_clap')] == [_origin('c')]
        assert rig.find('save_neural') == [] and rig.find('save_lyrics') == []

    def test_a_peer_value_that_cannot_be_saved_is_computed_locally(self, rig, caplog):
        caplog.set_level(logging.ERROR)
        rig.save_clap_error = RuntimeError('clap_embedding write refused')
        session = FakeSession(supply=_supply(keys=('c', 'l')))
        _run_track(BACKFILL, p2p=session)
        assert 'clap' in rig.names() and 'neural' in rig.names()
        assert 'lyrics' not in rig.names()
        assert rig.find('other_features') == []
        assert [r for r in caplog.records if 'Could not save the CLAP supplied by peer' in r.getMessage()]

    def test_a_neural_blob_the_row_cannot_take_is_computed_locally(self, rig):
        rig.save_neural_ok = False
        _run_track(BACKFILL, p2p=FakeSession(supply=_supply(keys=('n',))))
        assert 'neural' in rig.names()

    @pytest.mark.parametrize('error', [OperationalError('db gone'), InterfaceError('connection closed')])
    def test_a_database_connectivity_error_while_saving_propagates(self, rig, error):
        rig.save_clap_error = error
        with pytest.raises(type(error)):
            _run_track(BACKFILL, p2p=FakeSession(supply=_supply(keys=('c',))))

    def test_a_song_matched_to_an_existing_row_asks_for_that_row(self, rig):
        rig.identity = (TrackPlan(False, True, False, False, True), OTHER_SIG_ID, False)
        session = FakeSession(supply=_supply(keys=('c', 'n')))
        _run_track(ALL, p2p=session)

        assert session.asks == [{'item': OTHER_SIG_ID, 'server': 'srv-1', 'provider': 'prov-1', 'wants': ['c', 'n']}]
        assert rig.find('row_duration') == [('row_duration', OTHER_SIG_ID)]
        assert 'persist' not in rig.names()
        assert not {'clap', 'neural', 'lyrics'} & set(rig.names())
        assert rig.find('save_clap')[0][1] == OTHER_SIG_ID
        assert rig.find('save_neural')[0][1] == OTHER_SIG_ID

    def test_nothing_supplied_leaves_the_plain_analysis(self, rig):
        session = FakeSession(supply=None)
        _run_track(ALL, p2p=session)
        assert rig.names() == PLAIN_NEW_SONG
        assert len(session.asks) == 1


class TestNeverAsks:
    @pytest.mark.parametrize('item_id', [UNSIGNABLE_ID, 'prov-legacy-42', 'fp_2short'])
    def test_an_id_without_a_content_signature(self, rig, caplog, item_id):
        caplog.set_level(logging.WARNING)
        session = FakeSession(supply=_supply(), live=False)
        _run_track(BACKFILL, p2p=session, item_id=item_id)
        assert session.asks == []
        assert _codes(caplog, 7002) == []
        assert rig.find('row_duration') == []
        assert rig.names() == PLAIN_BACKFILL

    def test_a_new_song_that_got_an_unsignable_id(self, rig):
        rig.identity = (ALL._replace(musicnn=False), UNSIGNABLE_ID, True)
        session = FakeSession(supply=_supply())
        _run_track(ALL, p2p=session)
        assert session.asks == []
        assert rig.names() == PLAIN_NEW_SONG

    @pytest.mark.parametrize('duration', [None, 0.0, -1.0, float('nan'), float('inf'), 'unknown'])
    def test_an_unknown_duration(self, rig, caplog, duration):
        caplog.set_level(logging.WARNING)
        rig.row_duration = duration
        rig.musicnn_duration = duration
        session = FakeSession(supply=_supply(), live=False)
        _run_track(BACKFILL, p2p=session)
        _run_track(ALL, p2p=session)
        assert session.asks == []
        assert _codes(caplog, 7002) == []

    def test_a_plan_with_nothing_a_peer_can_supply(self, rig):
        session = FakeSession(supply=_supply())
        _run_track(TrackPlan(False, False, False, True, False), p2p=session)
        assert session.asks == []
        assert rig.find('row_duration') == []
