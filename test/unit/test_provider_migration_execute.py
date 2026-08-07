# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Provider-migration execution SQL sequence.

Covers the core migration executor that swaps provider ids across tables,
asserting the ordered SQL steps.

Main Features:
* Foreign keys are dropped before updates and re-added afterwards
* Orphan deletion runs before updates and workers are paused before starting
* app_config music-libraries row is written/deleted from the selected libraries
"""

import json
import logging
import os
import re
import sys
import importlib.util
import pytest
from unittest.mock import MagicMock, patch


def _load_tasks_mod():
    mod_name = 'tasks.provider_migration_tasks'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'tasks', 'provider_migration_tasks.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mig():
    mod = _load_tasks_mod()
    mod._RESTART_PUBLISH_RETRY_SECONDS = 0
    mod._RESTART_HANDSHAKE_RETRY_SECONDS = 0
    mod._real_await_worker_restart = mod._await_worker_restart
    mod._await_worker_restart = MagicMock(
        side_effect=lambda _conn, _session_id, request_id, **_kw: request_id
    )
    return mod


class TestFindFk:
    def test_returns_constraint_name_when_found(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = ('embedding_item_id_fkey',)
        name = mig.find_fk(cur, 'embedding', 'item_id')
        assert name == 'embedding_item_id_fkey'
        sql = cur.execute.call_args[0][0]
        assert 'information_schema' in sql
        assert 'FOREIGN KEY' in sql

    def test_returns_none_when_not_found(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = None
        name = mig.find_fk(cur, 'embedding', 'item_id')
        assert name is None


_META_ROWS = [
    (
        'new_1',
        '/target/music/new_1.flac',
        'Target Title',
        'Target Artist',
        'Target Album',
        'Target Album Artist',
        2024,
    ),
]


def _session_state(mapping, meta=None):
    return {
        'dry_run': {'matches': mapping},
        'manual_matches': {},
        'new_meta': meta or {},
    }


def _make_session_row(
    session_id=1, target='navidrome', creds=None, state=None, status='dry_run_ready'
):
    return (
        session_id,
        target,
        json.dumps(creds or {'url': 'http://nav.local', 'user': 'u', 'password': 'p'}),
        json.dumps(state or _session_state({'old_1': 'new_1'})),
        status,
    )


def _build_sql_handlers(mock_cur, session_row, meta_rows):
    session_snapshot = {'row': session_row}

    def _matches(up, *needles):
        return all(n in up for n in needles)

    def _set_one(value):
        mock_cur.fetchone.return_value = value

    def _set_all(value):
        mock_cur.fetchall.return_value = value

    def _set_completed(up, params):
        row = session_snapshot['row']
        session_snapshot['row'] = (
            row[0], row[1], row[2], params[0], 'completed'
        )
        _set_one((row[0],))

    def _set_restart_ack(up, params):
        row = session_snapshot['row']
        state = json.loads(row[3]) if isinstance(row[3], str) else dict(row[3])
        state['restart_acknowledged'] = True
        session_snapshot['row'] = (
            row[0], row[1], row[2], json.dumps(state), row[4]
        )
        _set_one((row[0],))

    return [
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'MIGRATION_TARGET_META'),
            lambda up, params: _set_one(
                ('migration_target_meta',) if meta_rows is not None else (None,)
            ),
        ),
        (
            lambda up: up.startswith('SELECT NEW_ID') and 'MIGRATION_TARGET_META' in up,
            lambda up, params: _set_all(list(meta_rows or [])),
        ),
        (
            lambda up: up.startswith('DELETE FROM TRACK_SERVER_MAP') and 'RETURNING' in up,
            lambda up, params: _set_all([]),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'MUSIC_SERVERS'),
            lambda up, params: _set_one((True,)),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'APP_CONFIG'),
            lambda up, params: _set_one((True,)),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'ARTIST_SERVER_MAP'),
            lambda up, params: _set_one(('artist_server_map',)),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'CHROMAPRINT'),
            lambda up, params: _set_one(('chromaprint',)),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'PLAYLIST'),
            lambda up, params: _set_one(('playlist',)),
        ),
        (
            lambda up: _matches(up, 'COUNT(DISTINCT P.PLAYLIST_NAME)'),
            lambda up, params: _set_one((2,)),
        ),
        (
            lambda up: up.startswith('SELECT STATUS FROM TASK_STATUS'),
            lambda up, params: _set_one(('RUNNING',)),
        ),
        (
            lambda up: up.startswith('UPDATE TASK_STATUS') and 'RETURNING TASK_ID' in up,
            lambda up, params: _set_one(((params or [None, None, 'task'])[2],)),
        ),
        (
            lambda up: up.startswith(
                "UPDATE MIGRATION_SESSION SET STATUS = 'COMPLETED'"
            ),
            _set_completed,
        ),
        (
            lambda up: up.startswith('SELECT STATUS, STATE FROM MIGRATION_SESSION'),
            lambda up, params: _set_one(
                (session_snapshot['row'][4], session_snapshot['row'][3])
            ),
        ),
        (
            lambda up: up.startswith('UPDATE MIGRATION_SESSION SET STATE = JSONB_SET')
            and "'{RESTART_ACKNOWLEDGED}'" in up,
            _set_restart_ack,
        ),
        (
            lambda up: _matches(up, 'FROM MIGRATION_SESSION', 'SELECT'),
            lambda up, params: _set_one(session_snapshot['row']),
        ),
    ]


def _install_fake_psycopg2(mig, session_row, meta_rows=None):
    mock_cur = MagicMock()
    executed = []
    handlers = _build_sql_handlers(mock_cur, session_row, meta_rows)

    def _execute(sql, params=None):
        sql_str = sql.strip() if isinstance(sql, str) else str(sql).strip()
        executed.append(sql_str)
        up = sql_str.upper()
        for predicate, apply_result in handlers:
            if predicate(up):
                apply_result(up, params)
                return

    mock_cur.execute.side_effect = _execute
    mock_cur.rowcount = 0
    mock_cur.__enter__ = lambda self: self
    mock_cur.__exit__ = lambda self, *a: None

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda self: self
    mock_conn.__exit__ = lambda self, *a: None

    mig._get_dedicated_conn = MagicMock(return_value=mock_conn)

    return mock_conn, mock_cur, executed


class TestExecuteProviderMigration:
    def test_runs_core_sql_sequence(self, mig):
        session_row = _make_session_row(
            session_id=42,
            state=_session_state({'old_1': 'new_1', 'old_2': 'new_2'}),
        )
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(42)

        joined = '\n'.join(executed).upper()
        assert 'PG_ADVISORY_XACT_LOCK' in joined
        assert 'CREATE TEMP TABLE ITEM_ID_MIGRATION_MAP' in joined
        assert 'UPDATE TRACK_SERVER_MAP' in joined
        assert 'DELETE FROM ARTIST_SERVER_MAP' in joined
        assert 'UPDATE MUSIC_SERVERS' in joined
        assert 'DELETE FROM APP_CONFIG' in joined
        assert 'INSERT INTO APP_CONFIG' not in joined
        assert 'UPDATE MIGRATION_SESSION' in joined

    def test_the_target_tags_are_written_onto_the_catalogue_rows(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row, meta_rows=_META_ROWS)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        score_updates = [s for s in upper if s.startswith('UPDATE SCORE')]
        assert len(score_updates) == 1, (
            "the target's tags must reach the catalogue exactly once"
        )
        set_clause = score_updates[0].split(' SET ', 1)[1].split(' FROM ', 1)[0]
        assert re.findall(r'(?:^|,)\s*([A-Z_]+)\s*=', set_clause) == [
            'TITLE', 'AUTHOR', 'ALBUM', 'ALBUM_ARTIST', 'YEAR'
        ], "only the display tags move; item_id and everything else stay put"

        path_updates = [
            s for s in upper
            if s.startswith('UPDATE TRACK_SERVER_MAP') and 'SET FILE_PATH' in s
        ]
        assert len(path_updates) == 1, "the new file path must be written on the binding"
        assert 'S.IS_DEFAULT' in path_updates[0], (
            "another server's file paths are none of this migration's business"
        )

    def test_no_target_metadata_leaves_the_catalogue_rows_alone(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        assert not any(s.startswith('UPDATE SCORE') for s in upper)
        assert not any(
            s.startswith('UPDATE TRACK_SERVER_MAP') and 'SET FILE_PATH' in s
            for s in upper
        )

    def test_only_the_tags_move_never_the_ids_vectors_or_indexes(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row, meta_rows=_META_ROWS)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        assert not any(s.startswith('DELETE FROM SCORE') for s in upper), (
            "a migration must never delete a song from the catalogue"
        )
        for stmt in upper:
            if stmt.startswith('UPDATE SCORE'):
                assert 'SET ITEM_ID' not in stmt, "canonical ids must never be rewritten"
        for table in ('EMBEDDING', 'CLAP_EMBEDDING', 'LYRICS_EMBEDDING', 'PLAYLIST'):
            assert not any(s.startswith(f'UPDATE {table} ') for s in upper)
        assert not any(s.startswith('UPDATE VOYAGER_INDEX_DATA') for s in upper)
        assert not any(s.startswith('UPDATE MAP_PROJECTION_DATA') for s in upper)
        assert not any('DROP CONSTRAINT' in s for s in upper)
        assert not any(s.startswith('DELETE FROM IVF_CELL') for s in upper)
        assert not any(s.startswith('DELETE FROM IVF_DIR') for s in upper)

    def test_unmatched_songs_are_unbound_from_the_server_not_deleted(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        unbind = [
            s for s in executed
            if s.upper().startswith('DELETE FROM TRACK_SERVER_MAP')
            and 'ITEM_ID_MIGRATION_MAP' in s.upper()
        ]
        assert unbind, "unmatched songs must be unbound from the default server"
        assert 'IS_DEFAULT' in unbind[0].upper(), "only the migrated server is unbound"

    def test_chromaprints_are_carried_onto_the_target_ids_not_left_under_dead_ones(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        carried = [s for s in upper if s.startswith('UPDATE CHROMAPRINT')]
        assert carried, "fingerprints must be repointed at the new provider ids"
        assert all('IS_DEFAULT' in s for s in carried), (
            "only the migrated server's fingerprints move"
        )

    def test_the_carry_is_staged_before_the_ids_it_reads_are_overwritten(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        staged = next(
            i for i, s in enumerate(upper)
            if s.startswith('INSERT INTO MIGRATION_CHROMAPRINT_CARRY')
        )
        repointed = next(
            i for i, s in enumerate(upper) if s.startswith('UPDATE TRACK_SERVER_MAP')
        )
        assert staged < repointed, (
            "the fingerprint carry must be staged while the old ids are still there"
        )

    def test_fingerprints_that_carry_nothing_are_deleted_not_orphaned(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        dropped = [
            s for s in executed
            if s.upper().startswith('DELETE FROM CHROMAPRINT')
        ]
        assert dropped, "uncarried fingerprints must not be left keyed by dead ids"
        assert 'MIGRATION_CHROMAPRINT_CARRY' in dropped[0].upper()
        assert 'IS_DEFAULT' in dropped[0].upper(), (
            "another server's fingerprints are none of this migration's business"
        )

    def test_the_fingerprint_rewrite_is_two_pass_like_the_mapping_rewrite(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        prefixed = [
            s for s in executed
            if s.upper().startswith('UPDATE CHROMAPRINT') and '|| K.NEW_ID' in s.upper()
        ]
        stripped = [
            s for s in executed
            if s.upper().startswith('UPDATE CHROMAPRINT') and 'SUBSTR(' in s.upper()
        ]
        assert prefixed and stripped, (
            "the fingerprint rewrite needs the same two passes through a temp prefix"
        )

    def test_the_unsignable_analysis_tier_survives_the_repoint(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        repoint = next(
            s for s in executed if s.upper().startswith('UPDATE TRACK_SERVER_MAP')
            and 'MATCH_TIER' in s.upper()
        )
        assert "'analysis'" in repoint.lower(), (
            "the repoint must preserve the unsignable marker, not overwrite it"
        )
        assert 'CASE' in repoint.upper()

    def test_rejects_session_not_in_dry_run_ready(self, mig):
        session_row = _make_session_row(status='in_progress')
        _install_fake_psycopg2(mig, session_row)

        with pytest.raises(Exception) as exc:
            mig.execute_provider_migration(1)
        assert 'dry_run_ready' in str(exc.value).lower() or 'status' in str(exc.value).lower()

    def test_migration_reports_its_task_status_instead_of_faking_a_worker_pause(self, mig, monkeypatch):
        session_row = _make_session_row(state=_session_state({'a': 'b'}))
        _install_fake_psycopg2(mig, session_row)

        reported = []
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(
            mig, '_report_migration',
            lambda task_id, status, progress, message, details=None: reported.append(
                (task_id, status)
            ),
        )
        monkeypatch.setattr(
            mig,
            '_finalize_restart_handshake',
            lambda _conn, _sid, _rid, task_id, _message, **_kw: reported.append(
                (task_id, 'SUCCESS')
            ),
        )

        mig.execute_provider_migration(1)

        assert [s for _t, s in reported] == ['RUNNING', 'SUCCESS', 'SUCCESS'], (
            'SUCCESS is recorded the moment the catalogue transaction commits, and '
            'again once the workers acknowledge the restart: the swap is durable at '
            'the first one, so nothing after it may report the run as failed'
        )
        assert {t for t, _s in reported} == {'mig-1'}

    def test_a_committed_migration_is_never_reported_as_failed(self, mig, monkeypatch):
        session_row = _make_session_row(state=_session_state({'a': 'b'}))
        _install_fake_psycopg2(mig, session_row)

        reported = []
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(
            mig, '_report_migration',
            lambda task_id, status, progress, message, details=None: reported.append(status),
        )

        def _handshake_never_completes(*_args, **_kwargs):
            raise RuntimeError('no worker acknowledged the restart')

        monkeypatch.setattr(mig, '_await_worker_restart', _handshake_never_completes)

        mig.execute_provider_migration(1)

        swap_is_durable = (
            'the catalogue swap is already durable, so telling the user the '
            'migration failed and their database is unchanged is a lie - and the '
            'handshake guard then refuses the re-run that message asks for'
        )
        assert 'FAIL' not in reported, swap_is_durable
        assert 'FAILURE' not in reported, swap_is_durable
        assert reported[-1] == 'SUCCESS'

    def test_a_rebuild_signalled_by_the_transaction_reaches_the_summary(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _install_fake_psycopg2(mig, session_row)

        with patch.object(mig, '_run_migration_transaction', return_value=True):
            rebuilt = mig.execute_provider_migration(1)
        with patch.object(mig, '_run_migration_transaction', return_value=False):
            untouched = mig.execute_provider_migration(1)

        assert rebuilt['index_rebuild_needed'] is True
        assert untouched['index_rebuild_needed'] is False


class TestMigrationWritesTheRegistryOnly:

    def _write(self, mig, selected_libraries):
        cur = MagicMock()
        executed = []
        params = []

        def _execute(sql, p=None):
            executed.append(sql.strip() if isinstance(sql, str) else str(sql))
            params.append(p)
            cur.fetchone.return_value = (True,)

        cur.execute.side_effect = _execute
        cur.rowcount = 0

        target_creds = {'url': 'http://nav.local', 'user': 'u', 'password': 'p'}
        mig._write_provider_to_default_server(
            cur,
            'navidrome',
            target_creds,
            selected_libraries=selected_libraries,
        )
        return executed, params

    def _default_row_update(self, executed, params):
        for sql, p in zip(executed, params):
            if 'UPDATE music_servers' in sql and 'is_default' in sql:
                return p
        raise AssertionError('the default music_servers row was never updated')

    def test_target_provider_and_creds_land_in_the_registry(self, mig):
        executed, params = self._write(mig, selected_libraries=['A'])
        server_type, creds, libraries = self._default_row_update(executed, params)
        assert server_type == 'navidrome'
        assert creds.adapted == {'url': 'http://nav.local', 'user': 'u', 'password': 'p'}
        assert libraries == 'A'

    def test_none_selection_clears_the_library_filter(self, mig):
        executed, params = self._write(mig, selected_libraries=None)
        assert self._default_row_update(executed, params)[2] == ''

    def test_empty_list_selection_also_clears_it(self, mig):
        executed, params = self._write(mig, selected_libraries=[])
        assert self._default_row_update(executed, params)[2] == ''

    def test_non_empty_selection_is_comma_joined(self, mig):
        executed, params = self._write(mig, selected_libraries=['Main Music', 'Podcasts'])
        assert self._default_row_update(executed, params)[2] == 'Main Music,Podcasts'

    def test_whitespace_only_entries_are_filtered(self, mig):
        executed, params = self._write(
            mig, selected_libraries=['Main Music', '  ', '', 'Podcasts']
        )
        assert self._default_row_update(executed, params)[2] == 'Main Music,Podcasts'

    def test_legacy_media_keys_are_purged_from_app_config(self, mig):
        import config

        cur = MagicMock()
        executed = []
        params = []

        def _execute(sql, p=None):
            executed.append(sql.strip() if isinstance(sql, str) else str(sql))
            params.append(p)
            cur.fetchone.return_value = (True,)

        cur.execute.side_effect = _execute
        cur.rowcount = 3

        mig._purge_media_keys_from_app_config(cur)

        deletes = [
            (sql, p) for sql, p in zip(executed, params)
            if 'DELETE FROM app_config' in sql
        ]
        assert len(deletes) == 1
        assert deletes[0][1] == (sorted(config.MEDIASERVER_CONFIG_KEYS),)
        assert not any('INSERT INTO app_config' in sql for sql in executed)


class TestExecuteProviderMigrationForwardsSelectedLibraries:
    def test_state_selected_libraries_reaches_write_provider(self, mig):
        state = _session_state({'old_1': 'new_1'})
        state['selected_libraries'] = ['Main', 'Extra']
        session_row = _make_session_row(state=state)
        _install_fake_psycopg2(mig, session_row)

        with patch.object(mig, '_run_migration_transaction') as mock_tx:
            mig.execute_provider_migration(42)

        assert mock_tx.called
        kwargs = mock_tx.call_args.kwargs
        assert kwargs.get('selected_libraries') == ['Main', 'Extra']

    def test_missing_state_selected_libraries_forwarded_as_none(self, mig):
        state = _session_state({'old_1': 'new_1'})
        session_row = _make_session_row(state=state)
        _install_fake_psycopg2(mig, session_row)

        with patch.object(mig, '_run_migration_transaction') as mock_tx:
            mig.execute_provider_migration(1)

        kwargs = mock_tx.call_args.kwargs
        assert kwargs.get('selected_libraries') is None


class TestMigrationClearsStaleArtistIds:

    def test_default_servers_artist_rows_are_deleted(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        deletes = [s for s in executed if 'DELETE FROM artist_server_map' in s]
        assert len(deletes) == 1
        assert 'music_servers s' in deletes[0]
        assert 's.is_default' in deletes[0]

    def test_missing_table_is_tolerated(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = (None,)
        executed = []
        cur.execute.side_effect = lambda sql, p=None: executed.append(sql)

        mig._clear_default_server_artist_map(cur)

        assert not any('DELETE FROM artist_server_map' in s for s in executed)


class TestTheMigrationSurvivesTheRestartItTriggers:
    def test_success_is_recorded_only_after_restart_ack(self, mig, monkeypatch):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _install_fake_psycopg2(mig, session_row)

        order = []
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(
            mig, '_report_migration',
            lambda task_id, status, progress, message, details=None: (
                order.append(status) or True
            ),
        )
        monkeypatch.setattr(
            mig,
            '_await_worker_restart',
            lambda _conn, _session, request_id, **_kw: (
                order.append('restart_ack') or request_id
            ),
        )
        monkeypatch.setattr(
            mig,
            '_finalize_restart_handshake',
            lambda *_args, **_kwargs: order.append('SUCCESS'),
        )

        mig.execute_provider_migration(1)

        assert order[-2:] == ['restart_ack', 'SUCCESS']

    def test_a_retry_of_an_applied_migration_reports_success_not_failure(
        self, mig, monkeypatch
    ):
        completed_state = {
            'post_migration': {'matched': 1},
            'exec_task_id': 'mig-1',
            'alignment_task_id': 'align-1',
            'restart_request_id': 'restart-1',
            'restart_acknowledged': False,
        }
        session_row = _make_session_row(status='completed', state=completed_state)
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        reported = []
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(
            mig, '_report_migration',
            lambda task_id, status, progress, message, details=None: reported.append(
                (status, details)
            ),
        )

        reloaded = []
        monkeypatch.setattr(mig, '_restart_request_result', lambda _request_id: None)
        monkeypatch.setattr(
            mig,
            '_await_worker_restart',
            lambda *a, **k: reloaded.append(k.get('alignment_task_id')) or 'restart-1',
        )
        monkeypatch.setattr(
            mig,
            '_finalize_restart_handshake',
            lambda _conn, _sid, _rid, _task_id, _message, **kw: reported.append(
                ('SUCCESS', kw.get('details'))
            ),
        )

        result = mig.execute_provider_migration(1)

        assert result['ok'] is True
        assert result['already_applied'] is True
        assert reported and reported[-1][0] == 'SUCCESS'
        assert not any(
            s.upper().startswith('UPDATE TRACK_SERVER_MAP') for s in executed
        ), "a retry must not repoint anything a second time"
        assert reloaded == ['align-1'], (
            "the first run may have died of an OOM before it ever reached the "
            "reload, so the retry must still refresh config and publish the restart"
        )

    def test_a_retry_does_not_queue_a_second_alignment(self, mig, monkeypatch):
        from tasks import multiserver_sync as sync

        calls = []
        monkeypatch.setattr(
            sync, 'enqueue_server_alignment', lambda **kw: calls.append(kw)
        )

        mig._enqueue_post_migration_alignment(None)

        assert calls == []

    def test_a_blip_in_the_restart_publish_is_retried_not_abandoned(
        self, mig, monkeypatch
    ):
        import types

        monkeypatch.setattr(mig, '_RESTART_PUBLISH_RETRY_SECONDS', 0)
        attempts = []
        fake_restart = types.ModuleType('restart_manager')

        def _flaky(request_id=None):
            attempts.append(1)
            return len(attempts) >= 2

        fake_restart.publish_restart_request = _flaky
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        assert mig._publish_restart_with_retries('restart-1') is True
        assert len(attempts) == 2

    def test_an_undelivered_restart_request_is_reported_loudly(self, mig, monkeypatch, caplog):
        import types

        monkeypatch.setattr(mig, '_RESTART_PUBLISH_RETRY_SECONDS', 0)
        fake_restart = types.ModuleType('restart_manager')
        fake_restart.publish_restart_request = lambda request_id=None: False
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        with caplog.at_level(logging.ERROR):
            assert mig._publish_restart_with_retries('restart-1') is False

        assert 'RESTART AUDIOMUSE MANUALLY' in caplog.text

    def test_a_raising_restart_manager_is_retried_too(self, mig, monkeypatch, caplog):
        import types

        monkeypatch.setattr(mig, '_RESTART_PUBLISH_RETRY_SECONDS', 0)
        calls = []
        fake_restart = types.ModuleType('restart_manager')

        def _boom(request_id=None):
            calls.append(1)
            raise RuntimeError('the control plane is down')

        fake_restart.publish_restart_request = _boom
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        with caplog.at_level(logging.ERROR):
            assert mig._publish_restart_with_retries('restart-1') is False

        assert len(calls) == mig._RESTART_PUBLISH_ATTEMPTS

    def test_the_alignment_row_is_committed_with_the_migration(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        rows = [s for s in executed if s.upper().startswith('INSERT INTO TASK_STATUS')]
        assert rows, "the alignment intent must be written inside the transaction"
        commit_marker = next(
            i for i, s in enumerate(executed)
            if s.upper().startswith('UPDATE MIGRATION_SESSION')
        )
        staged = next(
            i for i, s in enumerate(executed)
            if s.upper().startswith('INSERT INTO TASK_STATUS')
        )
        assert staged < commit_marker

    def test_the_enqueue_adopts_the_row_the_transaction_committed(self, mig, monkeypatch):
        from tasks import multiserver_sync as sync

        calls = []
        monkeypatch.setattr(
            sync, 'enqueue_server_alignment',
            lambda **kwargs: calls.append(kwargs) or kwargs.get('task_id'),
        )

        mig._enqueue_post_migration_alignment('align-42')

        assert calls[0]['task_id'] == 'align-42', (
            "the queued job must carry the id already committed, or the row and the "
            "job describe two different alignments"
        )


class TestRestartHandshakeStateMachine:
    def test_commit_marker_is_written_under_cancel_start_lock(self, mig):
        cur = MagicMock()
        cur.fetchone.side_effect = [('RUNNING',), ('mig-1',)]

        mig._stage_restart_handshake(cur, 'mig-1', 7, 'restart-7')

        sql = [call.args[0] for call in cur.execute.call_args_list]
        assert 'pg_advisory_xact_lock' in sql[0]
        assert 'FOR UPDATE' in sql[1]
        assert sql[2].startswith('UPDATE task_status')
        details = json.loads(cur.execute.call_args_list[2].args[1][1])
        assert details == {
            'message': 'Provider swap committed; waiting for worker restart acknowledgement.',
            'status_message': (
                'Provider swap committed; waiting for worker restart acknowledgement.'
            ),
            'phase': 'restart_handshake',
            'provider_migration_committed': True,
            'migration_session_id': 7,
            'restart_request_id': 'restart-7',
        }

    def test_cancel_tombstone_before_commit_forces_full_rollback(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = ('REVOKED',)

        with pytest.raises(RuntimeError, match='cancelled or lost'):
            mig._stage_restart_handshake(cur, 'mig-1', 7, 'restart-7')

        assert not any(
            call.args[0].startswith('UPDATE task_status')
            for call in cur.execute.call_args_list
        )

    def test_completed_state_retains_every_recovery_id(self, mig, monkeypatch):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _conn, cur, _executed = _install_fake_psycopg2(mig, session_row)
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(mig, '_report_migration', lambda *a, **k: True)
        monkeypatch.setattr(
            mig,
            '_await_worker_restart',
            lambda _conn, _session, request_id, **_kw: request_id,
        )

        mig.execute_provider_migration(1)

        completion = next(
            call
            for call in cur.execute.call_args_list
            if call.args[0].startswith('UPDATE migration_session SET status = \'completed\'')
        )
        state = json.loads(completion.args[1][0])
        assert state['exec_task_id'] == 'mig-1'
        assert state['alignment_task_id']
        assert state['restart_request_id'] == 'provider-migration-restart:mig-1'
        assert state['restart_acknowledged'] is False

    def test_unknown_ack_keeps_same_job_live_until_it_recovers(self, mig, monkeypatch):
        persisted = []
        sleeps = []
        monkeypatch.setattr(mig, '_post_commit_reload', lambda *a, **k: False)
        monkeypatch.setattr(mig, '_publish_restart_with_retries', lambda _rid: True)
        monkeypatch.setattr(mig, '_restart_request_result', lambda _rid: None)
        monkeypatch.setattr(
            mig,
            '_persist_completed_restart_state',
            lambda _conn, _sid, rid, **kw: persisted.append((rid, kw)),
        )
        monkeypatch.setattr(mig.time, 'sleep', lambda seconds: sleeps.append(seconds))

        result = mig._real_await_worker_restart(MagicMock(), 7, 'restart-7')

        assert result == 'restart-7'
        assert sleeps == [0]
        assert persisted == []

    def test_definitive_negative_rotates_persisted_id_before_republish(
        self, mig, monkeypatch
    ):
        import types

        persisted = []
        fake_restart = types.ModuleType('restart_manager')
        fake_restart.new_control_request_id = lambda: 'restart-8'
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)
        monkeypatch.setattr(mig, '_post_commit_reload', lambda *a, **k: False)
        monkeypatch.setattr(mig, '_publish_restart_with_retries', lambda rid: rid == 'restart-8')
        monkeypatch.setattr(mig, '_restart_request_result', lambda _rid: False)
        monkeypatch.setattr(
            mig,
            '_persist_completed_restart_state',
            lambda _conn, _sid, rid, **kw: persisted.append((rid, kw)),
        )
        monkeypatch.setattr(mig.time, 'sleep', lambda _seconds: None)

        result = mig._real_await_worker_restart(MagicMock(), 7, 'restart-7')

        assert result == 'restart-8'
        assert persisted == [
            ('restart-8', {'acknowledged': False}),
        ]

    def test_success_cannot_regress_on_delayed_retry(self, mig, monkeypatch):
        import types
        import database
        from contextlib import nullcontext

        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchone.return_value = ('SUCCESS',)
        db = MagicMock()
        db.cursor.return_value = cur
        save = MagicMock()
        fake_flask = types.ModuleType('flask_app')
        fake_flask.app = types.SimpleNamespace(app_context=lambda: nullcontext())
        monkeypatch.setitem(sys.modules, 'flask_app', fake_flask)
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(database, 'save_task_status', save)

        result = mig._report_migration(
            'mig-1', 'FAIL', 100, 'late retry failed'
        )

        assert result == 'success'
        save.assert_not_called()
        db.commit.assert_called_once()

    @pytest.mark.parametrize(('existing', 'result'), [(None, 'missing'), ('REVOKED', 'revoked')])
    def test_orphan_or_revoked_job_cannot_start(self, mig, monkeypatch, existing, result):
        import types
        import database
        from contextlib import nullcontext

        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchone.return_value = (existing,) if existing else None
        db = MagicMock()
        db.cursor.return_value = cur
        save = MagicMock()
        fake_flask = types.ModuleType('flask_app')
        fake_flask.app = types.SimpleNamespace(app_context=lambda: nullcontext())
        monkeypatch.setitem(sys.modules, 'flask_app', fake_flask)
        monkeypatch.setattr(database, 'get_db', lambda: db)
        monkeypatch.setattr(database, 'save_task_status', save)

        assert mig._report_migration('mig-1', 'RUNNING', 0, 'start') == result
        save.assert_not_called()

    def test_revoked_root_aborts_before_opening_migration_connection(self, mig, monkeypatch):
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(mig, '_report_migration', lambda *a, **k: 'revoked')
        conn = MagicMock()
        monkeypatch.setattr(mig, '_get_dedicated_conn', conn)

        with pytest.raises(RuntimeError, match='no live execution claim'):
            mig.execute_provider_migration(7)

        conn.assert_not_called()


class TestPlaylistReportIsInformationalOnly:

    def test_a_failing_count_cannot_abort_the_migration(self, mig):
        cur = MagicMock()
        executed = []

        def _execute(sql, params=None):
            executed.append(sql)
            if 'count(DISTINCT p.playlist_name)' in sql:
                raise RuntimeError('column p.server_id does not exist')

        cur.execute.side_effect = _execute
        cur.fetchone.return_value = ('playlist',)

        mig._report_playlists_bound_to_default_server(cur)

        assert any(s.startswith('SAVEPOINT') for s in executed)
        assert any(s.startswith('ROLLBACK TO SAVEPOINT') for s in executed), (
            "the transaction must be left usable for the rest of the migration"
        )

    def test_a_clean_count_releases_its_savepoint(self, mig):
        cur = MagicMock()
        executed = []
        cur.execute.side_effect = lambda sql, params=None: executed.append(sql)
        cur.fetchone.side_effect = [('playlist',), (3,)]

        mig._report_playlists_bound_to_default_server(cur)

        assert any(s.startswith('RELEASE SAVEPOINT') for s in executed)
        assert not any(s.startswith('ROLLBACK') for s in executed)


class TestChromaprintCarryTolerance:
    def test_missing_chromaprint_table_is_tolerated(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = (None,)
        executed = []
        cur.execute.side_effect = lambda sql, p=None: executed.append(sql)

        staged = mig._stage_chromaprint_carry(cur)
        mig._repoint_chromaprint(cur, staged)

        assert staged is False
        assert not any('chromaprint' in s and 'to_regclass' not in s for s in executed)

    def test_nothing_is_rewritten_when_staging_was_skipped(self, mig):
        cur = MagicMock()
        executed = []
        cur.execute.side_effect = lambda sql, p=None: executed.append(sql)

        mig._repoint_chromaprint(cur, False)

        assert executed == []


class TestPostMigrationAlignment:

    def test_the_alignment_is_queued_before_the_restart_that_kills_the_worker(
        self, mig, monkeypatch
    ):
        import types

        order = []
        monkeypatch.setattr(
            mig, '_enqueue_post_migration_alignment',
            lambda *a, **k: order.append('align'),
        )
        fake_restart = types.ModuleType('restart_manager')
        fake_restart.publish_restart_request = (
            lambda request_id=None: order.append('restart') or True
        )
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        mig._post_commit_reload(MagicMock(), restart_request_id='restart-1')

        assert order == ['align', 'restart'], (
            "queue the alignment first: the restart can kill this very worker"
        )

    def test_the_alignment_is_asked_of_the_sweep_layer_not_reimplemented(
        self, mig, monkeypatch
    ):
        from tasks import multiserver_sync as sync

        calls = []
        monkeypatch.setattr(
            sync, 'enqueue_server_alignment',
            lambda **kwargs: calls.append(kwargs) or 'sweep-task-1',
        )

        mig._enqueue_post_migration_alignment('align-1')

        assert len(calls) == 1, "the sweep module owns the enqueue"
        assert calls[0].get('message')

    def test_a_failed_enqueue_never_fails_the_migration(self, mig, monkeypatch):
        from tasks import multiserver_sync as sync

        def _boom(**kwargs):
            raise RuntimeError("the alignment enqueue is down")

        monkeypatch.setattr(sync, 'enqueue_server_alignment', _boom)

        mig._enqueue_post_migration_alignment('align-1')

    def test_the_user_is_told_how_to_align_by_hand_when_none_was_queued(
        self, mig, monkeypatch, caplog
    ):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, 'enqueue_server_alignment', lambda **kwargs: None)

        with caplog.at_level(logging.WARNING):
            mig._enqueue_post_migration_alignment('align-1')

        assert 'Align' in caplog.text


class TestMigrationWarnsOnMissingTargetMetadata:
    def test_zero_meta_rows_is_logged(self, mig, caplog):
        cur = MagicMock()
        cur.fetchone.return_value = ('migration_target_meta',)
        cur.fetchall.return_value = []

        with caplog.at_level(logging.WARNING):
            assert mig._load_new_meta_from_table(cur, 7) == {}

        assert 'no target metadata rows' in caplog.text

    def test_missing_table_is_logged(self, mig, caplog):
        cur = MagicMock()
        cur.fetchone.return_value = (None,)

        with caplog.at_level(logging.WARNING):
            assert mig._load_new_meta_from_table(cur, 7) == {}

        assert 'migration_target_meta does not exist' in caplog.text


class TestRestartHandshakeRecoverySchemaGuard:
    def test_recovery_is_a_no_op_when_migration_session_does_not_exist_yet(
        self, monkeypatch
    ):
        import tasks.provider_migration_tasks as mig

        executed = []

        class _Cursor:
            def execute(self, sql, params=None):
                executed.append(sql)

            def fetchone(self):
                return (None,)

            def fetchall(self):
                raise AssertionError(
                    "the candidate query must not run without migration_session"
                )

            def close(self):
                pass

        conn = MagicMock()
        conn.cursor.return_value = _Cursor()
        monkeypatch.setattr(mig, '_get_dedicated_conn', lambda: conn)

        assert mig.recover_provider_migration_restart_handshakes() == 0

        assert len(executed) == 1
        assert 'to_regclass' in executed[0]
        conn.close.assert_called_once()


class TestMigrationSuccessFinalization:
    def test_the_committed_success_row_is_finalized_on_the_same_raw_connection(
        self, monkeypatch
    ):
        import database
        import tasks.provider_migration_tasks as mig

        class _Cursor:
            def execute(self, sql, params=None):
                pass

            def fetchone(self):
                return ('completed', {'restart_request_id': 'req-1'})

            def fetchall(self):
                return []

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        conn = MagicMock()
        conn.cursor.return_value = _Cursor()

        recorded = {}
        collapsed = {}
        monkeypatch.setattr(
            database, 'record_task_history',
            lambda *a, **k: recorded.update({'args': a, 'kwargs': k}),
        )
        monkeypatch.setattr(
            database, '_collapse_finished_task',
            lambda *a: collapsed.update({'args': a}),
        )
        monkeypatch.setattr(
            database, 'save_task_status',
            lambda *a, **k: pytest.fail('save_task_status needs an app context here'),
        )

        mig._finalize_restart_handshake(
            conn, 7, 'req-1', 'root-1', 'Provider migration applied.',
        )

        conn.commit.assert_called_once()
        assert recorded['args'][0] == 'root-1'
        assert recorded['args'][2] == mig.TASK_STATUS_SUCCESS
        assert recorded['kwargs']['conn'] is conn
        assert recorded['kwargs']['details']['message'] == 'Provider migration applied.'
        assert collapsed['args'][0] is conn
        assert collapsed['args'][1] == 'root-1'

    def test_a_failed_replay_never_undoes_the_committed_success(self, monkeypatch):
        import database
        import tasks.provider_migration_tasks as mig

        class _Cursor:
            rowcount = 0

            def execute(self, sql, params=None):
                pass

            def fetchone(self):
                return ('completed', {'restart_request_id': 'req-1'})

            def fetchall(self):
                return []

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        conn = MagicMock()
        conn.cursor.return_value = _Cursor()
        monkeypatch.setattr(
            database, 'save_task_status',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('db gone')),
        )

        mig._finalize_restart_handshake(
            conn, 7, 'req-1', 'root-1', 'Provider migration applied.',
        )
        conn.rollback.assert_not_called()
