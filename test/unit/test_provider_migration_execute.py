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
    # Production waits between restart-publish attempts; the suite must not. The
    # retry tests assert the attempt COUNT, which this does not touch.
    mod._RESTART_PUBLISH_RETRY_SECONDS = 0
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


def _id_map_lookup(rows, params):
    name = params[0] if params else None
    match = next((r for r in (rows or []) if r[0] == name), None)
    return (match[1],) if match else None


def _build_sql_handlers(mock_cur, session_row, ivf_rows, mproj_rows, authors, lyrics_exists):
    def _matches(up, *needles):
        return all(n in up for n in needles)

    def _set_one(value):
        mock_cur.fetchone.return_value = value

    def _set_all(value):
        mock_cur.fetchall.return_value = value

    return [
        (
            lambda up: _matches(up, 'INFORMATION_SCHEMA', 'FOREIGN KEY'),
            lambda up, params: _set_one(
                ('{}_item_id_fkey'.format(params[0] if params else 'embedding'),)
            ),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'LYRICS_EMBEDDING'),
            lambda up, params: _set_one((lyrics_exists,)),
        ),
        (
            lambda up: _matches(up, 'TO_REGCLASS', 'MIGRATION_TARGET_META'),
            lambda up, params: _set_one((None,)),
        ),
        # The registry and the legacy config table both exist in a real install:
        # the migration repoints music_servers and purges app_config's copies.
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
            lambda up: _matches(up, 'FROM MIGRATION_SESSION', 'SELECT'),
            lambda up, params: _set_one(session_row),
        ),
        (
            lambda up: up.startswith('SELECT DISTINCT INDEX_NAME FROM VOYAGER_INDEX_DATA'),
            lambda up, params: _set_all([(r[0],) for r in (ivf_rows or [])]),
        ),
        (
            lambda up: up.startswith('SELECT ID_MAP_JSON FROM VOYAGER_INDEX_DATA'),
            lambda up, params: _set_one(_id_map_lookup(ivf_rows, params)),
        ),
        (
            lambda up: up.startswith('SELECT INDEX_NAME, ID_MAP_JSON FROM VOYAGER_INDEX_DATA'),
            lambda up, params: _set_all([]),
        ),
        (
            lambda up: up.startswith('SELECT DISTINCT INDEX_NAME FROM MAP_PROJECTION_DATA'),
            lambda up, params: _set_all([(r[0],) for r in (mproj_rows or [])]),
        ),
        (
            lambda up: up.startswith('SELECT ID_MAP_JSON FROM MAP_PROJECTION_DATA'),
            lambda up, params: _set_one(_id_map_lookup(mproj_rows, params)),
        ),
        (
            lambda up: up.startswith('SELECT INDEX_NAME, ID_MAP_JSON FROM MAP_PROJECTION_DATA'),
            lambda up, params: _set_all([]),
        ),
        (
            lambda up: _matches(up, 'SELECT DISTINCT', 'SCORE'),
            lambda up, params: _set_all([(a,) for a in (authors or [])]),
        ),
    ]


def _install_fake_psycopg2(
    mig, session_row, ivf_rows=None, mproj_rows=None, authors=None, lyrics_exists=False
):
    mock_cur = MagicMock()
    executed = []
    handlers = _build_sql_handlers(
        mock_cur, session_row, ivf_rows, mproj_rows, authors, lyrics_exists
    )

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

    fake_redis = MagicMock()
    fake_redis.get.return_value = None
    mig._get_redis = MagicMock(return_value=fake_redis)

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
        # The migration, in one statement: the song keeps its canonical id and only
        # the provider id it is reachable by on the default server changes.
        assert 'UPDATE TRACK_SERVER_MAP' in joined
        # Artist ids are the one thing the matcher cannot repoint, so they are cleared.
        assert 'DELETE FROM ARTIST_SERVER_MAP' in joined
        # The registry is the only home of media-server settings: the default
        # row is repointed and the legacy app_config copies are cleared, never
        # rewritten (they would keep serving the OLD provider on the next boot).
        assert 'UPDATE MUSIC_SERVERS' in joined
        assert 'DELETE FROM APP_CONFIG' in joined
        assert 'INSERT INTO APP_CONFIG' not in joined
        assert 'UPDATE MIGRATION_SESSION' in joined

    def test_the_centralized_catalogue_is_never_touched(self, mig):
        """`score` holds one row per distinct recording, keyed by the fp_2 hash of its
        own AUDIO. That id is a property of the song, not of any server, so a provider
        swap can neither delete it nor rewrite it. A song's analysis is expensive and
        irreplaceable; where the file happens to live is not.

        This used to DELETE unmatched score rows (taking their embeddings with them via
        ON DELETE CASCADE) and rewrite score.item_id into the target provider's track
        id - the pre-canonicalization design, where item_id WAS the provider id.
        """
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        assert not any(s.startswith('DELETE FROM SCORE') for s in upper), (
            "a migration must never delete a song from the catalogue"
        )
        # score is only ever UPDATEd for metadata, never for item_id.
        for stmt in upper:
            if stmt.startswith('UPDATE SCORE'):
                assert 'SET ITEM_ID' not in stmt, "canonical ids must never be rewritten"
        for table in ('EMBEDDING', 'CLAP_EMBEDDING', 'LYRICS_EMBEDDING', 'PLAYLIST'):
            assert not any(s.startswith(f'UPDATE {table} ') for s in upper)
        # No item_id moves, so the FKs never need dropping and the similarity indexes
        # stay valid: a provider swap no longer forces a full re-index.
        assert not any('DROP CONSTRAINT' in s for s in upper)
        assert not any(s.startswith('DELETE FROM IVF_CELL') for s in upper)
        assert not any(s.startswith('DELETE FROM IVF_DIR') for s in upper)

    def test_unmatched_songs_are_unbound_from_the_server_not_deleted(self, mig):
        """An unmatched song loses its mapping to THIS server and nothing else: its
        score row, its embeddings and its mappings to any other server survive."""
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
        """`chromaprint` is keyed by the (server_id, provider_track_id) pair the
        migration rewrites. Left alone, every fingerprint on the server becomes
        unreachable AND a target that recycles one of those ids inherits the wrong
        one - analysis reads the stored print back and never recomputes it. The
        audio did not change, so the print follows the song."""
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
        """The staging join reads the OLD provider ids out of track_server_map. Run
        it after the repoint and it reads the new ids and carries nothing."""
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
        """A row keyed by an id the server no longer has is unreachable through
        track_server_map and is exactly what lets a recycled id serve a stale
        fingerprint later. It goes; the backfill recomputes if the song comes back."""
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
        """A re-point onto the SAME provider (a Navidrome id rotation) permutes ids
        rather than replacing them, so a one-pass UPDATE trips the primary key the
        moment it assigns an id another row still holds."""
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
        """'analysis' is not provenance: it is the only record that a catalogue row
        yields no usable signature and can never be relabelled. Flatten it and the
        startup canonicalizer counts the row as legacy again and re-hashes the whole
        catalogue on the next boot, relabelling nothing."""
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
        """The old code claimed to "pause and drain the workers" and did nothing at
        all: send_stop_signal does not exist in RQ 2.x, the drain loop broke out
        after one second, and the migration:paused key had no reader anywhere. The
        migration is now VISIBLE in task_status instead, so get_active_main_task can
        keep an analysis or a sweep from running through its destructive transaction.
        """
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

        mig.execute_provider_migration(1)

        assert [s for _t, s in reported] == ['STARTED', 'SUCCESS']
        assert {t for t, _s in reported} == {'mig-1'}

    def test_index_id_maps_are_left_alone_because_ids_never_move(self, mig):
        """The similarity indexes are keyed by the canonical item_id. A migration no
        longer rewrites item_ids, so the indexes stay valid across a provider swap and
        need neither relabelling nor a rebuild - which is what this used to do."""
        ivf_rows = [('ivf_main', json.dumps({'0': 'old_1'}))]
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row, ivf_rows=ivf_rows)

        result = mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        assert not any(s.startswith('UPDATE VOYAGER_INDEX_DATA') for s in upper)
        assert not any(s.startswith('UPDATE MAP_PROJECTION_DATA') for s in upper)
        assert result['index_rebuild_needed'] is False


class TestMigrationWritesTheRegistryOnly:
    """The music_servers registry is the ONLY home of media-server settings.

    The migration points its default row at the target provider and clears any
    legacy app_config copies, instead of maintaining a second one that would
    keep serving the OLD provider (config projects the registry row) and leave
    stale credentials behind until the next restart.
    """

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
    """The default server's artist ids belong to the OLD provider after a migration.

    Track ids are repointed (the matcher produced a new id for each), but artists
    have no such mapping, so the default server's artist_server_map rows are
    cleared and rebuilt by the next analysis. Secondary servers did not migrate
    and keep theirs.
    """

    def test_default_servers_artist_rows_are_deleted(self, mig):
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        deletes = [s for s in executed if 'DELETE FROM artist_server_map' in s]
        assert len(deletes) == 1
        # Scoped to the default server only.
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
    def test_success_is_recorded_before_the_restart_is_published(self, mig, monkeypatch):
        """The restart can kill this very job. Published before SUCCESS is written,
        the row stays non-terminal, RQ retries the job, the retry fails the
        dry_run_ready gate (the session is 'completed' now) and reports FAILURE over
        a migration that actually succeeded."""
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _install_fake_psycopg2(mig, session_row)

        order = []
        monkeypatch.setattr(mig, '_migration_task_id', lambda: 'mig-1')
        monkeypatch.setattr(
            mig, '_report_migration',
            lambda task_id, status, progress, message, details=None: order.append(status),
        )
        monkeypatch.setattr(
            mig, '_post_commit_reload', lambda *a, **k: order.append('reload')
        )

        mig.execute_provider_migration(1)

        assert order[-1] == 'reload', "the restart must come after SUCCESS is written"
        assert 'SUCCESS' in order

    def test_a_retry_of_an_applied_migration_reports_success_not_failure(
        self, mig, monkeypatch
    ):
        """RQ requeues an abandoned job under the SAME task id, so a migration that
        committed and was then killed by its own restart comes back here. Raising on
        the dry_run_ready gate made that retry overwrite the SUCCESS row with
        FAILURE (save_task_status protects only REVOKED), telling the user a
        completed migration had failed."""
        session_row = _make_session_row(status='completed')
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
        monkeypatch.setattr(
            mig, '_post_commit_reload', lambda *a, **k: reloaded.append(1)
        )

        result = mig.execute_provider_migration(1)

        assert result['ok'] is True
        assert result['already_applied'] is True
        assert reported and reported[-1][0] == 'SUCCESS'
        assert not any(
            s.upper().startswith('UPDATE TRACK_SERVER_MAP') for s in executed
        ), "a retry must not repoint anything a second time"
        assert reloaded, (
            "the first run may have died of an OOM before it ever reached the "
            "reload, so the retry must still refresh config and publish the restart"
        )

    def test_a_retry_does_not_queue_a_second_alignment(self, mig, monkeypatch):
        """Its first run already committed an alignment row; the janitor revives
        that one. Queueing another would run the same full-refresh sweep twice."""
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
        """The request is a Redis publish, so a failure is a Redis blip. Giving up on
        the first one leaves the registry pointing at the new provider while every
        running worker still holds the old one."""
        import types

        monkeypatch.setattr(mig, '_RESTART_PUBLISH_RETRY_SECONDS', 0)
        attempts = []
        fake_restart = types.ModuleType('restart_manager')

        def _flaky():
            attempts.append(1)
            return len(attempts) >= 2

        fake_restart.publish_restart_request = _flaky
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        assert mig._publish_restart_with_retries() is True
        assert len(attempts) == 2

    def test_an_undelivered_restart_request_is_reported_loudly(self, mig, monkeypatch, caplog):
        """The swap is only half applied until the workers restart: the registry
        points at the new provider while running workers still hold the old one."""
        import types

        monkeypatch.setattr(mig, '_RESTART_PUBLISH_RETRY_SECONDS', 0)
        fake_restart = types.ModuleType('restart_manager')
        fake_restart.publish_restart_request = lambda: False
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        with caplog.at_level(logging.ERROR):
            assert mig._publish_restart_with_retries() is False

        assert 'RESTART AUDIOMUSE MANUALLY' in caplog.text

    def test_a_raising_restart_manager_is_retried_too(self, mig, monkeypatch, caplog):
        import types

        monkeypatch.setattr(mig, '_RESTART_PUBLISH_RETRY_SECONDS', 0)
        calls = []
        fake_restart = types.ModuleType('restart_manager')

        def _boom():
            calls.append(1)
            raise RuntimeError('redis is down')

        fake_restart.publish_restart_request = _boom
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        with caplog.at_level(logging.ERROR):
            assert mig._publish_restart_with_retries() is False

        assert len(calls) == mig._RESTART_PUBLISH_ATTEMPTS

    def test_the_alignment_row_is_committed_with_the_migration(self, mig):
        """A crash between the commit and the enqueue would otherwise leave no
        record that an alignment is owed, and the artist ids the swap cleared stay
        empty forever. The row rides the migration's own transaction, so the janitor
        can always find it."""
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


class TestPlaylistReportIsInformationalOnly:
    """Counting stored playlists is a log line. A schema where that query fails must
    not abort an otherwise good migration, and in Postgres a failed statement
    poisons the whole transaction unless it is fenced by a savepoint."""

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
    """A provider swap CLEARS the server's artist ids (the matcher maps tracks, not
    artists) and cannot refresh a path the target did not report. Only an alignment
    rebuilds those, and the migration is the one default-server change that never
    queued one - so the ids stayed empty until the user happened to re-analyze.
    The artist lookup has no fallback: it returns nothing at all meanwhile.
    """

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
        fake_restart.publish_restart_request = lambda: order.append('restart')
        monkeypatch.setitem(sys.modules, 'restart_manager', fake_restart)

        mig._post_commit_reload(MagicMock())

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
            raise RuntimeError("redis is down")

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
