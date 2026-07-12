# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Provider-migration execution SQL sequence and id-map rewrite.

Covers the core migration executor that swaps provider ids across tables,
asserting the ordered SQL steps and the JSON/list id-map rewriting.

Main Features:
* rewrite_id_map swaps values, drops unmapped entries and handles list form
* Foreign keys are dropped before updates and re-added afterwards
* Orphan deletion runs before updates and workers are paused before starting
* app_config music-libraries row is written/deleted from the selected libraries
"""

import json
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
    return _load_tasks_mod()


class TestRewriteIdMapJson:
    def test_swaps_values_leaves_int_keys(self, mig):
        old = json.dumps({'0': 'old_a', '1': 'old_b', '2': 'old_c'})
        mapping = {'old_a': 'new_a', 'old_b': 'new_b', 'old_c': 'new_c'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == {'0': 'new_a', '1': 'new_b', '2': 'new_c'}

    def test_drops_entries_with_no_mapping(self, mig):
        old = json.dumps({'0': 'keep', '1': 'orphan', '2': 'keep2'})
        mapping = {'keep': 'new1', 'keep2': 'new2'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == {'0': 'new1', '2': 'new2'}
        assert '1' not in parsed

    def test_empty_input_returns_empty(self, mig):
        assert mig.rewrite_id_map_json('', {'a': 'b'}) == ''
        assert mig.rewrite_id_map_json(None, {'a': 'b'}) is None

    def test_empty_mapping_drops_everything(self, mig):
        old = json.dumps({'0': 'a', '1': 'b'})
        new = mig.rewrite_id_map_json(old, {})
        parsed = json.loads(new)
        assert parsed == {}

    def test_list_format_rewrites_in_place(self, mig):
        old = json.dumps(['old_a', 'old_b', 'old_c'])
        mapping = {'old_a': 'new_a', 'old_b': 'new_b', 'old_c': 'new_c'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == ['new_a', 'new_b', 'new_c']

    def test_list_format_orphans_become_none(self, mig):
        old = json.dumps(['keep', 'orphan', 'keep2'])
        mapping = {'keep': 'new1', 'keep2': 'new2'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == ['new1', None, 'new2']
        assert len(parsed) == 3

    def test_list_format_empty_mapping(self, mig):
        old = json.dumps(['a', 'b', 'c'])
        new = mig.rewrite_id_map_json(old, {})
        parsed = json.loads(new)
        assert parsed == [None, None, None]

    def test_unknown_top_level_type_is_left_alone(self, mig):
        old = json.dumps('scalar_value')
        new = mig.rewrite_id_map_json(old, {'scalar_value': 'new'})
        assert new == old


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

    mig._drain_workers_or_timeout = MagicMock()

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
        assert 'DELETE FROM SCORE' in joined
        assert 'ALTER TABLE EMBEDDING DROP CONSTRAINT' in joined
        assert 'ALTER TABLE CLAP_EMBEDDING DROP CONSTRAINT' in joined
        assert 'UPDATE SCORE' in joined
        assert 'UPDATE PLAYLIST' in joined
        assert 'UPDATE EMBEDDING' in joined
        assert 'UPDATE CLAP_EMBEDDING' in joined
        assert joined.count('ADD CONSTRAINT') >= 2
        assert 'DELETE FROM ARTIST_INDEX_DATA' in joined
        assert 'DELETE FROM ARTIST_COMPONENT_PROJECTION' in joined
        assert 'DELETE FROM ARTIST_MAPPING' in joined
        # The registry is the only home of media-server settings: the default
        # row is repointed and the legacy app_config copies are cleared, never
        # rewritten (they would keep serving the OLD provider on the next boot).
        assert 'UPDATE MUSIC_SERVERS' in joined
        assert 'DELETE FROM APP_CONFIG' in joined
        assert 'INSERT INTO APP_CONFIG' not in joined
        assert 'UPDATE MIGRATION_SESSION' in joined

    def test_delete_orphans_runs_before_updates(self, mig):
        session_row = _make_session_row(
            state=_session_state({'old_1': 'new_1'}),
        )
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        delete_idx = next(i for i, s in enumerate(upper) if s.startswith('DELETE FROM SCORE'))
        update_score_idx = next(i for i, s in enumerate(upper) if s.startswith('UPDATE SCORE'))
        assert delete_idx < update_score_idx, "orphan delete must precede score rewrite"

    def test_fk_drop_before_update_then_readd_after(self, mig):
        session_row = _make_session_row(state=_session_state({'a': 'b'}))
        _, _, executed = _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        upper = [s.upper() for s in executed]
        drop_idx = next(
            i for i, s in enumerate(upper) if 'ALTER TABLE EMBEDDING DROP CONSTRAINT' in s
        )
        upd_idx = next(i for i, s in enumerate(upper) if s.startswith('UPDATE EMBEDDING'))
        readd_idx = next(
            i for i, s in enumerate(upper) if 'ALTER TABLE EMBEDDING' in s and 'ADD CONSTRAINT' in s
        )
        assert drop_idx < upd_idx < readd_idx

    def test_rejects_session_not_in_dry_run_ready(self, mig):
        session_row = _make_session_row(status='in_progress')
        _install_fake_psycopg2(mig, session_row)

        with pytest.raises(Exception) as exc:
            mig.execute_provider_migration(1)
        assert 'dry_run_ready' in str(exc.value).lower() or 'status' in str(exc.value).lower()

    def test_pauses_workers_before_starting(self, mig):
        session_row = _make_session_row(state=_session_state({'a': 'b'}))
        _install_fake_psycopg2(mig, session_row)

        mig.execute_provider_migration(1)

        fake_redis = mig._get_redis.return_value
        assert fake_redis.set.called
        paused_call_args = fake_redis.set.call_args
        assert paused_call_args[0][0] == 'migration:paused'
        assert fake_redis.delete.called
        assert 'migration:paused' in [c[0][0] for c in fake_redis.delete.call_args_list]

    def test_vec_id_map_rewrite_happens(self, mig):
        ivf_rows = [('ivf_main', json.dumps({'0': 'old_1'}))]
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _install_fake_psycopg2(mig, session_row, ivf_rows=ivf_rows)

        mig.execute_provider_migration(1)

        calls = mig._get_dedicated_conn.return_value.cursor.return_value.execute.call_args_list
        sqls = [c[0][0] for c in calls]
        upd_ivf = [
            s
            for s in sqls
            if 'UPDATE voyager_index_data' in s or 'UPDATE VOYAGER_INDEX_DATA' in s.upper()
        ]
        assert len(upd_ivf) >= 1


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
