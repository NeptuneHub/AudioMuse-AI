"""Unit tests for tasks.provider_migration_tasks.

Tests structured in layers:
  1. Pure functions: ``rewrite_id_map_json`` — JSON transform.
  2. Reflective helpers: ``find_fk`` — mocked psycopg2 cursor.
  3. Orchestration: ``execute_provider_migration`` — mocked psycopg2 connection
     with assertion that SQL statements are issued in the expected order.

Uses _import_module bypass.
"""
import json
import os
import sys
import importlib.util
import pytest
from unittest.mock import MagicMock, patch, call


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


# ---------------------------------------------------------------------------
# Pure function: rewrite_id_map_json
# ---------------------------------------------------------------------------

class TestRewriteIdMapJson:
    def test_swaps_values_leaves_int_keys(self, mig):
        old = json.dumps({'0': 'old_a', '1': 'old_b', '2': 'old_c'})
        mapping = {'old_a': 'new_a', 'old_b': 'new_b', 'old_c': 'new_c'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == {'0': 'new_a', '1': 'new_b', '2': 'new_c'}

    def test_drops_entries_with_no_mapping(self, mig):
        # If an old_id was deleted as an orphan (not in mapping), drop it from the map
        # so no dead pointers remain.
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
        # map_projection_data.id_map_json is a flat list where position N
        # corresponds to row N of the projection matrix. The rewrite must
        # keep the list length (and therefore row alignment) intact.
        old = json.dumps(['old_a', 'old_b', 'old_c'])
        mapping = {'old_a': 'new_a', 'old_b': 'new_b', 'old_c': 'new_c'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == ['new_a', 'new_b', 'new_c']

    def test_list_format_orphans_become_none(self, mig):
        # Can't drop slots from a list — that would desync row positions
        # from the projection matrix. Orphan slots become None and the
        # consumer in app_map.py falls through to compute on-the-fly.
        old = json.dumps(['keep', 'orphan', 'keep2'])
        mapping = {'keep': 'new1', 'keep2': 'new2'}
        new = mig.rewrite_id_map_json(old, mapping)
        parsed = json.loads(new)
        assert parsed == ['new1', None, 'new2']
        assert len(parsed) == 3  # length preserved

    def test_list_format_empty_mapping(self, mig):
        old = json.dumps(['a', 'b', 'c'])
        new = mig.rewrite_id_map_json(old, {})
        parsed = json.loads(new)
        assert parsed == [None, None, None]

    def test_unknown_top_level_type_is_left_alone(self, mig):
        # Defensive: if someone stores a JSON scalar we don't know how to
        # rewrite, leave the blob untouched rather than crashing the job.
        old = json.dumps('scalar_value')
        new = mig.rewrite_id_map_json(old, {'scalar_value': 'new'})
        assert new == old  # unchanged


# ---------------------------------------------------------------------------
# find_fk — reflects the actual FK constraint name
# ---------------------------------------------------------------------------

class TestFindFk:
    def test_returns_constraint_name_when_found(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = ('embedding_item_id_fkey',)
        name = mig.find_fk(cur, 'embedding', 'item_id')
        assert name == 'embedding_item_id_fkey'
        # Sanity: ensure the SELECT was against information_schema
        sql = cur.execute.call_args[0][0]
        assert 'information_schema' in sql
        assert 'FOREIGN KEY' in sql

    def test_returns_none_when_not_found(self, mig):
        cur = MagicMock()
        cur.fetchone.return_value = None
        name = mig.find_fk(cur, 'embedding', 'item_id')
        assert name is None


# ---------------------------------------------------------------------------
# execute_provider_migration — sequence assertions
# ---------------------------------------------------------------------------

def _session_state(mapping, meta=None):
    """Build a fake migration_session.state JSON blob."""
    return {
        'dry_run':        {'matches': mapping},
        'manual_matches': {},
        'new_meta':       meta or {},
    }


def _make_session_row(session_id=1, target='navidrome',
                      creds=None, state=None, status='dry_run_ready'):
    """Single migration_session row as a tuple (id, target_type, target_creds, state, status)."""
    return (
        session_id,
        target,
        json.dumps(creds or {'url': 'http://nav.local', 'user': 'u', 'password': 'p'}),
        json.dumps(state or _session_state({'old_1': 'new_1'})),
        status,
    )


def _install_fake_psycopg2(mig, session_row, voyager_rows=None, mproj_rows=None,
                           authors=None, mulan_exists=False):
    """Install a fake psycopg2 connection on the module + mock out redis, probe, etc.

    Returns (mock_conn, mock_cursor, executed_sql) for assertions.
    """
    mock_cur = MagicMock()
    executed = []  # list of SQL strings actually run

    def _execute(sql, params=None):
        sql_str = sql.strip() if isinstance(sql, str) else str(sql).strip()
        executed.append(sql_str)
        # Canned responses in the order they're asked for:
        up = sql_str.upper()
        if 'INFORMATION_SCHEMA' in up and 'FOREIGN KEY' in up:
            mock_cur.fetchone.return_value = ('{}_item_id_fkey'.format(params[0] if params else 'embedding'),)
        elif 'TO_REGCLASS' in up and 'MULAN_EMBEDDING' in up:
            mock_cur.fetchone.return_value = (mulan_exists,)
        elif 'FROM MIGRATION_SESSION' in up and 'SELECT' in up:
            mock_cur.fetchone.return_value = session_row
        elif 'FROM VOYAGER_INDEX_DATA' in up and 'SELECT' in up:
            mock_cur.fetchall.return_value = voyager_rows or []
        elif 'FROM MAP_PROJECTION_DATA' in up and 'SELECT' in up:
            mock_cur.fetchall.return_value = mproj_rows or []
        elif 'SELECT DISTINCT' in up and 'SCORE' in up:
            mock_cur.fetchall.return_value = [(a,) for a in (authors or [])]

    mock_cur.execute.side_effect = _execute
    mock_cur.__enter__ = lambda self: self
    mock_cur.__exit__  = lambda self, *a: None

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda self: self
    mock_conn.__exit__  = lambda self, *a: None

    # Patch the module's attributes in-place
    mig._get_dedicated_conn = MagicMock(return_value=mock_conn)

    # Redis stub
    fake_redis = MagicMock()
    fake_redis.get.return_value = None
    mig._get_redis = MagicMock(return_value=fake_redis)

    # RQ worker stop: no-op (no live workers in test)
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
        # Advisory lock taken
        assert 'PG_ADVISORY_XACT_LOCK' in joined
        # Temp table created
        assert 'CREATE TEMP TABLE ITEM_ID_MIGRATION_MAP' in joined
        # Orphans deleted before rewrite
        assert 'DELETE FROM SCORE' in joined
        # FK drops
        assert 'ALTER TABLE EMBEDDING DROP CONSTRAINT' in joined
        assert 'ALTER TABLE CLAP_EMBEDDING DROP CONSTRAINT' in joined
        # UPDATE ... FROM rewrites
        assert 'UPDATE SCORE' in joined
        assert 'UPDATE PLAYLIST' in joined
        assert 'UPDATE EMBEDDING' in joined
        assert 'UPDATE CLAP_EMBEDDING' in joined
        # FK re-add
        assert joined.count('ADD CONSTRAINT') >= 2
        # Artist tables truncated
        assert 'DELETE FROM ARTIST_INDEX_DATA' in joined
        assert 'DELETE FROM ARTIST_COMPONENT_PROJECTION' in joined
        assert 'DELETE FROM ARTIST_MAPPING' in joined
        # app_config upsert (provider credentials)
        assert 'INSERT INTO APP_CONFIG' in joined
        # app_config table is created if missing (legacy DBs restored from
        # backups predating the setup wizard don't have this table; the
        # migration must create it transactionally before inserting).
        assert 'CREATE TABLE IF NOT EXISTS APP_CONFIG' in joined
        # Session marked complete
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
        drop_idx  = next(i for i, s in enumerate(upper)
                         if 'ALTER TABLE EMBEDDING DROP CONSTRAINT' in s)
        upd_idx   = next(i for i, s in enumerate(upper)
                         if s.startswith('UPDATE EMBEDDING'))
        readd_idx = next(i for i, s in enumerate(upper)
                         if 'ALTER TABLE EMBEDDING' in s and 'ADD CONSTRAINT' in s)
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

        # Redis set('migration:paused', '1', ...) should have been called
        fake_redis = mig._get_redis.return_value
        assert fake_redis.set.called
        paused_call_args = fake_redis.set.call_args
        assert paused_call_args[0][0] == 'migration:paused'
        # And eventually cleared post-commit
        assert fake_redis.delete.called
        assert 'migration:paused' in [c[0][0] for c in fake_redis.delete.call_args_list]

    def test_voyager_id_map_rewrite_happens(self, mig):
        # Seed one voyager row with an id_map containing the old id
        voyager_rows = [('voyager_main', json.dumps({'0': 'old_1'}))]
        session_row = _make_session_row(state=_session_state({'old_1': 'new_1'}))
        _install_fake_psycopg2(mig, session_row, voyager_rows=voyager_rows)

        mig.execute_provider_migration(1)

        # The migration should have issued an UPDATE voyager_index_data after rewriting
        # (we can't easily verify the exact JSON written without capturing params, but
        #  we can verify the statement was executed)
        executed_upper = '\n'.join(
            s.upper() for s in mig._get_dedicated_conn.return_value.cursor.return_value.execute.call_args_list
            if isinstance(s, str)
        )
        # Fallback: walk the call_args_list and check sql strings
        calls = mig._get_dedicated_conn.return_value.cursor.return_value.execute.call_args_list
        sqls = [c[0][0] for c in calls]
        upd_voyager = [s for s in sqls if 'UPDATE voyager_index_data' in s or 'UPDATE VOYAGER_INDEX_DATA' in s.upper()]
        assert len(upd_voyager) >= 1
