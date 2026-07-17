# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Playlist persistence: per-server replacement, name history, and pruning.

Exercises database.update_playlist_table and
database.prune_playlist_rows_for_missing_servers against a mocked connection,
asserting the table only ever holds the last clustering run per server.

Main Features:
* The scoped write deletes only the target server's rows (NULL scope for the
  legacy no-registry fallback) and bulk-inserts rows carrying the server_id
* Repeated (name, item) pairs collapse to one row before insert
* playlist_name_history holds the created names of the last
  PLAYLIST_NAME_HISTORY_ROUNDS clustering rounds per server (rounds share one
  transaction timestamp); a round with no playlists leaves history intact; the
  history write is savepoint-guarded best-effort so a missing table never
  fails the playlist persist
* A failed write rolls back and re-raises so callers never report success on
  unpersisted rows; a server deleted mid-run gets no ghost rows re-inserted
* Pruning removes rows for servers out of scope, keeping NULL rows only when
  a legacy None server is in scope; an empty scope deletes nothing
"""

from unittest.mock import MagicMock

import pytest

import database


def _capture(monkeypatch):
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cur
    monkeypatch.setattr(database, 'get_db', lambda: conn)
    inserted = []

    def fake_execute_values(cursor, sql, rows, page_size=100):
        inserted.append((sql, list(rows)))

    monkeypatch.setattr(database, 'execute_values', fake_execute_values)
    return conn, cur, inserted


class TestUpdatePlaylistTable:
    def test_update_playlist_table_deletes_only_the_target_servers_rows(self, monkeypatch):
        conn, cur, _inserted = _capture(monkeypatch)
        database.update_playlist_table({'Rock': [('i1', 'T1', 'A1')]}, 's1')
        assert cur.execute.call_args_list[0].args == (
            "DELETE FROM playlist WHERE server_id = %s", ('s1',)
        )
        assert conn.commit.called

    def test_update_playlist_table_bulk_inserts_rows_carrying_the_server_id(self, monkeypatch):
        _conn, _cur, inserted = _capture(monkeypatch)
        database.update_playlist_table(
            {'Rock': [('i1', 'T1', 'A1'), ('i2', 'T2', 'A2')]}, 's1'
        )
        assert len(inserted) == 2
        sql, rows = inserted[0]
        assert 'ON CONFLICT (playlist_name, item_id, server_id) DO NOTHING' in sql
        assert rows == [('Rock', 'i1', 'T1', 'A1', 's1'), ('Rock', 'i2', 'T2', 'A2', 's1')]
        history_sql, history_rows = inserted[1]
        assert 'INSERT INTO playlist_name_history' in history_sql
        assert history_rows == [('s1', 'Rock')]

    def test_update_playlist_table_with_none_server_deletes_only_null_rows(self, monkeypatch):
        _conn, cur, inserted = _capture(monkeypatch)
        database.update_playlist_table({'Rock': [('i1', 'T1', 'A1')]}, None)
        assert cur.execute.call_args_list[0].args == (
            "DELETE FROM playlist WHERE server_id IS NULL",
        )
        assert inserted[0][1] == [('Rock', 'i1', 'T1', 'A1', None)]

    def test_update_playlist_table_dedupes_repeated_name_item_pairs(self, monkeypatch):
        _conn, _cur, inserted = _capture(monkeypatch)
        database.update_playlist_table(
            {'Rock': [('i1', 'T1', 'A1'), ('i1', 'T1', 'A1'), ('i2', 'T2', 'A2')]}, 's1'
        )
        assert inserted[0][1] == [('Rock', 'i1', 'T1', 'A1', 's1'), ('Rock', 'i2', 'T2', 'A2', 's1')]

    def test_update_playlist_table_with_no_playlists_still_clears_the_servers_rows(self, monkeypatch):
        conn, cur, inserted = _capture(monkeypatch)
        database.update_playlist_table({}, 's1')
        assert cur.execute.call_args_list[0].args == (
            "DELETE FROM playlist WHERE server_id = %s", ('s1',)
        )
        assert inserted == []
        assert conn.commit.called


class TestPlaylistNameHistory:
    def test_recent_names_merge_history_and_current_without_duplicates(self, monkeypatch):
        _conn, cur, _inserted = _capture(monkeypatch)
        cur.fetchall.side_effect = [
            [('Pop Heartbreak_automatic',), ('Rock Party_automatic',)],
            [('Pop Heartbreak_automatic',), ('Jazz Focus_automatic',)],
        ]

        names = database.get_recent_playlist_names('s1', limit=10)

        assert names == [
            'Pop Heartbreak_automatic',
            'Rock Party_automatic',
            'Jazz Focus_automatic',
        ]
        assert cur.execute.call_args_list[0].args[1] == ('s1', 10)
        assert cur.execute.call_args_list[1].args[1] == ('s1',)

    def test_zero_history_limit_does_not_query(self, monkeypatch):
        _conn, cur, _inserted = _capture(monkeypatch)

        assert database.get_recent_playlist_names('s1', limit=0) == []
        cur.execute.assert_not_called()

    def test_history_prunes_to_the_last_three_rounds_by_round_timestamp(self, monkeypatch):
        import config

        _conn, cur, inserted = _capture(monkeypatch)
        database.update_playlist_table({'Rock': [('i1', 'T1', 'A1')]}, 's1')
        assert inserted[1][1] == [('s1', 'Rock')]
        prune_calls = [
            call
            for call in cur.execute.call_args_list
            if 'playlist_name_history' in call.args[0]
        ]
        assert len(prune_calls) == 1
        prune_sql, prune_params = prune_calls[0].args
        assert 'created_at NOT IN' in prune_sql
        assert 'ORDER BY created_at DESC LIMIT %s' in prune_sql
        assert prune_params == ('s1', 's1', config.PLAYLIST_NAME_HISTORY_ROUNDS)
        assert config.PLAYLIST_NAME_HISTORY_ROUNDS == 3

    def test_a_history_write_failure_does_not_fail_the_playlist_persist(self, monkeypatch):
        conn, cur, _inserted = _capture(monkeypatch)

        def flaky(cursor, sql, rows, page_size=100):
            if 'playlist_name_history' in sql:
                raise RuntimeError('relation "playlist_name_history" does not exist')

        monkeypatch.setattr(database, 'execute_values', flaky)
        database.update_playlist_table({'Rock': [('i1', 'T1', 'A1')]}, 's1')
        executed = [call.args[0] for call in cur.execute.call_args_list]
        assert 'ROLLBACK TO SAVEPOINT history_names_write' in executed
        assert conn.commit.called
        assert not conn.rollback.called

    def test_a_round_with_no_playlists_preserves_the_previous_history(self, monkeypatch):
        _conn, cur, _inserted = _capture(monkeypatch)
        database.update_playlist_table({}, 's1')
        executed = [call.args[0] for call in cur.execute.call_args_list]
        assert not any('playlist_name_history' in sql for sql in executed)

    def test_update_playlist_table_rolls_back_and_reraises_when_the_write_fails(self, monkeypatch):
        conn, _cur, _inserted = _capture(monkeypatch)

        def boom(cursor, sql, rows, page_size=100):
            raise RuntimeError('insert failed')

        monkeypatch.setattr(database, 'execute_values', boom)
        with pytest.raises(RuntimeError):
            database.update_playlist_table({'Rock': [('i1', 'T1', 'A1')]}, 's1')
        assert conn.rollback.called
        assert not conn.commit.called

    def test_update_playlist_table_skips_insert_when_the_server_vanished_mid_run(self, monkeypatch):
        conn, cur, inserted = _capture(monkeypatch)
        cur.fetchone.return_value = (False,)
        database.update_playlist_table({'Rock': [('i1', 'T1', 'A1')]}, 's1')
        assert inserted == []
        assert conn.commit.called


class TestPrunePlaylistRows:
    def test_prune_deletes_rows_for_servers_out_of_scope_including_null(self, monkeypatch):
        conn, cur, _inserted = _capture(monkeypatch)
        database.prune_playlist_rows_for_missing_servers(['s1', 's2'])
        sql, params = cur.execute.call_args_list[0].args
        assert sql == "DELETE FROM playlist WHERE server_id IS NULL OR server_id != ALL(%s)"
        assert params == (['s1', 's2'],)
        assert conn.commit.called

    def test_prune_keeps_null_rows_when_a_legacy_none_server_is_in_scope(self, monkeypatch):
        _conn, cur, _inserted = _capture(monkeypatch)
        database.prune_playlist_rows_for_missing_servers(['s1', None])
        sql, params = cur.execute.call_args_list[0].args
        assert sql == "DELETE FROM playlist WHERE server_id IS NOT NULL AND server_id != ALL(%s)"
        assert params == (['s1'],)

    def test_prune_with_only_the_legacy_none_server_deletes_all_registered_rows(self, monkeypatch):
        _conn, cur, _inserted = _capture(monkeypatch)
        database.prune_playlist_rows_for_missing_servers([None])
        assert cur.execute.call_args_list[0].args == (
            "DELETE FROM playlist WHERE server_id IS NOT NULL",
        )

    def test_prune_with_an_empty_scope_deletes_nothing(self, monkeypatch):
        conn, cur, _inserted = _capture(monkeypatch)
        database.prune_playlist_rows_for_missing_servers([])
        assert cur.execute.call_count == 0
        assert not conn.commit.called
