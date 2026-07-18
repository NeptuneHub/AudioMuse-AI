# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Table-derived, marker-free catalogue-id duplicate check.

Verifies that the check keys off score.duration alone: only fp_ groups with a
NULL-duration survivor are examined, real duplicates get their length stamped
(so they are never re-examined), false duplicates lose ONLY their map rows
(never catalogue rows), and an unreachable or unreliable server is skipped and
retried. No app_config flag is read or written, so the config cleanup can never
make it re-run.

Main Features:
* One-time via score.duration: nothing to check once every survivor is stamped.
* Real vs false classification by duration consensus; real -> stamp, false ->
  unmap.
* Unreachable/unreliable servers skipped, their groups left for the next start.
"""

import pytest

from tasks import duplicate_repair as dr


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0
        self._last = None

    def execute(self, sql, params=None):
        squashed = ' '.join(sql.split())
        self.conn.executed.append((squashed, params))
        self._last = squashed
        if squashed.startswith('DELETE FROM track_server_map'):
            server_id, item_ids = params
            groups = self.conn.state['groups']
            self.rowcount = sum(
                len(groups.get(server_id, {}).get(item_id, []))
                for item_id in item_ids
            )
        else:
            self.rowcount = 0

    def fetchone(self):
        if self._last and 'pg_try_advisory_lock' in self._last:
            return (self.conn.state.get('lock_free', True),)
        return None

    def close(self):
        pass


class FakeConn:
    def __init__(self, state):
        self.state = state
        self.executed = []
        self.commits = 0
        self.autocommit = None

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def harness(monkeypatch):
    state = {'groups': {}, 'durations': {}, 'servers': {}, 'stamped': []}
    monkeypatch.setattr(dr, '_groups_needing_check', lambda cur: state['groups'])
    monkeypatch.setattr(
        dr.registry, 'get_server',
        lambda server_id, conn=None: state['servers'].get(server_id),
    )
    monkeypatch.setattr(
        dr, '_stamp_real_durations',
        lambda cur, real_durations: state['stamped'].append(dict(real_durations)),
    )

    def durations_for(server):
        value = state['durations'][server['server_id']]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(dr, '_server_durations', durations_for)
    state['conn'] = FakeConn(state)
    return state


def _server_row(server_id):
    return {'server_id': server_id, 'name': server_id, 'server_type': 'jellyfin',
            'creds': {}}


def _deletes(conn):
    return [
        params for sql, params in conn.executed
        if sql.startswith('DELETE FROM track_server_map')
    ]


def test_reads_and_writes_no_app_config(harness, monkeypatch):
    # The old marker lived in app_config and got purged as an unknown key every
    # boot. The check must never touch app_config at all now.
    monkeypatch.delattr(dr, 'get_app_config_value', raising=False)
    monkeypatch.delattr(dr, 'set_app_config_value', raising=False)
    assert dr.repair_duplicate_track_maps(conn=harness['conn']) == {
        'checked': 0, 'real': 0, 'false': 0, 'removed': 0
    }


def test_no_null_duration_groups_is_instant_noop(harness):
    result = dr.repair_duplicate_track_maps(conn=harness['conn'])
    assert result == {'checked': 0, 'real': 0, 'false': 0, 'removed': 0}
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == []
    # No START banner, no server contact when there is nothing to check.
    assert not any(
        'START OF CATALOGUE' in str(sql) for sql, _p in harness['conn'].executed
    )


def test_real_duplicates_are_kept_and_stamped(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2', 'p3']}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 201.0, 'p3': 206.9}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == {'checked': 1, 'real': 1, 'false': 0, 'removed': 0}
    assert _deletes(harness['conn']) == []
    # The survivor's length is recorded so the group is never re-examined.
    assert harness['stamped'] == [{'fp_2aaa': 200.0}]


def test_false_duplicates_lose_only_their_map_rows(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {
        'fp_2aaa': ['p1', 'p2'],
        'fp_2bbb': ['p3', 'p4'],
    }}
    harness['durations']['srv'] = {
        'p1': 200.0, 'p2': 210.0,
        'p3': 300.0, 'p4': 300.5,
    }

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == {'checked': 2, 'real': 1, 'false': 1, 'removed': 2}
    deletes = _deletes(harness['conn'])
    assert len(deletes) == 1
    assert deletes[0] == ('srv', ['fp_2aaa'])
    assert harness['stamped'] == [{'fp_2bbb': 300.0}]
    touched = [
        sql for sql, _params in harness['conn'].executed
        if sql.startswith('UPDATE music_servers SET updated_at')
    ]
    assert touched, "unmapping must invalidate the availability cache token"


def test_missing_member_duration_makes_the_group_false(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2', 'p3', 'p4']}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0, 'p3': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['false'] == 1
    assert _deletes(harness['conn'])[0] == ('srv', ['fp_2aaa'])
    assert harness['stamped'] == [{}]


def test_unreachable_server_leaves_its_groups_for_next_start(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2']}}
    harness['durations']['srv'] = RuntimeError('server down')

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['removed'] == 0
    assert _deletes(harness['conn']) == []
    # nothing stamped => the group is still NULL-duration => retried next start
    assert harness['stamped'] == []


def test_unreliable_listing_skips_the_server(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {
        'fp_2aaa': ['p1', 'p2'],
        'fp_2bbb': ['p3', 'p4'],
    }}
    harness['durations']['srv'] = {'p1': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['removed'] == 0
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == []


def test_deleted_server_is_skipped(harness):
    harness['groups'] = {'gone': {'fp_2aaa': ['p1', 'p2']}}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['removed'] == 0
    assert harness['stamped'] == []


def test_one_bad_server_does_not_block_the_good_one(harness):
    harness['servers']['ok'] = _server_row('ok')
    harness['servers']['bad'] = _server_row('bad')
    harness['groups'] = {
        'ok': {'fp_2aaa': ['p1', 'p2']},
        'bad': {'fp_2bbb': ['p3', 'p4']},
    }
    harness['durations']['ok'] = {'p1': 100.0, 'p2': 400.0}
    harness['durations']['bad'] = RuntimeError('server down')

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['false'] == 1
    assert result['removed'] == 2
    assert _deletes(harness['conn'])[0][0] == 'ok'


def test_another_replica_holding_the_lock_skips(harness):
    harness['lock_free'] = False
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2']}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == {'skipped': 'locked'}
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == []
