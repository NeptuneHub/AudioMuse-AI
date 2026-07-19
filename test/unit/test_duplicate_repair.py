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
        self._last_params = params
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
        if self._last and self._last.startswith('SELECT count(*) FROM track_server_map'):
            server_id = self._last_params[0]
            # default 0 => the "if mapped" guard is inert => the server is
            # processed; a test wanting the unreliable-skip sets a big count.
            return (self.conn.state.get('mapped', {}).get(server_id, 0),)
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
    state = {'groups': {}, 'durations': {}, 'servers': {}, 'stamped': [],
             'mapped': {}, 'old_scheme': True, 'relabelled': 0}
    def _grouped(cur):
        shaped = {}
        for server_id, items in state['groups'].items():
            shaped[server_id] = {
                item_id: value if isinstance(value, tuple) else (value, [])
                for item_id, value in items.items()
            }
        return shaped

    monkeypatch.setattr(dr, '_groups_needing_check', _grouped)
    monkeypatch.setattr(dr, '_old_scheme_rows_exist', lambda cur: state['old_scheme'])
    from tasks import fingerprint_canonicalize as fc
    monkeypatch.setattr(
        fc, 'relabel_scheme_to_current',
        lambda cur, only_with_duration=True: state['relabelled'],
    )
    monkeypatch.setattr(
        dr.registry, 'get_server',
        lambda server_id, conn=None: state['servers'].get(server_id),
    )
    monkeypatch.setattr(
        dr, '_stamp_durations',
        lambda cur, durations_to_write: state['stamped'].append(dict(durations_to_write)),
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


def _totals(**kw):
    base = {'checked': 0, 'backfilled': 0, 'no_length': 0,
            'real': 0, 'false': 0, 'removed': 0, 'relabelled': 0}
    base.update(kw)
    return base


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
    assert dr.repair_duplicate_track_maps(conn=harness['conn']) == _totals()


def test_no_null_duration_rows_is_instant_noop(harness):
    result = dr.repair_duplicate_track_maps(conn=harness['conn'])
    assert result == _totals()
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == []
    # No START banner, no server contact when there is nothing to check.
    assert not any(
        'START OF CATALOGUE' in str(sql) for sql, _p in harness['conn'].executed
    )


def test_single_file_rows_get_their_length_backfilled(harness):
    # The 88% case: rows mapping ONE file just need their length stamped so they
    # can be a duration-confirmed merge target for a future copy. They are never
    # unmapped, even when the group set has no duplicates at all.
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1'], 'fp_2bbb': ['p2']}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 314.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == _totals(checked=2, backfilled=2)
    assert _deletes(harness['conn']) == [], "a single-file row is never unmapped"
    assert harness['stamped'] == [{'fp_2aaa': 200.0, 'fp_2bbb': 314.0}]


def test_prefetched_durations_are_reused_without_relisting(harness):
    # A mixed upgrade: the legacy migration already listed this server this same
    # boot and handed its durations here. The repair MUST reuse them and never
    # list the same server a second time. The fetcher is armed to raise, so if it
    # were called at all the test would fail.
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1'], 'fp_2bbb': ['p2']}}
    harness['durations']['srv'] = RuntimeError('must not re-list an already-listed server')

    result = dr.repair_duplicate_track_maps(
        conn=harness['conn'],
        prefetched_durations={'srv': {'p1': 200.0, 'p2': 314.0}},
    )

    assert result == _totals(checked=2, backfilled=2)
    assert harness['stamped'] == [{'fp_2aaa': 200.0, 'fp_2bbb': 314.0}]


def test_prefetch_covers_one_server_the_other_is_still_listed(harness):
    # Only the server the legacy migration listed is reused; a different server
    # with older-scheme rows is listed normally.
    harness['servers']['a'] = _server_row('a')
    harness['servers']['b'] = _server_row('b')
    harness['groups'] = {'a': {'fp_2aaa': ['p1']}, 'b': {'fp_2bbb': ['p2']}}
    harness['durations']['a'] = RuntimeError('server a was prefetched, must not re-list')
    harness['durations']['b'] = {'p2': 150.0}

    result = dr.repair_duplicate_track_maps(
        conn=harness['conn'],
        prefetched_durations={'a': {'p1': 200.0}},
    )

    assert result == _totals(checked=2, backfilled=2)
    stamped = {k: v for row in harness['stamped'] for k, v in row.items()}
    assert stamped == {'fp_2aaa': 200.0, 'fp_2bbb': 150.0}


def test_single_file_with_no_server_length_gets_the_sentinel(harness):
    # A single file the server reports no length for is stamped with the 0
    # sentinel so the whole catalogue is not re-listed for it on every boot; it
    # is NOT unmapped.
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1'], 'fp_2bbb': ['p2']}}
    harness['durations']['srv'] = {'p1': 200.0}  # p2 unknown

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == _totals(checked=2, backfilled=1, no_length=1)
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == [{'fp_2aaa': 200.0, 'fp_2bbb': dr._NO_SERVER_DURATION}]


def test_real_duplicates_are_kept_and_stamped(harness):
    tol = dr.config.DURATION_TOLERANCE_SECONDS
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2', 'p3']}}
    # Spans exactly the tolerance, so it stays a REAL group whatever the tolerance is.
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0, 'p3': 200.0 + tol}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == _totals(checked=1, real=1)
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

    assert result == _totals(checked=2, real=1, false=1, removed=2)
    deletes = _deletes(harness['conn'])
    assert len(deletes) == 1
    assert deletes[0] == ('srv', ['fp_2aaa'])
    assert harness['stamped'] == [{'fp_2bbb': 300.0}]
    touched = [
        sql for sql, _params in harness['conn'].executed
        if sql.startswith('UPDATE music_servers SET updated_at')
    ]
    assert touched, "unmapping must invalidate the availability cache token"


def test_missing_member_duration_makes_a_multi_file_group_false(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2', 'p3', 'p4']}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0, 'p3': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['false'] == 1
    assert _deletes(harness['conn'])[0] == ('srv', ['fp_2aaa'])
    assert harness['stamped'] == [{}]


def test_same_folder_distinct_files_are_split_even_when_durations_agree(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': (['p1', 'p2'], [
        '/media/music/Artist/Album/01 - One.flac',
        '/media/music/Artist/Album/02 - Two.flac',
    ])}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['false'] == 1
    assert _deletes(harness['conn'])[0] == ('srv', ['fp_2aaa'])


def test_same_recording_across_folders_survives_the_folder_rule(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': (['p1', 'p2'], [
        '/media/music/Artist/Album A/01 - Song.flac',
        '/media/music/Artist/Album B/05 - Song.flac',
    ])}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == _totals(checked=1, real=1)
    assert _deletes(harness['conn']) == []


def test_same_file_mapped_twice_is_not_a_folder_conflict(harness):
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': (['p1', 'p2'], [
        '/media/music/Artist/Album/01 - Song.flac',
        '/media/music/Artist/Album/01 - Song.flac',
    ])}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == _totals(checked=1, real=1)
    assert _deletes(harness['conn']) == []


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
    # The server has 1000 mapped tracks but the listing returned only 2: a broken
    # fetch, skip and retry. (Reliability is fetched-vs-mapped, not vs NULL rows.)
    harness['servers']['srv'] = _server_row('srv')
    harness['mapped']['srv'] = 1000
    harness['groups'] = {'srv': {
        'fp_2aaa': ['p1', 'p2'],
        'fp_2bbb': ['p3', 'p4'],
    }}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['removed'] == 0
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == []


def test_orphan_null_rows_do_not_skip_a_healthy_server(harness):
    # THE BUG: the NULL rows are orphans (not in the server listing), but the
    # server IS healthy (its whole catalogue came back). It must NOT be skipped -
    # the orphans get the sentinel so the catalogue is never re-listed for them.
    harness['servers']['srv'] = _server_row('srv')
    harness['mapped']['srv'] = 500  # a real catalogue...
    harness['groups'] = {'srv': {'fp_2orph1': ['x1'], 'fp_2orph2': ['x2']}}
    # ...but the server returns 480 OTHER tracks and none of the two orphans
    harness['durations']['srv'] = {'other%d' % i: 100.0 + i for i in range(480)}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == _totals(checked=2, no_length=2)
    assert _deletes(harness['conn']) == [], "orphan single-file rows are never unmapped"
    assert harness['stamped'] == [
        {'fp_2orph1': dr._NO_SERVER_DURATION, 'fp_2orph2': dr._NO_SERVER_DURATION}
    ]


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


def test_servers_are_fetched_in_parallel(harness, monkeypatch):
    import threading

    n = 3
    barrier = threading.Barrier(n, timeout=8)
    for i in range(n):
        sid = 'srv%d' % i
        harness['servers'][sid] = _server_row(sid)
        harness['groups'][sid] = {'fp_2%s' % (chr(97 + i) * 4): ['pa', 'pb']}
        harness['durations'][sid] = {'pa': 200.0, 'pb': 200.0}

    def barrier_fetch(server):
        # Only returns once ALL n servers are fetching at the same instant.
        # A sequential fetch arrives one at a time -> the barrier times out ->
        # BrokenBarrierError -> the server is skipped (real stays 0). Parallel
        # fetch has all n threads reach the barrier together and pass.
        barrier.wait()
        return harness['durations'][server['server_id']]

    monkeypatch.setattr(dr, '_server_durations', barrier_fetch)

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result['real'] == n, "all servers must be listed concurrently, not one by one"
    assert result['checked'] == n


def test_another_replica_holding_the_lock_skips(harness):
    harness['lock_free'] = False
    harness['servers']['srv'] = _server_row('srv')
    harness['groups'] = {'srv': {'fp_2aaa': ['p1', 'p2']}}
    harness['durations']['srv'] = {'p1': 200.0, 'p2': 200.0}

    result = dr.repair_duplicate_track_maps(conn=harness['conn'])

    assert result == {'skipped': 'locked'}
    assert _deletes(harness['conn']) == []
    assert harness['stamped'] == []
