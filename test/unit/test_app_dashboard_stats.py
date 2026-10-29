# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Dashboard snapshot contract: what the stats blob may and may not say.

Guards the rules the dashboard is built on: every number is either CATALOG or
per-SERVER, a percentage may only reach 100% when the work behind it is
genuinely complete, and the two refresh cadences stay split.

Main Features:
* A failed count is never published as a real zero (it would show "0 analyzed"
  until the next refresh)
* The tautological musicnn "analyzed %" stays out of the payload
* The per-server block travels inside the snapshot, not the request path
* The template cannot regress to a percentage that rounds up into a false 100%
* The FAST counts refresh every 60s; the distribution charts (Genres, Moods,
  Tempo) stay hourly with their own timestamp; neither walks a media server
* Recent tasks shows one line per task type (its last run, its last failure and
  how many runs the history holds), and a cancelled run is never a failure;
  /api/dashboard/task_history lists one type's runs, newest first
"""

import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app_dashboard as dash
import config
from error.error_dictionary import ERR_INVALID_REQUEST


def _cursor_with(mood_rows=(('happy:0.9,sad:0.1', 'danceable:0.5'),),
                 tempo_row=(1, 2, 3, 4, 120.0)):
    """A cursor whose named-cursor mood scan and tempo query both succeed, so a
    test can vary only the counts it cares about."""
    cur = MagicMock()
    scan = MagicMock()
    scan.__iter__ = lambda self: iter(list(mood_rows))
    cur.connection.cursor.return_value.__enter__.return_value = scan
    cur.fetchone.return_value = tempo_row
    return cur


class TestRefreshInterval:
    def test_fast_refresh_is_60s_with_no_db_probe(self, monkeypatch):
        def _fail_if_db_touched():
            raise AssertionError('refresh interval must not query the DB')
        monkeypatch.setattr(dash, 'get_db', _fail_if_db_touched)
        assert dash.dashboard_refresh_interval() == 60

    def test_charts_stay_on_their_own_hourly_cadence(self):
        assert dash.DASHBOARD_CHARTS_REFRESH_INTERVAL_SECONDS == 3600
        assert (dash.DASHBOARD_CHARTS_REFRESH_INTERVAL_SECONDS
                > dash.DASHBOARD_REFRESH_INTERVAL_SECONDS)


class TestSnapshotCompleteness:
    def test_a_failed_count_blocks_the_whole_snapshot(self, monkeypatch):
        monkeypatch.setattr(dash, '_collect_music_server_metrics',
                            lambda cur, total_songs=None: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None:
                            None if 'clap_embedding' in sql else 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert metrics['clap_indexed'] is None
        assert metrics['_complete'] is False

    def test_a_complete_fast_block_is_publishable(self, monkeypatch):
        monkeypatch.setattr(dash, '_collect_music_server_metrics',
                            lambda cur, total_songs=None: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert metrics['_complete'] is True


class TestCadenceSplit:
    def test_fast_block_carries_no_distribution_chart_keys(self, monkeypatch):
        monkeypatch.setattr(dash, '_collect_music_server_metrics',
                            lambda cur, total_songs=None: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert 'top_genre' not in metrics
        assert 'moods_coverage' not in metrics
        assert 'tempo_profile' not in metrics
        assert 'total_songs' in metrics

    def test_charts_block_carries_the_charts_and_its_own_timestamp(self):
        metrics = dash._collect_charts_metrics(_cursor_with())

        assert 'top_genre' in metrics
        assert 'moods_coverage' in metrics
        assert 'tempo_profile' in metrics
        assert metrics['charts_updated_at']
        assert 'total_songs' not in metrics
        assert metrics['top_genre'][0]['label'] == 'happy'


class TestMoodsCoverageIsAPartitionNotASumOfScores:
    """Issue #826: the pie used to sum every song's raw emotional scores. Those
    scores are CLAP audio-vs-text cosine similarities remapped by (sim+1)/2, so
    they all sit in a narrow band well above 0; summing them converges on
    N*mean(label) and every slice lands at 1/6 +- <1%, the more so the LARGER the
    library. Counting each song once under its winning label removes the shared
    offset and makes the six slices a real partition."""

    def test_each_song_is_counted_once_under_its_winning_label(self):
        rows = (
            ('happy:0.9', 'danceable:0.61,sad:0.55'),
            ('happy:0.9', 'danceable:0.60,sad:0.58'),
            ('happy:0.9', 'danceable:0.52,sad:0.64'),
        )

        metrics = dash._collect_charts_metrics(_cursor_with(mood_rows=rows))

        counts = {r['label']: r['count'] for r in metrics['moods_coverage']}
        assert counts == {'danceable': 2, 'sad': 1}
        assert sum(counts.values()) == len(rows)

    def test_no_row_reports_a_summed_score(self):
        metrics = dash._collect_charts_metrics(_cursor_with())

        assert metrics['moods_coverage']
        for row in metrics['moods_coverage']:
            assert 'score' not in row
            assert isinstance(row['count'], int)

    def test_a_near_tie_library_still_separates_instead_of_flattening(self):
        flat = tuple(
            ('happy:0.9', 'danceable:0.62,sad:0.61,party:0.60')
            for _ in range(3)
        ) + (('happy:0.9', 'danceable:0.60,sad:0.61,party:0.62'),)

        metrics = dash._collect_charts_metrics(_cursor_with(mood_rows=flat))

        counts = {r['label']: r['count'] for r in metrics['moods_coverage']}
        assert counts == {'danceable': 3, 'party': 1}

    def test_a_song_awaiting_clap_is_left_out_entirely(self):
        zero = 'danceable:0.00,aggressive:0.00,happy:0.00,party:0.00,relaxed:0.00,sad:0.00'
        rows = (
            ('happy:0.9', zero),
            ('happy:0.9', zero),
            ('happy:0.9', 'relaxed:0.58,sad:0.55'),
        )

        metrics = dash._collect_charts_metrics(_cursor_with(mood_rows=rows))

        counts = {r['label']: r['count'] for r in metrics['moods_coverage']}
        assert counts == {'relaxed': 1}

    def test_tempo_and_energy_are_not_emotional_labels(self):
        rows = (('happy:0.9',
                 'tempo_normalized:0.98,energy_normalized:0.95,relaxed:0.60'),)

        metrics = dash._collect_charts_metrics(_cursor_with(mood_rows=rows))

        assert [r['label'] for r in metrics['moods_coverage']] == ['relaxed']

    def test_biggest_slice_comes_first(self):
        rows = (
            ('happy:0.9', 'sad:0.70,danceable:0.50'),
            ('happy:0.9', 'sad:0.70,danceable:0.50'),
            ('happy:0.9', 'danceable:0.80,sad:0.50'),
        )

        metrics = dash._collect_charts_metrics(_cursor_with(mood_rows=rows))

        assert metrics['moods_coverage'][0] == {'label': 'sad', 'count': 2}


class TestSnapshotContract:
    def test_no_tautological_musicnn_percentage(self, monkeypatch):
        monkeypatch.setattr(dash, '_collect_music_server_metrics',
                            lambda cur, total_songs=None: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert 'musicnn_indexed' not in metrics

    def test_per_server_block_rides_in_the_snapshot(self, monkeypatch):
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)
        monkeypatch.setattr(
            dash, '_collect_music_server_metrics',
            lambda cur, total_songs=None: [
                {'name': 'Jellyfin', 'unique_songs': 5, 'resolved': 5}],
        )

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert metrics['music_servers'][0]['name'] == 'Jellyfin'

    def test_summary_never_recomputes_the_heavy_aggregates(self):
        import inspect

        src = inspect.getsource(dash.dashboard_summary)
        assert '_collect_fast_metrics' not in src
        assert '_collect_charts_metrics' not in src
        assert '_collect_music_server_metrics' not in src
        assert 'FROM score' not in src


class TestOrphanAndOverlapRows:
    @staticmethod
    def _cursor(server_rows, distinct_mapped):
        cur = MagicMock()
        cur.fetchall.return_value = server_rows
        cur.fetchone.return_value = (distinct_mapped,)
        return cur

    _TWO_SERVERS = [
        ('s1', 'Jellyfin', 'jellyfin', True, 183732, 181366),
        ('s2', 'Plex', 'plex', False, 2212, 2166),
    ]

    def test_unbound_songs_become_a_trailing_orphan_row(self, monkeypatch):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = self._cursor(self._TWO_SERVERS, 183380)

        servers = dash._collect_music_server_metrics(cur, total_songs=186135)

        orphan = servers[-1]
        assert orphan['name'] == 'Orphan'
        assert orphan['is_orphan'] is True
        assert orphan['unique_songs'] == 2755
        assert orphan['duplicate_copies'] == 0

    def test_shared_songs_become_a_negative_overlap_row(self, monkeypatch):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = self._cursor(self._TWO_SERVERS, 183380)

        servers = dash._collect_music_server_metrics(cur, total_songs=186135)

        overlap = [s for s in servers if s.get('is_overlap')]
        assert len(overlap) == 1
        assert overlap[0]['name'] == 'On multiple servers'
        assert overlap[0]['unique_songs'] == -152

    def test_unique_column_sums_to_the_catalogue_total(self, monkeypatch):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = self._cursor(self._TWO_SERVERS, 183380)

        servers = dash._collect_music_server_metrics(cur, total_songs=186135)

        assert sum(s['unique_songs'] for s in servers) == 186135

    def test_orphan_derived_by_arithmetic_never_scans_score(self, monkeypatch):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = self._cursor(self._TWO_SERVERS, 183380)

        dash._collect_music_server_metrics(cur, total_songs=186135)

        executed = " ".join(str(call.args[0]) for call in cur.execute.call_args_list)
        assert "FROM score" not in executed

    def test_total_songs_unavailable_withholds_only_the_orphan_row(self, monkeypatch):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = self._cursor(self._TWO_SERVERS, 183380)

        servers = dash._collect_music_server_metrics(cur, total_songs=None)

        assert [s['name'] for s in servers] == ['Jellyfin', 'Plex', 'On multiple servers']
        assert servers[-1]['unique_songs'] == -152
        assert cur.connection.rollback.call_count == 0

    def test_no_synthetic_rows_when_disjoint_and_fully_bound(self, monkeypatch):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = self._cursor([('s1', 'Jellyfin', 'jellyfin', True, 100, 100)], 100)

        servers = dash._collect_music_server_metrics(cur, total_songs=100)

        assert [s['name'] for s in servers] == ['Jellyfin']
        assert all(not s.get('is_orphan') and not s.get('is_overlap') for s in servers)


class TestTemplateCannotRoundUpToOneHundred:
    """The false 100% lived in the template, so guard it there. There is no JS
    runner in this repo, so this asserts the constructs that caused it are gone
    rather than executing the formatter."""

    @staticmethod
    def _script():
        html = Path(__file__).resolve().parents[2] / 'templates' / 'dashboard.html'
        return html.read_text(encoding='utf-8')

    def test_no_round_half_up_on_a_percentage(self):
        src = self._script()
        assert not re.search(r'Math\.round\s*\(\s*100', src)
        assert not re.search(r'Math\.min\s*\(\s*100\s*,', src)
        assert 'clampPct' not in src

    def test_the_shared_floor_formatter_is_present(self):
        src = self._script()
        assert 'function ratio(' in src
        assert 'function slicePct(' in src
        assert 'Math.floor(1000' in src

    def test_dashboard_is_not_server_scoped_by_the_picker(self):
        """The picker appended ?server= to a dashboard endpoint that ignores it,
        so switching servers silently changed nothing."""
        js = (Path(__file__).resolve().parents[2]
              / 'static' / 'server_selector.js').read_text(encoding='utf-8')
        assert "'/api/dashboard/'" in js


def _history_row(task_id, task_type, status, minute, **extra):
    row = {
        'task_id': task_id, 'task_type': task_type, 'status': status,
        'duration_seconds': 2.5, 'note': f'{task_id} note',
        'recorded_at': datetime(2026, 10, 12, 10, minute),
    }
    row.update(extra)
    return row


class TestRecentTasksShowOneLinePerTaskType:
    """A plugin task scheduled every minute filled the old 10-row list in 10
    minutes and pushed AudioMuse's own last analysis, clustering or cleaning
    run out of sight (#944). One line per task type keeps every type visible,
    with or without plugins. The row-level cases run against a real Postgres in
    test/integration/test_task_history_retention_integration.py."""

    @staticmethod
    def _collect(monkeypatch, rows=()):
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        cur = MagicMock()
        cur.fetchall.return_value = list(rows)
        recent = dash._collect_task_metrics(cur)
        sql, params = cur.execute.call_args.args
        return ' '.join(sql.split()), params, recent

    def test_each_task_type_gets_one_line_with_its_last_run(self, monkeypatch):
        sql, _params, _recent = self._collect(monkeypatch)

        assert (
            'SELECT DISTINCT ON (task_type) * FROM shown '
            'ORDER BY task_type, recorded_at DESC, id DESC'
        ) in sql
        assert sql.endswith('ORDER BY l.recorded_at DESC, l.id DESC'), (
            'the type that ran last comes first'
        )
        assert "task_type IS NOT NULL AND task_type <> '' AND task_type <> 'unknown'" in sql

    def test_the_last_failure_is_a_failed_run_never_a_cancelled_one(self, monkeypatch):
        sql, params, _recent = self._collect(monkeypatch)

        assert 'WHERE status = %s' in sql
        assert params == (config.TASK_STATUS_FAILURE,)
        assert config.TASK_STATUS_REVOKED not in params, (
            'cancelling a task is not an error: the task was cancelled successfully'
        )

    def test_a_line_carries_its_last_run_its_last_failure_and_its_run_count(self, monkeypatch):
        rows = [
            _history_row(
                'plugin-9', 'plugin.demo.sync', 'SUCCESS', 59, runs=100,
                failure_task_id='plugin-4', failure_note='Timed out.',
                failure_recorded_at=datetime(2026, 10, 12, 10, 54),
            ),
            _history_row(
                'analysis-1', 'main_analysis', 'REVOKED', 2, runs=3,
                failure_task_id=None, failure_note=None, failure_recorded_at=None,
            ),
        ]

        _sql, _params, recent = self._collect(monkeypatch, rows)

        assert [line['task_type'] for line in recent] == ['plugin.demo.sync', 'main_analysis']
        plugin, analysis = recent
        assert plugin['task_id'] == 'plugin-9'
        assert plugin['status'] == 'SUCCESS'
        assert plugin['runs'] == 100
        assert plugin['timestamp'] == dash.to_local_str(datetime(2026, 10, 12, 10, 59))
        assert plugin['last_failure'] == {
            'task_id': 'plugin-4', 'note': 'Timed out.',
            'timestamp': dash.to_local_str(datetime(2026, 10, 12, 10, 54)),
        }
        assert analysis['last_failure'] is None
        assert analysis['status'] == 'REVOKED'
        assert analysis['runs'] == 3


class TestTheTaskHistoryEndpoint:
    @pytest.fixture
    def client(self):
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(dash.dashboard_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @staticmethod
    def _mock_db(monkeypatch, rows):
        cur = MagicMock()
        cur.fetchall.return_value = rows
        cur.fetchone.return_value = (True,)
        conn = MagicMock()
        conn.cursor.return_value = cur
        monkeypatch.setattr(dash, 'get_db', lambda: conn)
        return cur

    def test_a_task_type_is_required(self, client, monkeypatch):
        self._mock_db(monkeypatch, [])

        resp = client.get('/api/dashboard/task_history')

        assert resp.status_code == 400
        assert resp.get_json()['error_code'] == ERR_INVALID_REQUEST

    def test_it_lists_one_types_runs_newest_first_up_to_the_per_type_cap(
            self, client, monkeypatch):
        monkeypatch.setattr(config, 'TASK_HISTORY_MAX_ROWS_PER_TYPE', 25)
        cur = self._mock_db(monkeypatch, [
            _history_row('analysis-2', 'main_analysis', 'FAIL', 30),
            _history_row('analysis-1', 'main_analysis', 'SUCCESS', 1),
        ])

        resp = client.get('/api/dashboard/task_history?task_type=main_analysis')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['task_type'] == 'main_analysis'
        assert [run['task_id'] for run in data['runs']] == ['analysis-2', 'analysis-1']
        assert data['runs'][0]['status'] == 'FAIL'
        assert data['runs'][0]['note'] == 'analysis-2 note'
        sql, params = cur.execute.call_args.args
        sql = ' '.join(sql.split())
        assert 'WHERE task_type = %s ORDER BY recorded_at DESC, id DESC LIMIT %s' in sql
        assert params == ('main_analysis', 25)
