# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Dashboard snapshot contract: what the hourly stats blob may and may not say.

Guards the two normalization rules the dashboard is built on: every number is
either CATALOG or per-SERVER, and a percentage may only reach 100% when the work
behind it is genuinely complete.

Main Features:
* A failed count is never published as a real zero (it would show "0 analyzed"
  for a full hour)
* The tautological musicnn "analyzed %" stays out of the payload
* The per-server block travels inside the hourly snapshot, not the request path
* The template cannot regress to a percentage that rounds up into a false 100%
* The refresh cadence drops to 5 minutes while a server's library size is
  still unmeasured, and stays hourly otherwise (or when the probe fails)
* The scheduled refresh counts each server's library itself (skipping while
  nothing is analyzed, and isolating per-server fetch failures)
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

from flask import Flask

import app_dashboard as dash


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


class TestServerTrackCountRefresh:
    def _wire(self, monkeypatch, analyzed, servers, catalogues, table_exists=True):
        from tasks import multiserver_sync
        from tasks.mediaserver import registry

        cur = MagicMock()
        cur.fetchone.return_value = (analyzed,)
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: table_exists)
        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: servers)
        fetched = []

        def fake_fetch(server):
            fetched.append(server['server_id'])
            result = catalogues[server['server_id']]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(multiserver_sync, 'fetch_server_catalogue', fake_fetch)
        stored = []
        monkeypatch.setattr(
            multiserver_sync, '_store_server_track_count',
            lambda db, server_id, count: stored.append((server_id, count)),
        )
        monkeypatch.setattr(dash, 'get_db', lambda: MagicMock())
        return cur, fetched, stored

    def test_each_server_library_is_counted_and_stored(self, monkeypatch):
        cur, fetched, stored = self._wire(
            monkeypatch, analyzed=39,
            servers=[{'server_id': 's1', 'name': 'One'}, {'server_id': 's2', 'name': 'Two'}],
            catalogues={'s1': [{'id': 'a'}, {'id': 'b'}], 's2': [{'id': 'n'}]},
        )
        dash._refresh_server_track_counts(cur)
        assert fetched == ['s1', 's2']
        assert stored == [('s1', 2), ('s2', 1)]

    def test_empty_catalogue_skips_all_provider_fetches(self, monkeypatch):
        cur, fetched, stored = self._wire(
            monkeypatch, analyzed=0,
            servers=[{'server_id': 's1', 'name': 'One'}],
            catalogues={'s1': [{'id': 'a'}]},
        )
        dash._refresh_server_track_counts(cur)
        assert fetched == []
        assert stored == []

    def test_one_failing_server_does_not_block_the_others(self, monkeypatch):
        cur, fetched, stored = self._wire(
            monkeypatch, analyzed=39,
            servers=[{'server_id': 's1', 'name': 'One'}, {'server_id': 's2', 'name': 'Two'}],
            catalogues={'s1': RuntimeError('server down'), 's2': [{'id': 'n'}]},
        )
        dash._refresh_server_track_counts(cur)
        assert stored == [('s2', 1)]

    def test_missing_registry_table_is_a_noop(self, monkeypatch):
        cur, fetched, stored = self._wire(
            monkeypatch, analyzed=39,
            servers=[{'server_id': 's1', 'name': 'One'}],
            catalogues={'s1': [{'id': 'a'}]},
            table_exists=False,
        )
        dash._refresh_server_track_counts(cur)
        assert fetched == []
        assert stored == []


class TestRefreshInterval:
    def _db_with_unmeasured(self, monkeypatch, unmeasured, table_exists=True):
        cur = MagicMock()
        cur.fetchone.return_value = (unmeasured,)
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(dash, 'get_db', lambda: db)
        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: table_exists)

    def test_unmeasured_server_speeds_up_refresh_to_five_minutes(self, monkeypatch):
        self._db_with_unmeasured(monkeypatch, 2)
        assert dash.dashboard_refresh_interval(Flask('t')) == 300

    def test_all_measured_servers_keep_hourly_refresh(self, monkeypatch):
        self._db_with_unmeasured(monkeypatch, 0)
        assert dash.dashboard_refresh_interval(Flask('t')) == 3600

    def test_missing_registry_table_keeps_hourly_refresh(self, monkeypatch):
        self._db_with_unmeasured(monkeypatch, 5, table_exists=False)
        assert dash.dashboard_refresh_interval(Flask('t')) == 3600

    def test_probe_failure_falls_back_to_hourly(self, monkeypatch):
        def boom():
            raise RuntimeError('db down')

        monkeypatch.setattr(dash, 'get_db', boom)
        assert dash.dashboard_refresh_interval(Flask('t')) == 3600


class TestSnapshotCompleteness:
    def test_a_failed_count_blocks_the_whole_snapshot(self, monkeypatch):
        """A transient DB error must not be published as a real 0. The old code
        used a 0-on-failure count for clap, so one blip pinned "CLAP: 0 (0.0%)"
        on screen until the next hourly refresh an hour later."""
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None:
                            None if 'clap_embedding' in sql else 10)

        metrics = dash._collect_content_metrics(_cursor_with())

        assert metrics['clap_indexed'] is None
        assert metrics['_complete'] is False

    def test_a_complete_snapshot_is_publishable(self, monkeypatch):
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_content_metrics(_cursor_with())

        assert metrics['_complete'] is True


class TestSnapshotContract:
    def test_no_tautological_musicnn_percentage(self, monkeypatch):
        """A song only enters `score` when it is analyzed, and its embedding row
        is written in the same transaction, so musicnn/total is ~100% by
        construction. Publishing it invited a permanent, meaningless 100%."""
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_content_metrics(_cursor_with())

        assert 'musicnn_indexed' not in metrics

    def test_per_server_block_rides_in_the_hourly_snapshot(self, monkeypatch):
        """The per-server counts are a big query (GROUP BY over track_server_map
        plus an anti-join over score). They belong to the snapshot tier, never to
        the 30s request path."""
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)
        monkeypatch.setattr(
            dash, '_collect_music_server_metrics',
            lambda cur: [{'name': 'Jellyfin', 'server_songs': None, 'resolved': 5}],
        )

        metrics = dash._collect_content_metrics(_cursor_with())

        assert metrics['music_servers'][0]['name'] == 'Jellyfin'

    def test_summary_never_recomputes_the_heavy_aggregates(self):
        """dashboard_summary must not reach for a scan of `score`. It reads the
        precomputed snapshot and three cheap tables, nothing else."""
        import inspect

        src = inspect.getsource(dash.dashboard_summary)
        assert '_collect_content_metrics' not in src
        assert '_collect_music_server_metrics' not in src
        assert 'FROM score' not in src


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
        # Math.round(100 * x / y) printed "100%" for anything >= 99.5%.
        assert not re.search(r'Math\.round\s*\(\s*100', src)
        # Math.min(100, ...) hid a genuine overshoot instead of surfacing it.
        assert not re.search(r'Math\.min\s*\(\s*100\s*,', src)
        # clampPct was the toFixed(1) helper that turned 99.97% into "100.0%".
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
