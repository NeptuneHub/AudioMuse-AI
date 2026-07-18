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
* The FAST metrics refresh every 60s; the heavy mood scan stays hourly; neither
  walks a media server
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


class TestRefreshInterval:
    def test_fast_refresh_is_60s_with_no_db_probe(self, monkeypatch):
        # The fast tier is cheap enough to recompute every 60s, and it must not
        # touch the DB just to decide when to run again.
        monkeypatch.setattr(dash, 'get_db', lambda: (_ for _ in ()).throw(
            AssertionError('refresh interval must not query the DB')))
        assert dash.dashboard_refresh_interval(Flask('t')) == 60

    def test_mood_scan_stays_on_its_own_hourly_cadence(self):
        # The one heavy full-table scan runs far less often than the fast tier.
        assert dash.DASHBOARD_MOOD_REFRESH_INTERVAL_SECONDS == 3600
        assert (dash.DASHBOARD_MOOD_REFRESH_INTERVAL_SECONDS
                > dash.DASHBOARD_REFRESH_INTERVAL_SECONDS)


class TestSnapshotCompleteness:
    def test_a_failed_count_blocks_the_whole_snapshot(self, monkeypatch):
        # A transient DB error must not be published as a real 0. The old code
        # used a 0-on-failure count for clap, so one blip pinned "CLAP: 0 (0.0%)"
        # on screen until the next refresh.
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None:
                            None if 'clap_embedding' in sql else 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert metrics['clap_indexed'] is None
        assert metrics['_complete'] is False

    def test_a_complete_fast_block_is_publishable(self, monkeypatch):
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert metrics['_complete'] is True


class TestCadenceSplit:
    def test_fast_block_carries_no_mood_scan_keys(self, monkeypatch):
        # The Genres / Moods Coverage data is the ONE heavy full-table scan; it
        # must not ride in the 60s fast block.
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert 'top_genre' not in metrics
        assert 'moods_coverage' not in metrics
        # The cheap tempo profile DOES stay in the fast block.
        assert 'tempo_profile' in metrics

    def test_mood_block_carries_only_the_chart_keys(self):
        metrics = dash._collect_mood_metrics(_cursor_with())

        assert 'top_genre' in metrics
        assert 'moods_coverage' in metrics
        assert 'total_songs' not in metrics
        # 'happy' is the dominant label of the single mocked row.
        assert metrics['top_genre'][0]['label'] == 'happy'


class TestSnapshotContract:
    def test_no_tautological_musicnn_percentage(self, monkeypatch):
        # A song only enters `score` when it is analyzed, and its embedding row
        # is written in the same transaction, so musicnn/total is ~100% by
        # construction. Publishing it invited a permanent, meaningless 100%.
        monkeypatch.setattr(dash, '_collect_music_server_metrics', lambda cur: [])
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert 'musicnn_indexed' not in metrics

    def test_per_server_block_rides_in_the_snapshot(self, monkeypatch):
        # The per-server counts are a GROUP BY over track_server_map. They belong
        # to the snapshot tier, never to the 30s request path.
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 10)
        monkeypatch.setattr(
            dash, '_collect_music_server_metrics',
            lambda cur: [{'name': 'Jellyfin', 'unique_songs': 5, 'resolved': 5}],
        )

        metrics = dash._collect_fast_metrics(_cursor_with())

        assert metrics['music_servers'][0]['name'] == 'Jellyfin'

    def test_summary_never_recomputes_the_heavy_aggregates(self):
        # dashboard_summary must not reach for a scan of `score`. It reads the
        # precomputed snapshot and three cheap tables, nothing else.
        import inspect

        src = inspect.getsource(dash.dashboard_summary)
        assert '_collect_fast_metrics' not in src
        assert '_collect_mood_metrics' not in src
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
