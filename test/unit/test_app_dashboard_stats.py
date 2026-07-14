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
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

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
