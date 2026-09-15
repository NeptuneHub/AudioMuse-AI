# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-model index coverage for the setup wizard: bands and sources.

The Machine Learning Models section shows, under each model, how much of the
library that model's index can find. The number itself is never shown: a
200k-song library missing a hundred songs reads as complete, because a 99.9%
figure only sends users hunting for the rounding. These tests pin the band
edges and the cheap sources the wizard reads on every open: one directory
header substring per index, the loaded neural pack, and two indexed counts.

Main Features:
* A library missing a handful of songs out of 200k is still the top band
* The bands split at twenty, sixty, eighty and ninety-five percent; nothing
  indexed is band 0
* An index larger than the catalogue after a cleaning stays in the top band
* The lyrics denominator is the songs that have lyrics, not the whole catalogue
* An unloaded neural pack counts as nothing searchable, never as a failure
* A failed header read leaves that model out; a failed count hides every bar
* The paged IVF item count reads one header substring of a single or segmented
  directory; a missing or foreign directory counts as zero, a failed read as unknown
* The wizard API hands out bands only, never the counts behind them, and
  nothing of this touches the dashboard or a background thread
"""

import os
from unittest.mock import MagicMock

import numpy as np

import app_setup
import config
import database
from tasks import neural_fingerprint_index as nfi
from tasks import paged_ivf

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), encoding='utf-8') as handle:
        return handle.read()


def _db_with_counts(total, with_lyrics, fail=False):
    db = MagicMock()
    cur = MagicMock()
    db.cursor.return_value.__enter__.return_value = cur
    answers = {}

    def execute(sql, params=None):
        if fail:
            raise RuntimeError('count failed')
        answers['last'] = with_lyrics if 'lyrics_embedding' in sql else total

    cur.execute.side_effect = execute
    cur.fetchone.side_effect = lambda: (answers['last'],)
    return db


def _indexed(per_index):
    def fake(conn, index_name):
        return per_index[index_name]
    return fake


def _header_conn(header_bytes, raises=False):
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    if raises:
        cur.execute.side_effect = RuntimeError('boom')
    cur.fetchone.return_value = None if header_bytes is None else (memoryview(header_bytes),)
    return conn, cur


def test_a_library_missing_a_handful_of_songs_out_of_200k_is_still_the_top_band():
    assert app_setup.model_coverage_level(199_900, 200_000) == 5


def test_the_bands_split_at_twenty_sixty_eighty_and_ninety_five_percent():
    assert app_setup.model_coverage_level(1, 100) == 1
    assert app_setup.model_coverage_level(19, 100) == 1
    assert app_setup.model_coverage_level(20, 100) == 2
    assert app_setup.model_coverage_level(59, 100) == 2
    assert app_setup.model_coverage_level(60, 100) == 3
    assert app_setup.model_coverage_level(79, 100) == 3
    assert app_setup.model_coverage_level(80, 100) == 4
    assert app_setup.model_coverage_level(94, 100) == 4
    assert app_setup.model_coverage_level(95, 100) == 5
    assert app_setup.model_coverage_level(100, 100) == 5
    assert app_setup.MODEL_COVERAGE_BANDS == (0.2, 0.6, 0.8, 0.95)


def test_nothing_indexed_or_an_empty_catalogue_is_the_empty_band():
    assert app_setup.model_coverage_level(0, 100) == 0
    assert app_setup.model_coverage_level(None, 100) == 0
    assert app_setup.model_coverage_level(10, 0) == 0
    assert app_setup.model_coverage_level(10, None) == 0


def test_an_index_larger_than_the_catalogue_after_a_cleaning_stays_in_the_top_band():
    assert app_setup.model_coverage_level(110, 100) == 5


def test_the_lyrics_denominator_is_the_songs_with_lyrics_not_the_whole_catalogue(monkeypatch):
    monkeypatch.setattr(database, 'get_db', lambda: _db_with_counts(total=1000, with_lyrics=400))
    monkeypatch.setattr(paged_ivf, 'paged_ivf_item_count', _indexed({
        config.INDEX_NAME: 1000, 'clap_index': 450, 'lyrics_index': 380,
    }))
    monkeypatch.setattr(nfi, 'indexed_track_count', lambda: 1000)

    levels = app_setup.model_coverage_levels()

    assert levels == {'musicnn': 5, 'clap': 2, 'lyrics': 5, 'neural-fingerprint': 5}


def test_an_unloaded_neural_pack_counts_as_nothing_searchable_never_as_a_failure(monkeypatch):
    monkeypatch.setattr(database, 'get_db', lambda: _db_with_counts(total=1000, with_lyrics=400))
    monkeypatch.setattr(paged_ivf, 'paged_ivf_item_count', lambda conn, name: 1000)
    monkeypatch.setattr(nfi, 'indexed_track_count', lambda: None)

    levels = app_setup.model_coverage_levels()

    assert levels['neural-fingerprint'] == 0
    assert levels['musicnn'] == 5


def test_a_failed_header_read_leaves_that_model_out_and_the_others_in(monkeypatch):
    monkeypatch.setattr(database, 'get_db', lambda: _db_with_counts(total=1000, with_lyrics=400))
    monkeypatch.setattr(
        paged_ivf, 'paged_ivf_item_count',
        lambda conn, name: None if name == 'clap_index' else 1000,
    )
    monkeypatch.setattr(nfi, 'indexed_track_count', lambda: 1000)

    levels = app_setup.model_coverage_levels()

    assert 'clap' not in levels
    assert levels['musicnn'] == 5


def test_a_failed_count_or_a_missing_database_hides_every_bar(monkeypatch):
    monkeypatch.setattr(database, 'get_db', lambda: _db_with_counts(1, 1, fail=True))
    monkeypatch.setattr(paged_ivf, 'paged_ivf_item_count', lambda conn, name: 1000)
    monkeypatch.setattr(nfi, 'indexed_track_count', lambda: 1000)
    assert app_setup.model_coverage_levels() == {}

    def _no_database():
        raise RuntimeError('no database')
    monkeypatch.setattr(database, 'get_db', _no_database)
    assert app_setup.model_coverage_levels() == {}


def test_the_wizard_api_hands_out_bands_only_never_the_counts(monkeypatch):
    monkeypatch.setattr(database, 'get_db', lambda: _db_with_counts(total=200_000, with_lyrics=120_000))
    monkeypatch.setattr(paged_ivf, 'paged_ivf_item_count', lambda conn, name: 199_900)
    monkeypatch.setattr(nfi, 'indexed_track_count', lambda: 30_000)

    levels = app_setup.model_coverage_levels()

    assert set(levels) == set(app_setup.MODEL_COVERAGE_MODELS)
    assert all(isinstance(level, int) and 0 <= level <= 5 for level in levels.values())
    source = _read('app_setup.py')
    assert "'model_coverage': model_coverage_levels()" in source


def test_nothing_of_this_touches_the_dashboard_or_a_background_thread():
    assert 'model_coverage' not in _read('app_dashboard.py')
    assert 'model_coverage' not in _read('app.py')
    assert 'coverage' not in _read('app.py')


def test_the_paged_ivf_item_count_reads_one_header_substring_of_the_directory():
    n_items = 23
    blob = paged_ivf.pack_directory(
        np.zeros((3, 4), dtype=np.float32), np.zeros(n_items, dtype=np.uint32),
        ['item-%d' % n for n in range(n_items)], 4, 'angular',
    )
    conn, cur = _header_conn(blob[:paged_ivf._HEADER_SIZE])

    assert paged_ivf.paged_ivf_item_count(conn, 'music_library') == n_items
    sql, params = cur.execute.call_args[0]
    assert 'substring(blob_data from 1 for %d)' % paged_ivf._HEADER_SIZE in sql
    assert 'blob_data FROM' not in sql
    assert params == ('music_library__ivf_dir', r'music\_library\_\_ivf\_dir\_1\_%')


def test_a_missing_or_foreign_directory_counts_as_zero_and_a_failed_read_as_unknown():
    conn, _cur = _header_conn(None)
    assert paged_ivf.paged_ivf_item_count(conn, 'clap_index') == 0

    conn, _cur = _header_conn(b'NOPE' + bytes(paged_ivf._HEADER_SIZE - 4))
    assert paged_ivf.paged_ivf_item_count(conn, 'clap_index') == 0

    conn, _cur = _header_conn(b'AMIV')
    assert paged_ivf.paged_ivf_item_count(conn, 'clap_index') == 0

    conn, _cur = _header_conn(b'', raises=True)
    assert paged_ivf.paged_ivf_item_count(conn, 'clap_index') is None
    conn.rollback.assert_called_once()


def test_the_neural_count_is_the_loaded_packs_tracks_and_none_when_unloaded(monkeypatch):
    monkeypatch.setitem(nfi._STATE, 'pack', None)
    assert nfi.indexed_track_count() is None

    pack = MagicMock()
    pack.live_tracks = 42
    monkeypatch.setitem(nfi._STATE, 'pack', pack)
    assert nfi.indexed_track_count() == 42
