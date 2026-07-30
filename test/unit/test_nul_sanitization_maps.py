# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""NUL and control-character hardening on the server-map and provider-id write paths.

Verifies every Postgres bind site that carries media-server tag text applies the
shared ``sanitize_string_for_db`` transform before the value reaches psycopg2,
so one dirty tag can neither crash mogrify nor desync the sweep's staged ids
from the ids the registry stored sanitized.

Main Features:
* Artist map upserts strip NUL from names/ids and re-dedup names that collide after sanitizing
* Track map upserts sanitize ids and paths before the provider-id dedup key is built
* The sweep prune stages present ids with the identical transform the registry applies
* Sweep metadata staging keys rows by the sanitized id so collisions cannot violate the temp PK
* Migration id maps and chromaprint reads/writes sanitize provider ids the same way
"""

from unittest.mock import MagicMock, patch

from sanitization import sanitize_string_for_db


def _mock_db():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


class TestUpsertArtistMapsSanitization:
    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_strips_nul_from_name_and_id(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_artist_maps(
            'srv1', {'Artist\x00Name': 'id\x00123'}, conn=conn
        )

        assert written == 1
        rows = mock_ev.call_args_list[0][0][2]
        assert rows == [('ArtistName', 'srv1', 'id123')]

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_strips_control_chars_like_score_author(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        registry.upsert_artist_maps('srv1', {'Art\x01ist\x1f': 'id1'}, conn=conn)

        rows = mock_ev.call_args_list[0][0][2]
        assert rows[0][0] == 'Artist'

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_dedups_names_that_collide_after_sanitize(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_artist_maps(
            'srv1', {'AB': 'id1', 'A\x00B': 'id2'}, conn=conn
        )

        assert written == 1
        rows = mock_ev.call_args_list[0][0][2]
        assert [r[0] for r in rows] == ['AB']

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_drops_entries_empty_after_sanitize(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_artist_maps('srv1', {'\x00': 'id1'}, conn=conn)

        assert written == 0
        assert not mock_ev.called


class TestUpsertTrackMapsSanitization:
    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_track_maps_sanitizes_ids_and_paths(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_track_maps(
            'srv1',
            {'prov\x00id': ('fp_item\x00', 'tier1', '/music/a\x00.flac')},
            conn=conn,
        )

        assert written == 1
        rows = mock_ev.call_args_list[0][0][2]
        assert rows == [('fp_item', 'srv1', 'provid', 'tier1', '/music/a.flac')]

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_track_maps_dedups_ids_that_collide_after_sanitize(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_track_maps(
            'srv1',
            {'id1': ('fp_a', None, None), 'id\x001': ('fp_b', None, None)},
            conn=conn,
        )

        assert written == 1
        rows = mock_ev.call_args_list[0][0][2]
        assert [r[2] for r in rows] == ['id1']

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_track_maps_drops_rows_whose_id_is_empty_after_sanitize(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_track_maps(
            'srv1', {'\x00': ('fp_a', None, None)}, conn=conn
        )

        assert written == 0
        assert not mock_ev.called


class TestSweepUsesSameTransformAsRegistry:
    def test_strip_nul_delegates_to_shared_sanitizer(self):
        from tasks import multiserver_sync

        dirty = 'a\x00b\x01c'
        assert multiserver_sync._strip_nul(dirty) == sanitize_string_for_db(dirty)
        assert multiserver_sync._strip_nul(dirty) == 'abc'
        assert multiserver_sync._strip_nul(7) == 7
        assert multiserver_sync._strip_nul(None) is None

    @patch('tasks.multiserver_sync.execute_values')
    def test_prune_present_ids_use_same_transform_as_upsert(self, mock_ev):
        from tasks import multiserver_sync

        conn, cur = _mock_db()
        cur.fetchone.return_value = (0,)
        cur.rowcount = 0
        dirty = 'prov\x00id\x01x'
        multiserver_sync.prune_stale_mappings(conn, 'srv1', [dirty, 'clean'])

        staged = mock_ev.call_args_list[0][0][2]
        assert staged == [(sanitize_string_for_db(dirty),), ('clean',)]

    @patch('tasks.multiserver_sync.execute_values')
    def test_stage_track_metadata_dedups_ids_that_collide_after_sanitize(self, mock_ev):
        from tasks import multiserver_sync

        conn, _ = _mock_db()
        tracks = [
            {'id': 'id1', 'album': 'Album\x00One', 'album_artist': 'AA',
             'year': 2020, 'rating': 5, 'path': '/p\x001'},
            {'id': 'id\x001', 'album': 'Album Two', 'album_artist': 'BB',
             'year': 2021, 'rating': 4, 'path': '/p2'},
        ]
        multiserver_sync._stage_track_metadata(conn, tracks)

        staged = mock_ev.call_args_list[0][0][2]
        assert staged == [('id1', 'Album Two', 'BB', 2021, 4, '/p2')]


class TestMigrationMapSanitization:
    def test_migration_map_sanitizes_ids_like_new_meta(self):
        from tasks.provider_migration_tasks import _populate_migration_map_table

        cur = MagicMock()
        _populate_migration_map_table(cur, {'old\x00id': 'new\x00id'})

        insert_call = cur.execute.call_args_list[1]
        assert insert_call[0][1] == ['oldid', 'newid']

    def test_cleaned_libraries_drops_names_empty_after_sanitize(self):
        from tasks.provider_migration_tasks import _cleaned_libraries_value

        assert _cleaned_libraries_value(['Mus\x00ic', '\x00', 'a,b']) == 'Music'


class TestChromaprintIdSanitization:
    @patch('database.get_db')
    def test_persist_chromaprint_strips_nul_from_provider_track_id(self, mock_get_db):
        from database import persist_chromaprint

        conn, cur = _mock_db()
        mock_get_db.return_value = conn
        persist_chromaprint('srv1', 'prov\x00id', b'\x01\x02')

        params = cur.execute.call_args_list[0][0][1]
        assert params[1] == 'provid'

    @patch('database.get_db')
    def test_get_chromaprint_looks_up_with_sanitized_id(self, mock_get_db):
        from database import get_chromaprint

        conn, cur = _mock_db()
        cur.fetchone.return_value = None
        mock_get_db.return_value = conn
        get_chromaprint('srv1', 'prov\x00id')

        params = cur.execute.call_args_list[0][0][1]
        assert params[1] == 'provid'
