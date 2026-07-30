# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""NUL and control-character hardening across the provider-id write AND read domains.

Verifies that every Postgres bind site carrying media-server tag text applies the
shared ``sanitize_string_for_db`` transform, and that every compare/read site that
probes those columns (translation, work map, duration dicts) sanitizes its probe
the same way, so stored and fetched ids always live in one domain. DB-sourced
keys (score item_id, migration old_id) stay raw: they were stored sanitized or
predate the transform, and mutating them would break FK joins.

Main Features:
* Artist map upserts strip NUL/C0 from names and ids with a deterministic collision tiebreak
* Track map upserts sanitize provider ids and paths but keep the DB-sourced item_id raw
* reverse_translate_ids and artist lookups bind sanitized ids and key results by caller input
* The sweep prune stages a deduped post-sanitize present set; metadata staging keys clean ids
* Analysis id seams, migration id maps, duration dicts and chromaprint use the same transform
* The one-time init_db scrub targets exactly the C0 class and is gated by an app_config marker
"""

import os
import re
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import make_mock_connection
from sanitization import sanitize_string_for_db


def _mock_db():
    cur = MagicMock()
    return make_mock_connection(cur), cur


class TestSanitizeStringForDb:
    def test_single_pass_matches_two_pass_reference(self):
        probe = ''.join(chr(c) for c in range(0x00, 0x20)) + '\x7f caf\xe9 \r\n\t'
        reference = re.sub(
            r'[\x01-\x08\x0B-\x0C\x0E-\x1F]', '', probe.replace('\x00', '')
        )
        assert sanitize_string_for_db(probe) == reference

    def test_loud_variant_warns_only_when_value_changed(self):
        import sanitization

        with patch.object(sanitization.logger, 'warning') as warn:
            assert sanitization.sanitize_string_for_db_loud('a\x01b', 'field') == 'ab'
            assert warn.call_count == 1
            assert sanitization.sanitize_string_for_db_loud('clean', 'field') == 'clean'
            assert warn.call_count == 1


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
    def test_upsert_artist_maps_strips_c0_controls_but_keeps_del_and_cr(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        registry.upsert_artist_maps(
            'srv1', {'Art\x01ist\x1f': 'id1', 'Keep\x7f\rMe': 'id2'}, conn=conn
        )

        names = {r[0] for r in mock_ev.call_args_list[0][0][2]}
        assert names == {'Artist', 'Keep\x7f\rMe'}

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_collision_prefers_already_clean_name(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_artist_maps(
            'srv1', {'A\x00B': 'id_dirty', 'AB': 'id_clean'}, conn=conn
        )

        assert written == 1
        rows = mock_ev.call_args_list[0][0][2]
        assert rows == [('AB', 'srv1', 'id_clean')]

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_collision_tiebreak_is_input_order_independent(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        registry.upsert_artist_maps(
            'srv1', {'AB': 'id_clean', 'A\x00B': 'id_dirty'}, conn=conn
        )

        rows = mock_ev.call_args_list[0][0][2]
        assert rows == [('AB', 'srv1', 'id_clean')]

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_artist_maps_drops_entries_empty_after_sanitize(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_artist_maps('srv1', {'\x00': 'id1'}, conn=conn)

        assert written == 0
        assert not mock_ev.called


class TestUpsertTrackMapsSanitization:
    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_track_maps_sanitizes_provider_id_and_path(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        written = registry.upsert_track_maps(
            'srv1',
            {'prov\x00id': ('fp_item', 'tier1', '/music/a\x00.flac')},
            conn=conn,
        )

        assert written == 1
        rows = mock_ev.call_args_list[0][0][2]
        assert rows == [('fp_item', 'srv1', 'provid', 'tier1', '/music/a.flac')]

    @patch('tasks.mediaserver.registry.execute_values')
    def test_upsert_track_maps_keeps_db_sourced_item_id_raw(self, mock_ev):
        from tasks.mediaserver import registry

        conn, _ = _mock_db()
        registry.upsert_track_maps(
            'srv1', {'provid': ('fp_leg\x01acy', None, None)}, conn=conn
        )

        rows = mock_ev.call_args_list[0][0][2]
        assert rows[0][0] == 'fp_leg\x01acy'

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


class TestTranslationReadSitesSanitize:
    @patch('tasks.mediaserver.registry.get_default_server')
    def test_reverse_translate_ids_binds_sanitized_ids_and_keys_by_input(self, mock_default):
        from tasks.mediaserver import registry

        mock_default.return_value = {'server_id': 'srv1'}
        conn, cur = _mock_db()
        cur.fetchall.return_value = [('provid', 'fp_1')]

        result = registry.reverse_translate_ids(
            ['prov\x01id', 'unknown'], server_id='srv1', conn=conn
        )

        bound = cur.execute.call_args_list[0][0][1]
        assert bound == ('srv1', ['provid', 'unknown'])
        assert result == {'prov\x01id': 'fp_1', 'unknown': 'unknown'}

    @patch('tasks.mediaserver.registry.get_default_server_id')
    def test_artist_lookup_matches_sanitized_rows_keyed_by_caller_input(self, mock_default_id):
        from tasks.mediaserver import registry

        mock_default_id.return_value = 'srv1'
        conn, cur = _mock_db()
        cur.fetchall.return_value = [('artid1', 'AC/DC')]

        result = registry.artist_names_for_ids(['art\x01id1'], conn=conn)

        bound = cur.execute.call_args_list[0][0][1]
        assert bound == ('srv1', ['artid1'])
        assert result == {'art\x01id1': 'AC/DC'}


class TestSweepUsesSameTransformAsRegistry:
    def test_strip_nul_delegates_to_shared_sanitizer(self):
        from tasks import multiserver_sync

        dirty = 'a\x00b\x01c'
        assert multiserver_sync._strip_nul(dirty) == sanitize_string_for_db(dirty)
        assert multiserver_sync._strip_nul(dirty) == 'abc'
        assert multiserver_sync._strip_nul(7) == 7
        assert multiserver_sync._strip_nul(None) is None

    @patch('tasks.multiserver_sync.execute_values')
    def test_prune_present_ids_dedup_post_sanitize_and_drop_empty(self, mock_ev):
        from tasks import multiserver_sync

        conn, cur = _mock_db()
        cur.fetchone.return_value = (0,)
        cur.rowcount = 0
        multiserver_sync.prune_stale_mappings(
            conn, 'srv1', ['prov\x00id', 'provid', '\x01', 'clean']
        )

        staged = mock_ev.call_args_list[0][0][2]
        assert sorted(staged) == [('clean',), ('provid',)]

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


class TestAnalysisIdSeamsSanitize:
    def test_provider_item_id_sanitizes_control_chars(self):
        from tasks.analysis.song import provider_item_id

        assert provider_item_id({'Id': 'prov\x01id'}) == 'provid'
        assert provider_item_id({'id': 'prov\x00id'}) == 'provid'

    def test_catalog_item_id_sanitizes_fallback_provider_id(self):
        from tasks.analysis.song import catalog_item_id

        assert catalog_item_id({'_catalog_item_id': 'fp_1'}) == 'fp_1'
        assert catalog_item_id({'Id': 'prov\x01id'}) == 'provid'

    def test_str_ids_sanitizes_before_db_bind(self):
        from tasks.analysis.helper import _str_ids

        assert _str_ids(['a\x00b', 'c\x1fd', 'clean']) == ['ab', 'cd', 'clean']


class TestDurationDictsShareTheIdDomain:
    @patch('tasks.duplicate_repair.provider_probe')
    @patch('tasks.duplicate_repair.ms_context')
    def test_server_durations_keys_are_sanitized(self, _ctx, probe):
        from tasks.duplicate_repair import _server_durations

        probe.fetch_all_tracks.return_value = [{'id': 'prov\x01id', 'duration': 100.0}]
        result = _server_durations({'server_type': 'jellyfin', 'creds': {}})

        assert result == {'provid': 100.0}

    def test_group_duration_probes_with_sanitized_provider_ids(self):
        from tasks.duplicate_repair import _group_duration

        assert _group_duration(['prov\x01id'], {'provid': 42.0}) == 42.0

    def test_migration_duration_lookup_matches_sanitized_ids(self):
        from tasks.fingerprint_canonicalize import _durations_for_rows

        cur = MagicMock()
        cur.fetchall.side_effect = [[], [('fp_1', 'prov\x01id')]]
        durations = _durations_for_rows(cur, ['fp_1'], [0], {'provid': 77.0}, 'srv1')

        assert durations == {'fp_1': 77.0}


class TestMigrationMapSanitization:
    def test_migration_map_keeps_db_sourced_old_id_raw_and_sanitizes_new_id(self):
        from tasks.provider_migration_tasks import _populate_migration_map_table

        cur = MagicMock()
        _populate_migration_map_table(cur, {'old\x01id': 'new\x01id'})

        insert_call = cur.execute.call_args_list[1]
        assert insert_call[0][1] == ['old\x01id', 'newid']

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


class TestLegacyMapIdScrub:
    def test_scrub_skips_when_marker_present(self):
        from database import _scrub_control_chars_from_map_ids

        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        _scrub_control_chars_from_map_ids(cur)

        assert cur.execute.call_count == 1

    def test_scrub_targets_exactly_the_c0_class_and_sets_marker(self):
        from database import _scrub_control_chars_from_map_ids, _MAP_ID_SCRUB_MARKER

        cur = MagicMock()
        cur.fetchone.return_value = None
        cur.rowcount = 0
        _scrub_control_chars_from_map_ids(cur)

        klass = cur.execute.call_args_list[1][0][1][0]
        assert klass.startswith('[') and klass.endswith(']')
        chars = set(klass[1:-1])
        assert chars == {
            chr(c) for c in (*range(0x01, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20))
        }
        last = cur.execute.call_args_list[-1]
        assert _MAP_ID_SCRUB_MARKER in last[0][1]
