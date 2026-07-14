# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multi-server registry, context, translation and matcher-tier behaviour.

Covers the concurrent-server support added on top of the historical
single-server design, asserting that an unset context reproduces the default
behaviour and that a bound server overrides credentials and library filters.

Main Features:
* Active-server context accessors, nesting, reset isolation, and the
  bound-server-base credential merge.
* Credential masking/merge, config-derived creds, and registry row normalization.
* Matcher tiers and safe canonical/provider id translation.
* Server-scope resolution (``servers_for_scope`` / ``has_secondary_servers``).
* BoundServer runs dispatcher calls inside the selected server's context.
* Sweep alignment: keyset pagination, per-server failure isolation, pruning
  and catalogue-cache bounds.
* Registry mutators roll back and re-raise on write failures; canonicalization
  restores session settings on caller-provided connections.
"""

import gc
import json
import logging
import weakref

import pytest
from unittest.mock import MagicMock


class TestServerContext:
    def test_unset_context_returns_defaults(self):
        from tasks.mediaserver import context

        assert context.active_type('jellyfin') == 'jellyfin'
        assert context.active_creds() is None
        assert context.active_creds({'url': 'x'}) == {'url': 'x'}
        assert context.active_libraries('libs') == 'libs'
        assert context.active_server_id() is None

    def test_use_server_overrides_then_restores(self):
        from tasks.mediaserver import context

        server = {
            'server_id': 's1',
            'server_type': 'plex',
            'creds': {'url': 'u', 'token': 't'},
            'music_libraries': 'onlythis',
        }
        with context.use_server(server):
            assert context.active_type('jellyfin') == 'plex'
            assert context.active_creds() == {'url': 'u', 'token': 't'}
            assert context.active_libraries('libs') == 'onlythis'
            assert context.active_server_id() == 's1'
        assert context.active_type('jellyfin') == 'jellyfin'
        assert context.active_creds() is None
        assert context.active_server_id() is None

    def test_nested_use_server_restores_inner(self):
        from tasks.mediaserver import context

        outer = {'server_id': 'a', 'server_type': 'navidrome', 'creds': {'url': 'a'}, 'music_libraries': ''}
        inner = {'server_id': 'b', 'server_type': 'plex', 'creds': {'url': 'b'}, 'music_libraries': ''}
        with context.use_server(outer):
            with context.use_server(inner):
                assert context.active_server_id() == 'b'
            assert context.active_server_id() == 'a'
        assert context.active_server_id() is None

    def test_use_server_none_falls_back_to_config(self):
        from tasks.mediaserver import context

        with context.use_server(None):
            assert context.active_type('jellyfin') == 'jellyfin'
            assert context.active_creds() is None

    def test_bound_server_creds_win_over_empty_caller_fields(self):
        from tasks.mediaserver import context

        server = {
            'server_id': 's1',
            'server_type': 'plex',
            'creds': {'url': 'http://secondary', 'token': 'stok'},
            'music_libraries': '',
        }
        with context.use_server(server):
            merged = context.active_creds({'url': '', 'token': 'caller-token'})
            assert merged == {'url': 'http://secondary', 'token': 'caller-token'}
            assert context.active_creds() == {'url': 'http://secondary', 'token': 'stok'}

    def test_active_creds_merge_matrix(self):
        from tasks.mediaserver import context

        server = {
            'server_id': 's1',
            'server_type': 'jellyfin',
            'creds': {'url': 'u', 'token': 't', 'user_id': 'id'},
            'music_libraries': '',
        }
        with context.use_server(server):
            assert context.active_creds({'token': 'T2'}) == {
                'url': 'u', 'token': 'T2', 'user_id': 'id'
            }
            assert context.active_creds({'url': '', 'token': '', 'user_id': ''}) == {
                'url': 'u', 'token': 't', 'user_id': 'id'
            }
            assert context.active_creds(None) == {'url': 'u', 'token': 't', 'user_id': 'id'}
        assert context.active_creds({'token': 'T2'}) == {'token': 'T2'}
        assert context.active_creds(None) is None


class TestCredHelpers:
    def test_mask_hides_secret_fields(self):
        import app_server_context as asc

        masked = asc.mask_creds({'url': 'http://x', 'user': 'me', 'token': 'secret', 'password': 'pw'})
        assert masked['url'] == 'http://x'
        assert masked['user'] == 'me'
        assert masked['token'] == asc.CRED_MASK
        assert masked['password'] == asc.CRED_MASK

    def test_mask_leaves_empty_secret_empty(self):
        import app_server_context as asc

        masked = asc.mask_creds({'url': 'http://x', 'token': ''})
        assert masked['token'] == ''

    def test_merge_preserves_masked_secret(self):
        import app_server_context as asc

        existing = {'url': 'http://old', 'token': 'realsecret'}
        incoming = {'url': 'http://new', 'token': asc.CRED_MASK}
        merged = asc.merge_creds(existing, incoming)
        assert merged['url'] == 'http://new'
        assert merged['token'] == 'realsecret'

    def test_merge_accepts_new_secret(self):
        import app_server_context as asc

        merged = asc.merge_creds({'token': 'old'}, {'token': 'brandnew'})
        assert merged['token'] == 'brandnew'


class TestRequestServerResolution:
    def test_reads_server_from_json_body_when_helper_gets_no_data(self, monkeypatch):
        from flask import Flask
        import app_server_context as context
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry,
            'get_server',
            lambda server_id, conn=None: {'server_id': server_id} if server_id == 'secondary' else None,
        )
        monkeypatch.setattr(registry, 'get_server_by_name', lambda name, conn=None: None)
        app = Flask(__name__)
        with app.test_request_context(
            '/search', method='POST', json={'server': 'secondary'}
        ):
            assert context.resolve_request_server_id() == 'secondary'


class TestRegistryPureHelpers:
    def test_creds_from_config_per_type(self, monkeypatch):
        import config
        from tasks.mediaserver import registry

        monkeypatch.setattr(config, 'NAVIDROME_URL', 'http://nd', raising=False)
        monkeypatch.setattr(config, 'NAVIDROME_USER', 'user1', raising=False)
        monkeypatch.setattr(config, 'NAVIDROME_PASSWORD', 'pw1', raising=False)
        assert registry.creds_from_config('navidrome') == {
            'url': 'http://nd', 'user': 'user1', 'password': 'pw1'
        }

        monkeypatch.setattr(config, 'PLEX_URL', 'http://plex', raising=False)
        monkeypatch.setattr(config, 'PLEX_TOKEN', 'ptok', raising=False)
        assert registry.creds_from_config('plex') == {'url': 'http://plex', 'token': 'ptok'}

    def test_normalize_row(self):
        from tasks.mediaserver import registry

        row = {
            'server_id': 's1', 'name': 'Home', 'server_type': 'jellyfin',
            'creds': {'url': 'u', 'token': 't'}, 'music_libraries': None,
            'is_default': True,
        }
        norm = registry.normalize_row(row)
        assert norm['music_libraries'] == ''
        assert norm['is_default'] is True
        assert norm['creds'] == {'url': 'u', 'token': 't'}

    def test_translate_ids_identity_for_default(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        conn = MagicMock()
        assert registry.translate_ids(['A', 'B'], None, conn=conn) == {'A': 'A', 'B': 'B'}
        assert registry.translate_ids(['A', 'B'], 'def', conn=conn) == {'A': 'A', 'B': 'B'}

    def test_translate_ids_secondary_uses_map(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = [('A', 'provA')]
        conn = MagicMock()
        conn.cursor.return_value = cursor
        result = registry.translate_ids(['A', 'B'], 'sec', conn=conn)
        assert result == {'A': 'provA'}

    def test_default_never_leaks_unmapped_canonical_id(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value = cursor

        assert registry.translate_ids(['fp_deadbeef', 'legacy-provider-id'], None, conn=conn) == {
            'legacy-provider-id': 'legacy-provider-id'
        }


class TestServerScopes:
    @staticmethod
    def _server(server_id, default=False):
        return {
            'server_id': server_id, 'name': server_id, 'server_type': 'jellyfin',
            'creds': {}, 'music_libraries': '', 'is_default': default,
        }

    def test_empty_registry_means_legacy_default(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: [])
        assert registry.servers_for_scope('all') == [None]
        assert registry.servers_for_scope('default') == [None]

    def test_registry_failure_means_legacy_default(self, monkeypatch):
        from tasks.mediaserver import registry

        def boom(conn=None):
            raise RuntimeError('registry down')

        monkeypatch.setattr(registry, 'list_servers', boom)
        assert registry.servers_for_scope('all') == [None]

    def test_default_scope_returns_only_default(self, monkeypatch):
        from tasks.mediaserver import registry

        default = self._server('a', default=True)
        secondary = self._server('b')
        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: [default, secondary])
        assert registry.servers_for_scope('default') == [default]
        assert registry.servers_for_scope('all') == [default, secondary]

    def test_specific_scope_matches_id_or_name(self, monkeypatch):
        from tasks.mediaserver import registry

        default = self._server('a', default=True)
        secondary = self._server('b')
        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: [default, secondary])
        assert registry.servers_for_scope('b') == [secondary]
        assert registry.servers_for_scope('B') == [secondary]
        assert registry.servers_for_scope('nope') == []

    def test_has_secondary_servers_queries_registry(self):
        from tasks.mediaserver import registry

        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone.return_value = (True,)
        assert registry.has_secondary_servers(conn=conn) is True
        cursor.fetchone.return_value = (False,)
        assert registry.has_secondary_servers(conn=conn) is False

    def test_has_secondary_servers_cached_until_invalidated(self, monkeypatch):
        from tasks.mediaserver import registry

        registry.invalidate_server_cache()
        cursor = MagicMock()
        cursor.fetchone.return_value = (True,)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        monkeypatch.setattr(registry, 'get_db', lambda: conn)
        try:
            assert registry.has_secondary_servers() is True
            assert registry.has_secondary_servers() is True
            assert cursor.execute.call_count == 1
            registry.invalidate_server_cache()
            assert registry.has_secondary_servers() is True
            assert cursor.execute.call_count == 2
        finally:
            registry.invalidate_server_cache()


class TestRegistryMutatorRollback:
    def test_add_server_rolls_back_and_reraises_on_insert_failure(self):
        from tasks.mediaserver import registry

        db = MagicMock()
        cur = db.cursor.return_value
        cur.fetchone.return_value = (True,)

        def explode(sql, params=None):
            if sql.startswith('INSERT INTO music_servers'):
                raise RuntimeError('insert failed')

        cur.execute.side_effect = explode
        with pytest.raises(RuntimeError, match='insert failed'):
            registry.add_server('Home', 'jellyfin', {'url': 'u'}, conn=db)
        db.rollback.assert_called_once()
        db.commit.assert_not_called()

    def test_set_default_rolls_back_and_reraises_on_update_failure(self):
        from tasks.mediaserver import registry

        db = MagicMock()
        cur = db.cursor.return_value

        def explode(sql, params=None):
            if 'SET is_default = TRUE' in sql:
                raise RuntimeError('update failed')

        cur.execute.side_effect = explode
        with pytest.raises(RuntimeError, match='update failed'):
            registry.set_default('sid', conn=db)
        db.rollback.assert_called_once()
        db.commit.assert_not_called()


class TestMatcherTiers:
    def test_path_is_top_tier(self):
        from tasks.provider_migration_matcher import match_tracks

        old = [{
            'item_id': 'A', 'title': 't', 'author': 'a', 'album': 'al',
            'file_path': '/music/same.flac',
        }]
        new = [
            {'id': 'by_path', 'title': 'zzz', 'artist': 'q', 'album': 'w', 'path': '/music/same.flac'},
            {'id': 'by_meta', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/other.flac'},
        ]
        result = match_tracks(old, new)
        assert result['matches']['A'] == 'by_path'
        assert result['match_tiers']['A'] == 'path'

    def test_metadata_fallback_when_paths_differ(self):
        from tasks.provider_migration_matcher import match_tracks

        old = [{
            'item_id': 'A', 'title': 't', 'author': 'a', 'album': 'al',
            'file_path': '/jellyfin/x.flac',
        }]
        new = [{'id': 'n1', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/navidrome/y.flac'}]
        result = match_tracks(old, new)
        assert result['matches'] == {'A': 'n1'}
        assert result['match_tiers']['A'] == 'exact_meta'


class TestBoundServer:
    def test_for_server_runs_call_in_context(self, monkeypatch):
        from tasks import mediaserver
        from tasks.mediaserver import registry, context

        fake_ctx = {
            'server_id': 's2', 'server_type': 'plex',
            'creds': {'url': 'u', 'token': 't'}, 'music_libraries': 'lib',
        }
        monkeypatch.setattr(registry, 'context_for', lambda sid, conn=None: fake_ctx)

        captured = {}

        def fake_get_all_songs(user_creds=None, provider_type=None, apply_filter=True):
            captured['type'] = context.active_type('none')
            captured['creds'] = context.active_creds()
            return []

        monkeypatch.setattr(mediaserver, 'get_all_songs', fake_get_all_songs)
        mediaserver.for_server('s2').get_all_songs()
        assert captured['type'] == 'plex'
        assert captured['creds'] == {'url': 'u', 'token': 't'}
        assert context.active_type('none') == 'none'

    def test_default_server_uses_config_path(self, monkeypatch):
        from tasks import mediaserver
        from tasks.mediaserver import registry, context

        monkeypatch.setattr(registry, 'context_for', lambda sid, conn=None: None)
        captured = {}

        def fake_get_all_songs(user_creds=None, provider_type=None, apply_filter=True):
            captured['type'] = context.active_type('fallback')
            captured['creds'] = context.active_creds()
            return []

        monkeypatch.setattr(mediaserver, 'get_all_songs', fake_get_all_songs)
        mediaserver.for_server(None).get_all_songs()
        assert captured['type'] == 'fallback'
        assert captured['creds'] is None


class TestFingerprintAsId:
    def test_default_translates_via_map_with_identity_fallback(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = [('fp_1', 'prov1')]
        conn = MagicMock()
        conn.cursor.return_value = cursor
        result = registry.translate_ids(['fp_1', 'raw2'], None, conn=conn)
        assert result == {'fp_1': 'prov1', 'raw2': 'raw2'}

    def test_default_identity_when_no_default_server(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: None)
        conn = MagicMock()
        assert registry.translate_ids(['a', 'b'], None, conn=conn) == {'a': 'a', 'b': 'b'}

    def test_default_dropped_canonical_ids_log_warning(self, monkeypatch, caplog):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with caplog.at_level(logging.WARNING):
            result = registry.translate_ids(['fp_deadbeef', 'legacy'], None, conn=conn)
        assert result == {'legacy': 'legacy'}
        assert 'no mapping on the default server' in caplog.text


class TestReverseTranslation:
    def test_default_maps_known_and_falls_back_to_identity(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = [('jelly1', 'fp_1')]
        conn = MagicMock()
        conn.cursor.return_value = cursor
        result = registry.reverse_translate_ids(['jelly1', 'legacy2'], None, conn=conn)
        assert result == {'jelly1': 'fp_1', 'legacy2': 'legacy2'}

    def test_secondary_drops_unknown(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = [('nav1', 'fp_1')]
        conn = MagicMock()
        conn.cursor.return_value = cursor
        result = registry.reverse_translate_ids(['nav1', 'ghost'], 'sec', conn=conn)
        assert result == {'nav1': 'fp_1'}


class TestCanonicalInputIds:
    def test_provider_ids_resolve_and_canonical_pass_through(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry,
            'reverse_translate_ids',
            lambda ids, server_id=None, conn=None: {'nav1': 'fp_a'},
        )
        result = registry.canonical_input_ids(['nav1', 'fp_b', 'ghost'], 'sec')
        assert result == {'nav1': 'fp_a', 'fp_b': 'fp_b', 'ghost': 'ghost'}

    def test_registry_failure_falls_back_to_identity(self, monkeypatch):
        from tasks.mediaserver import registry

        def boom(ids, server_id=None, conn=None):
            raise RuntimeError('registry down')

        monkeypatch.setattr(registry, 'reverse_translate_ids', boom)
        result = registry.canonical_input_ids(['x', 'y'])
        assert result == {'x': 'x', 'y': 'y'}

    def test_empty_input_returns_empty_mapping(self):
        from tasks.mediaserver import registry

        assert registry.canonical_input_ids([]) == {}
        assert registry.canonical_input_ids([None, '']) == {}


class TestSonicFingerprintProviderRecency:
    def test_last_played_uses_provider_id_for_canonical_song(self, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry,
            'reverse_translate_ids',
            lambda ids, server_id=None, conn=None: {'jelly1': 'fp_a'},
        )
        mapping = registry.canonical_input_ids(['jelly1'], None)
        provider_by_canonical = {c: p for p, c in mapping.items()}
        assert provider_by_canonical.get('fp_a', 'fp_a') == 'jelly1'
        assert provider_by_canonical.get('fp_unknown', 'fp_unknown') == 'fp_unknown'


class TestAnalysisCanonicalResolution:
    def test_attaches_known_canonical_ids_and_keeps_unknown_temporary(self, monkeypatch):
        from tasks import analysis_helper as helper
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry,
            'reverse_translate_ids',
            lambda ids, server_id, conn=None: {'provider-known': 'fp_known'},
        )
        tracks = [{'Id': 'provider-known'}, {'Id': 'provider-new'}]

        helper.attach_catalog_item_ids(tracks, server_id='server-b')

        assert [helper.catalog_item_id(track) for track in tracks] == [
            'fp_known', 'provider-new'
        ]


class TestServerWorkMap:
    """load_server_work_map: ONE scan per server replacing per-album queries."""

    def _cursor(self, pages_by_sql):
        executed = []
        cur = MagicMock()
        state = {'sql': None}

        def execute(sql, params=None):
            executed.append((sql, params))
            state['sql'] = sql

        def fetchall():
            key = 'legacy' if 'FROM score s\n' in state['sql'] or (
                'NOT LIKE' in state['sql']
            ) else 'mapped'
            pages = pages_by_sql.get(key, [])
            return pages.pop(0) if pages else []

        cur.execute.side_effect = execute
        cur.fetchall.side_effect = fetchall
        return cur, executed

    def _patch_db(self, monkeypatch, cur):
        from tasks import analysis_helper as helper

        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        db_cm = MagicMock()
        db_cm.__enter__.return_value = conn
        db_cm.__exit__.return_value = False
        monkeypatch.setattr(helper, 'get_db', lambda: db_cm)

    def test_mask_bits_reflect_what_each_track_already_has(self, monkeypatch):
        from tasks import analysis_helper as helper

        cur, _ = self._cursor({'mapped': [[
            ('p-done', True, True, True),
            ('p-no-embedding', False, True, True),
            ('p-no-clap', True, False, True),
            ('p-no-lyrics', True, True, False),
        ]]})
        self._patch_db(monkeypatch, cur)

        work_map = helper.load_server_work_map('srv', False, True, True)
        done = helper.work_done_bits(True, True)

        assert work_map['p-done'] & done == done
        assert not work_map['p-no-embedding'] & helper.WORK_MUSICNN
        assert not work_map['p-no-clap'] & helper.WORK_CLAP
        assert not work_map['p-no-lyrics'] & helper.WORK_LYRICS
        assert work_map['p-no-clap'] & done != done

    def test_disabled_features_are_not_required(self, monkeypatch):
        from tasks import analysis_helper as helper

        cur, executed = self._cursor({'mapped': [[('p1', True, True, True)]]})
        self._patch_db(monkeypatch, cur)

        work_map = helper.load_server_work_map('srv', False, False, False)
        done = helper.work_done_bits(False, False)

        assert done == helper.WORK_MUSICNN
        assert work_map['p1'] & done == done
        sql = executed[0][0]
        assert 'clap_embedding' not in sql
        assert 'lyrics_embedding' not in sql

    def test_album_work_masks_is_a_bounded_per_album_query(self, monkeypatch):
        """The per-album fallback yields the same bits as the bulk scan but via a
        bounded ANY() query (no keyset LIMIT), so a bulk-scan failure degrades to
        per-album checks instead of aborting the phase."""
        from tasks import analysis_helper as helper

        cur, executed = self._cursor({'mapped': [[
            ('p-done', True, True, True),
            ('p-no-clap', True, False, True),
        ]]})
        self._patch_db(monkeypatch, cur)

        masks = helper.album_work_masks(['p-done', 'p-no-clap'], 'srv', False, True, True)
        done = helper.work_done_bits(True, True)

        assert masks['p-done'] & done == done
        assert masks['p-no-clap'] & done != done
        sql = executed[0][0]
        assert '= ANY(%s)' in sql
        assert 'LIMIT' not in sql

    def test_album_work_masks_default_server_includes_legacy(self, monkeypatch):
        from tasks import analysis_helper as helper

        cur, _ = self._cursor({
            'mapped': [[('p1', True, True, True)]],
            'legacy': [[('legacy-id', True, True, True)]],
        })
        self._patch_db(monkeypatch, cur)
        masks = helper.album_work_masks(['p1', 'legacy-id'], 'srv-def', True, True, True)
        assert 'p1' in masks and 'legacy-id' in masks

    def test_default_server_also_sees_legacy_rows_but_a_secondary_does_not(
        self, monkeypatch
    ):
        from tasks import analysis_helper as helper

        cur, _ = self._cursor({
            'mapped': [[('p1', True, True, True)]],
            'legacy': [[('legacy-id', True, True, True)]],
        })
        self._patch_db(monkeypatch, cur)
        default_map = helper.load_server_work_map('srv-def', True, True, True)

        cur2, _ = self._cursor({
            'mapped': [[('p1', True, True, True)]],
            'legacy': [[('legacy-id', True, True, True)]],
        })
        self._patch_db(monkeypatch, cur2)
        secondary_map = helper.load_server_work_map('srv-b', False, True, True)

        assert 'legacy-id' in default_map
        assert 'legacy-id' not in secondary_map
        assert 'p1' in secondary_map

    def test_keyset_pagination_advances_past_the_last_row(self, monkeypatch):
        from tasks import analysis_helper as helper

        cur, executed = self._cursor({'mapped': [
            [('p1', True, True, True), ('p2', True, True, True)],
            [('p3', True, True, True)],
            [],
        ]})
        self._patch_db(monkeypatch, cur)

        work_map = helper.load_server_work_map('srv', False, True, True, chunk_size=2)

        assert sorted(work_map) == ['p1', 'p2', 'p3']
        assert [params[-2] for _sql, params in executed] == ['', 'p2', 'p3']
        assert all(params[-1] == 2 for _sql, params in executed)


class TestSingleTranslationPoint:
    def test_dispatcher_translates_once_for_bound_server(self, monkeypatch):
        from tasks import mediaserver
        from tasks.mediaserver import registry

        calls = {}

        class FakeProvider:
            @staticmethod
            def create_instant_playlist(name, ids, creds=None):
                calls['ids'] = list(ids)
                return {'Id': 'p1'}

        monkeypatch.setattr(mediaserver, '_provider', lambda provider_type=None: FakeProvider)
        seen = []

        def fake_translate(ids, sid, conn=None):
            seen.append(sid)
            return {'fp_a': 'nav1'}

        monkeypatch.setattr(registry, 'translate_ids', fake_translate)
        ctx = {'server_id': 'sec', 'server_type': 'navidrome', 'creds': {'url': 'u'}, 'music_libraries': ''}
        with mediaserver.use_server(ctx):
            result = mediaserver.create_instant_playlist('P', ['fp_a', 'fp_b'])
        assert result == {'Id': 'p1'}
        assert calls['ids'] == ['nav1']
        assert seen == ['sec']

    def test_endpoint_helper_passes_untranslated_ids_to_dispatcher(self, monkeypatch):
        import app_server_context as asc
        from tasks import mediaserver
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry, 'translate_ids', lambda ids, sid, conn=None: {'fp_a': 'nav1'}
        )
        captured = {}

        class Bound:
            def create_instant_playlist(self, name, ids, creds=None):
                captured['ids'] = list(ids)
                return {'Id': 'p9'}

        monkeypatch.setattr(mediaserver, 'for_server', lambda sid, conn=None: Bound())
        info = asc.create_instant_playlist_for_server('P', ['fp_a', 'fp_b'], 'sec')
        assert captured['ids'] == ['fp_a', 'fp_b']
        assert info['mapped'] == 1
        assert info['skipped'] == 1
        assert info['result'] == {'Id': 'p9'}

    def test_no_available_tracks_raises(self, monkeypatch):
        import app_server_context as asc
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'translate_ids', lambda ids, sid, conn=None: {})
        with pytest.raises(ValueError):
            asc.create_instant_playlist_for_server('P', ['fp_a'], 'sec')


def _legacy_cursor(legacy_rows, canonical_rows=()):
    """A cursor over a catalogue of legacy rows (+ already-canonical ones).

    _build_mapping COUNTs each kind, streams both through server-side cursors to
    hash their signatures a batch at a time, then fetches embeddings BACK for
    the candidate pairs it asks the cosine to confirm.
    """
    canonical_rows = list(canonical_rows)
    legacy_rows = list(legacy_rows)
    blobs = {str(item_id): blob for item_id, blob in canonical_rows + legacy_rows}

    class ScanCursor:
        def __init__(self, rows):
            self._batches = [rows, []]
            self.itersize = None

        def execute(self, sql, params=None):
            pass

        def fetchmany(self, size):
            return self._batches.pop(0) if self._batches else []

        def close(self):
            pass

    class FetchCursor:
        def __init__(self):
            self._rows = []

        def execute(self, sql, params=None):
            wanted = list(params[0]) if params else []
            self._rows = [(i, blobs[i]) for i in wanted if i in blobs]

        def fetchall(self):
            return self._rows

        def close(self):
            pass

    class FakeConn:
        def __init__(self):
            self._scans = [list(canonical_rows), list(legacy_rows)]

        def cursor(self, name=None):
            if name is None:
                return FetchCursor()
            return ScanCursor(self._scans.pop(0) if self._scans else [])

    cursor = MagicMock()
    cursor.connection = FakeConn()
    # Two COUNTs: legacy first, then already-canonical.
    cursor.fetchone.side_effect = [(len(legacy_rows),), (len(canonical_rows),)]
    return cursor


class TestIndexRepoint:
    """A relabel renames tracks; it must not rebuild a single index."""

    def _patched(self, monkeypatch, blob, stored, invalidated):
        from tasks import index_build_helpers, paged_ivf

        monkeypatch.setattr(
            index_build_helpers, 'load_segmented_blob',
            lambda conn, table, name: blob if name.startswith('music_library') else None,
        )
        monkeypatch.setattr(
            index_build_helpers, 'store_segmented_blob',
            lambda conn, table, name, data, max_part_size_mb=None: stored.__setitem__(name, data),
        )
        monkeypatch.setattr(
            paged_ivf, 'invalidate_global_cell_cache', invalidated.append
        )

    def test_repoints_ids_and_leaves_every_vector_alone(self, monkeypatch):
        import numpy as np
        from tasks import fingerprint_canonicalize as canonicalize
        from tasks.paged_ivf import pack_directory, unpack_directory

        centroids = np.arange(8, dtype=np.float32).reshape(2, 4)
        id2cell = np.array([0, 1, 0], dtype=np.uint32)
        blob = pack_directory(
            centroids, id2cell, ['jf_1', 'jf_2', 'jf_3'], 4, 'angular',
            normalized=True, storage_dtype=1,
        )
        stored, invalidated = {}, []
        self._patched(monkeypatch, blob, stored, invalidated)

        cursor = MagicMock()
        cursor.fetchall.return_value = [('main_map', json.dumps(['jf_1', 'jf_2', 'jf_3']))]
        canonicalize._repoint_indexes(
            cursor, {'jf_1': 'fp_2aa', 'jf_3': 'fp_2aa'}  # jf_3 merged INTO jf_1's row
        )

        written = stored['music_library__ivf_dir']
        new_centroids, new_id2cell, new_ids, dim, metric, normalized, dtype = (
            unpack_directory(written)
        )
        assert new_ids == ['fp_2aa', 'jf_2', 'fp_2aa']
        # Not one vector, cell assignment or centroid may move.
        assert np.array_equal(new_centroids, centroids)
        assert np.array_equal(new_id2cell, id2cell)
        assert (dim, metric, normalized, dtype) == (4, 'angular', True, 1)
        assert invalidated == ['music_library']

        # The map projection's id list is rewritten in place too.
        update = [c for c in cursor.execute.call_args_list if 'UPDATE' in c.args[0]]
        assert json.loads(update[0].args[1][0]) == ['fp_2aa', 'jf_2', 'fp_2aa']

    def test_no_renames_writes_nothing(self, monkeypatch):
        from tasks import fingerprint_canonicalize as canonicalize

        stored, invalidated = {}, []
        self._patched(monkeypatch, b'', stored, invalidated)
        cursor = MagicMock()
        canonicalize._repoint_indexes(cursor, {})
        assert stored == {} and invalidated == []
        cursor.execute.assert_not_called()

    def test_a_broken_index_does_not_abort_the_migration(self, monkeypatch):
        from tasks import fingerprint_canonicalize as canonicalize
        from tasks import index_build_helpers

        def explode(conn, table, name):
            raise RuntimeError('corrupt directory blob')

        monkeypatch.setattr(index_build_helpers, 'load_segmented_blob', explode)
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        canonicalize._repoint_indexes(cursor, {'jf_1': 'fp_2aa'})


class TestEmbeddingCanonicalization:
    def test_builds_canonical_ids_from_stored_embeddings(self):
        import numpy as np
        from tasks import simhash
        from tasks import fingerprint_canonicalize as canonicalize

        embedding = np.sin(np.arange(200, dtype=np.float32)).tobytes()
        cursor = _legacy_cursor([('legacy-provider-id', embedding)])
        mapping, duplicate_mapping = canonicalize._build_mapping(cursor)
        assert mapping == {
            'legacy-provider-id': simhash.canonical_id_str(simhash.embedding_signature(embedding))
        }
        assert duplicate_mapping == {}

    def test_same_audio_copies_merge(self):
        import numpy as np
        from tasks import fingerprint_canonicalize as canonicalize

        embedding = np.sin(np.arange(200, dtype=np.float32)).tobytes()
        cursor = _legacy_cursor([
            ('copy-one', embedding),
            ('copy-two', embedding),
        ])
        mapping, duplicate_mapping = canonicalize._build_mapping(cursor)
        assert list(mapping.keys()) == ['copy-one']
        assert duplicate_mapping == {'copy-two': next(iter(mapping.values()))}

    def test_same_signature_different_audio_never_merges(self):
        import numpy as np
        from tasks import simhash
        from tasks import fingerprint_canonicalize as canonicalize

        half = simhash.SIGNATURE_BITS // 2
        first = np.concatenate(
            [np.full(half, 1.0), np.full(half, -1.0)]
        ).astype(np.float32)
        second = first.copy()
        second[0:half:2] = 2.0
        second[1:half:2] = 0.1
        second[half::2] = -2.0
        second[half + 1::2] = -0.1
        assert simhash.embedding_signature(first) == simhash.embedding_signature(second)
        assert simhash.cosine_distance(first, second) > 0.01

        cursor = _legacy_cursor([
            ('copy-one', first.tobytes()),
            ('copy-two', second.tobytes()),
        ])
        mapping, duplicate_mapping = canonicalize._build_mapping(cursor)
        assert duplicate_mapping == {}
        assert set(mapping.keys()) == {'copy-one', 'copy-two'}
        assert mapping['copy-one'] != mapping['copy-two']

    def test_preserves_recovered_default_provider_id(self):
        from tasks.fingerprint_canonicalize import _default_provider_ids

        cursor = MagicMock()
        cursor.fetchall.return_value = [('legacy-score-id', 'current-jellyfin-id')]

        result = _default_provider_ids(cursor, 'default-server', {'legacy-score-id': 'fp_hash'})

        assert result == {'legacy-score-id': 'current-jellyfin-id'}

    def test_passed_conn_session_settings_restored(self, monkeypatch):
        from tasks import fingerprint_canonicalize as canonicalize

        class SessionCursor:
            def __init__(self, conn):
                self._conn = conn
                self._last_sql = None

            def execute(self, sql, params=None):
                self._last_sql = sql
                self._conn.executed.append((sql, params))

            def fetchone(self):
                if self._last_sql == "SHOW statement_timeout":
                    return ('600s',)
                return (None,)

            def close(self):
                pass

        class SessionConn:
            def __init__(self):
                self._autocommit = True
                self.autocommit_events = []
                self.executed = []
                self.commits = 0

            @property
            def autocommit(self):
                return self._autocommit

            @autocommit.setter
            def autocommit(self, value):
                self._autocommit = value
                self.autocommit_events.append(value)

            def cursor(self):
                return SessionCursor(self)

            def commit(self):
                self.commits += 1

            def rollback(self):
                pass

        monkeypatch.setattr(canonicalize, '_build_mapping', lambda cur: ({}, {}))
        monkeypatch.setattr(
            canonicalize.registry, 'get_default_server_id', lambda conn=None: 'sid'
        )
        conn = SessionConn()

        result = canonicalize.canonicalize_fingerprinted_ids(conn=conn)

        assert result == {'relabelled': 0, 'duplicates': 0}
        sqls = [sql for sql, _params in conn.executed]
        # The advisory lock makes exactly one replica relabel; the others wait
        # and then find nothing to do.
        assert sqls == [
            "SHOW statement_timeout",
            "SET statement_timeout = 0",
            "SELECT pg_advisory_xact_lock(%s)",
            "SET statement_timeout = %s",
        ]
        assert conn.executed[3][1] == ('600s',)
        assert conn.autocommit_events == [False, True]
        assert conn.autocommit is True
        assert conn.commits >= 1


class TestSweepAlignment:
    def test_iter_unmapped_local_rows_keyset_pagination(self):
        from tasks import multiserver_sync as sync

        pages = [
            [
                ('a1', 't1', 'au1', 'al1', 'aa1', '/p1'),
                ('a2', 't2', 'au2', 'al2', 'aa2', '/p2'),
            ],
            [
                ('a3', 't3', 'au3', 'al3', 'aa3', '/p3'),
                ('a4', 't4', 'au4', 'al4', 'aa4', '/p4'),
            ],
            [
                ('a5', 't5', 'au5', 'al5', 'aa5', '/p5'),
            ],
            [],
        ]
        executed = []
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.side_effect = list(pages)
        conn = MagicMock()
        conn.cursor.return_value = cursor

        chunks = list(sync._iter_unmapped_local_rows(conn, 'srv', chunk_size=2))

        assert [len(chunk) for chunk in chunks] == [2, 2, 1]
        assert [row['item_id'] for chunk in chunks for row in chunk] == [
            'a1', 'a2', 'a3', 'a4', 'a5'
        ]
        assert chunks[0][0] == {
            'item_id': 'a1', 'title': 't1', 'author': 'au1',
            'album': 'al1', 'album_artist': 'aa1', 'file_path': '/p1',
        }
        assert len(executed) == 4
        assert all('ORDER BY s.item_id LIMIT %s' in sql for sql, _params in executed)
        assert [params[0] for _sql, params in executed] == ['', 'a2', 'a4', 'a5']
        assert all(params[1] == 'srv' and params[2] == 2 for _sql, params in executed)

    def test_sweep_all_isolates_per_server_failures(self, monkeypatch):
        from tasks import multiserver_sync as sync
        import config

        servers = [
            {'server_id': 's1', 'name': 'One', 'server_type': 'navidrome', 'creds': {},
             'music_libraries': '', 'is_default': False, 'enabled': True},
            {'server_id': 's2', 'name': 'Two', 'server_type': 'plex', 'creds': {},
             'music_libraries': '', 'is_default': False, 'enabled': True},
        ]
        monkeypatch.setattr(sync.registry, 'list_servers', lambda conn=None: servers)
        reports = []
        monkeypatch.setattr(
            sync, '_make_reporter',
            lambda task_id, label: (
                lambda message, progress, task_state=None: reports.append(
                    (message, progress, task_state)
                )
            ),
        )
        monkeypatch.setattr(
            sync, '_make_cancel_check', lambda task_id: (lambda: None, lambda: None)
        )

        def fake_sweep(server, db, report, base, span, cancel, full_refresh=False):
            if server['server_id'] == 's1':
                raise RuntimeError('provider down')
            return {'server_id': server['server_id'], 'matched': 3}

        monkeypatch.setattr(sync, '_sweep_one', fake_sweep)
        db = MagicMock()

        results = sync.sweep_all_secondary_servers(task_id='tid', conn=db)

        assert results == [
            {'server_id': 's1', 'error': 'sweep failed'},
            {'server_id': 's2', 'matched': 3},
        ]
        db.rollback.assert_called_once()
        assert reports[-1][2] == config.TASK_STATUS_SUCCESS
        assert reports[-1][1] == 100
        db.close.assert_not_called()

    def test_aligned_server_is_noop_without_fetch(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 5)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 0)
        fetched = []
        monkeypatch.setattr(
            sync.provider_probe, 'fetch_all_tracks',
            lambda *a, **k: fetched.append(1) or [],
        )
        summary = sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1', 'creds': {}},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None,
        )
        assert summary['aligned'] is True
        assert fetched == []

    def test_unmapped_rows_matched_and_written(self, monkeypatch):
        from tasks import multiserver_sync as sync

        rows = [{
            'item_id': 'fp_1', 'title': 't', 'author': 'a', 'album': 'al',
            'album_artist': 'a', 'file_path': '/x.flac',
        }]
        target = [{'id': 'nav1', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/x.flac'}]
        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 1)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 1)
        monkeypatch.setattr(sync, '_iter_unmapped_local_rows', lambda conn, sid, **k: iter([rows]))
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', lambda *a, **k: target)
        written = {}
        monkeypatch.setattr(
            sync, '_write_matches',
            lambda db, sid, result: written.update(result['matches']) or len(result['matches']),
        )
        summary = sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1', 'creds': {}},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None,
        )
        assert summary['matched'] == 1
        assert summary['pruned'] == 0
        assert written == {'fp_1': 'nav1'}

    def test_collect_artist_maps_requires_name_and_id(self):
        from tasks import multiserver_sync as sync

        tracks = [
            {'artist': 'Art', 'artist_id': 'a1'},
            {'artist': None, 'album_artist': 'Alb', 'artist_id': 'a2'},
            {'artist': 'NoId', 'artist_id': None},
            {'artist': 'Art', 'artist_id': 'a9'},
        ]
        assert sync._collect_artist_maps(tracks) == {'Art': 'a9', 'Alb': 'a2'}

    def test_sweep_aligns_artists_and_metadata_from_fetched_catalogue(self, monkeypatch):
        from tasks import multiserver_sync as sync

        target = [
            {'id': 'p1', 'title': 't', 'artist': 'Art', 'artist_id': 'a9',
             'album': 'al', 'path': '/x.flac'},
            {'id': 'p2', 'title': 't2', 'artist': 'Art', 'artist_id': 'a9',
             'album': 'al', 'path': '/y.flac'},
        ]
        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 1)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 1)
        monkeypatch.setattr(sync, '_iter_unmapped_local_rows', lambda conn, sid, **k: iter([]))
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', lambda *a, **k: target)
        monkeypatch.setattr(sync, '_write_matches', lambda db, sid, result: 0)
        staged = []
        monkeypatch.setattr(
            sync, '_stage_track_metadata',
            lambda db, tracks: staged.append([t['id'] for t in tracks]),
        )
        refreshed = {}
        monkeypatch.setattr(
            sync, '_refresh_mapped_metadata',
            lambda db, sid, is_default: refreshed.update(
                {'sid': sid, 'default': is_default}
            ) or 7,
        )
        written = {}
        monkeypatch.setattr(
            sync.registry, 'upsert_artist_maps',
            lambda sid, mapping, conn=None: written.update({sid: dict(mapping)})
            or len(mapping),
        )
        summary = sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1',
             'creds': {}, 'is_default': False},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None,
        )
        assert staged == [['p1', 'p2']]
        assert written == {'s1': {'Art': 'a9'}}
        assert refreshed == {'sid': 's1', 'default': False}
        assert summary['artists'] == 1
        assert summary['refreshed'] == 7

    def test_write_artist_maps_empty_is_noop(self, monkeypatch):
        from tasks import multiserver_sync as sync

        calls = []
        monkeypatch.setattr(
            sync.registry, 'upsert_artist_maps',
            lambda sid, mapping, conn=None: calls.append((sid, mapping)) or len(mapping),
        )
        assert sync._write_artist_maps(MagicMock(), {'server_id': 's1'}, {}) == 0
        assert calls == []
        assert sync._write_artist_maps(
            MagicMock(), {'server_id': 's1', 'is_default': False}, {'A': '1'}
        ) == 1
        assert calls == [('s1', {'A': '1'})]

    def test_refresh_mapped_metadata_file_path_only_for_default(self):
        from tasks import multiserver_sync as sync

        def run(is_default):
            executed = []
            cur = MagicMock()
            cur.execute.side_effect = lambda sql, params=None: executed.append(
                (str(sql), params)
            )
            cur.fetchone.return_value = ('sweep_track_meta',)  # guard: temp table exists
            cur.rowcount = 3
            db = MagicMock()
            db.cursor.return_value = cur
            assert sync._refresh_mapped_metadata(db, 'srv', is_default) == 3
            db.commit.assert_called_once()
            update_sql = next(sql for sql, p in executed if p == ('srv',))
            assert any('DROP TABLE' in sql for sql, _p in executed)
            return update_sql

        assert 'file_path' not in run(False)
        assert 'file_path' in run(True)

    def test_refresh_mapped_metadata_skips_when_stage_table_absent(self):
        from tasks import multiserver_sync as sync

        cur = MagicMock()
        cur.fetchone.return_value = (None,)  # to_regclass -> table not staged
        db = MagicMock()
        db.cursor.return_value = cur
        assert sync._refresh_mapped_metadata(db, 'srv', True) == 0
        db.commit.assert_not_called()

    def test_strip_nul_removes_null_bytes(self):
        from tasks import multiserver_sync as sync

        assert sync._strip_nul('a\x00b') == 'ab'
        assert sync._strip_nul('clean') == 'clean'
        assert sync._strip_nul(None) is None
        assert sync._strip_nul(2020) == 2020
        # A tag with a NUL must not blow up artist-map collection.
        maps = sync._collect_artist_maps(
            [{'artist': 'AC\x00DC', 'artist_id': 'id\x001'}]
        )
        assert maps == {'ACDC': 'id1'}

    def test_full_refresh_binds_server_filters_and_prunes(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 3)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 0)
        monkeypatch.setattr(sync, '_iter_unmapped_local_rows', lambda conn, sid, **k: iter([]))
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        seen = {}

        def fake_fetch(stype, creds, apply_filter=False):
            seen['apply_filter'] = apply_filter
            seen['bound_server'] = sync.ms_context.active_server_id()
            return [{'id': 'nav1'}]

        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', fake_fetch)

        def fake_prune(db, sid, present_ids, refused=None):
            seen['pruned_for'] = sid
            seen['present_ids'] = present_ids
            return 2

        monkeypatch.setattr(sync, 'prune_stale_mappings', fake_prune)
        monkeypatch.setattr(sync, '_write_matches', lambda db, sid, result: 0)
        summary = sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1',
             'creds': {}, 'music_libraries': 'Rock', 'is_default': False, 'enabled': True},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None, full_refresh=True,
        )
        assert seen['apply_filter'] is True
        assert seen['bound_server'] == 's1'
        assert seen['pruned_for'] == 's1'
        assert seen['present_ids'] == {'nav1'}
        assert summary['pruned'] == 2

    def test_fetched_catalogue_is_released_after_indexing(self, monkeypatch):
        from tasks import multiserver_sync as sync

        class TrackList(list):
            pass

        rows = [{
            'item_id': 'fp_1', 'title': 't', 'author': 'a', 'album': 'al',
            'album_artist': 'a', 'file_path': '/x.flac',
        }]
        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 1)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 1)
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        monkeypatch.setattr(sync, '_write_matches', lambda db, sid, result: 0)
        holder = {}

        def fake_fetch(*a, **k):
            tracks = TrackList(
                {'id': f'nav{i}', 'title': 't', 'artist': 'a', 'album': 'al',
                 'path': f'/x{i}.flac'}
                for i in range(3)
            )
            holder['ref'] = weakref.ref(tracks)
            return tracks

        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', fake_fetch)
        released = {}

        def fake_iter(conn, sid, **k):
            gc.collect()
            released['catalogue_freed'] = holder['ref']() is None
            return iter([rows])

        monkeypatch.setattr(sync, '_iter_unmapped_local_rows', fake_iter)
        sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1', 'creds': {}},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None,
        )
        assert released.get('catalogue_freed') is True

    def test_chunked_matching_never_maps_one_provider_track_twice(self, monkeypatch):
        from tasks import multiserver_sync as sync

        row = {
            'title': 't', 'author': 'a', 'album': 'al',
            'album_artist': 'a', 'file_path': '/x.flac',
        }
        chunk1 = [dict(row, item_id='fp_1')]
        chunk2 = [dict(row, item_id='fp_2')]
        target = [{'id': 'nav1', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/x.flac'}]
        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 2)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 2)
        monkeypatch.setattr(
            sync, '_iter_unmapped_local_rows', lambda conn, sid, **k: iter([chunk1, chunk2])
        )
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', lambda *a, **k: target)
        written = {}
        monkeypatch.setattr(
            sync, '_write_matches',
            lambda db, sid, result: written.update(result['matches']) or len(result['matches']),
        )
        summary = sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1', 'creds': {}},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None,
        )
        assert written == {'fp_1': 'nav1'}
        assert summary['matched'] == 1

    def test_enqueue_sweep_supersedes_active_sweeps_with_all_server_alignment(self, monkeypatch):
        import app_music_servers as msrv

        # No cleaning run in flight: the sweep and cleaning both prune
        # track_server_map, so an alignment refuses while cleaning is live.
        monkeypatch.setattr(msrv, 'get_active_main_task', lambda task_type=None: None)
        cancelled = []
        monkeypatch.setattr(
            msrv, '_cancel_active_sweeps', lambda: cancelled.append('old-task') or ['old-task']
        )
        saved = {}
        monkeypatch.setattr(
            msrv, 'save_task_status',
            lambda task_id, task_type, status, **kw: saved.update(
                {'task_id': task_id, 'task_type': task_type, 'status': status}
            ),
        )
        enqueued = {}

        def fake_enqueue(func, **kwargs):
            enqueued['func'] = func
            enqueued.update(kwargs)

        monkeypatch.setattr(msrv.rq_queue_high, 'enqueue', fake_enqueue)
        task_id = msrv._enqueue_sweep()
        assert cancelled == ['old-task']
        assert enqueued['func'] == 'tasks.multiserver_sync.sweep_all_secondary_servers'
        assert enqueued['job_id'] == task_id
        assert saved['task_type'] == 'server_sweep'

    def test_enqueue_sweep_refuses_while_a_cleaning_run_is_live(self, monkeypatch):
        """Both prune track_server_map against a catalogue snapshot taken minutes
        earlier, so an overlap lets one delete the mappings the other just wrote."""
        import app_music_servers as msrv

        active = {'task_id': 'clean-1', 'task_type': 'cleaning', 'status': 'PROGRESS'}
        monkeypatch.setattr(msrv, 'get_active_main_task', lambda task_type=None: active)
        enqueued = []
        monkeypatch.setattr(
            msrv.rq_queue_high, 'enqueue', lambda *a, **k: enqueued.append(a)
        )
        cancelled = []
        monkeypatch.setattr(
            msrv, '_cancel_active_sweeps', lambda: cancelled.append(1) or []
        )

        assert msrv._enqueue_sweep() is None
        assert enqueued == []
        # It must not supersede the sweeps it is not going to replace, either.
        assert cancelled == []

    def test_cancel_active_sweeps_revokes_each_non_terminal_sweep(self, monkeypatch):
        import app_music_servers as msrv
        import config

        cur = MagicMock()
        cur.fetchall.return_value = [('t1',), ('t2',)]
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(msrv, 'get_db', lambda: db)
        revoked = []
        monkeypatch.setattr(
            msrv, 'save_task_status',
            lambda task_id, task_type, status, **kw: revoked.append(
                (task_id, task_type, status)
            ),
        )
        started_job = MagicMock()
        started_job.get_status.return_value = 'started'
        queued_job = MagicMock()
        queued_job.get_status.return_value = 'queued'
        jobs = {'t1': started_job, 't2': queued_job}

        class _FakeJob:
            @staticmethod
            def fetch(task_id, connection=None):
                return jobs[task_id]

        monkeypatch.setattr(msrv, 'Job', _FakeJob)
        stopped = []
        monkeypatch.setattr(
            msrv, 'send_stop_job_command', lambda conn, task_id: stopped.append(task_id)
        )
        assert msrv._cancel_active_sweeps() == ['t1', 't2']
        assert revoked == [
            ('t1', 'server_sweep', config.TASK_STATUS_REVOKED),
            ('t2', 'server_sweep', config.TASK_STATUS_REVOKED),
        ]
        assert stopped == ['t1']
        queued_job.cancel.assert_called_once()
        started_job.cancel.assert_not_called()

    def test_recover_abandoned_sweeps_replaces_dead_sweep(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_recovery_state', {'last': -10000.0})
        cur = MagicMock()
        cur.fetchall.return_value = [('dead-sweep',)]
        executed = []
        cur.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(sync, 'connect_raw', lambda: db)
        monkeypatch.setattr(sync, '_sweep_job_state', lambda task_id: 'dead')
        enqueued = {}

        def fake_enqueue(func, **kwargs):
            enqueued['func'] = func
            enqueued.update(kwargs)

        import app_helper
        monkeypatch.setattr(app_helper.rq_queue_high, 'enqueue', fake_enqueue)
        new_task_id = sync.recover_abandoned_sweeps()
        assert new_task_id is not None
        assert enqueued['func'] == 'tasks.multiserver_sync.sweep_all_secondary_servers'
        assert enqueued['job_id'] == new_task_id
        revoke_calls = [e for e in executed if e[0].startswith('UPDATE task_status')]
        assert revoke_calls and revoke_calls[0][1][-1] == ['dead-sweep']

    def test_recover_abandoned_sweeps_leaves_healthy_sweeps_alone(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_recovery_state', {'last': -10000.0})
        cur = MagicMock()
        cur.fetchall.return_value = [('live-sweep',)]
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(sync, 'connect_raw', lambda: db)
        monkeypatch.setattr(sync, '_sweep_job_state', lambda task_id: 'alive')
        import app_helper
        called = []
        monkeypatch.setattr(
            app_helper.rq_queue_high, 'enqueue',
            lambda *a, **k: called.append(1),
        )
        assert sync.recover_abandoned_sweeps() is None
        assert called == []

    def test_recover_abandoned_sweeps_recovers_rows_whose_rq_job_vanished(self, monkeypatch):
        """A sweep row whose job is gone from Redis entirely used to be SKIPPED, on
        the theory that the batch-start cleanup would archive it. That cleanup no
        longer touches sweeps (starting an analysis used to silently revoke a running
        one), so nothing retired these rows and the servers panel showed a phantom
        alignment stuck at N% forever."""
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_recovery_state', {'last': -10000.0})
        cur = MagicMock()
        cur.fetchall.return_value = [('never-enqueued-sweep',)]
        executed = []
        cur.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(sync, 'connect_raw', lambda: db)
        monkeypatch.setattr(sync, '_sweep_job_state', lambda task_id: 'missing')
        import app_helper
        called = []
        monkeypatch.setattr(
            app_helper.rq_queue_high, 'enqueue',
            lambda *a, **k: called.append(1),
        )
        assert sync.recover_abandoned_sweeps() is not None
        assert called == [1]
        assert [e for e in executed if e[0].startswith('UPDATE task_status')]

    def test_reap_orphaned_tasks_fails_rows_with_no_rq_job_behind_them(self, monkeypatch):
        """A main row is committed BEFORE its job is enqueued. If Redis is down at
        that moment, the PENDING row survives with nothing behind it, and
        get_active_main_task counts it as live - so every later Start returns 409,
        forever. Nothing used to retire it."""
        from rq.exceptions import NoSuchJobError
        from tasks import multiserver_sync as sync

        cur = MagicMock()
        cur.fetchall.return_value = [('ghost-analysis',)]
        executed = []
        cur.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(sync, 'connect_raw', lambda: db)

        import rq.job

        def _no_job(task_id, connection=None):
            raise NoSuchJobError(task_id)

        monkeypatch.setattr(rq.job.Job, 'fetch', staticmethod(_no_job))

        assert sync.reap_orphaned_tasks() == 1
        updates = [e for e in executed if e[0].startswith('UPDATE task_status')]
        assert len(updates) == 1
        assert 'FAILURE' in updates[0][1]
        assert ['ghost-analysis'] in updates[0][1]

        # Sweeps are excluded: recover_abandoned_sweeps re-enqueues those instead.
        select = executed[0]
        assert sync.SWEEP_TASK_TYPE in select[1]

    def test_recover_abandoned_sweeps_backs_off_after_enqueue(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_recovery_state', {'last': -10000.0})
        cur = MagicMock()
        cur.fetchall.return_value = [('dead-sweep',)]
        db = MagicMock()
        db.cursor.return_value = cur
        connections = []
        monkeypatch.setattr(sync, 'connect_raw', lambda: connections.append(1) or db)
        monkeypatch.setattr(sync, '_sweep_job_state', lambda task_id: 'dead')
        import app_helper
        enqueued = []
        monkeypatch.setattr(
            app_helper.rq_queue_high, 'enqueue',
            lambda *a, **k: enqueued.append(1),
        )
        assert sync.recover_abandoned_sweeps() is not None
        assert sync.recover_abandoned_sweeps() is None
        assert enqueued == [1]
        assert connections == [1]

    def test_dashboard_metrics_measure_each_server_against_its_own_catalogue(self, monkeypatch):
        import app_dashboard as dash

        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        monkeypatch.setattr(dash, '_safe_count', lambda cur, sql, params=None: 188057)
        # The legacy count must distinguish "counted zero" from "query failed",
        # or a transient error latches the scan as done forever.
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 25)
        cur = MagicMock()
        # Columns: server_id, name, type, is_default, track_count, rows_total, unique_songs
        cur.fetchall.return_value = [
            ('s1', 'Jellyfin', 'jellyfin', True, None, 188032, 188000),
            ('s2', 'PLEX', 'plex', False, 120, 46, 46),
            ('s3', 'Fresh', 'navidrome', False, None, 0, 0),
        ]
        rows = dash._collect_music_server_metrics(cur)
        # Default server: legacy add-on (25) lifts unique_songs and resolved; the
        # 32-file gap between rows_total and unique_songs is duplicate copies.
        assert rows[0]['unique_songs'] == 188025
        assert rows[0]['duplicate_copies'] == 32
        assert rows[0]['resolved'] == 188057
        assert rows[0]['server_songs'] == 188057
        assert rows[1]['unique_songs'] == 46
        assert rows[1]['duplicate_copies'] == 0
        assert rows[1]['resolved'] == 46
        assert rows[1]['server_songs'] == 120
        assert rows[2]['server_songs'] is None
        assert all(r['catalogue_songs'] == 188057 for r in rows)

    def test_sweep_stores_server_track_count(self, monkeypatch):
        from tasks import multiserver_sync as sync

        target = [{'id': 'nav1', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/x.flac'}]
        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 1)
        monkeypatch.setattr(sync, 'unmapped_local_count', lambda conn, sid: 1)
        monkeypatch.setattr(sync, '_iter_unmapped_local_rows', lambda conn, sid, **k: iter([]))
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        monkeypatch.setattr(sync, '_write_matches', lambda db, sid, result: 0)
        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', lambda *a, **k: target)
        stored = {}
        monkeypatch.setattr(
            sync, '_store_server_track_count',
            lambda db, sid, count: stored.update({sid: count}),
        )
        sync._sweep_one(
            {'server_id': 's1', 'server_type': 'navidrome', 'name': 'N1', 'creds': {}},
            MagicMock(), lambda *a, **k: None, 5, 95, lambda: None,
        )
        assert stored == {'s1': 1}

    def test_prune_skipped_when_fetch_looks_partial(self, caplog):
        from tasks import multiserver_sync as sync

        cursor = MagicMock()
        cursor.fetchone.return_value = (100,)
        db = MagicMock()
        db.cursor.return_value = cursor
        target = {str(i) for i in range(10)}
        with caplog.at_level(logging.WARNING):
            assert sync.prune_stale_mappings(db, 's1', target) == 0
        assert 'pruning skipped' in caplog.text
        db.commit.assert_not_called()

    def test_a_refused_prune_is_reported_not_silently_zero(self):
        """A refusal returned 0, exactly like 'nothing to prune'. A library that
        really did shrink by more than half kept its stale mappings and said so
        nowhere the user could see."""
        from tasks import multiserver_sync as sync

        cursor = MagicMock()
        cursor.fetchone.return_value = (100,)
        db = MagicMock()
        db.cursor.return_value = cursor

        refused = []
        assert sync.prune_stale_mappings(
            db, 's1', {str(i) for i in range(10)}, refused=refused
        ) == 0
        # The counts are carried out so the UI can say WHICH server refused and why.
        assert refused == [(10, 100)]


class TestFirstRunSetupWizardServerApi:
    """The first-run wizard drives /api/servers before any admin exists.

    The auth barrier opens the registry API while setup is needed (it already
    opens /api/setup, which writes the same credentials); these tests pin the
    second gate - the blueprint's own admin check - to the same window, and the
    promotion of the first real server over init_db's credential-less seed row.
    """

    @staticmethod
    def _request_context(**kwargs):
        from flask import Flask

        return Flask('setup-wizard-test').test_request_context(
            '/api/servers', **kwargs
        )

    def test_mutations_allowed_while_setup_is_needed(self, monkeypatch):
        import app_music_servers as msrv
        import config
        from flask import g

        monkeypatch.setattr(config, 'AUTH_ENABLED', True, raising=False)
        with self._request_context():
            g.setup_needed = True
            assert msrv._is_admin_caller() is True
            assert msrv._forbid_non_admin() is None

    def test_mutations_forbidden_once_setup_is_complete(self, monkeypatch):
        import app_music_servers as msrv
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True, raising=False)
        with self._request_context():
            result = msrv._forbid_non_admin()
        assert result is not None
        assert result[1] == 403

    _SEED_ROW = {
        'server_id': 'seed', 'name': 'Jellyfin', 'server_type': 'jellyfin',
        'creds': {}, 'music_libraries': '', 'is_default': True,
    }
    _CONFIGURED_ROW = {
        'server_id': 'd1', 'name': 'Navidrome', 'server_type': 'navidrome',
        'creds': {'url': 'http://nd:4533', 'user': 'u', 'password': 'p'},
        'music_libraries': '', 'is_default': True,
    }

    def test_credential_less_seed_row_is_a_placeholder(self, monkeypatch):
        import app_music_servers as msrv

        monkeypatch.setattr(
            msrv.registry, 'get_default_server', lambda conn=None: self._SEED_ROW
        )
        assert msrv._placeholder_default() == self._SEED_ROW

    def test_configured_default_is_not_a_placeholder(self, monkeypatch):
        import app_music_servers as msrv

        monkeypatch.setattr(
            msrv.registry, 'get_default_server', lambda conn=None: self._CONFIGURED_ROW
        )
        assert msrv._placeholder_default() is None

    def _add_plex(self, monkeypatch, default_row, mapped=0):
        import app_music_servers as msrv
        from flask import g

        created = {
            'server_id': 'new', 'name': 'Plex', 'server_type': 'plex',
            'creds': {'url': 'http://plex:32400', 'token': 'tok'},
            'music_libraries': '', 'is_default': True,
        }
        added = {}
        deleted = []

        def fake_add(**kwargs):
            added.update(kwargs)
            return 'new'

        monkeypatch.setattr(msrv.registry, 'get_default_server', lambda conn=None: default_row)
        monkeypatch.setattr(msrv.registry, 'get_default_server_id', lambda conn=None: 'new')
        monkeypatch.setattr(msrv.registry, 'get_server_by_name', lambda name, conn=None: None)
        monkeypatch.setattr(msrv.registry, 'add_server', fake_add)
        monkeypatch.setattr(msrv.registry, 'get_server', lambda sid, conn=None: created)
        monkeypatch.setattr(msrv.registry, 'mapped_count', lambda sid, conn=None: mapped)
        monkeypatch.setattr(
            msrv.registry, 'delete_server',
            lambda sid, conn=None: deleted.append(sid) or True,
        )
        monkeypatch.setattr(msrv, '_apply_default_to_config', lambda: None)
        monkeypatch.setattr(msrv, '_enqueue_sweep', lambda *a, **k: 'sweep-1')

        with self._request_context(
            method='POST',
            json={
                'name': 'Plex', 'server_type': 'plex',
                'creds': {'url': 'http://plex:32400', 'token': 'tok'},
            },
        ):
            g.setup_needed = True
            _body, status = msrv.add_server()
        return added, deleted, status

    def test_first_real_server_replaces_and_removes_the_seed_row(self, monkeypatch):
        added, deleted, status = self._add_plex(monkeypatch, default_row=self._SEED_ROW)
        assert status == 201
        assert added['make_default'] is True
        assert deleted == ['seed']

    def test_seed_row_with_mappings_is_demoted_but_kept(self, monkeypatch):
        added, deleted, status = self._add_plex(
            monkeypatch, default_row=self._SEED_ROW, mapped=42
        )
        assert status == 201
        assert added['make_default'] is True
        assert deleted == []

    def test_added_server_stays_secondary_when_a_default_is_configured(self, monkeypatch):
        added, deleted, status = self._add_plex(
            monkeypatch, default_row=self._CONFIGURED_ROW
        )
        assert status == 201
        assert added['make_default'] is False
        assert deleted == []


class TestSonicFingerprintDefaultsPerServer:
    """The Sonic Fingerprint form must describe the SELECTED server.

    Its credential fields and pre-filled account come from /api/config/defaults;
    reading them off the config globals would describe the DEFAULT server, so a
    Navidrome secondary behind a Jellyfin default rendered the wrong fields and
    the generate call then rejected them.
    """

    @staticmethod
    def _call(monkeypatch, selected, server_row):
        import app_server_context
        import app_sonic_fingerprint as sf
        from flask import Flask
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            app_server_context, 'resolve_request_server_id',
            lambda data=None: selected,
        )
        monkeypatch.setattr(registry, 'get_server', lambda sid, conn=None: server_row)
        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: server_row)

        app = Flask('sonic-defaults-test')
        with app.test_request_context('/api/config/defaults'):
            return sf.get_media_server_defaults().get_json()

    def test_selected_navidrome_secondary_describes_itself(self, monkeypatch):
        payload = self._call(
            monkeypatch,
            selected='s2',
            server_row={
                'server_id': 's2', 'name': 'Nav', 'server_type': 'navidrome',
                'creds': {'url': 'http://nd', 'user': 'bob', 'password': 'p'},
                'music_libraries': '', 'is_default': False,
            },
        )
        assert payload['server_type'] == 'navidrome'
        assert payload['default_user'] == 'bob'
        assert 'password' not in payload

    def test_selected_emby_secondary_returns_its_user_id(self, monkeypatch):
        payload = self._call(
            monkeypatch,
            selected='s3',
            server_row={
                'server_id': 's3', 'name': 'Emb', 'server_type': 'emby',
                'creds': {'url': 'http://emby', 'user_id': 'uid-9', 'token': 'secret'},
                'music_libraries': '', 'is_default': False,
            },
        )
        assert payload['server_type'] == 'emby'
        assert payload['default_user_id'] == 'uid-9'
        assert 'token' not in payload
        assert 'secret' not in str(payload)

    def test_no_selection_describes_the_default_server(self, monkeypatch):
        payload = self._call(
            monkeypatch,
            selected=None,
            server_row={
                'server_id': 'd1', 'name': 'Main', 'server_type': 'jellyfin',
                'creds': {'url': 'http://jf', 'user_id': 'uid-1', 'token': 't'},
                'music_libraries': '', 'is_default': True,
            },
        )
        assert payload['server_type'] == 'jellyfin'
        assert payload['default_user_id'] == 'uid-1'

    def test_registry_failure_falls_back_to_config(self, monkeypatch):
        import app_server_context
        import app_sonic_fingerprint as sf
        from flask import Flask
        from tasks.mediaserver import registry

        def boom(conn=None):
            raise RuntimeError('registry down')

        monkeypatch.setattr(
            app_server_context, 'resolve_request_server_id', lambda data=None: None
        )
        monkeypatch.setattr(registry, 'get_default_server', boom)
        monkeypatch.setattr(sf, 'MEDIASERVER_TYPE', 'jellyfin')
        monkeypatch.setattr(sf, 'JELLYFIN_USER_ID', 'cfg-user')

        app = Flask('sonic-defaults-test')
        with app.test_request_context('/api/config/defaults'):
            payload = sf.get_media_server_defaults().get_json()
        assert payload['server_type'] == 'jellyfin'
        assert payload['default_user_id'] == 'cfg-user'


class TestRegistrySeeding:
    """A fresh install starts with a BLANK server table.

    init_db seeds the registry only from a legacy install that really has a
    reachable server configured; MEDIASERVER_TYPE merely defaulting to
    'jellyfin' with empty credentials is NOT a server, and seeding it put a
    phantom Jellyfin row in the setup wizard. Rows like that (from earlier
    builds) are removed at boot unless they own track mappings.
    """

    @staticmethod
    def _cursor(existing=0, rows=None):
        cur = MagicMock()
        cur.fetchone.return_value = (existing,)
        cur.fetchall.return_value = rows or []
        cur.rowcount = 0
        cur.executed = []
        cur.execute.side_effect = lambda sql, params=None: cur.executed.append((sql, params))
        return cur

    def test_unconfigured_install_seeds_nothing(self, monkeypatch):
        import config
        import database

        monkeypatch.setattr(config, 'MEDIASERVER_TYPE', 'jellyfin', raising=False)
        monkeypatch.setattr(config, 'JELLYFIN_URL', '', raising=False)
        monkeypatch.setattr(config, 'JELLYFIN_USER_ID', '', raising=False)
        monkeypatch.setattr(config, 'JELLYFIN_TOKEN', '', raising=False)

        cur = self._cursor(existing=0)
        database._seed_registry_from_legacy_config(cur)

        assert not any('INSERT INTO music_servers' in sql for sql, _p in cur.executed)

    def test_configured_legacy_install_is_migrated(self, monkeypatch):
        import config
        import database

        monkeypatch.setattr(config, 'MEDIASERVER_TYPE', 'jellyfin', raising=False)
        monkeypatch.setattr(config, 'JELLYFIN_URL', 'http://jf:8096', raising=False)
        monkeypatch.setattr(config, 'JELLYFIN_USER_ID', 'uid', raising=False)
        monkeypatch.setattr(config, 'JELLYFIN_TOKEN', 'tok', raising=False)

        cur = self._cursor(existing=0)
        database._seed_registry_from_legacy_config(cur)

        inserts = [p for sql, p in cur.executed if 'INSERT INTO music_servers' in sql]
        assert len(inserts) == 1
        assert inserts[0][2] == 'jellyfin'

    def test_existing_registry_is_never_reseeded(self, monkeypatch):
        import database

        cur = self._cursor(existing=1)
        database._seed_registry_from_legacy_config(cur)

        assert not any('INSERT INTO music_servers' in sql for sql, _p in cur.executed)

    def test_phantom_row_is_removed_at_boot(self):
        import database

        cur = self._cursor(rows=[('seed', 'Jellyfin', 'jellyfin', {})])
        database._drop_unconfigured_servers(cur)

        deletes = [(sql, p) for sql, p in cur.executed if 'DELETE FROM music_servers' in sql]
        assert len(deletes) == 1
        assert deletes[0][1] == (['seed'],)
        # Never drops a server that still owns catalogue bindings.
        assert 'NOT EXISTS' in deletes[0][0]
        assert 'track_server_map' in deletes[0][0]

    def test_configured_rows_are_left_alone(self):
        import database

        cur = self._cursor(rows=[
            ('d1', 'Nav', 'navidrome',
             {'url': 'http://nd', 'user': 'u', 'password': 'p'}),
        ])
        database._drop_unconfigured_servers(cur)

        assert not any('DELETE FROM music_servers' in sql for sql, _p in cur.executed)


class TestDashboardLegacyCountLatch:
    def test_failed_count_is_not_latched_as_done(self, monkeypatch):
        """A transient DB error must not be read as 'no legacy rows left'."""
        import app_dashboard as dash

        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        monkeypatch.setattr(dash, '_safe_count', lambda cur, sql, params=None: 100)
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: None)
        monkeypatch.setattr(dash, '_LEGACY_UNMAPPED_DONE', {})
        cur = MagicMock()
        cur.fetchall.return_value = [('d1', 'Main', 'jellyfin', True, None, 40, 40)]

        rows = dash._collect_music_server_metrics(cur)

        assert rows[0]['resolved'] == 40
        assert dash._LEGACY_UNMAPPED_DONE == {}

    def test_real_zero_retires_the_scan(self, monkeypatch):
        import app_dashboard as dash

        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        monkeypatch.setattr(dash, '_safe_count', lambda cur, sql, params=None: 100)
        monkeypatch.setattr(dash, '_counted_or_none', lambda cur, sql, params=None: 0)
        monkeypatch.setattr(dash, '_LEGACY_UNMAPPED_DONE', {})
        cur = MagicMock()
        cur.fetchall.return_value = [('d1', 'Main', 'jellyfin', True, None, 40, 40)]

        dash._collect_music_server_metrics(cur)

        assert dash._LEGACY_UNMAPPED_DONE == {'d1': True}


class TestLyrionFolderFilterIsAnchored:
    def test_substring_of_a_folder_name_does_not_match(self):
        from tasks.mediaserver import lyrion

        song = {'FilePath': '/music/kid rock anthology/01.flac', 'url': ''}
        assert lyrion._song_in_target_paths(song, {'rock'}) is False

    def test_real_folder_matches(self):
        from tasks.mediaserver import lyrion

        song = {'FilePath': '/music/rock/queen/01.flac', 'url': ''}
        assert lyrion._song_in_target_paths(song, {'rock'}) is True

    def test_full_configured_path_matches(self):
        from tasks.mediaserver import lyrion

        song = {'FilePath': '/music/myfolder/x.flac', 'url': ''}
        assert lyrion._song_in_target_paths(song, {'/music/myfolder'}) is True


class TestServerParamCoercion:
    def test_non_string_server_id_is_a_clean_400_not_a_crash(self, monkeypatch):
        from flask import Flask
        import app_server_context as ctx
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_server', lambda sid, conn=None: None)
        monkeypatch.setattr(registry, 'get_server_by_name', lambda name, conn=None: None)

        app = Flask('server-param-test')
        with app.test_request_context('/api/x', method='POST', json={'server': 12345}):
            with pytest.raises(ValueError):
                ctx.resolve_request_server_id()

    def test_structured_server_value_is_rejected(self):
        from flask import Flask
        import app_server_context as ctx

        app = Flask('server-param-test')
        with app.test_request_context('/api/x', method='POST', json={'server': ['a', 'b']}):
            with pytest.raises(ValueError):
                ctx.resolve_request_server_id()
