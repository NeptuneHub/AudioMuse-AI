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
            'is_default': True, 'enabled': True,
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
    def _server(server_id, enabled=True, default=False):
        return {
            'server_id': server_id, 'name': server_id, 'server_type': 'jellyfin',
            'creds': {}, 'music_libraries': '', 'is_default': default, 'enabled': enabled,
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

    def test_all_disabled_means_nothing_to_do(self, monkeypatch):
        from tasks.mediaserver import registry

        servers = [self._server('a', enabled=False, default=True)]
        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: servers)
        assert registry.servers_for_scope('all') == []
        assert registry.servers_for_scope('default') == []

    def test_default_scope_returns_only_enabled_default(self, monkeypatch):
        from tasks.mediaserver import registry

        default = self._server('a', default=True)
        secondary = self._server('b')
        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: [default, secondary])
        assert registry.servers_for_scope('default') == [default]
        assert registry.servers_for_scope('all') == [default, secondary]

    def test_default_scope_empty_when_default_disabled(self, monkeypatch):
        from tasks.mediaserver import registry

        servers = [self._server('a', enabled=False, default=True), self._server('b')]
        monkeypatch.setattr(registry, 'list_servers', lambda conn=None: servers)
        assert registry.servers_for_scope('default') == []
        assert registry.servers_for_scope('all') == [servers[1]]

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

    def test_album_needs_queries_canonical_ids(self, monkeypatch):
        from tasks import analysis_helper as helper

        tracks = [{'Id': 'provider-known'}, {'Id': 'provider-new'}]
        monkeypatch.setattr(
            helper,
            'attach_catalog_item_ids',
            lambda items, server_id=None: [
                item.update(_catalog_item_id=canonical)
                for item, canonical in zip(items, ('fp_known', 'provider-new'))
            ] or items,
        )
        queried = {}

        def existing_ids(ids):
            queried['ids'] = list(ids)
            return {'fp_known'}

        monkeypatch.setattr(
            helper,
            'get_existing_track_ids',
            existing_ids,
        )
        monkeypatch.setattr(helper, 'get_missing_ids_in_table', lambda table, ids: set())

        existing, needs_clap, needs_lyrics = helper.compute_album_needs(
            tracks, False, False, server_id='server-b'
        )

        assert queried['ids'] == ['fp_known', 'provider-new']
        assert existing == 1
        assert needs_clap is False
        assert needs_lyrics is False


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


def _legacy_cursor(legacy_rows):
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (len(legacy_rows),)
    cursor.fetchmany.side_effect = [legacy_rows, []]
    return cursor


class TestEmbeddingCanonicalization:
    def test_builds_canonical_ids_from_stored_embeddings(self):
        import numpy as np
        from tasks import simhash
        from tasks import fingerprint_canonicalize as canonicalize

        embedding = np.sin(np.arange(200, dtype=np.float32)).tobytes()
        cursor = _legacy_cursor([('legacy-provider-id', embedding)])
        mapping, duplicate_mapping = canonicalize._build_mapping(cursor)
        assert mapping == {
            'legacy-provider-id': simhash.embedding_canonical_id(embedding)
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

        result = canonicalize.canonicalize_fingerprinted_ids(conn=conn, rebuild=False)

        assert result == {'relabelled': 0, 'duplicates': 0}
        sqls = [sql for sql, _params in conn.executed]
        assert sqls == [
            "SHOW statement_timeout",
            "SET statement_timeout = 0",
            "SET statement_timeout = %s",
        ]
        assert conn.executed[2][1] == ('600s',)
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
        monkeypatch.setattr(sync, '_unmapped_local_count', lambda conn, sid: 0)
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
        monkeypatch.setattr(sync, '_unmapped_local_count', lambda conn, sid: 1)
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

    def test_full_refresh_binds_server_filters_and_prunes(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 3)
        monkeypatch.setattr(sync, '_unmapped_local_count', lambda conn, sid: 0)
        monkeypatch.setattr(sync, '_iter_unmapped_local_rows', lambda conn, sid, **k: iter([]))
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        seen = {}

        def fake_fetch(stype, creds, apply_filter=False):
            seen['apply_filter'] = apply_filter
            seen['bound_server'] = sync.ms_context.active_server_id()
            return [{'id': 'nav1'}]

        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', fake_fetch)

        def fake_prune(db, sid, present_ids):
            seen['pruned_for'] = sid
            seen['present_ids'] = present_ids
            return 2

        monkeypatch.setattr(sync, '_prune_stale_mappings', fake_prune)
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
        monkeypatch.setattr(sync, '_unmapped_local_count', lambda conn, sid: 1)
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
        monkeypatch.setattr(sync, '_unmapped_local_count', lambda conn, sid: 2)
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

        monkeypatch.setattr(msrv.rq_queue_default, 'enqueue', fake_enqueue)
        task_id = msrv._enqueue_sweep()
        assert cancelled == ['old-task']
        assert enqueued['func'] == 'tasks.multiserver_sync.sweep_all_secondary_servers'
        assert enqueued['job_id'] == task_id
        assert saved['task_type'] == 'server_sweep'

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
        monkeypatch.setattr(app_helper.rq_queue_default, 'enqueue', fake_enqueue)
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
            app_helper.rq_queue_default, 'enqueue',
            lambda *a, **k: called.append(1),
        )
        assert sync.recover_abandoned_sweeps() is None
        assert called == []

    def test_recover_abandoned_sweeps_skips_missing_inline_sweeps(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_recovery_state', {'last': -10000.0})
        cur = MagicMock()
        cur.fetchall.return_value = [('inline-sweep',)]
        executed = []
        cur.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(sync, 'connect_raw', lambda: db)
        monkeypatch.setattr(sync, '_sweep_job_state', lambda task_id: 'missing')
        import app_helper
        called = []
        monkeypatch.setattr(
            app_helper.rq_queue_default, 'enqueue',
            lambda *a, **k: called.append(1),
        )
        assert sync.recover_abandoned_sweeps() is None
        assert called == []
        assert not [e for e in executed if e[0].startswith('UPDATE task_status')]

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
            app_helper.rq_queue_default, 'enqueue',
            lambda *a, **k: enqueued.append(1),
        )
        assert sync.recover_abandoned_sweeps() is not None
        assert sync.recover_abandoned_sweeps() is None
        assert enqueued == [1]
        assert connections == [1]

    def test_dashboard_metrics_measure_each_server_against_its_own_catalogue(self, monkeypatch):
        import app_dashboard as dash

        monkeypatch.setattr(dash, '_table_exists', lambda cur, name: True)
        counts = iter([188057, 25])
        monkeypatch.setattr(dash, '_safe_count', lambda cur, sql, params=None: next(counts))
        cur = MagicMock()
        cur.fetchall.return_value = [
            ('s1', 'Jellyfin', 'jellyfin', True, True, None, 188032),
            ('s2', 'PLEX', 'plex', False, True, 120, 46),
            ('s3', 'Fresh', 'navidrome', False, True, None, 0),
        ]
        rows = dash._collect_music_server_metrics(cur)
        assert rows[0]['matched_songs'] == 188057
        assert rows[0]['server_songs'] == 188057
        assert rows[1]['matched_songs'] == 46
        assert rows[1]['server_songs'] == 120
        assert rows[2]['server_songs'] is None
        assert all(r['catalogue_songs'] == 188057 for r in rows)

    def test_sweep_stores_server_track_count(self, monkeypatch):
        from tasks import multiserver_sync as sync

        target = [{'id': 'nav1', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/x.flac'}]
        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 1)
        monkeypatch.setattr(sync, '_unmapped_local_count', lambda conn, sid: 1)
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
            assert sync._prune_stale_mappings(db, 's1', target) == 0
        assert 'pruning skipped' in caplog.text
        db.commit.assert_not_called()
