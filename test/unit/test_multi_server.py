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
* Active-server context accessors, nesting, and reset isolation.
* Credential masking/merge, config-derived creds, and registry row normalization.
* MBID matcher tier priority and id translation (identity for the default).
* BoundServer runs dispatcher calls inside the selected server's context.
"""

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


class TestMbidMatcherTier:
    def test_mbid_matches_even_when_metadata_differs(self):
        from tasks.provider_migration_matcher import match_tracks

        old = [{
            'item_id': 'A', 'title': 'Song', 'author': 'Art', 'album': 'Alb',
            'file_path': '/music/x.flac', 'mbid': 'MB-123',
        }]
        new = [{
            'id': 'n9', 'title': 'unrelated', 'artist': 'other', 'album': 'zzz',
            'path': '/other/y.flac', 'mbid': 'mb-123',
        }]
        result = match_tracks(old, new)
        assert result['matches'] == {'A': 'n9'}
        assert result['match_tiers']['A'] == 'mbid'
        assert result['tier_counts']['mbid'] == 1

    def test_mbid_takes_priority_over_path(self):
        from tasks.provider_migration_matcher import match_tracks

        old = [{
            'item_id': 'A', 'title': 't', 'author': 'a', 'album': 'al',
            'file_path': '/music/same.flac', 'mbid': 'ID1',
        }]
        new = [
            {'id': 'by_path', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/music/same.flac'},
            {'id': 'by_mbid', 'title': 'x', 'artist': 'y', 'album': 'z', 'path': '/nope.flac', 'mbid': 'id1'},
        ]
        result = match_tracks(old, new)
        assert result['matches']['A'] == 'by_mbid'
        assert result['match_tiers']['A'] == 'mbid'

    def test_absent_mbid_falls_back_to_existing_tiers(self):
        from tasks.provider_migration_matcher import match_tracks

        old = [{
            'item_id': 'A', 'title': 't', 'author': 'a', 'album': 'al',
            'file_path': '/music/same.flac', 'mbid': None,
        }]
        new = [{'id': 'n1', 'title': 't', 'artist': 'a', 'album': 'al', 'path': '/music/same.flac'}]
        result = match_tracks(old, new)
        assert result['matches'] == {'A': 'n1'}
        assert result['match_tiers']['A'] == 'path'


class TestBoundServer:
    def test_for_server_runs_call_in_context(self, monkeypatch):
        from tasks import mediaserver
        from tasks.mediaserver import registry, context

        fake_ctx = {
            'server_id': 's2', 'server_type': 'plex',
            'creds': {'url': 'u', 'token': 't'}, 'music_libraries': 'lib',
        }
        monkeypatch.setattr(registry, 'context_for', lambda sid: fake_ctx)

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

        monkeypatch.setattr(registry, 'context_for', lambda sid: None)
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
        import config
        from tasks.mediaserver import registry

        monkeypatch.setattr(config, 'CATALOG_FINGERPRINT_AS_ID', True, raising=False)
        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        cursor = MagicMock()
        cursor.fetchall.return_value = [('fp_1', 'prov1')]
        conn = MagicMock()
        conn.cursor.return_value = cursor
        result = registry.translate_ids(['fp_1', 'raw2'], None, conn=conn)
        assert result == {'fp_1': 'prov1', 'raw2': 'raw2'}

    def test_default_identity_when_flag_off(self, monkeypatch):
        import config
        from tasks.mediaserver import registry

        monkeypatch.setattr(config, 'CATALOG_FINGERPRINT_AS_ID', False, raising=False)
        monkeypatch.setattr(registry, 'get_default_server', lambda conn=None: {'server_id': 'def'})
        conn = MagicMock()
        assert registry.translate_ids(['a', 'b'], None, conn=conn) == {'a': 'a', 'b': 'b'}

    def test_needs_translation(self, monkeypatch):
        import config
        import app_server_context as asc
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'get_default_server_id', lambda conn=None: 'def')
        monkeypatch.setattr(config, 'CATALOG_FINGERPRINT_AS_ID', False, raising=False)
        assert asc.needs_translation(None) is False
        assert asc.needs_translation('def') is False
        assert asc.needs_translation('sec') is True
        monkeypatch.setattr(config, 'CATALOG_FINGERPRINT_AS_ID', True, raising=False)
        assert asc.needs_translation(None) is True
        assert asc.needs_translation('def') is True


class TestResolveRequestServer:
    def _flask_app(self):
        from flask import Flask
        return Flask(__name__)

    def test_reads_query_and_body_and_validates(self, monkeypatch):
        import app_server_context as asc
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry, 'get_server',
            lambda sid, conn=None: {'server_id': sid} if sid == 'known' else None,
        )
        app = self._flask_app()
        with app.test_request_context('/api/create_playlist?server=known'):
            assert asc.resolve_request_server_id() == 'known'
        with app.test_request_context('/api/create_playlist'):
            assert asc.resolve_request_server_id({'server': 'known'}) == 'known'
        with app.test_request_context('/api/create_playlist'):
            assert asc.resolve_request_server_id() is None
        with app.test_request_context('/api/create_playlist?server=ghost'):
            with pytest.raises(ValueError):
                asc.resolve_request_server_id()
