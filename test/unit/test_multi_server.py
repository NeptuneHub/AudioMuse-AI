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
* Matcher tiers and safe canonical/provider id translation.
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

    def test_needs_translation_always_true(self):
        import app_server_context as asc

        assert asc.needs_translation(None) is True
        assert asc.needs_translation('def') is True
        assert asc.needs_translation('sec') is True


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
        monkeypatch.setattr(
            helper, 'get_missing_chromaprint_ids', lambda ids: {'provider-new'}
        )

        existing, needs_clap, needs_lyrics, needs_chromaprint = helper.compute_album_needs(
            tracks, False, False, server_id='server-b'
        )

        assert queried['ids'] == ['fp_known', 'provider-new']
        assert existing == 1
        assert needs_clap is False
        assert needs_lyrics is False
        assert needs_chromaprint is True


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


class TestEmbeddingCanonicalization:
    def test_builds_canonical_ids_from_stored_embeddings(self):
        from tasks import audio_fingerprint as afp
        from tasks import fingerprint_canonicalize as canonicalize

        chromaprint = 'AQAAE0mUaEkSRZEGAA'
        cursor = MagicMock()
        cursor.fetchall.side_effect = [
            [('legacy-provider-id', chromaprint)],
            [],
        ]
        mapping, duplicate_mapping = canonicalize._build_mapping(cursor)
        assert mapping == {
            'legacy-provider-id': afp.chromaprint_canonical_id(chromaprint)
        }
        assert duplicate_mapping == {}

    def test_duplicate_embeddings_keep_one_canonical_row(self):
        from tasks import fingerprint_canonicalize as canonicalize

        chromaprint = 'AQAAE0mUaEkSRZEGAA'
        cursor = MagicMock()
        cursor.fetchall.side_effect = [
            [('copy-one', chromaprint), ('copy-two', chromaprint)],
            [],
        ]
        mapping, duplicate_mapping = canonicalize._build_mapping(cursor)
        assert list(mapping.keys()) == ['copy-one']
        assert duplicate_mapping == {'copy-two': next(iter(mapping.values()))}

    def test_preserves_recovered_default_provider_id(self):
        from tasks.fingerprint_canonicalize import _default_provider_ids

        cursor = MagicMock()
        cursor.fetchall.return_value = [('legacy-score-id', 'current-jellyfin-id')]

        result = _default_provider_ids(cursor, 'default-server', {'legacy-score-id': 'fp_hash'})

        assert result == {'legacy-score-id': 'current-jellyfin-id'}


class TestSweepAlignment:
    def test_aligned_server_is_noop_without_fetch(self, monkeypatch):
        from tasks import multiserver_sync as sync

        monkeypatch.setattr(sync, '_local_track_count', lambda conn: 5)
        monkeypatch.setattr(sync, '_unmapped_local_rows', lambda conn, sid: [])
        fetched = []
        monkeypatch.setattr(
            sync.provider_probe, 'fetch_all_tracks',
            lambda *a: fetched.append(1) or [],
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
        monkeypatch.setattr(sync, '_unmapped_local_rows', lambda conn, sid: rows)
        monkeypatch.setattr(sync, '_already_mapped_ids', lambda db, sid: set())
        monkeypatch.setattr(sync.provider_probe, 'fetch_all_tracks', lambda *a: target)
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
        assert written == {'fp_1': 'nav1'}
