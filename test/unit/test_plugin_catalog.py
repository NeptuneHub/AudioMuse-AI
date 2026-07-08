# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the plugin catalog resolution.

Drives ``_fetch_catalog`` with a mocked downloader to verify it follows a
catalog entry's ``pluginUrl`` to a flat ``plugin.json`` (the current format),
still accepts a legacy per-plugin manifest with a ``versions`` list and inline
``versions``, picks the newest compatible version, filters incompatible ones,
and records fetch errors.

Main Features:
* No network: the downloader and the installed-plugins query are monkeypatched.
* Covers the pluginUrl indirection plus backward compatibility and error paths.
"""

import json
import time
from urllib.parse import urlparse

import flask

import plugin.blueprint as blueprint
import plugin.net as net
import database


def _wire(monkeypatch, download):
    monkeypatch.setattr(blueprint, '_get_repos', lambda: ['https://example.com/manifest.json'])
    monkeypatch.setattr(blueprint, '_download', download)
    monkeypatch.setattr(database, 'list_plugins', lambda conn=None: [])


class TestCatalogResolution:
    def test_follows_manifest_url_and_picks_newest(self, monkeypatch):
        catalog = {'plugins': [{
            'id': 'song_counter', 'name': 'SongCounter', 'description': 'd', 'author': 'a',
            'manifestUrl': 'https://example.com/dist/song_counter/manifest.json',
        }]}
        per_plugin = {'id': 'song_counter', 'versions': [
            {'version': '1.0.0', 'min_core_version': '2.5.0', 'sourceUrl': 'https://e/1.0.0.zip', 'checksum': 'aaa'},
            {'version': '1.1.0', 'min_core_version': '2.5.0', 'sourceUrl': 'https://e/1.1.0.zip', 'checksum': 'bbb'},
        ]}

        def download(url, _max):
            return json.dumps(per_plugin if '/dist/' in url else catalog).encode()

        _wire(monkeypatch, download)
        plugins, errors = blueprint._fetch_catalog()
        assert errors == []
        assert len(plugins) == 1
        assert plugins[0]['latest_version'] == '1.1.0'
        assert plugins[0]['checksum'] == 'bbb'
        assert plugins[0]['source_url'].endswith('1.1.0.zip')
        assert plugins[0]['name'] == 'SongCounter'

    def test_follows_plugin_url_flat_plugin_json(self, monkeypatch):
        catalog = {'plugins': [{
            'id': 'song_counter',
            'pluginUrl': 'https://example.com/dist/song_counter/plugin.json',
        }]}
        plugin_json = {
            'id': 'song_counter', 'name': 'SongCounter', 'description': 'd', 'author': 'a',
            'version': '1.5.0', 'min_core_version': '2.5.0', 'changelog': 'c',
            'sourceUrl': 'https://e/song_counter_1.5.0.zip', 'checksum': 'abc',
        }

        def download(url, _max):
            return json.dumps(plugin_json if '/dist/' in url else catalog).encode()

        _wire(monkeypatch, download)
        plugins, errors = blueprint._fetch_catalog()
        assert errors == []
        assert len(plugins) == 1
        assert plugins[0]['id'] == 'song_counter'
        assert plugins[0]['name'] == 'SongCounter'
        assert plugins[0]['latest_version'] == '1.5.0'
        assert plugins[0]['checksum'] == 'abc'
        assert plugins[0]['source_url'].endswith('song_counter_1.5.0.zip')

    def test_plugin_url_versions_list_picks_newest_and_per_version_image(self, monkeypatch):
        catalog = {'plugins': [{
            'id': 'song_counter', 'name': 'SongCounter', 'author': 'NeptuneHub',
            'description': 'stable catalog description',
            'pluginUrl': 'https://example.com/dist/song_counter/plugin.json',
        }]}
        plugin_json = {
            'id': 'song_counter', 'name': 'SongCounter', 'author': 'NeptuneHub',
            'targets': ['flask'], 'requirements': ['matplotlib'],
            'versions': [
                {'version': '1.5.0', 'min_core_version': '2.5.0', 'changelog': 'new',
                 'imageUrl': 'https://img/1.5.0.png', 'sourceUrl': 'https://e/sc_1.5.0.zip', 'checksum': 'new'},
                {'version': '1.4.0', 'min_core_version': '2.4.0', 'changelog': 'old',
                 'imageUrl': 'https://img/1.4.0.png', 'sourceUrl': 'https://e/sc_1.4.0.zip', 'checksum': 'old'},
            ],
        }

        def download(url, _max):
            return json.dumps(plugin_json if '/dist/' in url else catalog).encode()

        _wire(monkeypatch, download)
        plugins, errors = blueprint._fetch_catalog()
        assert errors == []
        assert len(plugins) == 1
        best = plugins[0]
        assert best['latest_version'] == '1.5.0'
        assert best['checksum'] == 'new'
        assert best['source_url'].endswith('sc_1.5.0.zip')
        assert best['image_url'] == 'https://img/1.5.0.png'
        assert best['description'] == 'stable catalog description'
        assert best['author'] == 'NeptuneHub'

    def test_flat_plugin_json_incompatible_filtered(self, monkeypatch):
        catalog = {'plugins': [{'id': 'y', 'pluginUrl': 'https://example.com/dist/y/plugin.json'}]}
        plugin_json = {'id': 'y', 'name': 'Y', 'version': '3.0.0', 'min_core_version': '99.0.0',
                       'sourceUrl': 'https://e/y.zip', 'checksum': 'c'}

        def download(url, _max):
            return json.dumps(plugin_json if '/dist/' in url else catalog).encode()

        _wire(monkeypatch, download)
        plugins, _errors = blueprint._fetch_catalog()
        assert plugins == []

    def test_inline_versions_backward_compatible(self, monkeypatch):
        catalog = {'plugins': [{
            'id': 'x', 'name': 'X',
            'versions': [{'version': '2.0.0', 'min_core_version': '2.5.0', 'sourceUrl': 'https://e/x.zip', 'checksum': 'c'}],
        }]}
        _wire(monkeypatch, lambda url, _max: json.dumps(catalog).encode())
        plugins, errors = blueprint._fetch_catalog()
        assert errors == []
        assert len(plugins) == 1
        assert plugins[0]['latest_version'] == '2.0.0'

    def test_incompatible_version_filtered(self, monkeypatch):
        catalog = {'plugins': [{'id': 'y', 'name': 'Y', 'manifestUrl': 'https://example.com/dist/y/manifest.json'}]}
        per_plugin = {'id': 'y', 'versions': [
            {'version': '3.0.0', 'min_core_version': '99.0.0', 'sourceUrl': 'https://e/y.zip', 'checksum': 'c'},
        ]}

        def download(url, _max):
            return json.dumps(per_plugin if '/dist/' in url else catalog).encode()

        _wire(monkeypatch, download)
        plugins, _errors = blueprint._fetch_catalog()
        assert plugins == []

    def test_manifest_fetch_error_recorded(self, monkeypatch):
        catalog = {'plugins': [{'id': 'z', 'name': 'Z', 'manifestUrl': 'https://example.com/dist/z/manifest.json'}]}

        def download(url, _max):
            if '/dist/' in url:
                raise ValueError('boom')
            return json.dumps(catalog).encode()

        _wire(monkeypatch, download)
        plugins, errors = blueprint._fetch_catalog()
        assert plugins == []
        assert errors and 'boom' in errors[0]['error']

    def test_catalog_download_error_surfaces_clean_message(self, monkeypatch):
        def boom(url, _max):
            raise net.DownloadError('Could not reach raw.githubusercontent.com for the plugin download')

        monkeypatch.setattr(blueprint, '_get_repos', lambda: ['https://raw.githubusercontent.com/m.json'])
        monkeypatch.setattr(blueprint, '_download', boom)
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: [])
        plugins, errors = blueprint._fetch_catalog()
        assert plugins == []
        assert errors and 'raw.githubusercontent.com' in errors[0]['error']


class TestInstallErrorResponse:
    def _client(self):
        app = flask.Flask(__name__)
        app.register_blueprint(blueprint.plugins_bp)
        return app.test_client()

    def test_download_failure_returns_502_with_real_message(self, monkeypatch):
        monkeypatch.setattr(
            blueprint, '_resolve_install_source',
            lambda pid, version=None: ('https://raw.githubusercontent.com/x/y.zip', 'abc', 'repo', {'id': 'demo'}),
        )

        def boom(url, _max):
            raise net.DownloadError('Could not reach raw.githubusercontent.com for the plugin download')

        monkeypatch.setattr(blueprint, '_download', boom)
        resp = self._client().post('/api/plugins/install', json={'id': 'demo'})
        assert resp.status_code == 502
        assert resp.get_json()['error'] == 'Could not reach raw.githubusercontent.com for the plugin download'


class TestJsdelivrMirrorFallback:
    def test_raw_github_url_maps_to_jsdelivr(self):
        url = 'https://raw.githubusercontent.com/NeptuneHub/AudioMuse-AI-plugins/main/dist/song_counter/song_counter_1.5.1.zip'
        assert net._jsdelivr_mirror(url) == (
            'https://cdn.jsdelivr.net/gh/NeptuneHub/AudioMuse-AI-plugins@main/dist/song_counter/song_counter_1.5.1.zip'
        )

    def test_non_github_url_has_no_mirror(self):
        assert net._jsdelivr_mirror('https://example.com/manifest.json') is None
        assert net._jsdelivr_mirror('https://raw.githubusercontent.com/too/short') is None

    def test_refs_heads_prefix_is_normalized(self):
        url = 'https://raw.githubusercontent.com/u/r/refs/heads/main/manifest.json'
        assert net._jsdelivr_mirror(url) == 'https://cdn.jsdelivr.net/gh/u/r@main/manifest.json'
        tag = 'https://raw.githubusercontent.com/u/r/refs/tags/v1.0/manifest.json'
        assert net._jsdelivr_mirror(tag) == 'https://cdn.jsdelivr.net/gh/u/r@v1.0/manifest.json'

    def test_tokened_url_has_no_mirror(self):
        assert net._jsdelivr_mirror('https://raw.githubusercontent.com/u/r/main/f.json?token=abc') is None

    def test_fallback_used_when_raw_github_fails(self, monkeypatch):
        calls = []

        def fake_once(url, _max):
            calls.append(url)
            if urlparse(url).hostname == 'raw.githubusercontent.com':
                raise net.DownloadError('Timed out reaching raw.githubusercontent.com')
            return b'mirrored'

        monkeypatch.setattr(net, '_download_once', fake_once)
        data = net.download('https://raw.githubusercontent.com/u/r/main/manifest.json', 1024)
        assert data == b'mirrored'
        assert calls == [
            'https://raw.githubusercontent.com/u/r/main/manifest.json',
            'https://cdn.jsdelivr.net/gh/u/r@main/manifest.json',
        ]

    def test_error_mentions_both_hosts_when_mirror_also_fails(self, monkeypatch):
        def fake_once(url, _max):
            raise net.DownloadError('Timed out reaching ' + net._host(url))

        monkeypatch.setattr(net, '_download_once', fake_once)
        try:
            net.download('https://raw.githubusercontent.com/u/r/main/manifest.json', 1024)
            raise AssertionError('expected DownloadError')
        except net.DownloadError as exc:
            assert str(exc) == (
                'Timed out reaching raw.githubusercontent.com '
                '(the jsDelivr mirror also failed: Timed out reaching cdn.jsdelivr.net)'
            )

    def test_no_fallback_for_non_github_hosts(self, monkeypatch):
        calls = []

        def fake_once(url, _max):
            calls.append(url)
            raise net.DownloadError('down')

        monkeypatch.setattr(net, '_download_once', fake_once)
        try:
            net.download('https://example.com/manifest.json', 1024)
            raise AssertionError('expected DownloadError')
        except net.DownloadError:
            pass
        assert calls == ['https://example.com/manifest.json']


class TestCatalogAutoRefresh:
    def test_starts_once(self, monkeypatch):
        import threading as _threading
        started = []
        monkeypatch.setattr(blueprint, '_auto_refresh_started', False)
        monkeypatch.setattr(
            _threading, 'Thread',
            lambda **kw: started.append(kw.get('name')) or type('T', (), {'start': lambda self: None})(),
        )
        blueprint.start_catalog_auto_refresh()
        blueprint.start_catalog_auto_refresh()
        assert started == ['plugin-catalog-auto-refresh']


class TestCatalogCache:
    def _mock_store(self, monkeypatch):
        store = {}
        monkeypatch.setattr(database, 'set_app_config_value', lambda k, v: store.__setitem__(k, v))
        monkeypatch.setattr(database, 'get_app_config_value', lambda k: store.get(k))
        return store

    def test_store_and_load_roundtrip(self, monkeypatch):
        self._mock_store(monkeypatch)
        blueprint._store_catalog_cache([{'id': 'a', 'latest_version': '2.0.0'}], [])
        plugins, errors, at = blueprint._load_catalog_cache()
        assert plugins == [{'id': 'a', 'latest_version': '2.0.0'}]
        assert errors == []
        assert at > 0
        versions, _ = blueprint._cached_latest_versions()
        assert versions == {'a': '2.0.0'}

    def test_empty_store_with_errors_preserves_previous_plugins(self, monkeypatch):
        self._mock_store(monkeypatch)
        blueprint._store_catalog_cache([{'id': 'a', 'latest_version': '2.0.0'}], [])
        blueprint._store_catalog_cache([], [{'repo': 'r', 'error': 'down'}])
        plugins, errors, _ = blueprint._load_catalog_cache()
        assert plugins == [{'id': 'a', 'latest_version': '2.0.0'}]
        assert errors == [{'repo': 'r', 'error': 'down'}]

    def test_clean_empty_store_clears_delisted_plugins(self, monkeypatch):
        self._mock_store(monkeypatch)
        blueprint._store_catalog_cache([{'id': 'a', 'latest_version': '2.0.0'}], [])
        blueprint._store_catalog_cache([], [])
        plugins, errors, _ = blueprint._load_catalog_cache()
        assert plugins == []
        assert errors == []

    def test_load_survives_db_error(self, monkeypatch):
        def boom(_k):
            raise RuntimeError('no app context')
        monkeypatch.setattr(database, 'get_app_config_value', boom)
        assert blueprint._load_catalog_cache() == ([], [], 0.0)


class TestVersionHelpers:
    def test_pick_version_skips_entries_without_source_url(self, monkeypatch):
        monkeypatch.setattr(blueprint.config, 'APP_VERSION', 'v2.5.0')
        best = blueprint._pick_version([
            {'version': '2.0.0', 'min_core_version': '2.5.0'},
            {'version': '1.0.0', 'min_core_version': '2.5.0', 'sourceUrl': 'https://e/1.zip', 'checksum': 'c'},
        ])
        assert best['version'] == '1.0.0'

    def test_is_newer_version_numeric(self):
        assert blueprint._is_newer_version('1.5.1', '1.5.0') is True
        assert blueprint._is_newer_version('1.5.0', '1.5.1') is False
        assert blueprint._is_newer_version('1.0', '1.0.0') is False
        assert blueprint._is_newer_version('1.10.0', '1.9.0') is True
        assert blueprint._is_newer_version(None, '1.0.0') is False


class TestInstallVersionPin:
    def _client(self):
        app = flask.Flask(__name__)
        app.register_blueprint(blueprint.plugins_bp)
        return app.test_client()

    def test_unavailable_version_returns_409_without_side_effects(self, monkeypatch):
        cache = json.dumps({
            'at': time.time(),
            'plugins': [{'id': 'demo', 'name': 'Demo', 'latest_version': '1.5.0',
                         'source_url': 'https://e/sc_1.5.0.zip', 'checksum': 'abc',
                         'versions': [{'version': '1.5.0', 'sourceUrl': 'https://e/sc_1.5.0.zip',
                                       'checksum': 'abc'}]}],
            'errors': [],
        })
        monkeypatch.setattr(database, 'get_app_config_value', lambda k: cache)

        def boom(*_a, **_k):
            raise AssertionError('must not download when the requested version is unavailable')

        monkeypatch.setattr(blueprint, '_download', boom)
        resp = self._client().post('/api/plugins/install', json={'id': 'demo', 'version': '1.5.1'})
        assert resp.status_code == 409
        assert '1.5.1' in resp.get_json()['error']

    def test_rollback_to_listed_older_version(self, monkeypatch):
        cache = json.dumps({
            'at': time.time(),
            'plugins': [{'id': 'demo', 'name': 'Demo', 'latest_version': '1.5.1',
                         'source_url': 'https://e/sc_1.5.1.zip', 'checksum': 'new',
                         'versions': [
                             {'version': '1.5.1', 'sourceUrl': 'https://e/sc_1.5.1.zip', 'checksum': 'new'},
                             {'version': '1.4.0', 'sourceUrl': 'https://e/sc_1.4.0.zip', 'checksum': 'old'},
                         ]}],
            'errors': [],
        })
        monkeypatch.setattr(database, 'get_app_config_value', lambda k: cache)
        downloaded = {}
        monkeypatch.setattr(blueprint, '_download',
                            lambda url, cap: downloaded.__setitem__('url', url) or b'zipbytes')
        captured = {}

        def fake_install(package, meta, **kwargs):
            captured['meta'] = meta
            captured['checksum'] = kwargs.get('expected_checksum')
            return meta, True, None

        monkeypatch.setattr(blueprint.plugin_manager, 'install_package', fake_install)
        resp = self._client().post('/api/plugins/install', json={'id': 'demo', 'version': '1.4.0'})
        assert resp.status_code == 200
        assert downloaded['url'] == 'https://e/sc_1.4.0.zip'
        assert captured['meta']['version'] == '1.4.0'
        assert captured['checksum'] == 'old'

    def test_matching_version_proceeds(self, monkeypatch):
        monkeypatch.setattr(
            blueprint, '_resolve_install_source',
            lambda pid, version=None: ('https://e/sc_1.5.1.zip', 'abc', 'repo', {'id': 'demo', 'version': '1.5.1'}),
        )
        monkeypatch.setattr(blueprint, '_download', lambda url, cap: b'zipbytes')
        monkeypatch.setattr(
            blueprint.plugin_manager, 'install_package',
            lambda *a, **k: ({'id': 'demo', 'version': '1.5.1'}, True, None),
        )
        resp = self._client().post('/api/plugins/install', json={'id': 'demo', 'version': '1.5.1'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['status'] == 'ok'
        assert body['deps_ok'] is True

    def test_deps_failure_surfaces_in_install_response(self, monkeypatch):
        monkeypatch.setattr(
            blueprint, '_resolve_install_source',
            lambda pid, version=None: ('https://e/sc.zip', 'abc', 'repo', {'id': 'demo', 'version': '1.0.0'}),
        )
        monkeypatch.setattr(blueprint, '_download', lambda url, cap: b'zipbytes')
        monkeypatch.setattr(
            blueprint.plugin_manager, 'install_package',
            lambda *a, **k: ({'id': 'demo'}, False, 'ERROR: No matching distribution found for nosuchpkg'),
        )
        body = self._client().post('/api/plugins/install', json={'id': 'demo'}).get_json()
        assert body['status'] == 'ok'
        assert body['deps_ok'] is False
        assert 'nosuchpkg' in body['deps_error']


def _cache_payload(cached_version, at=None):
    return json.dumps({
        'at': time.time() if at is None else at,
        'plugins': [{'id': 'song_counter', 'name': 'SongCounter', 'latest_version': cached_version,
                     'source_url': 'https://e/sc.zip', 'checksum': 'c', 'installed_version': 'stale'}],
        'errors': [],
    })


class TestInstalledUpdateFlag:
    def _client(self):
        app = flask.Flask(__name__)
        app.register_blueprint(blueprint.plugins_bp)
        return app.test_client()

    def _wire(self, monkeypatch, installed_version, cached_version):
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: [{
            'id': 'song_counter', 'name': 'SongCounter', 'version': installed_version,
            'manifest': {}, 'settings': {}, 'enabled': True, 'requirements': [], 'load_status': 'ok',
        }])
        monkeypatch.setattr(blueprint.plugin_manager, 'registry', lambda: [])
        monkeypatch.setattr(blueprint.plugin_manager, 'get_settings_endpoint', lambda pid: None)
        monkeypatch.setattr(database, 'get_app_config_value', lambda k: _cache_payload(cached_version))

    def test_update_available_from_cache(self, monkeypatch):
        self._wire(monkeypatch, installed_version='1.5.0', cached_version='1.5.1')
        plugin = self._client().get('/api/plugins/installed').get_json()['plugins'][0]
        assert plugin['update_available'] is True
        assert plugin['latest_version'] == '1.5.1'

    def test_no_update_when_versions_match(self, monkeypatch):
        self._wire(monkeypatch, installed_version='1.5.1', cached_version='1.5.1')
        plugin = self._client().get('/api/plugins/installed').get_json()['plugins'][0]
        assert plugin['update_available'] is False

    def test_installed_never_fetches_the_catalog_inline(self, monkeypatch):
        self._wire(monkeypatch, installed_version='1.5.0', cached_version='1.5.1')
        monkeypatch.setattr(database, 'get_app_config_value', lambda k: None)

        def boom(*_a, **_k):
            raise AssertionError('api_installed must never fetch the catalog synchronously')

        monkeypatch.setattr(blueprint, '_fetch_catalog', boom)
        monkeypatch.setattr(blueprint, '_refresh_catalog_cache_async', lambda force=False: True)
        resp = self._client().get('/api/plugins/installed')
        assert resp.status_code == 200
        assert resp.get_json()['plugins'][0]['update_available'] is False


class TestCatalogEndpointServesCache:
    def _client(self):
        app = flask.Flask(__name__)
        app.register_blueprint(blueprint.plugins_bp)
        return app.test_client()

    def _wire(self, monkeypatch):
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: [{
            'id': 'song_counter', 'name': 'SongCounter', 'version': '1.5.0',
            'manifest': {}, 'settings': {}, 'enabled': True, 'requirements': [], 'load_status': 'ok',
        }])
        monkeypatch.setattr(database, 'get_app_config_value', lambda k: _cache_payload('1.5.1'))
        monkeypatch.setattr(blueprint, '_get_repos', lambda: ['https://example.com/manifest.json'])

        def boom(*_a, **_k):
            raise AssertionError('api_catalog must never download synchronously')

        monkeypatch.setattr(blueprint, '_download', boom)

    def test_catalog_served_from_cache_without_network(self, monkeypatch):
        self._wire(monkeypatch)
        data = self._client().get('/api/plugins/catalog').get_json()
        assert data['plugins'][0]['latest_version'] == '1.5.1'
        assert data['plugins'][0]['installed_version'] == '1.5.0'
        assert data['refreshing'] is False
        assert data['cached_at'] > 0

    def test_stale_installed_version_overwritten_from_db(self, monkeypatch):
        self._wire(monkeypatch)
        data = self._client().get('/api/plugins/catalog').get_json()
        assert data['plugins'][0]['installed_version'] == '1.5.0'

    def test_force_refresh_spawns_background_and_still_serves_cache(self, monkeypatch):
        self._wire(monkeypatch)
        called = {'force': None}
        monkeypatch.setattr(blueprint, '_refresh_catalog_cache_async',
                            lambda force=False: called.__setitem__('force', force) or True)
        data = self._client().get('/api/plugins/catalog?refresh=1').get_json()
        assert called['force'] is True
        assert data['refreshing'] is True
        assert data['plugins'][0]['latest_version'] == '1.5.1'
