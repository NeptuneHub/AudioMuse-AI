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
            lambda pid: ('https://raw.githubusercontent.com/x/y.zip', 'abc', 'repo', {'id': 'demo'}),
        )

        def boom(url, _max):
            raise net.DownloadError('Could not reach raw.githubusercontent.com for the plugin download')

        monkeypatch.setattr(blueprint, '_download', boom)
        resp = self._client().post('/api/plugins/install', json={'id': 'demo'})
        assert resp.status_code == 502
        assert 'raw.githubusercontent.com' in resp.get_json()['error']
