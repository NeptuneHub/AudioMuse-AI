# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the two-level plugin catalog resolution.

Drives ``_fetch_catalog`` with a mocked downloader to verify it follows a
per-plugin ``manifestUrl``, still accepts inline ``versions`` (legacy), picks the
newest compatible version, filters incompatible ones, and records fetch errors.

Main Features:
* No network: the downloader and the installed-plugins query are monkeypatched.
* Covers the manifestUrl indirection plus backward compatibility and error paths.
"""

import json

import plugin.blueprint as blueprint
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
