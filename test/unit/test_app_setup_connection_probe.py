# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Setup-wizard on-the-fly connection testing against user-typed values.

Covers the temporary config patch used by the wizard's "test connection" and
"list libraries" flows: the user-submitted server values must reach the
media-server probe call, and the saved config must always be restored
afterwards, on success and on failure alike.

Main Features:
* _merge_test_config prefers user values, keeps stored secrets for masked
  input, falls back to current config for missing keys, lowercases the type
* _patch_config_for_test applies values to config and _restore_config undoes it
* The probe call observes the user-typed URL/token/type, not the saved ones
* Config is restored after success, provider failure, and empty-result failure
"""

import pytest

import app_setup
import config
from error.error_manager import AudioMuseError


BASELINE = {
    'MEDIASERVER_TYPE': 'navidrome',
    'JELLYFIN_URL': 'https://saved-server:8096',
    'JELLYFIN_TOKEN': 'saved-token',
}

USER_VALUES = {
    'MEDIASERVER_TYPE': 'JELLYFIN',
    'JELLYFIN_URL': 'https://user-typed:8096',
    'JELLYFIN_TOKEN': 'user-token',
}


@pytest.fixture
def saved_config(monkeypatch):
    for key, value in BASELINE.items():
        monkeypatch.setattr(config, key, value)
    return dict(BASELINE)


def _assert_config_matches(expected):
    for key, value in expected.items():
        assert getattr(config, key) == value


class TestMergeTestConfig:
    def test_user_value_wins(self, saved_config):
        merged = app_setup._merge_test_config({'JELLYFIN_URL': 'https://user-typed:8096'})
        assert merged['JELLYFIN_URL'] == 'https://user-typed:8096'

    def test_missing_key_falls_back_to_current_config(self, saved_config):
        merged = app_setup._merge_test_config({})
        assert merged['JELLYFIN_URL'] == 'https://saved-server:8096'
        assert merged['JELLYFIN_TOKEN'] == 'saved-token'

    def test_masked_secret_keeps_stored_value(self, saved_config):
        merged = app_setup._merge_test_config({'JELLYFIN_TOKEN': '********'})
        assert merged['JELLYFIN_TOKEN'] == 'saved-token'

    def test_mediaserver_type_is_lowercased(self, saved_config):
        merged = app_setup._merge_test_config({'MEDIASERVER_TYPE': 'JELLYFIN'})
        assert merged['MEDIASERVER_TYPE'] == 'jellyfin'


class TestPatchAndRestore:
    def test_patch_applies_and_restore_undoes(self, saved_config):
        originals = app_setup._patch_config_for_test({'JELLYFIN_URL': 'https://patched'})
        assert config.JELLYFIN_URL == 'https://patched'
        assert originals['JELLYFIN_URL'] == 'https://saved-server:8096'
        app_setup._restore_config(originals)
        assert config.JELLYFIN_URL == 'https://saved-server:8096'


class TestConnectionProbeSeesUserValues:
    def _fake_probe(self, monkeypatch, captured, items):
        def fake_get_top_played_songs(limit):
            captured['limit'] = limit
            captured['MEDIASERVER_TYPE'] = config.MEDIASERVER_TYPE
            captured['JELLYFIN_URL'] = config.JELLYFIN_URL
            captured['JELLYFIN_TOKEN'] = config.JELLYFIN_TOKEN
            if isinstance(items, Exception):
                raise items
            return items
        monkeypatch.setattr(
            app_setup.mediaserver, 'get_top_played_songs', fake_get_top_played_songs
        )

    def test_probe_call_sees_user_typed_values(self, saved_config, monkeypatch):
        captured = {}
        self._fake_probe(monkeypatch, captured, [{'Id': '1'}, {'Id': '2'}])

        result = app_setup._test_media_server_connection(dict(USER_VALUES))

        assert captured['MEDIASERVER_TYPE'] == 'jellyfin'
        assert captured['JELLYFIN_URL'] == 'https://user-typed:8096'
        assert captured['JELLYFIN_TOKEN'] == 'user-token'
        assert captured['limit'] == config.PROBE_TOP_PLAYED_LIMIT
        assert result['type'] == 'jellyfin'
        assert result['probe_count'] == 2

    def test_config_restored_after_success(self, saved_config, monkeypatch):
        self._fake_probe(monkeypatch, {}, [{'Id': '1'}])
        app_setup._test_media_server_connection(dict(USER_VALUES))
        _assert_config_matches(saved_config)

    def test_config_restored_after_provider_failure(self, saved_config, monkeypatch):
        self._fake_probe(monkeypatch, {}, RuntimeError('connection refused'))
        args = dict(USER_VALUES)
        with pytest.raises(AudioMuseError):
            app_setup._test_media_server_connection(args)
        _assert_config_matches(saved_config)

    def test_config_restored_after_empty_result(self, saved_config, monkeypatch):
        self._fake_probe(monkeypatch, {}, [])
        args = dict(USER_VALUES)
        with pytest.raises(AudioMuseError):
            app_setup._test_media_server_connection(args)
        _assert_config_matches(saved_config)


class TestListLibrariesSeesUserValues:
    def test_list_libraries_sees_user_values_and_restores(self, saved_config, monkeypatch):
        captured = {}

        def fake_list_libraries(provider_type=None):
            captured['provider_type'] = provider_type
            captured['JELLYFIN_URL'] = config.JELLYFIN_URL
            captured['JELLYFIN_TOKEN'] = config.JELLYFIN_TOKEN
            return [{'id': 'lib1', 'name': 'Music'}]

        monkeypatch.setattr(app_setup.mediaserver, 'list_libraries', fake_list_libraries)

        result = app_setup._list_provider_libraries(dict(USER_VALUES))

        assert captured['provider_type'] == 'jellyfin'
        assert captured['JELLYFIN_URL'] == 'https://user-typed:8096'
        assert captured['JELLYFIN_TOKEN'] == 'user-token'
        assert result == [{'id': 'lib1', 'name': 'Music'}]
        _assert_config_matches(saved_config)
