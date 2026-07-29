# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every media server must be registered in EVERY layer, not just config.py.

config.MEDIASERVER_FIELDS_BY_TYPE is the single source of truth for which
providers exist. A provider added there but missed in one of the hardcoded
lists elsewhere (the dispatcher, the supported-type gates, the setup wizard
JavaScript, the HTML dropdowns, the parameters doc) fails silently: the backend
accepts the type while the UI cannot offer it, or the dispatcher calls a
backend function with a signature it does not have.

Main Features:
* Every backend module binds against every dispatcher call site, so an arity
  mismatch is caught at test time instead of crashing a cron playlist run
* The four Python supported-type gates agree with config
* The setup wizard and multi-server admin JavaScript define credential fields
  for every provider, and mark every secret field secret
* Both HTML dropdowns and the provider-migration credential blocks cover every
  provider, and docs/PARAMETERS.md documents every media-server config field
"""

import inspect
import re
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config

REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDERS = sorted(config.MEDIASERVER_FIELDS_BY_TYPE)

# How tasks/mediaserver/__init__.py calls into a backend: the attribute name,
# the number of POSITIONAL arguments it passes, and the keyword arguments.
DISPATCHER_CALLS = (
    ('get_recent_albums', 1, ()),
    ('get_tracks_from_album', 1, ('user_creds',)),
    ('download_track', 2, ()),
    ('get_all_songs', 0, ('user_creds', 'apply_filter')),
    ('list_libraries', 0, ('user_creds',)),
    ('search_albums', 1, ('user_creds',)),
    ('test_connection', 0, ('user_creds',)),
    ('get_playlist_by_name', 1, ()),
    ('get_all_playlists', 0, ()),
    ('get_playlist_track_ids', 1, ('user_creds',)),
    ('create_playlist', 2, ()),
    ('delete_playlist', 1, ()),
    ('create_instant_playlist', 3, ()),
    ('create_or_replace_playlist', 3, ()),
    ('get_top_played_songs', 2, ()),
    ('get_last_played_time', 2, ()),
    ('get_lyrics', 1, ('timeout',)),
)

# Lyrion is special-cased by the dispatcher and receives no user_creds.
LYRION_NO_CREDS = {
    'get_playlist_track_ids',
    'create_instant_playlist',
    'get_top_played_songs',
    'get_last_played_time',
}


def _read(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding='utf-8')


def _js_object_keys(source, object_name):
    start = source.index(object_name)
    depth = 0
    end = start
    for index in range(source.index('{', start), len(source)):
        if source[index] == '{':
            depth += 1
        elif source[index] == '}':
            depth -= 1
            if depth == 0:
                end = index
                break
    block = source[start:end]
    return set(re.findall(r'^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[', block, re.MULTILINE))


@pytest.mark.parametrize('provider', PROVIDERS)
def test_every_provider_has_a_backend_module(provider):
    assert import_module('tasks.mediaserver.' + provider) is not None


@pytest.mark.parametrize('provider', PROVIDERS)
def test_every_backend_binds_to_every_dispatcher_call_site(provider):
    backend = import_module('tasks.mediaserver.' + provider)
    failures = []
    for attribute, positional, keywords in DISPATCHER_CALLS:
        function = getattr(backend, attribute, None)
        if function is None:
            failures.append(f'{attribute} is missing')
            continue
        if provider == 'lyrion' and attribute in LYRION_NO_CREDS:
            keywords = tuple(k for k in keywords if k != 'user_creds')
            if attribute != 'get_playlist_track_ids':
                positional -= 1
        try:
            inspect.signature(function).bind(*range(positional), **{k: None for k in keywords})
        except TypeError as error:
            failures.append(
                f'{attribute}{inspect.signature(function)} does not accept '
                f'{positional} positional + {list(keywords)}: {error}'
            )
    assert not failures, f'{provider} backend does not match the dispatcher: ' + '; '.join(failures)


def _http_json(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


# What each backend has to be fed for list_libraries to reach its return, and the
# one library both JavaScript consumers must then be able to render.
LIST_LIBRARIES_FIXTURES = {
    'jellyfin': (
        'requests',
        lambda http: setattr(
            http, 'get', MagicMock(return_value=_http_json(
                [{'ItemId': '2', 'Name': 'Main', 'CollectionType': 'music'}]
            ))
        ),
    ),
    'emby': (
        'requests',
        lambda http: setattr(
            http, 'get', MagicMock(return_value=_http_json(
                [{'ItemId': '2', 'Name': 'Main', 'CollectionType': 'music'}]
            ))
        ),
    ),
    'plex': (
        'requests',
        lambda http: setattr(
            http, 'get', MagicMock(return_value=_http_json(
                {'MediaContainer': {'Directory': [
                    {'key': '2', 'title': 'Main', 'type': 'artist'}
                ]}}
            ))
        ),
    ),
    'navidrome': (
        '_navidrome_request',
        lambda stub: stub.configure_mock(
            return_value={'musicFolders': {'musicFolder': [{'id': '2', 'name': 'Main'}]}}
        ),
    ),
    'lyrion': (
        '_jsonrpc_request',
        lambda stub: stub.configure_mock(
            return_value={'folder_loop': [{'id': '2', 'filename': 'Main'}]}
        ),
    ),
    'ampache': (
        '_request',
        lambda stub: stub.configure_mock(return_value={'catalog': [{'id': 2, 'name': 'Main'}]}),
    ),
}


@pytest.mark.parametrize('provider', PROVIDERS)
def test_every_backend_lists_libraries_with_the_lowercase_keys_the_ui_reads(provider):
    backend = import_module('tasks.mediaserver.' + provider)
    attribute, prime = LIST_LIBRARIES_FIXTURES[provider]

    with patch.object(backend, attribute) as stub:
        prime(stub)
        libraries = backend.list_libraries()

    assert libraries, f'{provider} list_libraries returned nothing for a music library'
    for library in libraries:
        assert set(library) == {'id', 'name'}, (
            f"{provider} list_libraries returns {sorted(library)}; static/setup.js and "
            f"static/music_servers_admin.js both read lowercase 'id'/'name', so any "
            f"other shape renders an empty or '[object Object]' library picker"
        )
        assert library['name'], f'{provider} list_libraries returned a nameless library'


def test_dispatcher_provider_names_match_config():
    from tasks.mediaserver import _PROVIDER_NAMES

    assert set(_PROVIDER_NAMES) == set(PROVIDERS)


def test_music_servers_supported_types_match_config():
    from app_music_servers import _SUPPORTED_TYPES

    assert set(_SUPPORTED_TYPES) == set(PROVIDERS)


def test_provider_migration_supported_targets_match_config():
    from app_provider_migration import _SUPPORTED_TARGETS

    assert set(_SUPPORTED_TARGETS) == set(PROVIDERS)


def test_provider_probe_supported_providers_match_config():
    from tasks.provider_probe import _SUPPORTED_PROVIDERS

    assert set(_SUPPORTED_PROVIDERS) == set(PROVIDERS)


@pytest.mark.parametrize('provider', PROVIDERS)
def test_every_provider_field_maps_to_a_cred_key(provider):
    missing = [
        field
        for field in config.MEDIASERVER_FIELDS_BY_TYPE[provider]
        if field not in config.MEDIASERVER_CRED_KEY_BY_FIELD
    ]
    assert not missing, f'{provider} fields absent from MEDIASERVER_CRED_KEY_BY_FIELD: {missing}'


def test_setup_wizard_javascript_defines_fields_for_every_provider():
    keys = _js_object_keys(_read('static/setup.js'), 'serverFields')
    assert set(PROVIDERS) <= keys, f'static/setup.js serverFields is missing: {set(PROVIDERS) - keys}'


def test_music_servers_admin_javascript_defines_creds_for_every_provider():
    keys = _js_object_keys(_read('static/music_servers_admin.js'), 'CRED_FIELDS')
    assert set(PROVIDERS) <= keys, f'music_servers_admin.js CRED_FIELDS is missing: {set(PROVIDERS) - keys}'


def test_setup_wizard_javascript_preserves_every_provider_field_on_type_change():
    source = _read('static/setup.js')
    every_field = {f for fields in config.MEDIASERVER_FIELDS_BY_TYPE.values() for f in fields}
    block = re.search(r'var keys = \[(.*?)\];', source, re.DOTALL).group(1)
    listed = set(re.findall(r"'([A-Z0-9_]+)'", block))
    assert every_field <= listed, f'saveCurrentServerValues drops: {every_field - listed}'


def test_setup_wizard_javascript_marks_every_secret_field_secret():
    import app_setup

    source = _read('static/setup.js')
    block = re.search(r'var secretKeys = \[(.*?)\];', source, re.DOTALL).group(1)
    listed = set(re.findall(r"'([A-Z0-9_]+)'", block))
    mediaserver_secrets = {
        field
        for fields in config.MEDIASERVER_FIELDS_BY_TYPE.values()
        for field in fields
        if field in app_setup.SECRET_FIELDS
    }
    assert mediaserver_secrets <= listed, f'rendered in plain text: {mediaserver_secrets - listed}'


@pytest.mark.parametrize('provider', PROVIDERS)
def test_setup_html_offers_every_provider_in_both_dropdowns(provider):
    source = _read('templates/setup.html')
    assert source.count(f'<option value="{provider}">') >= 2, (
        f'{provider} missing from a MEDIASERVER_TYPE / ms-type dropdown in templates/setup.html'
    )


@pytest.mark.parametrize('provider', PROVIDERS)
def test_provider_migration_html_offers_every_provider_with_cred_fields(provider):
    source = _read('templates/provider_migration.html')
    assert f'<option value="{provider}">' in source, f'{provider} missing from the migration target dropdown'
    assert f'data-for="{provider}"' in source, f'{provider} has no credential block in the migration wizard'


@pytest.mark.parametrize('provider', PROVIDERS)
def test_parameters_doc_documents_every_mediaserver_field(provider):
    source = _read('docs/PARAMETERS.md')
    missing = [
        field for field in config.MEDIASERVER_FIELDS_BY_TYPE[provider] if f'`{field}`' not in source
    ]
    assert not missing, f'docs/PARAMETERS.md does not document: {missing}'


@pytest.mark.parametrize('provider', PROVIDERS)
def test_current_provider_creds_is_built_for_every_provider(provider, monkeypatch):
    import app_provider_migration

    monkeypatch.setattr(config, 'MEDIASERVER_TYPE', provider, raising=False)
    source_type, creds = app_provider_migration._current_provider_creds()
    assert source_type == provider
    expected = {
        config.MEDIASERVER_CRED_KEY_BY_FIELD[field]
        for field in config.MEDIASERVER_FIELDS_BY_TYPE[provider]
        if field in config.MEDIASERVER_CRED_KEY_BY_FIELD
    }
    assert set(creds) == expected
