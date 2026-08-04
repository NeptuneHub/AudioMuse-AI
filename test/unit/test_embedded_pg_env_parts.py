# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""POSTGRES_HOST/POSTGRES_PORT derivation in the Linux and macOS standalone builds.

config.py assembles every connection string from the five POSTGRES_* parts, so
in the standalone builds those parts must carry what the embedded server
actually reported (pgserver picks the socket directory at runtime and falls back
to a hashed runtime path when the pgdata path is too long for a Unix socket).

Main Features:
* A socket URL yields the reported socket directory, not the pgdata guess
* A TCP URL yields its host and port
* An empty or hostless URL falls back to the pgdata directory and 5432
* The parts round-trip through config's URL assembly back to the same host
"""

import importlib.util
import os
import sys

import pytest
from psycopg2.extensions import parse_dsn
from urllib.parse import quote

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

SOCKET_DIR = '/Users/me/Library/Application Support/AudioMuse-AI/pgdata'


def _load_env_module(platform_name):
    for entry in (REPO_ROOT, NATIVE_BUILD):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    mod_name = 'native_env_under_test_' + platform_name
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(NATIVE_BUILD, platform_name, 'env.py')
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        pytest.skip(f"{platform_name} env does not import on this platform: {exc!r}")
    return mod


@pytest.fixture(params=['linux', 'macos'])
def env_mod(request):
    return _load_env_module(request.param)


class TestPgConnParts:
    def test_socket_url_yields_the_reported_socket_directory(self, env_mod):
        url = f"postgresql://postgres:@/postgres?host={SOCKET_DIR}"
        assert env_mod._pg_conn_parts(url) == (SOCKET_DIR, '5432')

    def test_socket_directory_is_not_assumed_to_be_the_pgdata_directory(self, env_mod, monkeypatch):
        monkeypatch.setattr(env_mod.paths, 'pgdata_dir', lambda: '/never/used')
        url = "postgresql://postgres:@/postgres?host=/tmp/pgserver-3f2a1b/"
        assert env_mod._pg_conn_parts(url)[0] == '/tmp/pgserver-3f2a1b/'

    @pytest.mark.parametrize(
        'socket_dir',
        ['/tmp/Audio&Muse', '/tmp/Audio+Muse', '/tmp/Audio#Muse', '/tmp/Audio:Muse'],
        ids=['ampersand', 'plus', 'hash', 'colon'],
    )
    def test_raw_pgserver_socket_query_keeps_special_characters(self, env_mod, socket_dir):
        url = f'postgresql://postgres:@/postgres?host={socket_dir}'
        assert env_mod._pg_conn_parts(url) == (socket_dir, '5432')

    def test_tcp_url_yields_its_host_and_port(self, env_mod):
        assert env_mod._pg_conn_parts('postgresql://postgres:pw@127.0.0.1:5544/postgres') == (
            '127.0.0.1',
            '5544',
        )

    @pytest.mark.parametrize('url', ['', None, 'postgresql://postgres@/postgres'])
    def test_hostless_url_falls_back_to_the_pgdata_directory(self, env_mod, monkeypatch, url):
        monkeypatch.setattr(env_mod.paths, 'pgdata_dir', lambda: '/fallback/pgdata')
        assert env_mod._pg_conn_parts(url) == ('/fallback/pgdata', '5432')

    def test_derived_parts_rebuild_a_url_libpq_resolves_to_the_same_socket(self, env_mod):
        host, port = env_mod._pg_conn_parts(f"postgresql://postgres:@/postgres?host={SOCKET_DIR}")
        rebuilt = f"postgresql://postgres:@{quote(host, safe='[]:')}:{port}/postgres"
        assert parse_dsn(rebuilt)['host'] == SOCKET_DIR
        assert parse_dsn(rebuilt)['port'] == '5432'
