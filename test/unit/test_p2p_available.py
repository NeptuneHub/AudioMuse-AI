# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Whether this build can run P2P Sharing (p2p.available).

Flask and the node both ask this module, so it must answer from spec lookups
and a file lookup alone. These tests replace the spec lookup, point
TOR_BINARY at scripts in a temporary directory and read the module source.

Main Features:
* REQUIRED_MODULES names stem, trio, trio_websocket and coincurve;
  missing_modules lists the absent ones in that order, and a spec lookup that
  raises counts as absent
* tor_binary returns an executable path as it is, finds a bare name on the
  PATH, and answers None for a missing program, a file that is not
  executable, or an empty setting
* unavailable_reason names the missing libraries first, then a missing Tor
  program with the TOR_BINARY value, and is None when both are present
* the module imports nothing but the standard library and config
"""

import ast
import stat

import pytest

import config
from p2p import available

TOR_NAME = 'fake-tor-for-p2p-tests'


def _executable(path):
    path.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class TestMissingModules:
    def test_the_node_libraries(self):
        assert available.REQUIRED_MODULES == ('stem', 'trio', 'trio_websocket', 'coincurve')

    def test_nothing_is_missing_in_this_environment(self):
        assert available.missing_modules() == []

    def test_absent_modules_are_listed_in_order(self, monkeypatch):
        present = {'trio', 'coincurve'}
        monkeypatch.setattr(available.importlib.util, 'find_spec',
                            lambda name: object() if name in present else None)

        assert available.missing_modules() == ['stem', 'trio_websocket']

    @pytest.mark.parametrize('error', [ValueError('stem.__spec__ is None'), ImportError('broken parent')])
    def test_a_spec_lookup_that_raises_counts_as_missing(self, monkeypatch, error):
        def find_spec(name):
            if name == 'stem':
                raise error
            return object()

        monkeypatch.setattr(available.importlib.util, 'find_spec', find_spec)

        assert available.missing_modules() == ['stem']


class TestTorBinary:
    def test_an_executable_path_is_used_as_it_is(self, monkeypatch, tmp_path):
        tor = _executable(tmp_path / 'tor')
        monkeypatch.setattr(config, 'TOR_BINARY', str(tor))

        assert available.tor_binary() == str(tor)

    def test_a_bare_name_is_found_on_the_path(self, monkeypatch, tmp_path):
        tor = _executable(tmp_path / TOR_NAME)
        monkeypatch.setenv('PATH', str(tmp_path))
        monkeypatch.setattr(config, 'TOR_BINARY', TOR_NAME)

        assert available.tor_binary() == str(tor)

    def test_a_file_that_is_not_executable_is_not_the_tor_program(self, monkeypatch, tmp_path):
        tor = tmp_path / 'tor'
        tor.write_text('not a program', encoding='utf-8')
        tor.chmod(stat.S_IRUSR | stat.S_IWUSR)
        monkeypatch.setattr(config, 'TOR_BINARY', str(tor))

        assert available.tor_binary() is None

    @pytest.mark.parametrize('value', ['', '   ', None, 'no-such-tor-program-for-tests'])
    def test_a_missing_program_or_an_empty_setting_is_none(self, monkeypatch, tmp_path, value):
        monkeypatch.setenv('PATH', str(tmp_path))
        monkeypatch.setattr(config, 'TOR_BINARY', value)

        assert available.tor_binary() is None


class TestUnavailableReason:
    def test_the_missing_libraries_come_first(self, monkeypatch):
        monkeypatch.setattr(available, 'missing_modules', lambda: ['stem', 'coincurve'])
        monkeypatch.setattr(available, 'tor_binary', lambda: None)

        assert available.unavailable_reason() == 'this build lacks the Python libraries stem, coincurve'

    def test_then_a_missing_tor_program_with_its_setting(self, monkeypatch, tmp_path):
        monkeypatch.setattr(available, 'missing_modules', lambda: [])
        monkeypatch.setenv('PATH', str(tmp_path))
        monkeypatch.setattr(config, 'TOR_BINARY', 'no-such-tor-program-for-tests')

        assert available.unavailable_reason() == 'the Tor program (no-such-tor-program-for-tests) is not installed'

    def test_nothing_is_missing_means_no_reason(self, monkeypatch, tmp_path):
        tor = _executable(tmp_path / 'tor')
        monkeypatch.setattr(available, 'missing_modules', lambda: [])
        monkeypatch.setattr(config, 'TOR_BINARY', str(tor))

        assert available.unavailable_reason() is None


class TestImports:
    def test_only_the_standard_library_and_config_are_imported(self):
        with open(available.__file__, 'r', encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)

        assert imported == {'importlib.util', 'os', 'shutil', 'config'}
