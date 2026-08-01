# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Argv handling in the three standalone launchers.

Guards issue #827 end to end: a frozen launcher handed the argv of a
multiprocessing/loky worker must run that worker, never re-enter its own
menu-bar/tray/supervisor path (which on macOS opened a browser window per
spawned process), and must reject any other unrecognised argv.

Main Features:
* No launcher starts its user-facing UI when handed spawned-worker argv
* The macOS launcher rejects unknown arguments instead of opening a browser
* The macOS launcher still starts for a bare or Finder launch and for --role=
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

EXE = '/opt/audiomuse/AudioMuse-AI'

LOKY_WORKER_ARGV = [
    EXE,
    '-m', 'joblib.externals.loky.backend.popen_loky_posix',
    '--process-name', 'LokyProcess-3',
    '--pipe', '11',
]

UI_ENTRY_POINT = {
    'macos': '_run_menubar',
    'linux': '_run_supervisor',
    'windows': '_run_tray',
}


def _load_launcher(platform_name):
    for entry in (REPO_ROOT, NATIVE_BUILD):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    mod_name = 'native_launcher_under_test_' + platform_name
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(NATIVE_BUILD, platform_name, 'launcher.py')
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        pytest.skip(f"{platform_name} launcher does not import on this platform: {exc!r}")
    return mod


@pytest.fixture
def macos(monkeypatch):
    return _prepare(_load_launcher('macos'), monkeypatch)


def _prepare(mod, monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    return mod


def _block_ui(mod, platform_name, monkeypatch):
    started = []
    monkeypatch.setattr(
        mod, UI_ENTRY_POINT[platform_name],
        lambda *args, **kwargs: started.append(args) or 0,
    )
    return started


@pytest.mark.parametrize('platform_name', sorted(UI_ENTRY_POINT))
def test_no_launcher_starts_its_ui_when_handed_spawned_worker_argv(
    platform_name, monkeypatch
):
    mod = _prepare(_load_launcher(platform_name), monkeypatch)
    started = _block_ui(mod, platform_name, monkeypatch)
    ran = []
    monkeypatch.setattr(
        mod.frozen_children.runpy, 'run_module',
        lambda name, **kwargs: ran.append(name),
    )
    monkeypatch.setattr(sys, 'argv', LOKY_WORKER_ARGV)

    mod.main()

    assert ran == ['joblib.externals.loky.backend.popen_loky_posix']
    assert started == []


def test_the_macos_launcher_rejects_an_unknown_argument(macos, monkeypatch):
    started = _block_ui(macos, 'macos', monkeypatch)
    monkeypatch.setattr(sys, 'argv', [EXE, 'wat'])

    with pytest.raises(SystemExit) as excinfo:
        macos.main()

    assert excinfo.value.code == 2
    assert started == []


def test_the_macos_launcher_starts_the_menu_bar_with_no_arguments(macos, monkeypatch):
    started = _block_ui(macos, 'macos', monkeypatch)
    monkeypatch.setattr(sys, 'argv', [EXE])

    macos.main()

    assert len(started) == 1


def test_the_macos_launcher_starts_the_menu_bar_for_a_finder_launch(macos, monkeypatch):
    started = _block_ui(macos, 'macos', monkeypatch)
    monkeypatch.setattr(sys, 'argv', [EXE, '-psn_0_774521'])

    macos.main()

    assert len(started) == 1


def test_the_macos_launcher_still_dispatches_a_role(macos, monkeypatch):
    started = _block_ui(macos, 'macos', monkeypatch)
    roles = []
    monkeypatch.setattr(macos, '_run_role', roles.append)
    monkeypatch.setattr(sys, 'argv', [EXE, '--role=worker-default'])

    macos.main()

    assert roles == ['worker-default']
    assert started == []
