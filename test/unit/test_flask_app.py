# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Bundle resource-root resolution in flask_app._resource_root.

Covers how the static/template root is chosen when frozen by PyInstaller versus
running from source.

Main Features:
* Frozen with sys._MEIPASS set resolves to the bundle directory
* Frozen without _MEIPASS, and not frozen, both fall back to the module directory
* A falsy sys.frozen is treated as not frozen
"""

import os
import sys

import flask_app


def _module_dir():
    return os.path.dirname(os.path.abspath(flask_app.__file__))


def test_resource_root_frozen_with_meipass(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', '/bundle', raising=False)

    assert flask_app._resource_root() == '/bundle'


def test_resource_root_frozen_without_meipass(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.delattr(sys, '_MEIPASS', raising=False)

    assert flask_app._resource_root() == _module_dir()


def test_resource_root_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', '/bundle', raising=False)

    assert flask_app._resource_root() == _module_dir()


def test_resource_root_frozen_false(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', False, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', '/bundle', raising=False)

    assert flask_app._resource_root() == _module_dir()
