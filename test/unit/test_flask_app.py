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
