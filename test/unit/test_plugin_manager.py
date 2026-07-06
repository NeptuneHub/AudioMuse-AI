# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the plugin loader, registry, and API surface.

Exercises version compatibility, zip-slip-safe extraction and manifest parsing,
DB-materialization with a faked data layer, per-plugin load failure isolation,
and the author-facing API (context recording, table namespacing, settings).

Main Features:
* No real database or network: the data layer and connections are monkeypatched.
* Covers the security-critical zip validation and boot failure-isolation paths.
"""

import hashlib
import io
import sys
import zipfile

import pytest

import config
import database
import plugin.api as api
import plugin.manager as manager


def _make_zip(files, wrap_dir=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        for name, content in files.items():
            arc = f'{wrap_dir}/{name}' if wrap_dir else name
            zf.writestr(arc, content)
    return buffer.getvalue()


def _make_unsafe_zip(member):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        zf.writestr('plugin.json', '{"id": "demo"}')
        zf.writestr(member, 'x')
    return buffer.getvalue()


def _write_plugin(root, plugin_id, init_body):
    directory = root / plugin_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'plugin.json').write_text('{"id": "%s"}' % plugin_id, encoding='utf-8')
    (directory / '__init__.py').write_text(init_body, encoding='utf-8')


def _record(plugin_id, enabled=True, requirements=None, manifest=None):
    return {
        'id': plugin_id,
        'name': plugin_id,
        'version': '1.0.0',
        'manifest': manifest if manifest is not None else {},
        'checksum': 'x',
        'requirements': requirements or [],
        'enabled': enabled,
        'settings': {},
        'source_repo': None,
        'load_status': None,
        'menu_items': [],
        'cron_tasks': {},
        'onnx_providers': [],
        'error': None,
    }


class _DummyConn:
    def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset_namespace():
    def _clear():
        for name in [n for n in list(sys.modules)
                     if n == manager.NAMESPACE or n.startswith(manager.NAMESPACE + '.')]:
            sys.modules.pop(name, None)
    _clear()
    yield
    _clear()


class TestVersionCompare:
    def test_equal_is_compatible(self):
        assert manager.version_ge('v2.5.0', '2.5.0') is True

    def test_newer_current_is_compatible(self):
        assert manager.version_ge('v2.6.0', '2.5.0') is True

    def test_older_current_is_incompatible(self):
        assert manager.version_ge('v2.5.0', '2.6.0') is False

    def test_missing_requirement_is_compatible(self):
        assert manager.version_ge('2.5.0', None) is True
        assert manager.version_ge('2.5.0', '') is True

    def test_short_requirement(self):
        assert manager.version_ge('2.5.0', '2.4') is True


class TestZipSafety:
    def test_safe_members(self):
        assert manager._is_safe_member('a/b.py') is True
        assert manager._is_safe_member('plugin.json') is True

    def test_unsafe_members(self):
        assert manager._is_safe_member('../evil.py') is False
        assert manager._is_safe_member('/etc/passwd') is False
        assert manager._is_safe_member('a/../../b') is False

    def test_read_manifest_ok(self):
        pkg = _make_zip({'plugin.json': '{"id": "demo", "version": "1.0.0"}'})
        manifest = manager.read_manifest_from_bytes(pkg)
        assert manifest['id'] == 'demo'

    def test_read_manifest_wrapped_dir(self):
        pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': ''}, wrap_dir='demo')
        manifest = manager.read_manifest_from_bytes(pkg)
        assert manifest['id'] == 'demo'

    def test_read_manifest_missing(self):
        pkg = _make_zip({'__init__.py': ''})
        with pytest.raises(ValueError):
            manager.read_manifest_from_bytes(pkg)

    def test_read_manifest_rejects_zip_slip(self):
        pkg = _make_unsafe_zip('../evil.py')
        with pytest.raises(ValueError):
            manager.read_manifest_from_bytes(pkg)

    def test_safe_extract_rejects_zip_slip(self, tmp_path):
        pkg = _make_unsafe_zip('../evil.py')
        with pytest.raises(ValueError):
            manager._safe_extract(pkg, str(tmp_path / 'demo'))

    def test_safe_extract_writes_files(self, tmp_path):
        pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': 'X = 1\n'})
        target = tmp_path / 'demo'
        manager._safe_extract(pkg, str(target))
        assert (target / 'plugin.json').is_file()
        assert (target / '__init__.py').read_text(encoding='utf-8') == 'X = 1\n'


class TestSync:
    def test_materializes_enabled_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        enabled_pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': ''})
        enabled_sum = hashlib.md5(enabled_pkg).hexdigest()
        rows = [
            {'id': 'demo', 'name': 'Demo', 'version': '1.0.0', 'manifest': {'id': 'demo'},
             'checksum': enabled_sum, 'requirements': [], 'enabled': True, 'settings': {},
             'source_repo': None, 'load_status': None, 'installed_at': None, 'updated_at': None},
            {'id': 'off', 'name': 'Off', 'version': '1.0.0', 'manifest': {'id': 'off'},
             'checksum': 'y', 'requirements': [], 'enabled': False, 'settings': {},
             'source_repo': None, 'load_status': None, 'installed_at': None, 'updated_at': None},
        ]
        packages = {'demo': (enabled_sum, enabled_pkg)}
        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: rows)
        monkeypatch.setattr(database, 'get_plugin_package',
                            lambda pid, conn=None: packages.get(pid, (None, None)))

        mgr = manager.PluginManager()
        mgr.sync()

        assert (tmp_path / 'demo' / 'plugin.json').is_file()
        assert (tmp_path / 'demo' / '.checksum').read_text(encoding='utf-8').strip() == enabled_sum
        assert not (tmp_path / 'off').exists()
        assert set(mgr.records) == {'demo', 'off'}

    def test_sync_skips_reextract_when_checksum_matches(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': ''})
        checksum = hashlib.md5(pkg).hexdigest()
        rows = [{'id': 'demo', 'name': 'Demo', 'version': '1.0.0', 'manifest': {'id': 'demo'},
                 'checksum': checksum, 'requirements': [], 'enabled': True, 'settings': {},
                 'source_repo': None, 'load_status': None, 'installed_at': None, 'updated_at': None}]
        calls = {'n': 0}

        def _get_pkg(pid, conn=None):
            calls['n'] += 1
            return checksum, pkg

        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: rows)
        monkeypatch.setattr(database, 'get_plugin_package', _get_pkg)

        mgr = manager.PluginManager()
        mgr.sync()
        mgr.sync()
        target = tmp_path / 'demo'
        assert (target / '.checksum').read_text(encoding='utf-8').strip() == checksum


class TestLoadIsolation:
    def test_failure_isolated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        _write_plugin(tmp_path, 'goodp', 'def register(ctx):\n    pass\n')
        _write_plugin(tmp_path, 'badp', "def register(ctx):\n    raise RuntimeError('boom')\n")

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'goodp': _record('goodp'), 'badp': _record('badp')}
        mgr.load('worker')

        assert mgr.records['goodp']['load_status'] == 'ok'
        assert mgr.records['badp']['load_status'] == 'error'
        assert mgr.records['badp']['error']

    def test_incompatible_version_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        _write_plugin(tmp_path, 'futurep', 'def register(ctx):\n    pass\n')

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'futurep': _record('futurep', manifest={'min_core_version': '9.9.9'})}
        mgr.load('worker')

        assert mgr.records['futurep']['load_status'] == 'incompatible'

    def test_disabled_not_loaded(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        _write_plugin(tmp_path, 'offp', "def register(ctx):\n    raise RuntimeError('should not run')\n")

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'offp': _record('offp', enabled=False)}
        mgr.load('worker')

        assert mgr.records['offp']['load_status'] is None

    def test_settings_endpoint_captured(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        _write_plugin(tmp_path, 'setp', "def register(ctx):\n    ctx.set_settings_page('setp.settings')\n")

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'setp': _record('setp')}
        mgr.load('web')

        assert mgr.get_settings_endpoint('setp') == 'setp.settings'
        assert mgr.get_settings_endpoint('missing') is None

    def test_settings_detected_by_convention_and_menu_hidden(self, monkeypatch, tmp_path):
        from flask import Flask

        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        code = (
            "from flask import Blueprint\n"
            "bp = Blueprint('conv', __name__)\n"
            "@bp.route('/')\n"
            "def home():\n    return 'h'\n"
            "@bp.route('/settings')\n"
            "def settings():\n    return 's'\n"
            "def register(ctx):\n"
            "    ctx.add_blueprint(bp)\n"
            "    ctx.add_menu_item('Conv', 'conv.home')\n"
            "    ctx.add_menu_item('Conv Settings', 'conv.settings')\n"
        )
        _write_plugin(tmp_path, 'conv', code)

        app = Flask('convtest')
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'conv': _record('conv')}
        mgr.load('web', flask_app=app)

        assert mgr.get_settings_endpoint('conv') == 'conv.settings'
        assert [m['label'] for m in mgr.menu_items()] == ['Conv']


class TestRegistryLookups:
    def test_get_cron_task_and_onnx(self):
        mgr = manager.PluginManager()
        record = _record('demo')
        record['load_status'] = 'ok'
        record['cron_tasks'] = {'daily': {'dotted': 'audiomuse_plugins.demo.tasks.daily', 'queue': 'high'}}
        record['onnx_providers'] = [{'name': 'X', 'options': {}, 'position': 'before_cpu'}]
        mgr.records = {'demo': record}

        assert mgr.get_cron_task('plugin.demo.daily')['queue'] == 'high'
        assert mgr.get_cron_task('plugin.demo.missing') is None
        assert mgr.get_cron_task('analysis') is None
        assert mgr.get_onnx_providers()[0]['name'] == 'X'

    def test_menu_items_only_from_loaded(self):
        mgr = manager.PluginManager()
        ok = _record('ok')
        ok['load_status'] = 'ok'
        ok['menu_items'] = [{'label': 'A', 'endpoint': 'ok.home', 'admin_only': False}]
        broken = _record('broken')
        broken['load_status'] = 'error'
        broken['menu_items'] = [{'label': 'B', 'endpoint': 'broken.home', 'admin_only': False}]
        mgr.records = {'ok': ok, 'broken': broken}

        labels = [m['label'] for m in mgr.menu_items()]
        assert labels == ['A']


class TestApiSurface:
    def test_valid_plugin_id(self):
        assert api.valid_plugin_id('hello_world') is True
        assert api.valid_plugin_id('Hello') is False
        assert api.valid_plugin_id('9bad') is False
        assert api.valid_plugin_id('') is False

    def test_dotted_path(self):
        def f():
            pass
        f.__module__ = 'audiomuse_plugins.demo.tasks'
        assert api.dotted_path(f) == 'audiomuse_plugins.demo.tasks.f'
        assert api.dotted_path('a.b.c') == 'a.b.c'

    def test_context_records_by_target(self):
        ctx = api.PluginContext('demo', 'worker')

        def task():
            pass
        task.__module__ = 'audiomuse_plugins.demo.tasks'
        task.__name__ = 'task'

        ctx.add_menu_item('Hello', 'demo.home')
        ctx.add_cron_task('daily', task)
        ctx.register_onnx_provider('X', {'a': 1})

        assert ctx.menu_items[0]['endpoint'] == 'demo.home'
        assert ctx.cron_tasks['daily']['dotted'] == 'audiomuse_plugins.demo.tasks.task'
        assert ctx.onnx_providers[0]['options'] == {'a': 1}

    def test_table_namespacing_infers_plugin(self):
        namespace = {'__name__': 'audiomuse_plugins.demo.tasks', 'table': api.table}
        exec('def call():\n    return table("runs")', namespace)
        assert namespace['call']() == 'plugin_demo__runs'

    def test_table_rejects_bad_name(self):
        namespace = {'__name__': 'audiomuse_plugins.demo.tasks', 'table': api.table}
        exec('def call():\n    return table("Bad-Name")', namespace)
        with pytest.raises(ValueError):
            namespace['call']()

    def test_get_setting_reads_db(self, monkeypatch):
        monkeypatch.setattr(database, 'get_plugin_settings', lambda pid: {'greeting': 'hi'})
        namespace = {'__name__': 'audiomuse_plugins.demo', 'get_setting': api.get_setting}
        exec('def call():\n    return get_setting("greeting", "default")', namespace)
        assert namespace['call']() == 'hi'

    def test_get_setting_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(database, 'get_plugin_settings', lambda pid: {})
        namespace = {'__name__': 'audiomuse_plugins.demo', 'get_setting': api.get_setting}
        exec('def call():\n    return get_setting("missing", "default")', namespace)
        assert namespace['call']() == 'default'
