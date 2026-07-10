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
        """No-op close; the fake connection holds no resources."""


@pytest.fixture(autouse=True)
def _reset_namespace():
    def _clear():
        for name in [n for n in sys.modules
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


class TestCronTaskFallback:
    def test_worker_only_plugin_resolves_from_persisted_manifest(self):
        mgr = manager.PluginManager()
        mgr.records = {'jobber': _record('jobber', manifest={
            'targets': ['worker'],
            'cron_tasks': {'daily': {'dotted': 'audiomuse_plugins.jobber.tasks.daily', 'queue': 'default'}},
        })}
        task = mgr.get_cron_task('plugin.jobber.daily')
        assert task == {'dotted': 'audiomuse_plugins.jobber.tasks.daily', 'queue': 'default'}

    def test_loaded_registration_wins_over_manifest(self):
        mgr = manager.PluginManager()
        record = _record('jobber', manifest={
            'cron_tasks': {'daily': {'dotted': 'stale.path', 'queue': 'default'}},
        })
        record['cron_tasks'] = {'daily': {'dotted': 'audiomuse_plugins.jobber.tasks.daily', 'queue': 'high'}}
        mgr.records = {'jobber': record}
        assert mgr.get_cron_task('plugin.jobber.daily')['queue'] == 'high'

    def test_disabled_plugin_never_dispatches(self):
        mgr = manager.PluginManager()
        mgr.records = {'jobber': _record('jobber', enabled=False, manifest={
            'cron_tasks': {'daily': {'dotted': 'audiomuse_plugins.jobber.tasks.daily', 'queue': 'default'}},
        })}
        assert mgr.get_cron_task('plugin.jobber.daily') is None

    def test_malformed_persisted_entry_is_ignored(self):
        mgr = manager.PluginManager()
        mgr.records = {'jobber': _record('jobber', manifest={'cron_tasks': {'daily': 'not-a-dict'}})}
        assert mgr.get_cron_task('plugin.jobber.daily') is None


class TestRequirementPinning:
    def _lib_with(self, tmp_path, name, version):
        dist = tmp_path / '_lib' / f'{name}-{version}.dist-info'
        dist.mkdir(parents=True)
        (dist / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n', encoding='utf-8'
        )

    def test_changed_pin_triggers_reinstall(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        self._lib_with(tmp_path, 'matplotlib', '3.7.0')
        installed = {'specs': None}

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: installed.__setitem__('specs', specs) or True)
        mgr.records = {'withreq': _record('withreq', requirements=['matplotlib==3.9.0'])}

        mgr.ensure_requirements()

        assert installed['specs'] == ['matplotlib==3.9.0']

    def test_satisfied_pin_skips_pip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        self._lib_with(tmp_path, 'matplotlib', '3.9.0')
        calls = {'n': 0}

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: calls.__setitem__('n', calls['n'] + 1) or True)
        mgr.records = {'withreq': _record('withreq', requirements=['matplotlib==3.9.0'])}

        mgr.ensure_requirements()

        assert calls['n'] == 0

    def test_range_pin_is_validated(self):
        assert manager._valid_requirement('matplotlib==3.9.0') is True
        assert manager._valid_requirement('matplotlib>=3.8,<4') is True
        assert manager._valid_requirement('-r requirements.txt') is False


class TestPipInstallArgs:
    def test_pip_argv_includes_upgrade_and_prunes_stale_dist_info(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        stale = tmp_path / '_lib' / 'matplotlib-3.7.0.dist-info'
        stale.mkdir(parents=True)
        (stale / 'METADATA').write_text('Metadata-Version: 2.1\nName: matplotlib\nVersion: 3.7.0\n', encoding='utf-8')
        seen = {}

        def fake_run(argv, **kwargs):
            seen['argv'] = argv
            return None

        monkeypatch.setattr(manager.subprocess, 'run', fake_run)
        mgr = manager.PluginManager()
        assert mgr._pip_install(['matplotlib==3.9']) is True
        assert '--upgrade' in seen['argv']
        assert not stale.exists()


class TestReplaceDir:
    def test_replaces_existing_target(self, tmp_path):
        old = tmp_path / 'plug'
        old.mkdir()
        (old / 'stale.py').write_text('OLD = 1\n', encoding='utf-8')
        fresh = tmp_path / 'incoming'
        fresh.mkdir()
        (fresh / '__init__.py').write_text('NEW = 1\n', encoding='utf-8')
        manager._replace_dir(str(fresh), str(old))
        assert (old / '__init__.py').is_file()
        assert not (old / 'stale.py').exists()
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('.plugin_old_')]
        assert leftovers == []


class TestZipSafety:
    def test_safe_members(self):
        assert manager._is_safe_member('a/b.py') is True
        assert manager._is_safe_member('plugin.json') is True

    def test_unsafe_members(self):
        assert manager._is_safe_member('../evil.py') is False
        assert manager._is_safe_member('/etc/passwd') is False
        assert manager._is_safe_member('a/../../b') is False

    def test_safe_extract_rejects_zip_slip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        pkg = _make_unsafe_zip('../evil.py')
        target = str(tmp_path / 'demo')
        with pytest.raises(ValueError):
            manager._safe_extract(pkg, target)

    def test_safe_extract_rejects_target_escape(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path / 'plugins'))
        pkg = _make_zip({'plugin.json': '{"id": "demo"}'})
        with pytest.raises(ValueError):
            manager._safe_extract(pkg, str(tmp_path / 'outside'))

    def test_safe_extract_writes_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': 'X = 1\n'})
        target = tmp_path / 'demo'
        manager._safe_extract(pkg, str(target))
        assert (target / 'plugin.json').is_file()
        assert (target / '__init__.py').read_text(encoding='utf-8') == 'X = 1\n'

    def test_plugin_path_rejects_invalid_ids(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        for bad in ['../evil', 'a/b', '', 'Upper', '.hidden', 'x/../..']:
            with pytest.raises(ValueError):
                manager._plugin_path(bad)

    def test_plugin_path_accepts_valid_id(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        assert manager._plugin_path('song_counter').endswith('song_counter')


def _plugin_row(plugin_id, checksum, source_url='https://example.com/x.zip', enabled=True,
                requirements=None, targets=None):
    manifest = {'id': plugin_id}
    if targets is not None:
        manifest['targets'] = targets
    return {
        'id': plugin_id, 'name': plugin_id, 'version': '1.0.0', 'manifest': manifest,
        'source_url': source_url, 'checksum': checksum, 'requirements': requirements or [],
        'enabled': enabled, 'settings': {}, 'source_repo': None, 'load_status': None,
        'installed_at': None, 'updated_at': None,
    }


class TestSync:
    def test_downloads_missing_code_from_source_url_with_warning(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': ''})
        checksum = hashlib.md5(pkg, usedforsecurity=False).hexdigest()
        rows = [
            _plugin_row('demo', checksum, source_url='https://example.com/demo.zip'),
            _plugin_row('off', 'y', source_url='https://example.com/off.zip', enabled=False),
        ]
        downloads = {'n': 0}

        def _fake_download(url, max_bytes):
            downloads['n'] += 1
            return pkg

        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: rows)
        monkeypatch.setattr(manager, '_download_url', _fake_download)

        mgr = manager.PluginManager()
        with caplog.at_level('WARNING'):
            mgr.sync()

        assert (tmp_path / 'demo' / 'plugin.json').is_file()
        assert (tmp_path / 'demo' / '.checksum').read_text(encoding='utf-8').strip() == checksum
        assert not (tmp_path / 'off').exists()
        assert downloads['n'] == 1
        assert any('was not found on disk' in r.message for r in caplog.records)

    def test_does_not_download_when_code_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        pkg = _make_zip({'plugin.json': '{"id": "demo"}', '__init__.py': ''})
        checksum = hashlib.md5(pkg, usedforsecurity=False).hexdigest()
        manager._safe_extract(pkg, str(tmp_path / 'demo'))
        (tmp_path / 'demo' / '.checksum').write_text(checksum, encoding='utf-8')

        def _boom(url, max_bytes):
            raise AssertionError('should not re-download when code is present')

        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: [_plugin_row('demo', checksum)])
        monkeypatch.setattr(manager, '_download_url', _boom)

        mgr = manager.PluginManager()
        mgr.sync()
        assert mgr.records['demo']['load_status'] != 'error'

    def test_missing_code_and_no_source_url_is_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins',
                            lambda conn=None: [_plugin_row('demo', 'abc', source_url=None)])

        mgr = manager.PluginManager()
        mgr.sync()
        assert mgr.records['demo']['load_status'] == 'error'


class TestBootDbWait:
    def test_retries_until_db_ready(self, monkeypatch):
        monkeypatch.setattr(config, 'PLUGIN_BOOT_DB_WAIT_SECONDS', 60)
        monkeypatch.setattr(manager.time, 'sleep', lambda _s: None)
        calls = {'n': 0}

        def _connect():
            calls['n'] += 1
            if calls['n'] < 3:
                raise OSError('connection refused')
            return _DummyConn()

        monkeypatch.setattr(database, 'connect_raw', _connect)
        manager._wait_for_db()
        assert calls['n'] == 3

    def test_gives_up_after_deadline(self, monkeypatch):
        monkeypatch.setattr(config, 'PLUGIN_BOOT_DB_WAIT_SECONDS', 0)
        monkeypatch.setattr(manager.time, 'sleep', lambda _s: None)

        def _connect():
            raise OSError('connection refused')

        monkeypatch.setattr(database, 'connect_raw', _connect)
        with pytest.raises(OSError):
            manager._wait_for_db()


class TestRequirements:
    def test_reinstall_warning_when_lib_missing(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        (tmp_path / '_lib').mkdir()
        installed = {'specs': None}

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: installed.__setitem__('specs', specs) or True)
        mgr.records = {'withreq': _record('withreq', requirements=['matplotlib'])}

        with caplog.at_level('WARNING'):
            mgr.ensure_requirements()

        assert installed['specs'] == ['matplotlib']
        assert any('missing or version-mismatched' in r.message and 'withreq' in r.message
                   for r in caplog.records)

    def test_pip_skipped_when_dep_already_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        dist = tmp_path / '_lib' / 'matplotlib-3.8.0.dist-info'
        dist.mkdir(parents=True)
        (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: matplotlib\nVersion: 3.8.0\n', encoding='utf-8')
        calls = {'n': 0}

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: calls.__setitem__('n', calls['n'] + 1) or True)
        mgr.records = {'withreq': _record('withreq', requirements=['matplotlib'])}

        mgr.ensure_requirements()

        assert calls['n'] == 0

    def test_pip_runs_when_pinned_version_mismatches(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        dist = tmp_path / '_lib' / 'matplotlib-3.7.0.dist-info'
        dist.mkdir(parents=True)
        (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: matplotlib\nVersion: 3.7.0\n', encoding='utf-8')
        installed = {'specs': None}

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: installed.__setitem__('specs', specs) or True)
        mgr.records = {'withreq': _record('withreq', requirements=['matplotlib==3.9.0'])}

        mgr.ensure_requirements()

        assert installed['specs'] == ['matplotlib==3.9.0']


class TestTargets:
    def test_default_targets_are_both(self):
        assert manager._plugin_targets({}) == {'flask', 'worker'}
        assert manager._plugin_targets({'targets': []}) == {'flask', 'worker'}
        assert manager._plugin_targets({'targets': ['garbage']}) == {'flask', 'worker'}

    def test_explicit_targets_are_honored(self):
        assert manager._plugin_targets({'targets': ['flask']}) == {'flask'}
        assert manager._plugin_targets({'targets': ['web', 'online']}) == {'flask'}
        assert manager._plugin_targets({'targets': ['worker', 'batch']}) == {'worker'}

    def test_string_targets_are_treated_as_a_single_token(self):
        assert manager._plugin_targets({'targets': 'worker'}) == {'worker'}
        assert manager._plugin_targets({'targets': 'flask'}) == {'flask'}

    def test_worker_skips_flask_only_code(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        rows = [_plugin_row('flaskonly', 'abc', source_url='https://example.com/f.zip', targets=['flask'])]
        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: rows)

        def _boom(url, max_bytes):
            raise AssertionError('worker must not download a flask-only plugin')

        monkeypatch.setattr(manager, '_download_url', _boom)

        mgr = manager.PluginManager()
        mgr.sync(role='worker')
        assert 'flaskonly' in mgr.records
        assert mgr.records['flaskonly']['load_status'] != 'error'

    def test_worker_downloads_worker_targeted_code(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        pkg = _make_zip({'plugin.json': '{"id": "job"}', '__init__.py': ''})
        checksum = hashlib.md5(pkg, usedforsecurity=False).hexdigest()
        rows = [_plugin_row('job', checksum, source_url='https://example.com/job.zip', targets=['worker'])]
        downloads = {'n': 0}

        def _fake(url, max_bytes):
            downloads['n'] += 1
            return pkg

        monkeypatch.setattr(database, 'connect_raw', lambda: _DummyConn())
        monkeypatch.setattr(database, 'list_plugins', lambda conn=None: rows)
        monkeypatch.setattr(manager, '_download_url', _fake)

        mgr = manager.PluginManager()
        mgr.sync(role='worker')
        assert downloads['n'] == 1

    def test_worker_skips_flask_only_requirements(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)

        mgr = manager.PluginManager()
        calls = {'specs': None}
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: calls.__setitem__('specs', specs) or True)
        mgr.records = {'flaskonly': _record('flaskonly', requirements=['matplotlib'],
                                            manifest={'targets': ['flask']})}

        mgr.ensure_requirements(role='worker')

        assert calls['specs'] is None

    def test_worker_does_not_load_flask_only_plugin(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        _write_plugin(tmp_path, 'flaskonly', 'raise RuntimeError("must not import on worker")\n')

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'flaskonly': _record('flaskonly', manifest={'targets': ['flask']})}
        mgr.load('worker')

        assert mgr.records['flaskonly']['load_status'] != 'error'


class TestInstallOrdering:
    def test_broadcast_fires_after_db_commit_before_local_pip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        pkg = _make_zip({'__init__.py': ''})
        order = []
        monkeypatch.setattr(database, 'upsert_plugin', lambda *a, **k: order.append('db_commit'))

        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, 'setup_namespace', lambda: None)
        monkeypatch.setattr(mgr, '_purge_modules', lambda pid: None)
        monkeypatch.setattr(mgr, '_materialize_one', lambda *a, **k: order.append('flask_code'))
        monkeypatch.setattr(mgr, '_install_specs', lambda *a, **k: order.append('flask_pip') or True)
        monkeypatch.setattr(mgr, 'run_install_hooks', lambda pid: order.append('flask_hooks'))

        _manifest, deps_ok, deps_error = mgr.install_package(
            pkg, {'id': 'demo', 'version': '1.0.0'}, source_url='https://example.com/demo.zip',
            on_registered=lambda pid: order.append('broadcast'),
        )

        assert order.index('db_commit') < order.index('flask_code') < order.index('broadcast')
        assert order.index('broadcast') < order.index('flask_pip') < order.index('flask_hooks')
        assert deps_ok is True
        assert deps_error is None
        assert mgr._runtime_dirty is True


class TestRestartPending:
    def _snapshot_mgr(self):
        mgr = manager.PluginManager()
        mgr.records = {'demo': _record('demo')}
        mgr.records['demo']['checksum'] = 'abc'
        mgr._boot_snapshot = {'demo': ('abc', True)}
        return mgr

    def test_unknown_before_load(self):
        mgr = manager.PluginManager()
        assert mgr.restart_pending([]) is None

    def test_false_when_db_matches_snapshot(self):
        mgr = self._snapshot_mgr()
        assert mgr.restart_pending([{'id': 'demo', 'checksum': 'abc', 'enabled': True}]) is False

    def test_true_on_checksum_change(self):
        mgr = self._snapshot_mgr()
        assert mgr.restart_pending([{'id': 'demo', 'checksum': 'NEW', 'enabled': True}]) is True

    def test_true_on_enable_flip_and_new_plugin(self):
        mgr = self._snapshot_mgr()
        assert mgr.restart_pending([{'id': 'demo', 'checksum': 'abc', 'enabled': False}]) is True
        assert mgr.restart_pending([
            {'id': 'demo', 'checksum': 'abc', 'enabled': True},
            {'id': 'other', 'checksum': 'x', 'enabled': True},
        ]) is True

    def test_true_after_uninstall_reinstall_same_version(self):
        mgr = self._snapshot_mgr()
        mgr._runtime_dirty = True
        assert mgr.restart_pending([{'id': 'demo', 'checksum': 'abc', 'enabled': True}]) is True


class TestAvailableCronTasks:
    def test_merges_memory_and_manifest_and_skips_disabled(self):
        mgr = manager.PluginManager()
        loaded = _record('loaded')
        loaded['cron_tasks'] = {'live': {'dotted': 'audiomuse_plugins.loaded.t.live', 'queue': 'default'}}
        persisted = _record('persisted', manifest={
            'targets': ['worker'],
            'cron_tasks': {'nightly': {'dotted': 'audiomuse_plugins.persisted.t.nightly', 'queue': 'high'}},
        })
        off = _record('off', enabled=False, manifest={
            'cron_tasks': {'never': {'dotted': 'audiomuse_plugins.off.t.never', 'queue': 'default'}},
        })
        mgr.records = {'loaded': loaded, 'persisted': persisted, 'off': off}
        tasks = mgr.available_cron_tasks()
        assert [t['task_type'] for t in tasks] == ['plugin.loaded.live', 'plugin.persisted.nightly']


class TestInstallFromManifest:
    def _mgr(self, monkeypatch, tmp_path, stored):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        monkeypatch.setattr(database, 'upsert_plugin',
                            lambda *a, **k: stored.update({'args': a}))
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, 'setup_namespace', lambda: None)
        monkeypatch.setattr(mgr, '_purge_modules', lambda pid: None)
        monkeypatch.setattr(mgr, '_materialize_one', lambda *a, **k: None)
        monkeypatch.setattr(mgr, '_install_specs', lambda *a, **k: True)
        monkeypatch.setattr(mgr, 'run_install_hooks', lambda pid: None)
        return mgr

    def test_stores_manifest_from_catalog_not_zip(self, monkeypatch, tmp_path):
        stored = {}
        mgr = self._mgr(monkeypatch, tmp_path, stored)
        pkg = _make_zip({'__init__.py': ''})
        manifest = {'id': 'demo', 'name': 'Demo', 'version': '2.0.0',
                    'min_core_version': '2.5.0', 'requirements': []}
        mgr.install_package(pkg, manifest, source_url='https://e/demo.zip')
        # upsert_plugin(plugin_id, name, version, manifest, ...)
        assert stored['args'][0] == 'demo'
        assert stored['args'][2] == '2.0.0'
        assert stored['args'][3] is manifest

    def test_rejects_checksum_mismatch(self, monkeypatch, tmp_path):
        mgr = self._mgr(monkeypatch, tmp_path, {})
        pkg = _make_zip({'__init__.py': ''})
        with pytest.raises(ValueError):
            mgr.install_package(pkg, {'id': 'demo'}, source_url='https://e/demo.zip',
                                expected_checksum='deadbeef')

    def test_rejects_incompatible_min_core(self, monkeypatch, tmp_path):
        mgr = self._mgr(monkeypatch, tmp_path, {})
        pkg = _make_zip({'__init__.py': ''})
        with pytest.raises(ValueError):
            mgr.install_package(pkg, {'id': 'demo', 'min_core_version': '9.9.9'},
                                source_url='https://e/demo.zip')

    def test_rejects_invalid_id(self, monkeypatch, tmp_path):
        mgr = self._mgr(monkeypatch, tmp_path, {})
        pkg = _make_zip({'__init__.py': ''})
        with pytest.raises(ValueError):
            mgr.install_package(pkg, {'id': 'Bad Id'}, source_url='https://e/demo.zip')


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

    def test_bad_menu_endpoint_warns_but_still_loads(self, monkeypatch, tmp_path, caplog):
        from flask import Flask

        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        code = (
            "from flask import Blueprint\n"
            "bp = Blueprint('bad', __name__)\n"
            "@bp.route('/')\n"
            "def home():\n    return 'h'\n"
            "def register(ctx):\n"
            "    ctx.add_blueprint(bp)\n"
            "    ctx.add_menu_item('Bad', 'bad.does_not_exist')\n"
        )
        _write_plugin(tmp_path, 'bad', code)

        app = Flask('badtest')
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'bad': _record('bad')}
        with caplog.at_level('WARNING'):
            mgr.load('web', flask_app=app)

        assert mgr.records['bad']['load_status'] == 'ok'
        assert any('not registered' in r.message for r in caplog.records)


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
            """Stub; the test only inspects its __module__ and __name__."""
        f.__module__ = 'audiomuse_plugins.demo.tasks'
        assert api.dotted_path(f) == 'audiomuse_plugins.demo.tasks.f'
        assert api.dotted_path('a.b.c') == 'a.b.c'

    def test_context_records_by_target(self):
        ctx = api.PluginContext('demo', 'worker')

        def task():
            """Stub; the test only inspects its __module__ and __name__."""
        task.__module__ = 'audiomuse_plugins.demo.tasks'
        task.__name__ = 'task'

        ctx.add_menu_item('Hello', 'demo.home')
        ctx.add_cron_task('daily', task)
        ctx.register_onnx_provider('X', {'a': 1})
        ctx.on_song_analyzed(task)

        assert ctx.menu_items[0]['endpoint'] == 'demo.home'
        assert ctx.cron_tasks['daily']['dotted'] == 'audiomuse_plugins.demo.tasks.task'
        assert ctx.onnx_providers[0]['options'] == {'a': 1}
        assert ctx.song_analyzed_hooks == [task]

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


class TestSongAnalyzedHooks:
    def _mgr(self, hooks_by_plugin):
        mgr = manager.PluginManager()
        mgr.records = {
            pid: {'load_status': 'ok', 'song_analyzed_hooks': hooks}
            for pid, hooks in hooks_by_plugin.items()
        }
        return mgr

    def test_aggregates_ok_records_only(self):
        def a(payload):
            """Stub listener."""

        def b(payload):
            """Stub listener."""

        mgr = self._mgr({'a': [a], 'b': [b]})
        mgr.records['b']['load_status'] = 'error'
        assert mgr.song_analyzed_hooks() == [a]

    def test_run_calls_all_and_isolates_failures(self):
        seen = []

        def good(payload):
            seen.append('good')

        def bad(payload):
            raise RuntimeError('boom')

        def good2(payload):
            seen.append('good2')

        mgr = self._mgr({'p': [good, bad, good2]})
        mgr.run_song_analyzed({'item_id': 'x'})
        assert seen == ['good', 'good2']

    def test_run_is_noop_when_no_hooks(self):
        mgr = manager.PluginManager()
        mgr.records = {}
        mgr.run_song_analyzed({'item_id': 'x'})

    def test_multiple_plugins_run_in_sequence_and_isolated(self):
        seen = []

        def a1(payload):
            seen.append('a1')

        def a2(payload):
            seen.append('a2')

        def b_bad(payload):
            raise RuntimeError('boom')

        def c1(payload):
            seen.append('c1')

        mgr = self._mgr({'a': [a1, a2], 'b': [b_bad], 'c': [c1]})
        mgr.run_song_analyzed({'item_id': 'x'})
        assert seen == ['a1', 'a2', 'c1']

    def test_deps_failed_plugin_hooks_stay_active(self):
        def a(payload):
            """Stub listener."""

        mgr = self._mgr({'a': [a]})
        mgr.records['a']['load_status'] = 'deps_failed'
        assert mgr.song_analyzed_hooks() == [a]


class TestMultiPluginConflicts:
    def test_conflicting_pin_surfaces_deps_error_on_the_unmet_plugin(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'PLUGIN_ALLOW_PIP', True)
        dist = tmp_path / '_lib' / 'matplotlib-3.9.0.dist-info'
        dist.mkdir(parents=True)
        (dist / 'METADATA').write_text(
            'Metadata-Version: 2.1\nName: matplotlib\nVersion: 3.9.0\n', encoding='utf-8'
        )
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_pip_install', lambda specs: True)
        mgr.records = {
            'wants39': _record('wants39', requirements=['matplotlib==3.9.0']),
            'wants37': _record('wants37', requirements=['matplotlib==3.7.0']),
        }

        mgr.ensure_requirements()

        assert 'deps_error' not in mgr.records['wants39']
        assert 'matplotlib==3.7.0' in mgr.records['wants37']['deps_error']
        assert 'conflicting version' in mgr.records['wants37']['deps_error']

    def test_load_persists_deps_failed_but_plugin_still_loads(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        _write_plugin(tmp_path, 'conflicted', 'def register(ctx):\n    pass\n')
        persisted = {}

        mgr = manager.PluginManager()
        monkeypatch.setattr(
            mgr, '_persist_status',
            lambda pid, status, role=None, error=None: persisted.__setitem__(pid, (status, error)),
        )
        record = _record('conflicted')
        record['deps_error'] = 'unsatisfied requirement(s) after install: matplotlib==3.7.0'
        mgr.records = {'conflicted': record}
        mgr.load('worker')

        assert record['load_status'] == 'deps_failed'
        assert 'matplotlib==3.7.0' in record['error']
        assert persisted['conflicted'][0] == 'deps_failed'

    def test_blueprint_named_after_plugin_id_loads_without_warning(self, monkeypatch, tmp_path, caplog):
        import flask

        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        _write_plugin(tmp_path, 'goodname', (
            "from flask import Blueprint\n"
            "bp = Blueprint('goodname', __name__)\n"
            "def register(ctx):\n"
            "    ctx.add_blueprint(bp)\n"
        ))
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'goodname': _record('goodname')}
        with caplog.at_level('WARNING'):
            mgr.load('web', flask_app=flask.Flask(__name__))
        assert mgr.records['goodname']['load_status'] == 'ok'
        assert not any('names its Blueprint' in r.message for r in caplog.records)

    def test_blueprint_not_named_after_plugin_id_warns_but_loads(self, monkeypatch, tmp_path, caplog):
        import flask

        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        _write_plugin(tmp_path, 'oddname', (
            "from flask import Blueprint\n"
            "bp = Blueprint('something_else', __name__)\n"
            "def register(ctx):\n"
            "    ctx.add_blueprint(bp)\n"
        ))
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'oddname': _record('oddname')}
        with caplog.at_level('WARNING'):
            mgr.load('web', flask_app=flask.Flask(__name__))
        assert mgr.records['oddname']['load_status'] == 'ok'
        assert any('names its Blueprint' in r.message for r in caplog.records)

    def test_duplicate_blueprint_name_fails_with_clear_error_and_isolates(self, monkeypatch, tmp_path):
        import flask

        monkeypatch.setattr(config, 'PLUGINS_DIR', str(tmp_path))
        monkeypatch.setattr(config, 'PLUGINS_ENABLED', True)
        monkeypatch.setattr(config, 'APP_VERSION', 'v2.5.0')
        body = (
            "from flask import Blueprint\n"
            "bp = Blueprint('dup', __name__)\n"
            "def register(ctx):\n"
            "    ctx.add_blueprint(bp)\n"
        )
        _write_plugin(tmp_path, 'firstp', body)
        _write_plugin(tmp_path, 'secondp', body)

        app = flask.Flask(__name__)
        mgr = manager.PluginManager()
        monkeypatch.setattr(mgr, '_persist_status', lambda *a, **k: None)
        mgr.records = {'firstp': _record('firstp'), 'secondp': _record('secondp')}
        mgr.load('web', flask_app=app)

        assert mgr.records['firstp']['load_status'] == 'ok'
        assert mgr.records['secondp']['load_status'] == 'error'
        assert 'rename the Blueprint' in mgr.records['secondp']['error']


class TestRunSongAnalyzedHookHelper:
    class _PM:
        def __init__(self, listeners):
            self._listeners = listeners
            self.received = None

        def enabled(self):
            return True

        def song_analyzed_hooks(self):
            return self._listeners

        def run_song_analyzed(self, payload):
            self.received = payload

    def test_noop_when_no_listeners(self, monkeypatch):
        import tasks.analysis_helper as ah
        pm = self._PM([])
        monkeypatch.setattr('plugin.manager.plugin_manager', pm)
        ah.run_song_analyzed_hook({'Id': '1'}, '/tmp/a.mp3', None, None, None, None, 'alb', 'Album', 'run-7')
        assert pm.received is None

    def test_builds_and_forwards_payload(self, monkeypatch):
        import tasks.analysis_helper as ah
        pm = self._PM([lambda payload: None])
        monkeypatch.setattr('plugin.manager.plugin_manager', pm)
        ah.run_song_analyzed_hook(
            {'Id': 42, 'Name': 'Song', 'AlbumArtist': 'Artist', 'Album': 'Alb'},
            '/tmp/a.mp3', {'tempo': 120}, None, None, {'happy': 0.9}, 'alb-id', 'Album', 'run-7',
        )
        assert pm.received['item_id'] == '42'
        assert pm.received['run_id'] == 'run-7'
        assert pm.received['audio_path'] == '/tmp/a.mp3'
        assert pm.received['metadata']['title'] == 'Song'
        assert pm.received['metadata']['artist'] == 'Artist'
        assert pm.received['metadata']['album_id'] == 'alb-id'
        assert pm.received['analysis'] == {'tempo': 120}
        assert pm.received['top_moods'] == {'happy': 0.9}
