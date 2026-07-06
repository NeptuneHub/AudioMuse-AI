# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Plugin loader, registry, and lifecycle manager.

Materializes installed plugin code from the canonical ``plugins`` DB table into
the local ``PLUGINS_DIR`` cache, optionally pip-installs declared requirements,
imports each plugin through the ``audiomuse_plugins`` namespace package, and
invokes ``register(ctx)`` with per-plugin failure isolation so a bad plugin can
never stop the app from booting. Also installs/uninstalls packages and dispatches
plugin cron/RQ tasks inside a Flask app context.

Main Features:
* ``sync`` + ``ensure_requirements`` + ``load`` boot sequence shared by web and workers.
* Zip-slip-safe extraction, md5/size/version validation, and DB-backed canonical storage.
* ``run_plugin_task`` runs a plugin task by dotted path inside an app context.
"""

import hashlib
import importlib
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile

import config
import database
from plugin.api import PluginContext, NAMESPACE, valid_plugin_id

logger = logging.getLogger(__name__)

_MANIFEST_NAME = 'plugin.json'


def _parse_version(value):
    text = str(value or '').strip().lstrip('vV')
    nums = []
    for part in re.split(r'[.\-+]', text):
        match = re.match(r'\d+', part)
        nums.append(int(match.group()) if match else 0)
    return tuple(nums) if nums else (0,)


def version_ge(current, required):
    if not required:
        return True
    return _parse_version(current) >= _parse_version(required)


def _is_safe_member(name):
    if not name:
        return True
    normalized = name.replace('\\', '/')
    if normalized.startswith('/') or (len(normalized) > 1 and normalized[1] == ':'):
        return False
    return '..' not in normalized.split('/')


def _validate_zip_safe(zip_file):
    for member in zip_file.namelist():
        if not _is_safe_member(member):
            raise ValueError(f'Unsafe path in plugin package: {member}')


def read_manifest_from_bytes(package_bytes):
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
        _validate_zip_safe(zf)
        candidates = [n for n in zf.namelist() if n.rsplit('/', 1)[-1] == _MANIFEST_NAME]
        candidates.sort(key=lambda n: n.count('/'))
        if not candidates:
            raise ValueError('plugin.json not found in package')
        import json
        return json.loads(zf.read(candidates[0]))


def _read_marker(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        return None


def _write_marker(path, value):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(str(value))


def _resolve_extract_root(staging):
    if os.path.isfile(os.path.join(staging, _MANIFEST_NAME)):
        return staging
    entries = os.listdir(staging)
    if len(entries) == 1:
        inner = os.path.join(staging, entries[0])
        if os.path.isdir(inner) and os.path.isfile(os.path.join(inner, _MANIFEST_NAME)):
            return inner
    return staging


def _safe_extract(package_bytes, target):
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix='.plugin_stage_', dir=parent)
    try:
        with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
            _validate_zip_safe(zf)
            zf.extractall(staging)
        root = _resolve_extract_root(staging)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        os.replace(root, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class PluginManager:
    def __init__(self):
        self.records = {}
        self._loaded_role = None

    def enabled(self):
        return bool(config.PLUGINS_ENABLED)

    def setup_namespace(self):
        os.makedirs(config.PLUGINS_DIR, exist_ok=True)
        module = sys.modules.get(NAMESPACE)
        if module is None:
            module = types.ModuleType(NAMESPACE)
            module.__path__ = []
            module.__package__ = NAMESPACE
            sys.modules[NAMESPACE] = module
        if not hasattr(module, '__path__'):
            module.__path__ = []
        if config.PLUGINS_DIR not in module.__path__:
            module.__path__.append(config.PLUGINS_DIR)
        lib_dir = os.path.join(config.PLUGINS_DIR, '_lib')
        if os.path.isdir(lib_dir) and lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

    def _materialize_one(self, plugin_id, checksum, package_bytes):
        target = os.path.join(config.PLUGINS_DIR, plugin_id)
        marker = os.path.join(target, '.checksum')
        if os.path.isdir(target) and _read_marker(marker) == checksum:
            return
        _safe_extract(package_bytes, target)
        _write_marker(marker, checksum or '')

    def sync(self, conn=None):
        if not self.enabled():
            self.records = {}
            return
        own = conn is None
        connection = conn or database.connect_raw()
        try:
            rows = database.list_plugins(connection)
            records = {}
            for row in rows:
                record = dict(row)
                record.setdefault('load_status', row.get('load_status'))
                record['menu_items'] = []
                record['settings_endpoint'] = None
                record['cron_tasks'] = {}
                record['onnx_providers'] = []
                record['error'] = None
                records[row['id']] = record
                if row['enabled']:
                    try:
                        checksum, package = database.get_plugin_package(row['id'], connection)
                        if package is not None:
                            self._materialize_one(row['id'], checksum, package)
                    except Exception:
                        logger.exception('Failed to materialize plugin %s', row['id'])
                        record['load_status'] = 'error'
            self.records = records
        finally:
            if own:
                connection.close()

    def _pip_install(self, specs):
        if not specs:
            return True
        if not config.PLUGIN_ALLOW_PIP or getattr(sys, 'frozen', False):
            return False
        lib_dir = os.path.join(config.PLUGINS_DIR, '_lib')
        os.makedirs(lib_dir, exist_ok=True)
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--target', lib_dir, '--no-input', *specs],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except Exception:
            logger.exception('pip install failed for plugin requirements: %s', specs)
            return False
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        return True

    def ensure_requirements(self):
        if not self.enabled():
            return
        frozen = getattr(sys, 'frozen', False)
        specs = []
        for plugin_id, record in self.records.items():
            if record['enabled'] and record['requirements']:
                if frozen or not config.PLUGIN_ALLOW_PIP:
                    record['load_status'] = 'incompatible'
                    self._persist_status(plugin_id, 'incompatible')
                else:
                    specs.extend(record['requirements'])
        lib_dir = os.path.join(config.PLUGINS_DIR, '_lib')
        if not specs:
            if os.path.isdir(lib_dir) and lib_dir not in sys.path:
                sys.path.insert(0, lib_dir)
            return
        marker = os.path.join(lib_dir, '.specs')
        key = '\n'.join(sorted(set(specs)))
        if os.path.isdir(lib_dir) and _read_marker(marker) == key:
            if lib_dir not in sys.path:
                sys.path.insert(0, lib_dir)
            return
        if self._pip_install(sorted(set(specs))):
            try:
                _write_marker(marker, key)
            except OSError:
                logger.exception('Could not write plugin _lib specs marker')

    def _persist_status(self, plugin_id, status):
        try:
            connection = database.connect_raw()
            try:
                database.set_plugin_load_status(plugin_id, status, connection)
            finally:
                connection.close()
        except Exception:
            logger.exception('Failed to persist load_status for plugin %s', plugin_id)

    def _purge_modules(self, plugin_id):
        prefix = f'{NAMESPACE}.{plugin_id}'
        for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + '.')]:
            sys.modules.pop(name, None)

    def _import_plugin(self, plugin_id):
        return importlib.import_module(f'{NAMESPACE}.{plugin_id}')

    def _build_context(self, plugin_id, role):
        module = self._import_plugin(plugin_id)
        register = getattr(module, 'register', None)
        ctx = PluginContext(plugin_id, role)
        if callable(register):
            register(ctx)
        return ctx

    def load(self, role, flask_app=None):
        if not self.enabled():
            return
        self.setup_namespace()
        self._loaded_role = role
        for plugin_id, record in self.records.items():
            if not record['enabled']:
                continue
            if not version_ge(config.APP_VERSION, record['manifest'].get('min_core_version')):
                record['load_status'] = 'incompatible'
                self._persist_status(plugin_id, 'incompatible')
                continue
            if record.get('requirements') and getattr(sys, 'frozen', False):
                record['load_status'] = 'incompatible'
                self._persist_status(plugin_id, 'incompatible')
                continue
            try:
                ctx = self._build_context(plugin_id, role)
                record['menu_items'] = ctx.menu_items
                record['settings_endpoint'] = ctx.settings_endpoint
                record['cron_tasks'] = ctx.cron_tasks
                record['onnx_providers'] = ctx.onnx_providers
                if role == 'web' and flask_app is not None and ctx.blueprint is not None:
                    flask_app.register_blueprint(ctx.blueprint, url_prefix=f'/plugins/{plugin_id}')
                    if not record['settings_endpoint']:
                        candidate = f'{ctx.blueprint.name}.settings'
                        if candidate in flask_app.view_functions:
                            record['settings_endpoint'] = candidate
                if role == 'web':
                    self._run_hooks(ctx.flask_start, plugin_id, 'flask_start')
                elif role == 'worker':
                    self._run_hooks(ctx.worker_start, plugin_id, 'worker_start')
                record['load_status'] = 'ok'
                record['error'] = None
                self._persist_status(plugin_id, 'ok')
            except Exception as exc:
                logger.exception('Failed to load plugin %s', plugin_id)
                record['load_status'] = 'error'
                record['error'] = str(exc)
                self._persist_status(plugin_id, 'error')

    def _run_hooks(self, hooks, plugin_id, label):
        for hook in hooks or []:
            try:
                hook()
            except Exception:
                logger.exception('Plugin %s %s hook failed', plugin_id, label)

    def install_package(self, package_bytes, source_repo=None, expected_checksum=None):
        max_bytes = config.PLUGIN_MAX_DOWNLOAD_MB * 1024 * 1024
        if len(package_bytes) > max_bytes:
            raise ValueError(f'Plugin package exceeds {config.PLUGIN_MAX_DOWNLOAD_MB} MB limit')
        checksum = hashlib.md5(package_bytes).hexdigest()
        if expected_checksum and checksum.lower() != str(expected_checksum).lower():
            raise ValueError('Plugin package checksum mismatch')
        manifest = read_manifest_from_bytes(package_bytes)
        plugin_id = manifest.get('id')
        if not valid_plugin_id(plugin_id):
            raise ValueError('Invalid or missing plugin id (expected ^[a-z][a-z0-9_]{1,63}$)')
        if not version_ge(config.APP_VERSION, manifest.get('min_core_version')):
            raise ValueError(
                f"Plugin requires core >= {manifest.get('min_core_version')} (current {config.APP_VERSION})"
            )
        requirements = manifest.get('requirements') or []
        database.upsert_plugin(
            plugin_id,
            manifest.get('name') or plugin_id,
            manifest.get('version'),
            manifest,
            package_bytes,
            checksum,
            requirements,
            source_repo,
        )
        self.setup_namespace()
        self._purge_modules(plugin_id)
        self._materialize_one(plugin_id, checksum, package_bytes)
        self._pip_install(requirements)
        self.run_install_hooks(plugin_id)
        return manifest

    def run_install_hooks(self, plugin_id):
        from flask_app import app

        self.setup_namespace()
        self._purge_modules(plugin_id)
        try:
            ctx = self._build_context(plugin_id, 'install')
        except Exception:
            logger.exception('Plugin %s import failed during install hooks', plugin_id)
            return
        if not ctx.install_hooks:
            return
        with app.app_context():
            db = database.get_db()
            for hook in ctx.install_hooks:
                try:
                    hook(db)
                except Exception:
                    logger.exception('Plugin %s install hook failed', plugin_id)

    def uninstall(self, plugin_id, purge_data=False):
        database.delete_cron_rows_for_plugin(plugin_id)
        if purge_data:
            database.drop_plugin_data_tables(plugin_id)
        database.delete_plugin(plugin_id)
        target = os.path.join(config.PLUGINS_DIR, plugin_id)
        shutil.rmtree(target, ignore_errors=True)
        self._purge_modules(plugin_id)
        self.records.pop(plugin_id, None)

    def set_enabled(self, plugin_id, enabled):
        database.set_plugin_enabled(plugin_id, enabled)
        record = self.records.get(plugin_id)
        if record is not None:
            record['enabled'] = bool(enabled)

    def get_cron_task(self, task_type):
        if not task_type or not task_type.startswith('plugin.'):
            return None
        remainder = task_type[len('plugin.'):]
        plugin_id, _, name = remainder.partition('.')
        record = self.records.get(plugin_id)
        if not record:
            return None
        return record.get('cron_tasks', {}).get(name)

    def get_settings_endpoint(self, plugin_id):
        record = self.records.get(plugin_id)
        return record.get('settings_endpoint') if record else None

    def get_onnx_providers(self):
        providers = []
        for record in self.records.values():
            if record.get('load_status') == 'ok':
                providers.extend(record.get('onnx_providers', []))
        return providers

    def menu_items(self):
        items = []
        for record in self.records.values():
            if record.get('load_status') != 'ok':
                continue
            settings_ep = record.get('settings_endpoint')
            for entry in record.get('menu_items', []):
                if settings_ep and entry.get('endpoint') == settings_ep:
                    continue
                items.append(entry)
        return items

    def registry(self):
        summary = []
        for plugin_id, record in self.records.items():
            summary.append({
                'id': plugin_id,
                'name': record.get('name'),
                'version': record.get('version'),
                'enabled': bool(record.get('enabled')),
                'load_status': record.get('load_status'),
                'error': record.get('error'),
            })
        return summary


plugin_manager = PluginManager()


def run_plugin_task(dotted, *args, **kwargs):
    """RQ entrypoint: import a plugin task by dotted path and run it in an app context."""
    from flask_app import app

    plugin_manager.setup_namespace()
    module_path, _, fn_name = dotted.rpartition('.')
    with app.app_context():
        module = importlib.import_module(module_path)
        func = getattr(module, fn_name)
        return func(*args, **kwargs)


def boot(role, flask_app=None):
    """Run the full boot sequence for a process role ('web' or 'worker')."""
    if not plugin_manager.enabled():
        return
    try:
        plugin_manager.setup_namespace()
        plugin_manager.sync()
        plugin_manager.ensure_requirements()
        plugin_manager.load(role, flask_app=flask_app)
    except Exception:
        logger.exception('Plugin subsystem boot failed; continuing without plugins')
