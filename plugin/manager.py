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

import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import config
import database
from plugin import net
from plugin.api import PluginContext, NAMESPACE, valid_plugin_id

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None
try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None

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


def _plugin_path(plugin_id):
    """Resolve a plugin's directory under PLUGINS_DIR, rejecting any path escape.

    ``plugin_id`` is already constrained by ``valid_plugin_id`` at install time;
    this realpath + commonpath check is defense-in-depth so a crafted id can never
    resolve outside PLUGINS_DIR before the loader extracts or imports community code.
    """
    if not valid_plugin_id(plugin_id):
        raise ValueError(f'Invalid plugin id: {plugin_id!r}')
    base = os.path.realpath(config.PLUGINS_DIR)
    target = os.path.realpath(os.path.join(base, plugin_id))
    if os.path.commonpath([base, target]) != base:
        raise ValueError(f'Plugin path escapes the plugin directory: {plugin_id!r}')
    return target


def _replace_dir(root, target):
    """Swap a freshly extracted directory into place without deleting the live target first.

    The old directory is renamed aside (with a short retry, because on Windows an
    antivirus or indexer handle can briefly block the rename) and removed only after
    the swap succeeds, so a locked file can never leave a half-deleted plugin behind.
    """
    if not os.path.isdir(target):
        os.replace(root, target)
        return
    parent = os.path.dirname(target)
    aside = os.path.join(parent, f'.plugin_old_{os.getpid()}_{os.urandom(4).hex()}')
    last_error = None
    for _attempt in range(5):
        try:
            os.replace(target, aside)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            time.sleep(0.3)
    if last_error is not None:
        raise last_error
    os.replace(root, target)
    shutil.rmtree(aside, ignore_errors=True)


def _safe_extract(package_bytes, target):
    base = os.path.realpath(config.PLUGINS_DIR)
    target = os.path.realpath(target)
    if os.path.commonpath([base, target]) != base:
        raise ValueError('Plugin extraction target escapes the plugin directory')
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix='.plugin_stage_', dir=parent)
    try:
        with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
            _validate_zip_safe(zf)
            zf.extractall(staging)
        root = _resolve_extract_root(staging)
        _replace_dir(root, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _valid_requirement(spec):
    if not isinstance(spec, str) or not spec.strip() or spec.strip().startswith('-'):
        return False
    try:
        Requirement(spec)
        return True
    except Exception:
        return False


_PLUGIN_ROLES = ('flask', 'worker')

_LOADED_STATUSES = ('ok', 'deps_failed')


def _plugin_targets(manifest):
    raw = (manifest or {}).get('targets')
    if not raw:
        return set(_PLUGIN_ROLES)
    if isinstance(raw, str):
        raw = [raw]
    targets = set()
    for value in raw:
        token = str(value).strip().lower()
        if token in ('flask', 'web', 'online'):
            targets.add('flask')
        elif token in ('worker', 'batch'):
            targets.add('worker')
    return targets or set(_PLUGIN_ROLES)


def _role_target(role):
    return 'worker' if role == 'worker' else 'flask'


def _download_url(url, max_bytes):
    return net.download(url, max_bytes)


def _acquire_install_lock():
    """Take an OS-level lock file that serializes plugin writers across processes.

    The web process, both RQ workers, and the restart listener can all share one
    PLUGINS_DIR volume; a threading.Lock only guards a single interpreter. This
    file lock (fcntl on POSIX, msvcrt on Windows) serializes concurrent code
    extraction and pip installs into that shared directory across every process on
    the host, so two `pip install --target _lib` runs can never interleave.
    """
    os.makedirs(config.PLUGINS_DIR, exist_ok=True)
    fh = open(os.path.join(config.PLUGINS_DIR, '.install.lock'), 'a+')
    try:
        if _fcntl is not None:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
        elif _msvcrt is not None:
            fh.seek(0)
            while True:
                try:
                    _msvcrt.locking(fh.fileno(), _msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
    except Exception:
        fh.close()
        raise
    return fh


def _release_install_lock(fh):
    try:
        if _fcntl is not None:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
        elif _msvcrt is not None:
            fh.seek(0)
            try:
                _msvcrt.locking(fh.fileno(), _msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    finally:
        fh.close()


@contextlib.contextmanager
def _install_lock():
    fh = _acquire_install_lock()
    try:
        yield
    finally:
        _release_install_lock(fh)


class PluginManager:
    def __init__(self):
        self.records = {}
        self._loaded_role = None
        self._boot_snapshot = None
        self._runtime_dirty = False
        self.last_pip_error = None

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
            sys.path.append(lib_dir)

    def _materialize_one(self, plugin_id, checksum, package_bytes):
        target = _plugin_path(plugin_id)
        marker = os.path.join(target, '.checksum')
        if os.path.isdir(target) and _read_marker(marker) == checksum:
            return
        with _install_lock():
            if os.path.isdir(target) and _read_marker(marker) == checksum:
                return
            _safe_extract(package_bytes, target)
            _write_marker(marker, checksum or '')

    def _runs_here(self, record, role):
        if role is None:
            return True
        return _role_target(role) in _plugin_targets(record.get('manifest'))

    def sync(self, conn=None, role=None):
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
                record['song_analyzed_hooks'] = []
                record['error'] = None
                records[row['id']] = record
                if row['enabled'] and self._runs_here(record, role):
                    try:
                        self._ensure_code(row)
                    except Exception as exc:
                        logger.exception('Failed to provide code for plugin %s', row['id'])
                        record['load_status'] = 'error'
                        record['error'] = str(exc)
            self.records = records
        finally:
            if own:
                connection.close()

    def _ensure_code(self, row):
        plugin_id = row['id']
        checksum = row.get('checksum')
        target = _plugin_path(plugin_id)
        marker = os.path.join(target, '.checksum')
        if os.path.isdir(target) and (not checksum or _read_marker(marker) == checksum):
            return
        source_url = row.get('source_url')
        if not source_url:
            raise RuntimeError(
                f'plugin code for "{plugin_id}" is missing and it has no source_url to re-download it '
                '- reinstall it from the Plugins catalog'
            )
        logger.warning(
            'Installed plugin "%s" was not found on disk; re-downloading it from %s',
            plugin_id, source_url,
        )
        package = _download_url(source_url, config.PLUGIN_MAX_DOWNLOAD_MB * 1024 * 1024)
        got = hashlib.md5(package, usedforsecurity=False).hexdigest()
        if checksum and got.lower() != str(checksum).lower():
            raise ValueError(f'plugin "{plugin_id}" re-download checksum mismatch')
        self._materialize_one(plugin_id, checksum or got, package)

    def _prune_stale_dist_info(self, lib_dir, specs):
        """Remove old .dist-info dirs for the distributions about to be (re)installed.

        pip --target has no uninstall step: --upgrade replaces the package dir but
        leaves the previous version's differently-named dist-info behind, which
        would make the installed-version scan nondeterministic and re-trigger pip
        on every boot.
        """
        names = set()
        for spec in specs:
            try:
                names.add(canonicalize_name(Requirement(spec).name))
            except Exception:
                continue
        try:
            entries = os.listdir(lib_dir)
        except OSError:
            return
        for entry in entries:
            if not entry.endswith('.dist-info'):
                continue
            dist_name = entry[: -len('.dist-info')].rsplit('-', 1)[0]
            if canonicalize_name(dist_name) in names:
                shutil.rmtree(os.path.join(lib_dir, entry), ignore_errors=True)

    def _pip_install(self, specs):
        if not specs:
            return True
        if not config.PLUGIN_ALLOW_PIP or getattr(sys, 'frozen', False):
            return False
        if not all(_valid_requirement(s) for s in specs):
            logger.error('Refusing pip install: plugin requirements contain unsafe specifiers: %s', specs)
            return False
        lib_dir = os.path.join(config.PLUGINS_DIR, '_lib')
        os.makedirs(lib_dir, exist_ok=True)
        self._prune_stale_dist_info(lib_dir, specs)
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', '--target', lib_dir,
                 '--no-input', *specs],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except Exception as exc:
            stderr = getattr(exc, 'stderr', None) or b''
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            self.last_pip_error = (stderr.strip() or str(exc))[-400:]
            logger.exception('pip install failed for plugin requirements: %s', specs)
            return False
        self.last_pip_error = None
        if lib_dir not in sys.path:
            sys.path.append(lib_dir)
        importlib.invalidate_caches()
        return True

    def _ensure_lib_on_path(self):
        lib_dir = os.path.join(config.PLUGINS_DIR, '_lib')
        if os.path.isdir(lib_dir) and lib_dir not in sys.path:
            sys.path.append(lib_dir)

    def _installed_dist_versions(self):
        lib_dir = os.path.join(config.PLUGINS_DIR, '_lib')
        if not os.path.isdir(lib_dir):
            return {}
        versions = {}
        try:
            for dist in importlib.metadata.distributions(path=[lib_dir]):
                name = dist.metadata['Name'] if dist.metadata else None
                if name:
                    versions[canonicalize_name(name)] = dist.version
        except Exception:
            logger.exception('Could not read installed plugin dependencies from _lib')
        return versions

    def _missing_specs(self, specs):
        """Return the specs whose distribution is absent OR whose version pin is unmet in _lib.

        Unlike a name-only presence check, this parses each spec with packaging so a
        changed pin (matplotlib==3.7 -> matplotlib==3.9) is correctly seen as missing
        and reinstalled, instead of being silently satisfied by any matplotlib on disk.
        """
        have = self._installed_dist_versions()
        missing = []
        for spec in specs:
            try:
                req = Requirement(spec)
            except Exception:
                missing.append(spec)
                continue
            try:
                if req.marker is not None and not req.marker.evaluate():
                    continue
            except Exception:
                pass
            installed = have.get(canonicalize_name(req.name))
            if installed is None:
                missing.append(spec)
                continue
            if req.specifier:
                try:
                    satisfied = req.specifier.contains(installed, prereleases=True)
                except Exception:
                    satisfied = False
                if not satisfied:
                    missing.append(spec)
        return missing

    def _install_specs(self, requirements, plugin_ids=None):
        """Install the pip specs not already satisfied (by name AND version pin) in _lib.

        Satisfaction is checked against the distributions really installed in _lib,
        so a dependency is (re)installed only when genuinely missing or version-
        mismatched. The pip install runs under a cross-process file lock so the two
        RQ workers and the restart listener sharing one PLUGINS_DIR volume can never
        run overlapping installs into _lib. _lib is appended to the END of sys.path
        so a plugin dependency can never shadow a core AudioMuse-AI package.
        """
        specs = sorted({str(s) for s in (requirements or []) if _valid_requirement(s)})
        self._ensure_lib_on_path()
        if not specs:
            return True
        if getattr(sys, 'frozen', False) or not config.PLUGIN_ALLOW_PIP:
            return False
        if not self._missing_specs(specs):
            return True
        with _install_lock():
            missing = self._missing_specs(specs)
            if not missing:
                return True
            logger.warning(
                'Dependencies for plugin(s) %s are missing or version-mismatched; installing from PyPI: %s',
                ', '.join(sorted(plugin_ids)) if plugin_ids else '?',
                missing,
            )
            return self._pip_install(missing)

    def ensure_requirements(self, role=None):
        """Install pip requirements for enabled plugins that run on this role (missing ones only)."""
        if not self.enabled():
            return
        frozen = getattr(sys, 'frozen', False)
        specs = []
        plugins_with_reqs = []
        for plugin_id, record in self.records.items():
            if record['enabled'] and record['requirements'] and self._runs_here(record, role):
                if frozen or not config.PLUGIN_ALLOW_PIP:
                    record['load_status'] = 'incompatible'
                    self._persist_status(plugin_id, 'incompatible')
                else:
                    specs.extend(record['requirements'])
                    plugins_with_reqs.append(plugin_id)
        if not self._install_specs(specs, plugins_with_reqs) and len(plugins_with_reqs) > 1:
            for plugin_id in plugins_with_reqs:
                record = self.records[plugin_id]
                if not self._install_specs(record['requirements'], [plugin_id]):
                    logger.error(
                        'Dependency install failed for plugin %s: %s',
                        plugin_id, record['requirements'],
                    )
        for plugin_id in plugins_with_reqs:
            record = self.records[plugin_id]
            unmet = self._missing_specs(
                sorted({str(s) for s in record['requirements'] if _valid_requirement(s)})
            )
            if unmet:
                record['deps_error'] = (
                    'unsatisfied requirement(s) after install: ' + ', '.join(unmet) +
                    ' (another plugin may pin a conflicting version of the same package)'
                )
                logger.error('Plugin %s has %s', plugin_id, record['deps_error'])

    def _persist_status(self, plugin_id, status, role=None, error=None):
        if role is None and self._loaded_role is not None:
            role = _role_target(self._loaded_role)
        try:
            connection = database.connect_raw()
            try:
                database.set_plugin_load_status(plugin_id, status, connection, role=role, error=error)
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
        app_obj = flask_app
        if app_obj is None:
            from flask_app import app as app_obj
        for plugin_id, record in self.records.items():
            if not record['enabled']:
                continue
            if not self._runs_here(record, role):
                self._persist_status(plugin_id, None, error=None)
                continue
            if record.get('error'):
                self._persist_status(plugin_id, 'error', error=record.get('error'))
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
                with app_obj.app_context():
                    ctx = self._build_context(plugin_id, role)
                    record['menu_items'] = ctx.menu_items
                    record['settings_endpoint'] = ctx.settings_endpoint
                    record['cron_tasks'] = {**(ctx.tasks or {}), **(ctx.cron_tasks or {})}
                    record['onnx_providers'] = ctx.onnx_providers
                    record['song_analyzed_hooks'] = ctx.song_analyzed_hooks
                    if role == 'web' and flask_app is not None and ctx.blueprint is not None:
                        if ctx.blueprint.name != plugin_id:
                            logger.warning(
                                'Plugin %s names its Blueprint %r; the convention is to name it '
                                'after the plugin id, otherwise it can collide with another plugin',
                                plugin_id, ctx.blueprint.name,
                            )
                        if ctx.blueprint.name in flask_app.blueprints:
                            raise ValueError(
                                f'blueprint name "{ctx.blueprint.name}" is already registered by '
                                'another plugin or by the app; rename the Blueprint (use your plugin id)'
                            )
                        flask_app.register_blueprint(ctx.blueprint, url_prefix=f'/plugins/{plugin_id}')
                        if not record['settings_endpoint']:
                            candidate = f'{ctx.blueprint.name}.settings'
                            if candidate in flask_app.view_functions:
                                record['settings_endpoint'] = candidate
                    if role == 'web' and flask_app is not None:
                        self._warn_unknown_menu_endpoints(record['menu_items'], flask_app, plugin_id)
                    if role == 'web':
                        self._run_hooks(ctx.flask_start, plugin_id, 'flask_start')
                    elif role == 'worker':
                        self._run_hooks(ctx.worker_start, plugin_id, 'worker_start')
                deps_error = record.get('deps_error')
                if deps_error:
                    record['load_status'] = 'deps_failed'
                    record['error'] = deps_error
                    self._persist_status(plugin_id, 'deps_failed', error=deps_error)
                else:
                    record['load_status'] = 'ok'
                    record['error'] = None
                    self._persist_status(plugin_id, 'ok')
            except Exception as exc:
                logger.exception('Failed to load plugin %s', plugin_id)
                record['load_status'] = 'error'
                record['error'] = str(exc)
                self._persist_status(plugin_id, 'error', error=str(exc))
        self._boot_snapshot = {
            pid: (rec.get('checksum'), bool(rec.get('enabled')))
            for pid, rec in self.records.items()
        }

    def _run_hooks(self, hooks, plugin_id, label):
        for hook in hooks or []:
            try:
                hook()
            except Exception:
                logger.exception('Plugin %s %s hook failed', plugin_id, label)

    def _warn_unknown_menu_endpoints(self, menu_items, flask_app, plugin_id):
        for entry in menu_items or []:
            endpoint = entry.get('endpoint')
            if endpoint not in flask_app.view_functions:
                logger.warning(
                    'Plugin %s menu item "%s" points at endpoint %r which is not registered; '
                    'the link stays hidden until the endpoint resolves',
                    plugin_id, entry.get('label'), endpoint,
                )

    def install_package(self, package_bytes, manifest, source_url, source_repo=None,
                        expected_checksum=None, on_registered=None):
        """Install a code-only plugin zip using the manifest resolved from the catalog.

        The package holds only code; its metadata (id, name, version,
        min_core_version, requirements, targets) comes from ``manifest`` - the
        plugin.json the catalog pointed at - not from a file inside the zip. The
        download is size-capped and its md5 verified against ``expected_checksum``
        before anything is written.
        """
        max_bytes = config.PLUGIN_MAX_DOWNLOAD_MB * 1024 * 1024
        if len(package_bytes) > max_bytes:
            raise ValueError(f'Plugin package exceeds {config.PLUGIN_MAX_DOWNLOAD_MB} MB limit')
        checksum = hashlib.md5(package_bytes, usedforsecurity=False).hexdigest()
        if expected_checksum and checksum.lower() != str(expected_checksum).lower():
            raise ValueError('Plugin package checksum mismatch')
        try:
            with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
                _validate_zip_safe(zf)
        except zipfile.BadZipFile as exc:
            raise ValueError('Invalid plugin package: not a valid zip archive') from exc
        manifest = manifest or {}
        plugin_id = manifest.get('id')
        if not valid_plugin_id(plugin_id):
            raise ValueError('Invalid or missing plugin id (expected ^[a-z][a-z0-9_]{1,63}$)')
        if not version_ge(config.APP_VERSION, manifest.get('min_core_version')):
            raise ValueError(
                f"Plugin requires core >= {manifest.get('min_core_version')} (current {config.APP_VERSION})"
            )
        requirements = manifest.get('requirements') or []
        for spec in requirements:
            if not _valid_requirement(spec):
                raise ValueError(f'Invalid or unsafe plugin requirement: {spec!r}')
        previous_cron_tasks = None
        try:
            existing = database.get_plugin(plugin_id)
            if existing:
                previous_cron_tasks = (existing.get('manifest') or {}).get('cron_tasks')
        except Exception:
            previous_cron_tasks = None
        database.upsert_plugin(
            plugin_id,
            manifest.get('name') or plugin_id,
            manifest.get('version'),
            manifest,
            source_url,
            checksum,
            requirements,
            source_repo,
        )
        if previous_cron_tasks:
            try:
                database.set_plugin_cron_tasks(plugin_id, previous_cron_tasks)
            except Exception:
                logger.exception('Failed to carry over cron task declarations for plugin %s', plugin_id)
        self.setup_namespace()
        self._purge_modules(plugin_id)
        self._materialize_one(plugin_id, checksum, package_bytes)
        if on_registered:
            try:
                on_registered(plugin_id)
            except Exception:
                logger.exception('Plugin %s on_registered callback failed', plugin_id)
        deps_ok = self._install_specs(requirements, [plugin_id])
        deps_error = None
        pip_possible = config.PLUGIN_ALLOW_PIP and not getattr(sys, 'frozen', False)
        if not deps_ok and pip_possible:
            deps_error = self.last_pip_error or 'dependency install failed; check the container logs'
            self._persist_status(plugin_id, 'deps_failed', role='flask', error=deps_error)
        elif pip_possible:
            self._persist_status(plugin_id, None, role='flask', error=None)
            try:
                connection = database.connect_raw()
                try:
                    database.clear_plugin_deps_failed(plugin_id, connection)
                finally:
                    connection.close()
            except Exception:
                logger.exception('Failed to clear deps_failed status for plugin %s', plugin_id)
        self.run_install_hooks(plugin_id)
        self._runtime_dirty = True
        return manifest, deps_ok, deps_error

    def run_install_hooks(self, plugin_id):
        from flask_app import app

        self.setup_namespace()
        self._purge_modules(plugin_id)
        try:
            ctx = self._build_context(plugin_id, 'install')
        except Exception:
            logger.exception('Plugin %s import failed during install hooks', plugin_id)
            return
        try:
            database.set_plugin_cron_tasks(
                plugin_id, {**(ctx.tasks or {}), **(ctx.cron_tasks or {})}
            )
        except Exception:
            logger.exception('Failed to persist cron task declarations for plugin %s', plugin_id)
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
        if not valid_plugin_id(plugin_id):
            raise ValueError('Invalid plugin id')
        database.delete_cron_rows_for_plugin(plugin_id)
        if purge_data:
            database.drop_plugin_data_tables(plugin_id)
        database.delete_plugin(plugin_id)
        plugins_root = os.path.realpath(config.PLUGINS_DIR)
        target = os.path.realpath(os.path.join(plugins_root, plugin_id))
        if target != plugins_root and os.path.commonpath([plugins_root, target]) == plugins_root:
            shutil.rmtree(target, ignore_errors=True)
        self._purge_modules(plugin_id)
        self.records.pop(plugin_id, None)
        self._runtime_dirty = True

    def set_enabled(self, plugin_id, enabled):
        database.set_plugin_enabled(plugin_id, enabled)
        record = self.records.get(plugin_id)
        if record is not None:
            record['enabled'] = bool(enabled)

    def get_cron_task(self, task_type):
        """Resolve a plugin cron task, falling back to the declarations persisted at install.

        The web process dispatches cron but never imports a worker-only plugin, so its
        in-memory cron_tasks stay empty for those; the mapping stored in the manifest
        by run_install_hooks keeps them dispatchable.
        """
        if not task_type or not task_type.startswith('plugin.'):
            return None
        remainder = task_type[len('plugin.'):]
        plugin_id, _, name = remainder.partition('.')
        record = self.records.get(plugin_id)
        if not record or not record.get('enabled'):
            return None
        task = record.get('cron_tasks', {}).get(name)
        if task:
            return task
        persisted = (record.get('manifest') or {}).get('cron_tasks') or {}
        task = persisted.get(name)
        return task if isinstance(task, dict) and task.get('dotted') else None

    def available_cron_tasks(self):
        """Return every schedulable plugin cron task for the Scheduled Tasks page.

        Merges the in-memory registrations with the declarations persisted in the
        manifest at install time, so worker-only plugins are listed on the web
        process too.
        """
        items = []
        for plugin_id, record in self.records.items():
            if not record.get('enabled'):
                continue
            names = set(record.get('cron_tasks') or {})
            persisted = (record.get('manifest') or {}).get('cron_tasks') or {}
            names.update(
                name for name, task in persisted.items()
                if isinstance(task, dict) and task.get('dotted')
            )
            for name in sorted(names):
                items.append({
                    'task_type': f'plugin.{plugin_id}.{name}',
                    'plugin': plugin_id,
                    'task': name,
                })
        return sorted(items, key=lambda e: e['task_type'])

    def get_settings_endpoint(self, plugin_id):
        record = self.records.get(plugin_id)
        return record.get('settings_endpoint') if record else None

    def settings_endpoints(self):
        return {
            record['settings_endpoint']
            for record in self.records.values()
            if record.get('settings_endpoint')
        }

    def get_onnx_providers(self):
        providers = []
        for record in self.records.values():
            if record.get('load_status') in _LOADED_STATUSES:
                providers.extend(record.get('onnx_providers', []))
        return providers

    def song_analyzed_hooks(self):
        hooks = []
        for record in self.records.values():
            if record.get('load_status') in _LOADED_STATUSES:
                hooks.extend(record.get('song_analyzed_hooks', []))
        return hooks

    def run_song_analyzed(self, payload):
        """Call every registered on_song_analyzed hook with per-plugin isolation; a no-op when none."""
        for hook in self.song_analyzed_hooks():
            try:
                hook(payload)
            except Exception:
                logger.exception('Plugin song-analyzed hook %s failed', getattr(hook, '__module__', '?'))

    def menu_items(self):
        items = []
        for record in self.records.values():
            if record.get('load_status') not in _LOADED_STATUSES:
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

    def restart_pending(self, db_plugins):
        """True when this process no longer matches the DB registry it booted with.

        Compares the (checksum, enabled) snapshot captured at load time against the
        current rows, so the restart-required state survives page reloads and shows
        for every admin, not only the one who clicked. The dirty flag covers changes
        the snapshot cannot see, like uninstall + reinstall of the same version.
        None (unknown) before load().
        """
        if self._boot_snapshot is None:
            return None
        if self._runtime_dirty:
            return True
        current = {
            p['id']: (p.get('checksum'), bool(p.get('enabled')))
            for p in db_plugins
        }
        return current != self._boot_snapshot


plugin_manager = PluginManager()


def run_plugin_task(dotted, *args, **kwargs):
    """RQ entrypoint: import a plugin task by dotted path and run it in an app context.

    When the plugin code is missing on this worker's volume (fresh pod, plugin-sync
    missed), a throwaway manager re-materializes the enabled plugins once and the
    import is retried - the global manager's in-memory registrations (hooks, ONNX
    providers, cron tasks) are never touched, which matters on Windows where
    SimpleWorker runs jobs inside the long-lived worker process.
    Cron-enqueued jobs have a task_status row (created by the dispatcher); that row
    is transitioned to SUCCESS/FAILURE here so it can never sit PENDING forever.
    """
    from flask_app import app
    from rq import get_current_job

    plugin_manager.setup_namespace()
    module_path, _, fn_name = dotted.rpartition('.')
    try:
        job = get_current_job()
    except Exception:
        job = None
    task_id = job.id if job is not None else None
    with app.app_context():
        row = database.get_task_info_from_db(task_id) if task_id else None
        try:
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                recovery = PluginManager()
                recovery.setup_namespace()
                recovery.sync(role=None)
                recovery.ensure_requirements(role=None)
                importlib.invalidate_caches()
                module = importlib.import_module(module_path)
            func = getattr(module, fn_name)
            result = func(*args, **kwargs)
            if row:
                database.save_task_status(
                    task_id, row['task_type'], config.TASK_STATUS_SUCCESS, progress=100
                )
            return result
        except Exception as exc:
            if row:
                database.save_task_status(
                    task_id, row['task_type'], config.TASK_STATUS_FAILURE,
                    details={'error': str(exc)},
                )
            raise


_presync_lock = threading.Lock()


def _wait_for_db():
    """Block until the database accepts a connection, bounded by config.

    The RQ worker entrypoints boot the plugin subsystem before the Postgres pod is
    guaranteed to be up. Without this wait a startup 'connection refused' is caught
    by boot() and permanently disables plugins on that worker until it restarts. A
    no-op for the web process, which has already run init_db by the time it boots.
    """
    deadline = time.monotonic() + config.PLUGIN_BOOT_DB_WAIT_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            database.connect_raw().close()
            if attempt > 1:
                logger.info('Database is ready; loading plugins')
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            if attempt == 1:
                logger.warning(
                    'Database not ready yet; waiting up to %ss before loading plugins',
                    config.PLUGIN_BOOT_DB_WAIT_SECONDS,
                )
            time.sleep(config.PLUGIN_BOOT_DB_WAIT_INTERVAL)


def boot(role, flask_app=None):
    """Run the full boot sequence for a process role ('web' or 'worker')."""
    if not plugin_manager.enabled():
        return
    try:
        _wait_for_db()
        database.ensure_plugins_table()
        plugin_manager.setup_namespace()
        plugin_manager.sync(role=role)
        plugin_manager.ensure_requirements(role=role)
        plugin_manager.load(role, flask_app=flask_app)
    except Exception:
        logger.exception('Plugin subsystem boot failed; continuing without plugins')


def worker_presync():
    """Download plugin code and pip-install deps into this worker's own volume.

    Triggered by the Redis 'plugin-sync' broadcast at plugin install time so every
    worker container populates its PLUGINS_DIR (code) and _lib (dependencies)
    immediately, in parallel with the web process, instead of only at the next
    restart. The apply restart then reloads fast because ensure_requirements finds
    the dependencies already present. Serialized so overlapping broadcasts cannot
    run two pip installs into _lib at once.
    """
    if not plugin_manager.enabled():
        return
    with _presync_lock:
        try:
            _wait_for_db()
            database.ensure_plugins_table()
            plugin_manager.setup_namespace()
            plugin_manager.sync(role='worker')
            plugin_manager.ensure_requirements(role='worker')
            worker_plugins = sorted(
                pid for pid, record in plugin_manager.records.items()
                if record['enabled'] and plugin_manager._runs_here(record, 'worker')
            )
            if worker_plugins:
                logger.info('Worker plugins ready (code + dependencies): %s', ', '.join(worker_plugins))
            else:
                logger.info('No installed plugin targets this worker; nothing to install here')
        except Exception:
            logger.exception('Plugin pre-sync on worker failed; continuing')
