# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for the plugin manager admin UI and REST API.

Serves the `/plugins` admin page and the `/api/plugins/*` endpoints that browse
catalog repositories, install/update/uninstall/enable/disable plugins, edit
per-plugin settings, and trigger a restart to apply changes. Downloads are
SSRF-guarded, size-capped, and md5-verified before the package is stored in the
canonical `plugins` DB table.

Main Features:
* Catalog fetch/merge across configured repository manifests with compatibility filtering.
* Install/uninstall/enable/disable/settings/apply endpoints (admin-gated via app_auth path rules).
"""

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, render_template, jsonify, request, url_for

import config
import database
import restart_manager
from plugin import net
from ssrf_guard import validate_outbound_url
from plugin.manager import plugin_manager, version_ge, _parse_version

logger = logging.getLogger(__name__)

plugins_bp = Blueprint('plugins_bp', __name__, template_folder='templates')

_REPOS_KEY = 'PLUGIN_REPOS'
_CATALOG_CACHE_KEY = 'PLUGIN_CATALOG_CACHE'
_CATALOG_MAX_BYTES = 5 * 1024 * 1024

_GENERIC_ERROR = 'Operation failed. Check the container logs for details.'

_catalog_refresh_lock = threading.Lock()


def _store_catalog_cache(plugins, errors):
    """Persist the resolved catalog so the UI is served from the DB, never from a live fetch.

    A resolution that came back empty WITH fetch errors keeps the previously cached
    plugins (a transient outage never wipes the last known-good catalog); an empty
    result with no errors means every repo answered and truly lists nothing, so the
    cache clears and delisted plugins disappear.
    """
    if not plugins and errors:
        plugins = _load_catalog_cache()[0]
    payload = {'at': time.time(), 'plugins': plugins or [], 'errors': errors or []}
    try:
        database.set_app_config_value(_CATALOG_CACHE_KEY, json.dumps(payload))
    except Exception:
        logger.warning('Could not persist the plugin catalog cache')


def _load_catalog_cache():
    try:
        raw = database.get_app_config_value(_CATALOG_CACHE_KEY)
    except Exception:
        return [], [], 0.0
    if not raw:
        return [], [], 0.0
    try:
        payload = json.loads(raw)
        return (payload.get('plugins') or [], payload.get('errors') or [],
                float(payload.get('at') or 0.0))
    except (ValueError, TypeError):
        return [], [], 0.0


def _cached_latest_versions():
    plugins, _errors, at = _load_catalog_cache()
    return {p['id']: p.get('latest_version') for p in plugins if p.get('id')}, at


def _is_newer_version(latest, installed):
    """True when ``latest`` is numerically newer than ``installed`` (1.0 == 1.0.0)."""
    if not latest:
        return False
    latest_v = _parse_version(latest)
    installed_v = _parse_version(installed)
    width = max(len(latest_v), len(installed_v))
    return latest_v + (0,) * (width - len(latest_v)) > installed_v + (0,) * (width - len(installed_v))


def _refresh_catalog_cache_async(force=False):
    """Refresh the catalog cache in a background daemon thread.

    Every user-facing endpoint serves the cached catalog instantly and calls this;
    the actual GitHub fetches (which can be slow on clusters with broken pod DNS or
    filtered CDN routes) never block a request. Returns True when a refresh thread
    is running (freshly started or already in flight).
    """
    _plugins, _errors, cached_at = _load_catalog_cache()
    if not force and time.time() - cached_at < config.PLUGIN_CATALOG_CACHE_TTL:
        return _catalog_refresh_lock.locked()
    if not _catalog_refresh_lock.acquire(blocking=False):
        return True

    def _run():
        try:
            from flask_app import app
            with app.app_context():
                try:
                    plugins, errors = _fetch_catalog()
                    logger.info('Plugin catalog cache refreshed: %d plugins, %d repository errors',
                                len(plugins), len(errors))
                except Exception:
                    logger.warning('Background plugin catalog refresh failed')
        finally:
            _catalog_refresh_lock.release()

    try:
        threading.Thread(target=_run, name='plugin-catalog-refresh', daemon=True).start()
    except Exception:
        _catalog_refresh_lock.release()
        logger.exception('Could not start the plugin catalog refresh thread')
        return False
    return True


_auto_refresh_started = False


def start_catalog_auto_refresh():
    """Refresh the catalog cache at web startup and then every
    PLUGIN_CATALOG_REFRESH_INTERVAL seconds (default hourly), so new plugin
    versions surface on the Installed tab even if nobody opens the Catalog tab.
    """
    global _auto_refresh_started
    if _auto_refresh_started or not config.PLUGINS_ENABLED:
        return
    _auto_refresh_started = True

    def _loop():
        while True:
            try:
                _refresh_catalog_cache_async(force=True)
            except Exception:
                logger.exception('Scheduled plugin catalog refresh failed to start')
            time.sleep(config.PLUGIN_CATALOG_REFRESH_INTERVAL)

    threading.Thread(target=_loop, name='plugin-catalog-auto-refresh', daemon=True).start()
    logger.info('Plugin catalog auto-refresh scheduled: now and then every %s seconds',
                config.PLUGIN_CATALOG_REFRESH_INTERVAL)


def _pip_supported():
    return bool(config.PLUGIN_ALLOW_PIP) and not getattr(sys, 'frozen', False)


def _get_repos():
    raw = database.get_app_config_value(_REPOS_KEY)
    repos = []
    if raw:
        try:
            repos = [str(u) for u in json.loads(raw) if u]
        except (ValueError, TypeError):
            repos = []
    if config.PLUGIN_DEFAULT_REPO_URL and config.PLUGIN_DEFAULT_REPO_URL not in repos:
        repos.insert(0, config.PLUGIN_DEFAULT_REPO_URL)
    return repos


def _set_repos(repos):
    database.set_app_config_value(_REPOS_KEY, json.dumps(repos))


def _download(url, max_bytes):
    return net.download(url, max_bytes)


def _pick_version(versions, requested=None):
    compatible = []
    for entry in versions or []:
        min_core = entry.get('min_core_version') or entry.get('targetAbi')
        if not version_ge(config.APP_VERSION, min_core):
            continue
        if not (entry.get('sourceUrl') or entry.get('source_url')):
            continue
        if requested and str(entry.get('version')) != str(requested):
            continue
        compatible.append(entry)
    if not compatible:
        return None
    compatible.sort(key=lambda e: _parse_version(e.get('version')), reverse=True)
    return compatible[0]


def _versions_from_doc(doc):
    """Return the version list offered by a fetched ``plugin.json``.

    The current format is a ``plugin.json`` whose ``versions`` list holds every
    release (each entry carries ``version``/``min_core_version``/``changelog``/
    ``imageUrl``/``sourceUrl``/``checksum``), returned as-is. A ``plugin.json`` that
    instead describes a single release with flat top-level fields is wrapped into a
    one-item list for backward compatibility.
    """
    versions = doc.get('versions')
    if versions:
        return versions
    source_url = doc.get('sourceUrl') or doc.get('source_url')
    if source_url:
        return [{
            'version': doc.get('version'),
            'changelog': doc.get('changelog', ''),
            'min_core_version': doc.get('min_core_version') or doc.get('targetAbi'),
            'sourceUrl': source_url,
            'checksum': doc.get('checksum'),
        }]
    return None


def _resolve_versions(entry, errors):
    """Return (detail, versions) for a catalog entry.

    A catalog entry carries the stable identity (``id``/``name``/``author``/
    ``description``) plus a ``pluginUrl`` pointing at the plugin's own
    ``plugin.json`` (``manifestUrl`` and an inline ``versions`` list are still
    accepted). That file is fetched (SSRF-guarded) and holds the full ``versions``
    list with each release's download url, checksum, min_core_version and image,
    so there is no separate per-plugin manifest to keep in sync.
    """
    versions = entry.get('versions')
    if versions:
        return entry, versions
    detail_url = (entry.get('pluginUrl') or entry.get('plugin_url')
                  or entry.get('manifestUrl') or entry.get('manifest_url'))
    if not detail_url:
        return entry, None
    try:
        raw = _download(detail_url, _CATALOG_MAX_BYTES)
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            raise TypeError('plugin.json is not a JSON object')
    except Exception as exc:
        logger.warning('Failed to fetch plugin.json %s: %s', detail_url, exc)
        errors.append({'repo': detail_url, 'error': str(exc)})
        return entry, None
    return doc, _versions_from_doc(doc)


def _build_catalog_entry(repo_url, entry, installed):
    """Resolve one catalog entry to its best version. Runs in a worker thread.

    Returns ``(plugin_id, merged_dict_or_None, local_errors)``. Never raises: any
    failure is recorded in ``local_errors`` so one bad plugin cannot abort the fan-out.
    """
    plugin_id = entry.get('id')
    local_errors = []
    try:
        detail, versions = _resolve_versions(entry, local_errors)
        best = _pick_version(versions)
    except Exception as exc:
        logger.warning('Failed to resolve plugin %s: %s', plugin_id, exc)
        local_errors.append({'repo': plugin_id or repo_url, 'error': str(exc)})
        return plugin_id, None, local_errors
    if not best:
        return plugin_id, None, local_errors
    current = installed.get(plugin_id)
    compatible_versions = sorted(
        (
            {
                'version': e.get('version'),
                'min_core_version': e.get('min_core_version') or e.get('targetAbi'),
                'changelog': e.get('changelog', ''),
                'sourceUrl': e.get('sourceUrl') or e.get('source_url'),
                'checksum': e.get('checksum'),
                'requirements': e.get('requirements'),
                'targets': e.get('targets'),
            }
            for e in (versions or [])
            if version_ge(config.APP_VERSION, e.get('min_core_version') or e.get('targetAbi'))
            and (e.get('sourceUrl') or e.get('source_url'))
        ),
        key=lambda e: _parse_version(e.get('version')),
        reverse=True,
    )
    return plugin_id, {
        'id': plugin_id,
        'name': entry.get('name') or detail.get('name') or plugin_id,
        'description': entry.get('description') or detail.get('description', ''),
        'author': entry.get('author') or detail.get('author', ''),
        'image_url': best.get('imageUrl') or entry.get('imageUrl') or detail.get('imageUrl', ''),
        'latest_version': best.get('version'),
        'source_url': best.get('sourceUrl'),
        'checksum': best.get('checksum'),
        'changelog': best.get('changelog', ''),
        'min_core_version': best.get('min_core_version') or best.get('targetAbi'),
        'targets': detail.get('targets') or [],
        'requirements': detail.get('requirements') or [],
        'versions': compatible_versions,
        'source_repo': repo_url,
        'installed_version': current.get('version') if current else None,
    }, local_errors


def _fetch_catalog():
    installed = {p['id']: p for p in database.list_plugins()}
    errors = []
    failed_repos = set()
    pending = []
    for repo_url in _get_repos():
        try:
            raw = _download(repo_url, _CATALOG_MAX_BYTES)
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                raise TypeError('Repository catalog is not a JSON object')
        except net.DownloadError as exc:
            logger.warning('Failed to fetch plugin repo %s: %s', repo_url, exc)
            errors.append({'repo': repo_url, 'error': str(exc)})
            failed_repos.add(repo_url)
            continue
        except Exception as exc:
            logger.warning('Failed to fetch plugin repo %s: %s', repo_url, exc)
            errors.append({'repo': repo_url, 'error': 'Failed to fetch repository catalog'})
            failed_repos.add(repo_url)
            continue
        entries = doc.get('plugins')
        if not isinstance(entries, list):
            logger.warning('Plugin repo %s has no valid "plugins" list; skipping', repo_url)
            errors.append({'repo': repo_url, 'error': 'Repository catalog "plugins" is not a list'})
            failed_repos.add(repo_url)
            continue
        pending.extend(
            (repo_url, entry) for entry in entries
            if isinstance(entry, dict) and entry.get('id')
        )

    merged = {}
    failed_ids = set()
    if pending:
        workers = max(1, min(config.PLUGIN_CATALOG_FETCH_WORKERS, len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda item: _build_catalog_entry(item[0], item[1], installed), pending))
        for plugin_id, entry_data, local_errors in results:
            errors.extend(local_errors)
            if entry_data:
                merged[plugin_id] = entry_data
            elif local_errors and plugin_id:
                failed_ids.add(plugin_id)
    if failed_repos or failed_ids:
        previous, _prev_errors, _prev_at = _load_catalog_cache()
        for entry in previous:
            entry_id = entry.get('id')
            if entry_id in merged:
                continue
            if entry.get('source_repo') in failed_repos or entry_id in failed_ids:
                merged[entry_id] = entry
    result = list(merged.values())
    _store_catalog_cache(result, errors)
    return result, errors


@plugins_bp.route('/plugins', methods=['GET'])
def plugins_page():
    return render_template('plugins.html', title='AudioMuse-AI - Plugins', active='plugins')


@plugins_bp.route('/api/plugins/installed', methods=['GET'])
def api_installed():
    plugins = database.list_plugins()
    registry = {r['id']: r for r in plugin_manager.registry()}
    latest, _cached_at = _cached_latest_versions()
    for plugin in plugins:
        entry = registry.get(plugin['id'])
        plugin['error'] = entry.get('error') if entry else None
        plugin['requirements'] = plugin.get('requirements') or []
        latest_version = latest.get(plugin['id'])
        plugin['latest_version'] = latest_version
        plugin['update_available'] = _is_newer_version(latest_version, plugin.get('version'))
        endpoint = plugin_manager.get_settings_endpoint(plugin['id'])
        settings_url = None
        if endpoint:
            try:
                settings_url = url_for(endpoint)
            except Exception:
                settings_url = None
        plugin['settings_url'] = settings_url
    _refresh_catalog_cache_async()
    return jsonify({
        'plugins': plugins,
        'pip_supported': _pip_supported(),
        'restart_pending': plugin_manager.restart_pending(plugins),
    })


@plugins_bp.route('/api/plugins/catalog', methods=['GET'])
def api_catalog():
    """Serve the cached catalog instantly; the network refresh always runs in background.

    ``?refresh=1`` (the Refresh button) forces a background refresh regardless of the
    cache age. The response carries ``refreshing`` so the UI can poll until the
    background fetch lands, and ``cached_at`` for transparency.
    """
    force = request.args.get('refresh') in ('1', 'true')
    try:
        refreshing = _refresh_catalog_cache_async(force=force)
        plugins, errors, cached_at = _load_catalog_cache()
        installed = {p['id']: p.get('version') for p in database.list_plugins()}
        for entry in plugins:
            entry['installed_version'] = installed.get(entry.get('id'))
    except Exception:
        logger.exception('Failed to serve plugin catalog')
        return jsonify({'error': _GENERIC_ERROR}), 500
    return jsonify({
        'plugins': plugins,
        'repos': _get_repos(),
        'errors': errors,
        'cached_at': cached_at,
        'refreshing': bool(refreshing),
    })


def _install_manifest(match):
    """Build the manifest stored for an install from a resolved catalog entry.

    The zip is code-only, so this is the sole source of the plugin's metadata: the
    plugin.json top-level identity plus the fields of the chosen release.
    """
    return {
        'id': match['id'],
        'name': match.get('name') or match['id'],
        'author': match.get('author', ''),
        'description': match.get('description', ''),
        'version': match.get('latest_version'),
        'min_core_version': match.get('min_core_version'),
        'changelog': match.get('changelog', ''),
        'imageUrl': match.get('image_url', ''),
        'targets': match.get('targets') or [],
        'requirements': match.get('requirements') or [],
    }


class VersionUnavailableError(Exception):
    """The specific plugin version an install requested cannot be resolved right now."""


def _resolve_install_source(plugin_id, requested_version=None):
    """Return (source_url, checksum, source_repo, manifest) for a plugin to install.

    Resolves from the cached catalog first (instant - the user typically clicks
    Install right after seeing the catalog), falling back to a live fetch only when
    the cache does not know the plugin. If neither works, falls back to the
    source_url and manifest already stored for an installed plugin so a reinstall
    still works during an upstream outage. Returns all-None when nothing yields a
    source. With ``requested_version`` the matching release is resolved from the
    entry's compatible versions list (install a specific version / rollback);
    raises VersionUnavailableError when that exact release cannot be served.
    """
    catalog, _errors, _at = _load_catalog_cache()
    match = next((p for p in catalog if p.get('id') == plugin_id), None)
    if not match or not match.get('source_url'):
        try:
            catalog, _ = _fetch_catalog()
        except Exception:
            logger.exception('Catalog fetch failed while resolving install source for %s', plugin_id)
            catalog = []
        match = next((p for p in catalog if p.get('id') == plugin_id), None)
    if match and match.get('source_url'):
        meta = _install_manifest(match)
        source_url = match['source_url']
        checksum = match.get('checksum')
        if requested_version is not None and str(meta.get('version')) != str(requested_version):
            release = next(
                (e for e in (match.get('versions') or [])
                 if str(e.get('version')) == str(requested_version)
                 and (e.get('sourceUrl') or e.get('source_url'))),
                None,
            )
            if release is None:
                raise VersionUnavailableError(
                    f'Version {requested_version} of {plugin_id} is not currently available '
                    'from any configured repository; nothing was changed'
                )
            source_url = release.get('sourceUrl') or release.get('source_url')
            checksum = release.get('checksum')
            meta = {
                **meta,
                'version': release.get('version'),
                'min_core_version': release.get('min_core_version') or meta.get('min_core_version'),
                'changelog': release.get('changelog', ''),
            }
            if release.get('requirements') is not None:
                meta['requirements'] = release['requirements']
            if release.get('targets') is not None:
                meta['targets'] = release['targets']
        return source_url, checksum, match.get('source_repo'), meta
    existing = database.get_plugin(plugin_id)
    if existing and existing.get('source_url'):
        if requested_version is not None and str(existing.get('version')) != str(requested_version):
            raise VersionUnavailableError(
                f'Version {requested_version} of {plugin_id} is not currently available '
                'from any configured repository; nothing was changed'
            )
        logger.warning(
            'Plugin %s not resolvable from the catalog; reinstalling from its stored source_url', plugin_id
        )
        return (existing['source_url'], existing.get('checksum'), existing.get('source_repo'),
                existing.get('manifest') or {'id': plugin_id})
    return None, None, None, None


@plugins_bp.route('/api/plugins/install', methods=['POST'])
def api_install():
    data = request.get_json(silent=True) or {}
    plugin_id = data.get('id')
    if not plugin_id:
        return jsonify({'error': 'Missing required field: id'}), 400
    requested_version = data.get('version')
    try:
        source_url, checksum, source_repo, install_meta = _resolve_install_source(
            plugin_id, requested_version
        )
        if not source_url:
            return jsonify({'error': 'Plugin not found in any configured repository'}), 404
        package = _download(source_url, config.PLUGIN_MAX_DOWNLOAD_MB * 1024 * 1024)
        manifest, deps_ok, deps_error = plugin_manager.install_package(
            package, install_meta, source_url=source_url, source_repo=source_repo,
            expected_checksum=checksum,
            on_registered=lambda _pid: restart_manager.publish_plugin_sync_request(),
        )
        response = {
            'status': 'ok',
            'manifest': manifest,
            'restart_required': True,
            'deps_ok': bool(deps_ok),
        }
        if deps_error:
            response['deps_error'] = deps_error
        return jsonify(response)
    except VersionUnavailableError as exc:
        return jsonify({'error': str(exc)}), 409
    except net.DownloadError as exc:
        logger.warning('Plugin download failed for %s: %s', plugin_id, exc)
        return jsonify({'error': str(exc)}), 502
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('Plugin install failed')
        return jsonify({'error': _GENERIC_ERROR}), 500


@plugins_bp.route('/api/plugins/uninstall', methods=['POST'])
def api_uninstall():
    data = request.get_json(silent=True) or {}
    plugin_id = data.get('id')
    purge_data = bool(data.get('purge_data'))
    if not plugin_id:
        return jsonify({'error': 'Missing plugin id'}), 400
    try:
        plugin_manager.uninstall(plugin_id, purge_data=purge_data)
        return jsonify({'status': 'ok', 'restart_required': True})
    except Exception:
        logger.exception('Plugin uninstall failed for %s', plugin_id)
        return jsonify({'error': _GENERIC_ERROR}), 500


@plugins_bp.route('/api/plugins/enable', methods=['POST'])
def api_enable():
    return _set_enabled(True)


@plugins_bp.route('/api/plugins/disable', methods=['POST'])
def api_disable():
    return _set_enabled(False)


def _set_enabled(enabled):
    data = request.get_json(silent=True) or {}
    plugin_id = data.get('id')
    if not plugin_id:
        return jsonify({'error': 'Missing plugin id'}), 400
    try:
        plugin_manager.set_enabled(plugin_id, enabled)
        return jsonify({'status': 'ok', 'restart_required': True})
    except Exception:
        logger.exception('Plugin enable/disable failed for %s', plugin_id)
        return jsonify({'error': _GENERIC_ERROR}), 500


@plugins_bp.route('/api/plugins/settings/<plugin_id>', methods=['GET', 'POST'])
def api_settings(plugin_id):
    plugin = database.get_plugin(plugin_id)
    if not plugin:
        return jsonify({'error': 'Plugin not found'}), 404
    if request.method == 'GET':
        return jsonify({'id': plugin_id, 'settings': plugin['settings'], 'manifest': plugin['manifest']})
    data = request.get_json(silent=True) or {}
    settings = data.get('settings')
    if not isinstance(settings, dict):
        return jsonify({'error': 'settings must be an object'}), 400
    try:
        database.set_plugin_settings(plugin_id, settings)
        return jsonify({'status': 'ok'})
    except Exception:
        logger.exception('Failed to save settings for plugin %s', plugin_id)
        return jsonify({'error': _GENERIC_ERROR}), 500


@plugins_bp.route('/api/plugins/repos', methods=['GET', 'POST', 'DELETE'])
def api_repos():
    if request.method == 'GET':
        return jsonify({'repos': _get_repos(), 'default': config.PLUGIN_DEFAULT_REPO_URL})
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'Missing repository url'}), 400
    ok, message = validate_outbound_url(url)
    if not ok:
        return jsonify({'error': f'URL rejected: {message}'}), 400
    repos = [r for r in _get_repos() if r != config.PLUGIN_DEFAULT_REPO_URL]
    if request.method == 'POST':
        if url not in repos:
            repos.append(url)
    else:
        repos = [r for r in repos if r != url]
    _set_repos(repos)
    return jsonify({'status': 'ok', 'repos': _get_repos()})


@plugins_bp.route('/api/plugins/apply', methods=['POST'])
def api_apply():
    try:
        workers_published = restart_manager.publish_restart_request()
        flask_scheduled = restart_manager.schedule_flask_restart()
        if workers_published and flask_scheduled:
            return jsonify({'status': 'ok'})
        return jsonify({
            'status': 'partial',
            'workers_restart_published': bool(workers_published),
            'flask_restart_scheduled': bool(flask_scheduled),
        })
    except Exception:
        logger.exception('Failed to trigger plugin apply restart')
        return jsonify({'error': _GENERIC_ERROR}), 500
