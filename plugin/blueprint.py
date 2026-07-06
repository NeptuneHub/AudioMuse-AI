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
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Blueprint, render_template, jsonify, request, url_for

import config
import database
import restart_manager
from ssrf_guard import validate_outbound_url
from plugin.manager import plugin_manager, version_ge, _parse_version

logger = logging.getLogger(__name__)

plugins_bp = Blueprint('plugins_bp', __name__, template_folder='templates')

_REPOS_KEY = 'PLUGIN_REPOS'
_CATALOG_MAX_BYTES = 5 * 1024 * 1024

_GENERIC_ERROR = 'Operation failed. Check the container logs for details.'

_HTTP_SESSION = requests.Session()


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
    ok, message = validate_outbound_url(url)
    if not ok:
        raise ValueError(f'URL rejected: {message}')
    headers = {'User-Agent': f'AudioMuse-AI/{config.APP_VERSION}'}
    timeout = (config.PLUGIN_HTTP_CONNECT_TIMEOUT, config.PLUGIN_HTTP_READ_TIMEOUT)
    with _HTTP_SESSION.get(url, headers=headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        data = b''
        for chunk in resp.iter_content(chunk_size=65536):
            data += chunk
            if len(data) > max_bytes:
                raise ValueError('Download exceeds the configured size limit')
    return data


def _pick_version(versions, requested=None):
    compatible = []
    for entry in versions or []:
        min_core = entry.get('min_core_version') or entry.get('targetAbi')
        if not version_ge(config.APP_VERSION, min_core):
            continue
        if requested and str(entry.get('version')) != str(requested):
            continue
        compatible.append(entry)
    if not compatible:
        return None
    compatible.sort(key=lambda e: _parse_version(e.get('version')), reverse=True)
    return compatible[0]


def _resolve_versions(entry, errors):
    """Return (detail, versions) for a catalog entry.

    A catalog entry either carries an inline ``versions`` list (legacy single-file
    catalog) or a ``manifestUrl`` pointing at the plugin's own manifest, which is
    fetched (SSRF-guarded) and holds the detailed version history. The per-plugin
    manifest keeps the catalog static across plugin updates.
    """
    versions = entry.get('versions')
    if versions:
        return entry, versions
    manifest_url = entry.get('manifestUrl') or entry.get('manifest_url')
    if not manifest_url:
        return entry, None
    try:
        raw = _download(manifest_url, _CATALOG_MAX_BYTES)
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            raise TypeError('Manifest is not a JSON object')
    except Exception as exc:
        logger.warning('Failed to fetch plugin manifest %s: %s', manifest_url, exc)
        errors.append({'repo': manifest_url, 'error': str(exc)})
        return entry, None
    return doc, doc.get('versions')


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
    return plugin_id, {
        'id': plugin_id,
        'name': entry.get('name') or detail.get('name') or plugin_id,
        'description': entry.get('description') or detail.get('description', ''),
        'author': entry.get('author') or detail.get('author', ''),
        'image_url': entry.get('imageUrl') or detail.get('imageUrl', ''),
        'latest_version': best.get('version'),
        'source_url': best.get('sourceUrl'),
        'checksum': best.get('checksum'),
        'changelog': best.get('changelog', ''),
        'min_core_version': best.get('min_core_version') or best.get('targetAbi'),
        'source_repo': repo_url,
        'installed_version': current.get('version') if current else None,
    }, local_errors


def _fetch_catalog():
    installed = {p['id']: p for p in database.list_plugins()}
    errors = []
    pending = []
    for repo_url in _get_repos():
        try:
            raw = _download(repo_url, _CATALOG_MAX_BYTES)
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                raise TypeError('Repository catalog is not a JSON object')
        except Exception as exc:
            logger.warning('Failed to fetch plugin repo %s: %s', repo_url, exc)
            errors.append({'repo': repo_url, 'error': str(exc)})
            continue
        pending.extend((repo_url, entry) for entry in doc.get('plugins', []) if entry.get('id'))

    merged = {}
    if pending:
        workers = max(1, min(config.PLUGIN_CATALOG_FETCH_WORKERS, len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda item: _build_catalog_entry(item[0], item[1], installed), pending)
        for plugin_id, entry_data, local_errors in results:
            errors.extend(local_errors)
            if entry_data:
                merged[plugin_id] = entry_data
    return list(merged.values()), errors


@plugins_bp.route('/plugins', methods=['GET'])
def plugins_page():
    return render_template('plugins.html', title='AudioMuse-AI - Plugins', active='plugins')


@plugins_bp.route('/api/plugins/installed', methods=['GET'])
def api_installed():
    plugins = database.list_plugins()
    registry = {r['id']: r for r in plugin_manager.registry()}
    for plugin in plugins:
        entry = registry.get(plugin['id'])
        plugin['error'] = entry.get('error') if entry else None
        plugin['requirements'] = plugin.get('requirements') or []
        endpoint = plugin_manager.get_settings_endpoint(plugin['id'])
        settings_url = None
        if endpoint:
            try:
                settings_url = url_for(endpoint)
            except Exception:
                settings_url = None
        plugin['settings_url'] = settings_url
    return jsonify({'plugins': plugins, 'pip_supported': _pip_supported()})


@plugins_bp.route('/api/plugins/catalog', methods=['GET'])
def api_catalog():
    try:
        plugins, errors = _fetch_catalog()
    except Exception:
        logger.exception('Failed to build plugin catalog')
        return jsonify({'error': _GENERIC_ERROR}), 500
    return jsonify({'plugins': plugins, 'repos': _get_repos(), 'errors': errors})


@plugins_bp.route('/api/plugins/install', methods=['POST'])
def api_install():
    data = request.get_json(silent=True) or {}
    plugin_id = data.get('id')
    if not plugin_id:
        return jsonify({'error': 'Missing required field: id'}), 400
    try:
        catalog, _ = _fetch_catalog()
        match = next((p for p in catalog if p['id'] == plugin_id), None)
        if not match:
            return jsonify({'error': 'Plugin not found in any configured repository'}), 404
        source_url = match['source_url']
        checksum = match['checksum']
        source_repo = match['source_repo']
        package = _download(source_url, config.PLUGIN_MAX_DOWNLOAD_MB * 1024 * 1024)
        manifest = plugin_manager.install_package(
            package, source_url=source_url, source_repo=source_repo, expected_checksum=checksum
        )
        return jsonify({'status': 'ok', 'manifest': manifest, 'restart_required': True})
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
        restart_manager.publish_restart_request()
        restart_manager.schedule_flask_restart()
        return jsonify({'status': 'ok'})
    except Exception:
        logger.exception('Failed to trigger plugin apply restart')
        return jsonify({'error': _GENERIC_ERROR}), 500
