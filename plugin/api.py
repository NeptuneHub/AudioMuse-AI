# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Stable, author-facing plugin API surface.

The only module a plugin should import from. Exposes the registration context
handed to ``register(ctx)`` plus a sanctioned facade over the core app: database
access, task-status and queue helpers, per-plugin settings and table naming, and
read access to the core config. Keeps plugins from reaching into app internals.

Main Features:
* ``PluginContext`` accumulates flask-vs-worker component registrations.
* Facade helpers (``get_db``, ``get_setting``/``set_setting``, ``table``, ``enqueue``)
  auto-resolve the calling plugin id from the import namespace.
"""

import logging
import re
import sys

from flask import render_template, url_for

import config
import database
from database import (
    get_db,
    save_task_status,
    get_score_data_by_ids,
    get_tracks_by_ids,
)
from taskqueue import rq_queue_high, rq_queue_default
from config import (
    TASK_STATUS_PENDING,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
    TASK_STATUS_REVOKED,
)

NAMESPACE = 'audiomuse_plugins'

logger = logging.getLogger('audiomuse.plugin')

_ID_RE = re.compile(r'^[a-z][a-z0-9_]{1,63}$')
_NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,62}$')

__all__ = [
    'PluginContext', 'config', 'logger', 'get_db', 'save_task_status',
    'get_score_data_by_ids', 'get_tracks_by_ids', 'get_setting', 'set_setting',
    'table', 'enqueue', 'valid_plugin_id', 'dotted_path', 'render_page',
    'manage_plugins_url', 'rq_queue_high', 'rq_queue_default',
    'TASK_STATUS_PENDING', 'TASK_STATUS_STARTED', 'TASK_STATUS_PROGRESS',
    'TASK_STATUS_SUCCESS', 'TASK_STATUS_FAILURE', 'TASK_STATUS_REVOKED',
]


def render_page(body, title=None, active='plugins'):
    """Wrap a plugin's HTML in the AudioMuse-AI layout (sidebar nav + styling).

    Lets a plugin return a full page - with the app navigation preserved - from a
    single call: ``return render_page('<p>hi</p>', title='My Plugin')``. ``body``
    is rendered as-is (the plugin controls it); ``title`` shows as the page heading.
    """
    return render_template(
        'plugin_page.html',
        plugin_body=body,
        plugin_title=title,
        active=active,
        title=(title or 'AudioMuse-AI'),
    )


def manage_plugins_url():
    """Return the URL of the Manage Plugins admin page.

    Handy as the redirect target after a plugin's settings form saves, since the
    settings page is opened from that admin page.
    """
    return url_for('plugins_bp.plugins_page')


def valid_plugin_id(plugin_id):
    return bool(plugin_id) and bool(_ID_RE.match(str(plugin_id)))


def dotted_path(func):
    if isinstance(func, str):
        if '.' not in func:
            raise ValueError(f'Invalid dotted path: {func}')
        return func
    return f"{func.__module__}.{func.__name__}"


def _current_plugin_id():
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_globals.get('__name__', '')
        if name == NAMESPACE or name.startswith(NAMESPACE + '.'):
            rest = name[len(NAMESPACE):].lstrip('.')
            return rest.split('.')[0] if rest else None
        frame = frame.f_back
    return None


def table(name):
    """Return the namespaced table name ``plugin_<id>__<name>`` for the calling plugin."""
    pid = _current_plugin_id()
    if not pid:
        raise RuntimeError('table() must be called from plugin code')
    if not _NAME_RE.match(str(name)):
        raise ValueError('table name must match ^[a-z][a-z0-9_]*$')
    return f"plugin_{pid}__{name}"


def get_setting(key, default=None):
    """Return the DB-stored per-plugin setting for ``key`` or ``default``."""
    pid = _current_plugin_id()
    if not pid:
        return default
    settings = database.get_plugin_settings(pid)
    return settings.get(key, default)


def set_setting(key, value):
    """Persist a per-plugin setting override into the ``plugins.settings`` JSONB."""
    pid = _current_plugin_id()
    if not pid:
        raise RuntimeError('set_setting() must be called from plugin code')
    settings = database.get_plugin_settings(pid)
    settings[key] = value
    database.set_plugin_settings(pid, settings)
    return value


def enqueue(func, *args, queue='default', **kwargs):
    """Enqueue a plugin task by callable or dotted path, wrapped for app context."""
    q = rq_queue_high if queue == 'high' else rq_queue_default
    dotted = dotted_path(func)
    return q.enqueue(
        'plugin.manager.run_plugin_task',
        args=(dotted,) + tuple(args),
        kwargs=kwargs,
        job_timeout=-1,
    )


class PluginContext:
    """Registration sink passed to a plugin's ``register(ctx)``.

    A plugin declares where each component runs by which method it calls: the
    ``add_blueprint``/``add_menu_item``/``on_flask_start`` group activates on the
    Flask (online) container, while ``add_task``/``add_cron_task``/
    ``register_onnx_provider``/``on_worker_start``/``on_song_analyzed`` activate on
    the worker (batch) container. ``on_install`` runs once at install for schema setup.
    """

    def __init__(self, plugin_id, role):
        self.plugin_id = plugin_id
        self.role = role
        self.blueprint = None
        self.menu_items = []
        self.settings_endpoint = None
        self.cron_tasks = {}
        self.tasks = {}
        self.onnx_providers = []
        self.flask_start = []
        self.worker_start = []
        self.song_analyzed_hooks = []
        self.install_hooks = []

    def add_blueprint(self, blueprint):
        self.blueprint = blueprint

    def add_menu_item(self, label, endpoint, admin_only=False):
        self.menu_items.append({'label': label, 'endpoint': endpoint, 'admin_only': bool(admin_only)})

    def set_settings_page(self, endpoint):
        """Point the Manage Plugins 'Settings' button at this Flask endpoint.

        When set, the Settings button on the admin Plugins page opens this page
        instead of the generic JSON editor. No extra menu entry is created.
        """
        self.settings_endpoint = endpoint

    def add_task(self, name, func, queue='default'):
        self.tasks[name] = {'dotted': dotted_path(func), 'queue': queue}

    def add_cron_task(self, name, func, queue='default'):
        self.cron_tasks[name] = {'dotted': dotted_path(func), 'queue': queue}

    def register_onnx_provider(self, name, options=None, position='before_cpu'):
        self.onnx_providers.append({'name': name, 'options': options or {}, 'position': position})

    def on_flask_start(self, func):
        self.flask_start.append(func)

    def on_worker_start(self, func):
        self.worker_start.append(func)

    def on_song_analyzed(self, func):
        """Register a worker hook fired after a song finishes analysis and its results are saved.

        The hook receives one dict: ``item_id``, ``run_id`` (the analysis run's task
        id, shared by every song of one run), ``audio_path`` (the temp file, valid
        only during the call), ``metadata`` (title/artist/album/...), ``media_item``
        (the raw media-server track), ``analysis`` (tempo/key/scale/moods/energy or
        None), ``top_moods``, and ``musicnn_embedding``/``clap_embedding`` (or None).
        It runs on the worker inside an app context, so ``get_db``/``table`` work.
        """
        self.song_analyzed_hooks.append(func)

    def on_install(self, func):
        self.install_hooks.append(func)

    def register_analysis_provider(self, *args, **kwargs):
        logger.info(
            'Plugin %s called register_analysis_provider; alternative analysis models '
            'are a forward seam and are not wired in this version.',
            self.plugin_id,
        )
