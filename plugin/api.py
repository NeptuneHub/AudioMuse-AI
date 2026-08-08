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

import inspect
import logging
import re
import sys
import uuid

from flask import render_template, url_for

import config
import database
from database import (
    get_db,
    save_task_status,
    get_score_data_by_ids,
    get_tracks_by_ids,
)
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

# Analysis steps a plugin may replace wholesale with register_analysis_provider.
ANALYSIS_COMPONENTS = frozenset({'asr'})

# Where a plugin provider goes in the ONNX chain, see register_onnx_provider.
ONNX_POSITIONS = frozenset({'before_cuda', 'before_cpu'})

# Argument names plugin.manager.run_plugin_task consumes itself, so a plugin task
# function that declares one of them could never receive it.
RESERVED_TASK_PARAMS = frozenset({'server_scope', 'task_claim_required'})

__all__ = [
    'PluginContext', 'config', 'logger', 'get_db', 'save_task_status',
    'get_score_data_by_ids', 'get_tracks_by_ids', 'get_setting', 'set_setting',
    'table', 'enqueue', 'valid_plugin_id', 'dotted_path', 'render_page',
    'manage_plugins_url',
    'active_server_id', 'list_servers', 'use_server',
    'TASK_STATUS_PENDING', 'TASK_STATUS_STARTED', 'TASK_STATUS_PROGRESS',
    'TASK_STATUS_SUCCESS', 'TASK_STATUS_FAILURE', 'TASK_STATUS_REVOKED',
]


def render_page(body, title=None, active='plugins'):
    return render_template(
        'plugin_page.html',
        plugin_body=body,
        plugin_title=title,
        active=active,
        title=(title or 'AudioMuse-AI'),
    )


def manage_plugins_url():
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
    pid = _current_plugin_id()
    if not pid:
        raise RuntimeError('table() must be called from plugin code')
    if not _NAME_RE.match(str(name)):
        raise ValueError('table name must match ^[a-z][a-z0-9_]*$')
    return f"plugin_{pid}__{name}"


def get_setting(key, default=None):
    pid = _current_plugin_id()
    if not pid:
        return default
    settings = database.get_plugin_settings(pid)
    return settings.get(key, default)


def set_setting(key, value):
    pid = _current_plugin_id()
    if not pid:
        raise RuntimeError('set_setting() must be called from plugin code')
    settings = database.get_plugin_settings(pid)
    settings[key] = value
    database.set_plugin_settings(pid, settings)
    return value


class QueuedPluginTask(str):
    @property
    def id(self):
        return str(self)


def _reject_reserved_params(func, dotted):
    if not callable(func):
        return
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return
    reserved = sorted(
        name for name, param in params.items()
        if name in RESERVED_TASK_PARAMS
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    )
    if reserved:
        raise TypeError(
            f"plugin task {dotted} declares reserved argument names: {', '.join(reserved)}. "
            "AudioMuse-AI consumes server_scope and task_claim_required itself to run a plugin "
            "task once per music server, so your function can never receive them. Rename the "
            "parameter; call active_server_id() to learn which server the current run targets."
        )


def enqueue(func, *args, queue='default', **kwargs):
    import json

    import taskqueue

    task_id = str(uuid.uuid4())
    dotted = dotted_path(func)
    _reject_reserved_params(func, dotted)
    try:
        json.dumps({'args': list(args), 'kwargs': kwargs})
    except TypeError as exc:
        raise TypeError(
            f"plugin task {dotted} was queued with arguments that cannot be stored: {exc}. "
            "Task arguments must be JSON-serializable (str, int, float, bool, None, "
            "list, dict); pass an ISO string instead of a datetime, a list instead of a set."
        ) from exc
    taskqueue.enqueue(
        'plugin.manager.run_plugin_task',
        args=(dotted,) + tuple(args),
        kwargs=kwargs,
        task_id=task_id,
        task_type=f'plugin.{dotted}',
        queue=taskqueue.QUEUE_HIGH if queue == 'high' else taskqueue.QUEUE_DEFAULT,
        details={'message': 'Plugin task queued.'},
    )
    return QueuedPluginTask(task_id)


def active_server_id():
    from tasks.mediaserver import context as ms_context

    return ms_context.active_server_id()


def list_servers():
    from tasks.mediaserver import registry as ms_registry

    return ms_registry.list_servers()


def use_server(server_id):
    from tasks.mediaserver import context as ms_context, registry as ms_registry

    if not server_id:
        return ms_context.use_server(None)
    return ms_registry.bind({'server_id': server_id})


def _model_scope(value):
    if not value:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


class PluginContext:

    def __init__(self, plugin_id, role):
        self.plugin_id = plugin_id
        self.role = role
        self.blueprint = None
        self.menu_items = []
        self.settings_endpoint = None
        self.cron_tasks = {}
        self.tasks = {}
        self.onnx_providers = []
        self.analysis_providers = {}
        self.flask_start = []
        self.worker_start = []
        self.song_analyzed_hooks = []
        self.install_hooks = []

    def add_blueprint(self, blueprint):
        self.blueprint = blueprint

    def add_menu_item(self, label, endpoint, admin_only=False):
        self.menu_items.append({'label': label, 'endpoint': endpoint, 'admin_only': bool(admin_only)})

    def set_settings_page(self, endpoint):
        self.settings_endpoint = endpoint

    def add_task(self, name, func, queue='default'):
        dotted = dotted_path(func)
        _reject_reserved_params(func, dotted)
        self.tasks[name] = {'dotted': dotted, 'queue': queue}

    def add_cron_task(self, name, func, queue='default'):
        dotted = dotted_path(func)
        _reject_reserved_params(func, dotted)
        self.cron_tasks[name] = {'dotted': dotted, 'queue': queue}

    def register_onnx_provider(self, name, options=None, position='before_cpu',
                               only_models=None, exclude_models=None,
                               needs_static_shapes=False):
        if position not in ONNX_POSITIONS:
            logger.warning(
                "Plugin %s registered ONNX provider %s with unknown position %r; "
                "using 'before_cpu'. Valid positions: %s",
                self.plugin_id, name, position, sorted(ONNX_POSITIONS),
            )
            position = 'before_cpu'
        self.onnx_providers.append({
            'name': name,
            'options': options or {},
            'position': position,
            'only_models': _model_scope(only_models),
            'exclude_models': _model_scope(exclude_models),
            'needs_static_shapes': bool(needs_static_shapes),
        })

    def on_flask_start(self, func):
        self.flask_start.append(func)

    def on_worker_start(self, func):
        self.worker_start.append(func)

    def on_song_analyzed(self, func):
        self.song_analyzed_hooks.append(func)

    def on_install(self, func):
        self.install_hooks.append(func)

    def register_analysis_provider(self, component, factory, cache=True):
        if component not in ANALYSIS_COMPONENTS:
            logger.warning(
                'Plugin %s registered an analysis provider for unknown component %r; '
                'ignoring it. Known components: %s',
                self.plugin_id, component, sorted(ANALYSIS_COMPONENTS),
            )
            return
        if component in self.analysis_providers:
            logger.warning(
                'Plugin %s registered two analysis providers for %r; keeping the last one',
                self.plugin_id, component,
            )
        self.analysis_providers[component] = {'factory': factory, 'cache': bool(cache)}
