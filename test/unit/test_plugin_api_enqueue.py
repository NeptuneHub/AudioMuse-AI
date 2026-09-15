# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The plugin enqueue facade respects the queue guard.

A user-triggered plugin task must not start while a queue-guard task is live,
exactly like the manual batch starts; a plugin task enqueueing follow-up work
from inside its own running task is exempt because that task already holds the
guard.

Main Features:
* enqueue from a request context (no running task) refuses when the guard is busy
* enqueue from inside a running task skips the guard
* enqueue with a clear guard proceeds and queues
"""

import pytest
from unittest.mock import patch

import plugin.api


def _patches(mapping):
    from contextlib import ExitStack

    stack = ExitStack()
    for target, value in mapping.items():
        stack.enter_context(patch(target, side_effect=value))
    return stack


def test_enqueue_refuses_when_a_queue_guard_task_is_live():
    blocking = {'task_type': 'main_analysis', 'task_id': 'live-1', 'status': 'RUNNING'}
    with _patches({
        'plugin.api.dotted_path': lambda f: 'demo.sync',
        'taskqueue.current_task_id': lambda: None,
        'taskqueue.enqueue': lambda *a, **k: k['task_id'],
        'plugin.api.get_queue_blocking_task': lambda *a, **k: blocking,
    }):
        with pytest.raises(Exception) as exc:
            plugin.api.enqueue(lambda: None)
    assert 'main_analysis' in str(exc.value)


def test_enqueue_from_inside_a_running_task_skips_the_guard():
    enqueue = lambda *a, **k: k['task_id']  # noqa: E731
    with _patches({
        'plugin.api.dotted_path': lambda f: 'demo.sync',
        'taskqueue.current_task_id': lambda: 'plugin-run-1',
        'taskqueue.enqueue': enqueue,
        'plugin.api.get_queue_blocking_task': (
            lambda *a, **k: (_ for _ in ()).throw(AssertionError('guard skipped'))
        ),
    }):
        plugin.api.enqueue(lambda: None)


def test_enqueue_proceeds_when_the_guard_is_clear():
    calls = []
    with _patches({
        'plugin.api.dotted_path': lambda f: 'demo.sync',
        'taskqueue.current_task_id': lambda: None,
        'taskqueue.enqueue': (
            lambda *a, **k: calls.append((a[0], k['task_type'])) or k['task_id']
        ),
        'plugin.api.get_queue_blocking_task': lambda *a, **k: None,
    }):
        plugin.api.enqueue(lambda: None)
    assert calls == [('plugin.manager.run_plugin_task', 'plugin.demo.sync')]
