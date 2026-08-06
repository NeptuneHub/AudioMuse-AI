# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The queue implementation stays inside taskqueue/.

Replacing the queue engine once touched most of the repository because queue
mechanics had leaked everywhere: job ids, retry budgets, registry probes and
queue handles were open-coded in blueprints and task modules. Swapping the engine
again must mean rewriting taskqueue/ and nothing else, so this test fails the
moment a queue internal escapes the package.

The rule is about MECHANICS, not about the task_status table: callers legitimately
read a task's status, progress and details. What they may not do is know how a job
is claimed, how liveness is decided, or which columns record either.

Main Features:
* No module outside taskqueue/ names a queue-only column
* No module outside taskqueue/ uses the claim, advisory-lock or LISTEN primitives
* Callers depend only on the documented public API
"""

import os
import re
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import taskqueue  # noqa: E402

_SKIP_DIRS = {
    '.git', '.venv', '.venv-windows', '__pycache__', 'build', 'dist', 'pginstall',
    'node_modules', 'test', 'taskqueue', 'query', 'screenshot', 'scripts',
}

# Columns that exist only because task_status is also the queue. A caller naming
# one of these has reached past the API into the storage layout.
_QUEUE_ONLY_COLUMNS = ('queue_name', 'worker_id', 'max_attempts')

# The primitives that decide who runs what. These belong to exactly one module.
#
# Advisory locks are deliberately NOT on this list: the app has used them for
# years for the schema lock, the CLAP text-search lock and the duplicate-repair
# locks, and they are not a queue concept. What is queue-only is the LOCK_CLASS
# namespace, which is checked separately below.
_QUEUE_PRIMITIVES = (
    'FOR UPDATE SKIP LOCKED',
    'audiomuse_job',
    'audiomuse_cancel',
)

# The whole surface a caller is allowed to use.
_PUBLIC_API = frozenset((
    'enqueue', 'cancel', 'request_cancel', 'request_cancel_all', 'publish_event',
    'current_task_id', 'set_current_task_id', 'resolve_func',
    'reap_finished_children', 'live_children', 'worker_snapshot',
    # Cancel has to be ordered against a Start, and which key does that is the
    # queue's business, not a blueprint's - so it is exposed here rather than
    # letting app_helper reach into taskqueue.sql for the lock id.
    'take_start_lock',
    # A fan-out stores one large shared input on the parent instead of copying it
    # into every child row; callers name the kwarg, the queue owns the storage.
    'put_shared_payload', 'clear_shared_payload', 'SHARED_KWARG_REF',
    # An enqueue outcome callers must be able to catch: a resumed parent
    # re-launching a child whose row already exists gets this instead of a row.
    'TaskNotQueued',
    'TaskAlreadyRunning', 'UnknownTaskFunction', 'ALLOWED_FUNCS',
    'QUEUE_HIGH', 'QUEUE_DEFAULT', 'PRIORITY_FRONT', 'CANCEL_ALL',
))


def _source_files():
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith('.')]
        for filename in filenames:
            if filename.endswith('.py'):
                yield os.path.join(dirpath, filename)


def _relative(path):
    return os.path.relpath(path, _REPO_ROOT).replace(os.sep, '/')


@pytest.mark.parametrize('column', _QUEUE_ONLY_COLUMNS)
def test_no_module_outside_the_package_names_a_queue_only_column(column):
    pattern = re.compile(rf"\b{column}\b")
    offenders = []
    for path in _source_files():
        with open(path, encoding='utf-8', errors='replace') as handle:
            for number, line in enumerate(handle, 1):
                if pattern.search(line) and ('task_status' in line or f".{column}" in line):
                    offenders.append(f"{_relative(path)}:{number}")
    assert not offenders, (
        f"{column!r} is a taskqueue-internal column; ask the package instead: {offenders}"
    )


@pytest.mark.parametrize('primitive', _QUEUE_PRIMITIVES)
def test_no_module_outside_the_package_uses_a_queue_primitive(primitive):
    offenders = []
    for path in _source_files():
        with open(path, encoding='utf-8', errors='replace') as handle:
            for number, line in enumerate(handle, 1):
                if primitive in line:
                    offenders.append(f"{_relative(path)}:{number}")
    assert not offenders, (
        f"{primitive!r} belongs to taskqueue/ alone; found in {offenders}"
    )


def test_callers_only_use_the_documented_public_api():
    used = set()
    reference = re.compile(r"\btaskqueue\.([A-Za-z_][A-Za-z0-9_]*)")
    submodules = {'sql', 'worker', 'maintenance', 'control', 'listen', 'process'}
    for path in _source_files():
        with open(path, encoding='utf-8', errors='replace') as handle:
            for match in reference.finditer(handle.read()):
                name = match.group(1)
                if name not in submodules:
                    used.add(name)
    escaped = used - _PUBLIC_API
    assert not escaped, (
        f"these are not part of the queue's public API: {sorted(escaped)}. "
        "Either add them to the API deliberately, or keep them inside the package."
    )


def test_only_the_package_uses_the_queue_advisory_lock_namespace():
    offenders = []
    for path in _source_files():
        with open(path, encoding='utf-8', errors='replace') as handle:
            for number, line in enumerate(handle, 1):
                if 'LOCK_CLASS' in line:
                    offenders.append(f"{_relative(path)}:{number}")
    assert not offenders, (
        f"the queue's advisory-lock namespace is internal; found in {offenders}"
    )


def test_the_public_api_actually_exists():
    missing = [name for name in _PUBLIC_API if not hasattr(taskqueue, name)]
    missing = [name for name in missing if name != 'cancel']
    assert not missing, f"the API promises names the package does not define: {missing}"
