# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every queueable function must actually import, and nothing else may run.

A worker resolves `func` out of a database row and calls it, so a stale import
anywhere in that module's chain does not fail at lint time or at enqueue time -
it fails once, per job, on the worker, after the task has already been claimed.
That is exactly how a dead `from app_helper import redis_conn` in album.py
survived a clean `ruff check` and turned every album job into an instant
ImportError. Linters read files in isolation; this executes the import.

Main Features:
* Every entry in ALLOWED_FUNCS resolves to a real callable
* A dotted path outside the allowlist is refused before importlib sees it
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import taskqueue  # noqa: E402


@pytest.mark.parametrize('dotted', sorted(taskqueue.ALLOWED_FUNCS))
def test_every_allowed_function_imports_and_is_callable(dotted):
    resolved = taskqueue.resolve_func(dotted)
    assert callable(resolved), f"{dotted} resolved to something that cannot be called"


def test_a_function_outside_the_allowlist_is_refused():
    with pytest.raises(taskqueue.UnknownTaskFunction):
        taskqueue.resolve_func('os.system')


def test_the_allowlist_is_refused_before_the_module_is_imported():
    with pytest.raises(taskqueue.UnknownTaskFunction):
        taskqueue.resolve_func('this.module.does.not.exist')


def test_enqueue_refuses_a_function_outside_the_allowlist():
    with pytest.raises(taskqueue.UnknownTaskFunction):
        taskqueue.enqueue(
            'os.system', args=('echo hi',), task_id='x', task_type='bogus'
        )
