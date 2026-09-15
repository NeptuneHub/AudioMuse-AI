# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recognise and run the worker processes a frozen launcher spawns of itself.

``multiprocessing`` and joblib/loky start worker processes by re-running
``sys.executable`` with ``-m <module>``, ``-c <code>`` or
``--multiprocessing-fork``. Inside a PyInstaller bundle ``sys.executable`` is
the AudioMuse-AI binary and the bootloader drops those switches, so without
this module every spawned child re-enters the launcher as if the user had
started the app: on macOS that meant one "already running, open the browser"
per worker (issue #827), and on every platform the pool then died with
``TerminatedWorkerError`` and the artist similarity index was never rebuilt.
PyInstaller's own ``multiprocessing`` runtime hook only patches
``multiprocessing.freeze_support`` (which nothing called) and only knows the
stdlib spawn payloads, so loky's are handled here too.

Main Features:
* ``run_frozen_child`` runs the spawn payload named by argv and reports it.
* ``dispatch_child_invocation`` is the one entry the three launchers share: a
  spawn payload, a ``--run-restore`` runner or a ``--role=`` child, in that
  order, so no platform can forget one of them.
* Only ``multiprocessing`` and ``joblib.externals.loky`` targets are accepted.
* The child sees the argv tail its parent passed, without the spawn switch.
* ``-m`` payloads run with ``alter_sys=False`` so ``sys.modules["__main__"]``
  stays the launcher, matching the parent: loky's ``_fixup_main_from_path``
  compares the two and would otherwise re-run the bundle entry script from a
  path that does not exist on disk.
"""

import ast
import importlib
import runpy
import sys

import service_roles

MULTIPROCESSING_FORK_FLAG = "--multiprocessing-fork"

CHILD_MODULE_PREFIXES = (
    "multiprocessing.",
    "joblib.externals.loky.",
)


def _switch_value(argv, switch):
    for index in range(1, len(argv) - 1):
        if argv[index] == switch:
            return index, argv[index + 1]
    return None


def _allowed(module):
    return bool(module) and module.startswith(CHILD_MODULE_PREFIXES)


def _parse_main_call(code):
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    module = None
    call = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            continue
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if module is not None or names != ["main"]:
                return None
            module = node.module
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if call is not None:
                return None
            call = node.value
            continue
        return None
    if not _allowed(module) or call is None:
        return None
    if not isinstance(call.func, ast.Name) or call.func.id != "main":
        return None
    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                kwargs.update(ast.literal_eval(keyword.value))
            else:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    except (ValueError, TypeError):
        return None
    return module, args, kwargs


def _run_multiprocessing_fork(argv):
    rest = argv[2:]
    if rest and all("=" in arg for arg in rest):
        from multiprocessing.spawn import spawn_main

        kwds = {}
        for arg in rest:
            name, _, value = arg.partition("=")
            try:
                kwds[name] = None if value == "None" else int(value)
            except ValueError:
                return False
        sys.argv = list(argv)
        spawn_main(**kwds)
        return True
    if sys.platform == "win32" and len(rest) == 1 and rest[0].isdigit():
        from joblib.externals.loky.backend.popen_loky_win32 import main

        sys.argv = list(argv)
        main(pipe_handle=int(rest[0]))
        return True
    return False


def run_frozen_child(argv=None, frozen=None):
    if frozen is None:
        frozen = getattr(sys, "frozen", False)
    if not frozen:
        return False
    argv = list(sys.argv if argv is None else argv)

    if len(argv) >= 2 and argv[1] == MULTIPROCESSING_FORK_FLAG:
        return _run_multiprocessing_fork(argv)

    found = _switch_value(argv, "-m")
    if found is not None and _allowed(found[1]):
        index, module = found
        sys.argv = [argv[0]] + argv[index + 2:]
        runpy.run_module(module, run_name="__main__", alter_sys=False)
        return True

    found = _switch_value(argv, "-c")
    if found is not None:
        index, code = found
        parsed = _parse_main_call(code)
        if parsed is not None:
            module, args, kwargs = parsed
            sys.argv = ["-c"] + argv[index + 2:]
            importlib.import_module(module).main(*args, **kwargs)
            return True

    return False


def dispatch_child_invocation(run_role):
    if run_frozen_child():
        return True

    if "--run-restore" in sys.argv:
        index = sys.argv.index("--run-restore")
        from app_backup import _run_restore_runner

        sys.exit(_run_restore_runner(sys.argv[index + 1], sys.argv[index + 2]))

    role = service_roles.role_from_argv()
    if role:
        run_role(role)
        return True

    return False
