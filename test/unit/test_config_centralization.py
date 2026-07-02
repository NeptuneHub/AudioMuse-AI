# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Config-default centralization for RQ and instrumentation settings.

Asserts each tunable exists in config with its default, honors its environment
variable, and that importer modules reference config rather than re-reading env.
Also enforces the house rule repo-wide: a tunable's default lives ONLY in
config.py, so no other runtime module may re-read its env var with a second
default or use a getattr(config, ...) fallback default.

Main Features:
* config attributes exist with the documented default and coerce env overrides
* Importer modules still reference the config names they depend on
* Importers have no local os.environ.get or getattr-fallback default for those names
* Repo-wide: no runtime module re-reads a config-owned env var with a default
* Repo-wide: no runtime module uses getattr(config, 'NAME', default) fallbacks
"""

import ast
import importlib
import os
import re
import subprocess

import pytest

import config


_DEFAULTS = (
    ('RQ_MAX_JOBS', 50, 'RQ_MAX_JOBS', '7', 7),
    ('RQ_MAX_JOBS_HIGH', 100, 'RQ_MAX_JOBS_HIGH', '13', 13),
    ('RQ_LOGGING_LEVEL', 'INFO', 'RQ_LOGGING_LEVEL', 'debug', 'DEBUG'),
    ('RADIUS_INSTRUMENTATION', False, 'RADIUS_INSTRUMENTATION', 'True', True),
)


@pytest.mark.parametrize(
    'attr,default', [(d[0], d[1]) for d in _DEFAULTS], ids=[d[0] for d in _DEFAULTS]
)
def test_attribute_exists_with_default(attr, default):
    assert hasattr(config, attr), f"config.{attr} missing"
    env_var = next(d[2] for d in _DEFAULTS if d[0] == attr)
    saved = os.environ.pop(env_var, None)
    try:
        reloaded = importlib.reload(config)
        assert getattr(reloaded, attr) == default
    finally:
        if saved is not None:
            os.environ[env_var] = saved
        importlib.reload(config)


@pytest.mark.parametrize(
    'attr,_default,env_var,env_value,expected',
    list(_DEFAULTS),
    ids=[d[0] for d in _DEFAULTS],
)
def test_attribute_honors_env(attr, _default, env_var, env_value, expected, monkeypatch):
    monkeypatch.setenv(env_var, env_value)
    try:
        reloaded = importlib.reload(config)
        assert getattr(reloaded, attr) == expected
    finally:
        monkeypatch.delenv(env_var, raising=False)
        importlib.reload(config)


def _read_source(relative_path):
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    with open(os.path.join(repo_root, relative_path), encoding='utf-8') as fh:
        return fh.read()


_IMPORTERS = {
    'rq_worker.py': ('RQ_MAX_JOBS', 'RQ_LOGGING_LEVEL'),
    'rq_worker_high_priority.py': ('RQ_MAX_JOBS_HIGH', 'RQ_LOGGING_LEVEL'),
    'tasks/radius_walk_helper.py': ('RADIUS_INSTRUMENTATION',),
    'tasks/ivf_manager.py': ('RADIUS_INSTRUMENTATION',),
}


@pytest.mark.parametrize('relative_path,names', sorted(_IMPORTERS.items()))
def test_importer_references_config_name(relative_path, names):
    src = _read_source(relative_path)
    for name in names:
        assert name in src, f"{relative_path} no longer references {name}"


@pytest.mark.parametrize('relative_path,names', sorted(_IMPORTERS.items()))
def test_importer_has_no_local_default(relative_path, names):
    src = _read_source(relative_path)
    for name in names:
        env_pat = re.compile(r"os\.(?:environ\.get|getenv)\(\s*['\"]" + re.escape(name) + r"['\"]")
        assert not env_pat.search(src), (
            f"{relative_path} re-reads env for {name}; must use config.{name}"
        )
        getattr_pat = re.compile(r"getattr\(\s*config\s*,\s*['\"]" + re.escape(name) + r"['\"]\s*,")
        assert not getattr_pat.search(src), (
            f"{relative_path} has a getattr fallback default for {name}; "
            f"must use config.{name} directly"
        )


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

_SCAN_EXCLUDED_PREFIXES = (
    'test/',
    'screenshot/',
    'query/',
    'native-build/',
    'scripts/',
)


def _tracked_runtime_py_files():
    out = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=_REPO_ROOT).decode('utf-8')
    files = []
    for rel in out.splitlines():
        if not rel or rel == 'config.py':
            continue
        if rel.startswith(_SCAN_EXCLUDED_PREFIXES):
            continue
        files.append(rel)
    return files


def _is_os_environ_get(node):
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == 'getenv' and isinstance(func.value, ast.Name) and func.value.id == 'os':
        return True
    return (
        func.attr == 'get'
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == 'environ'
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == 'os'
    )


def _config_owned_env_names():
    with open(os.path.join(_REPO_ROOT, 'config.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_os_environ_get(node):
            if node.args and isinstance(node.args[0], ast.Constant):
                names.add(node.args[0].value)
    return names


def _parse(rel_path):
    with open(os.path.join(_REPO_ROOT, rel_path), encoding='utf-8') as fh:
        return ast.parse(fh.read(), filename=rel_path)


def test_config_owned_env_names_found():
    names = _config_owned_env_names()
    assert len(names) > 100, f'config.py env-var extraction looks broken ({len(names)} names)'


def _config_reread_violation(rel, node, owned):
    if not (isinstance(node, ast.Call) and _is_os_environ_get(node)):
        return None
    if len(node.args) < 2:
        return None
    if not (node.args and isinstance(node.args[0], ast.Constant)):
        return None
    name = node.args[0].value
    if name in owned:
        return f'{rel}:{node.lineno}: os.environ.get({name!r}, <default>)'
    return None


def test_no_module_rereads_config_env_with_default():
    owned = _config_owned_env_names()
    violations = []
    for rel in _tracked_runtime_py_files():
        for node in ast.walk(_parse(rel)):
            violation = _config_reread_violation(rel, node, owned)
            if violation:
                violations.append(violation)
    assert not violations, (
        'Env vars owned by config.py must not be re-read with a second default '
        '(import config.<NAME> instead, or read the env without a default):\n  '
        + '\n  '.join(sorted(violations))
    )


def test_no_module_uses_getattr_config_fallback():
    violations = []
    for rel in _tracked_runtime_py_files():
        for node in ast.walk(_parse(rel)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'getattr'
                and len(node.args) == 3
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == 'config'
                and isinstance(node.args[1], ast.Constant)
            ):
                continue
            violations.append(f'{rel}:{node.lineno}: getattr(config, {node.args[1].value!r}, <default>)')
    assert not violations, (
        'getattr(config, ..., default) fallbacks re-specify a default outside '
        'config.py; config always defines the attribute, so access it directly:\n  '
        + '\n  '.join(sorted(violations))
    )
