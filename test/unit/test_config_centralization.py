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

Main Features:
* config attributes exist with the documented default and coerce env overrides
* Importer modules still reference the config names they depend on
* Importers have no local os.environ.get or getattr-fallback default for those names
"""

import importlib
import os
import re

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
