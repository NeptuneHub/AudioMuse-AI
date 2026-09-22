# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The standalone build retries a model download instead of failing on one reset.

The Windows build failed on a single "connection forcibly closed by the remote
host" from the GitHub release CDN while downloading a model file, because
scripts/standalone/assemble_model.py ran `gh release download` exactly once.

Main Features:
* A download that fails and then succeeds is retried, with a backoff between tries
* A download that keeps failing still fails the build, loudly, after the retries
* A download that succeeds first time neither waits nor retries
"""

import importlib.util
import pathlib
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'standalone' / 'assemble_model.py'


@pytest.fixture
def assemble_model():
    spec = importlib.util.spec_from_file_location('assemble_model_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs(*codes):
    results = iter(codes)

    def run(cmd, check=False):
        code = next(results)
        if check and code:
            raise subprocess.CalledProcessError(code, cmd)
        return SimpleNamespace(returncode=code)

    return run


def test_a_reset_download_is_retried_until_it_succeeds(assemble_model):
    with (
        patch.object(assemble_model.subprocess, 'run', side_effect=_runs(1, 1, 0)) as run,
        patch.object(assemble_model.time, 'sleep') as sleep,
    ):
        assemble_model._gh_download('v1', 'owner/repo', 'model', ['a.onnx'])

    assert run.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == list(
        assemble_model._DOWNLOAD_RETRY_DELAYS[:2]
    )


def test_a_download_that_keeps_failing_still_fails_the_build(assemble_model):
    attempts = len(assemble_model._DOWNLOAD_RETRY_DELAYS) + 1
    with (
        patch.object(assemble_model.subprocess, 'run', side_effect=_runs(*([1] * attempts))) as run,
        patch.object(assemble_model.time, 'sleep'),
        pytest.raises(subprocess.CalledProcessError),
    ):
        assemble_model._gh_download('v1', 'owner/repo', 'model', ['a.onnx'])

    assert run.call_count == attempts


def test_a_first_time_success_neither_waits_nor_retries(assemble_model):
    with (
        patch.object(assemble_model.subprocess, 'run', side_effect=_runs(0)) as run,
        patch.object(assemble_model.time, 'sleep') as sleep,
    ):
        assemble_model._gh_download('v1', 'owner/repo', 'model', ['a.onnx', 'b.onnx'])

    assert run.call_count == 1
    sleep.assert_not_called()
    cmd = run.call_args.args[0]
    assert cmd[:3] == ['gh', 'release', 'download'] and '--clobber' in cmd
