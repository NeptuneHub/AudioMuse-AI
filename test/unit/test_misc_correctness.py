# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Assorted correctness guards for isolated numeric and config behaviors.

Groups small regression checks that each pin one previously broken behavior,
loading target modules in isolation with stubbed dependencies.

Main Features:
* Mistral request timeout scales linearly with the configured seconds
* GMM component count is capped at the available sample count and never crashes
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, Mock, patch

import numpy as np


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _ensure_namespace_pkg(name: str, sub_path: str) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, sub_path)]
    sys.modules[name] = pkg


def _ensure_mistralai_stub():
    if 'mistralai' in sys.modules:
        return
    try:
        import mistralai  # noqa: F401

        return
    except ImportError:
        pass
    mod = types.ModuleType('mistralai')
    mod.Mistral = MagicMock
    sys.modules['mistralai'] = mod


def _load_submodule(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_namespace_pkg('tasks', 'tasks')
_ensure_namespace_pkg('tasks.ai', 'tasks/ai')
_ensure_namespace_pkg('tasks.ai.providers', 'tasks/ai/providers')
_ensure_mistralai_stub()

mistral_mod = _load_submodule('tasks.ai.providers.mistral', 'tasks/ai/providers/mistral.py')


def _load_brainstorm_gmm():
    name = 'brainstorm_real_gmm_080'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO_ROOT, 'query', 'brainstorm_real_gmm_080.py')
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mock_mistral_client():
    mock_message = Mock()
    mock_message.content = "Generated Name"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_chat = Mock()
    mock_chat.complete.return_value = mock_response

    mock_client = Mock()
    mock_client.chat = mock_chat
    return mock_client, mock_chat


class TestMistralTimeoutScaling:
    def _invoke_and_capture_timeout(self, timeout_seconds):
        import config as cfg

        cfg.AI_REQUEST_TIMEOUT_SECONDS = timeout_seconds

        mock_client, mock_chat = _mock_mistral_client()
        with (
            patch('mistralai.Mistral', return_value=mock_client),
            patch.object(mistral_mod.time, 'sleep'),
        ):
            result = mistral_mod.generate_text(
                api_key="valid-key",
                model_name="ministral-3b-latest",
                full_prompt="Create a name",
                skip_delay=True,
            )
        assert result == "Generated Name"
        assert mock_chat.complete.called
        _, kwargs = mock_chat.complete.call_args
        return kwargs

    def test_timeout_is_30000_for_30_seconds(self):
        kwargs = self._invoke_and_capture_timeout(30)
        assert 'timeout_ms' in kwargs
        assert kwargs['timeout_ms'] == 30000
        assert kwargs['timeout_ms'] != 960

    def test_timeout_is_300000_for_default_300_seconds(self):
        kwargs = self._invoke_and_capture_timeout(300)
        assert kwargs['timeout_ms'] == 300000

    def test_timeout_scales_linearly(self):
        k15 = self._invoke_and_capture_timeout(15)
        k60 = self._invoke_and_capture_timeout(60)
        assert k15['timeout_ms'] == 15000
        assert k60['timeout_ms'] == 60000
        assert k60['timeout_ms'] == 4 * k15['timeout_ms']

    def test_timeout_not_hardcoded_960(self):
        for seconds in (1, 10, 45, 120):
            kwargs = self._invoke_and_capture_timeout(seconds)
            assert kwargs['timeout_ms'] == seconds * 1000


class TestGmmComponentCap:
    def test_k_capped_at_sample_count(self):
        mod = _load_brainstorm_gmm()
        X = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        best_gmm, best_k, bic_values = mod.fit_gmm_bic(X, k_range=range(2, 50))

        assert best_gmm is not None
        assert best_gmm.n_components <= len(X)
        assert best_k <= len(X)
        assert all(k <= len(X) for k in bic_values)

    def test_does_not_crash_with_tiny_X(self):
        mod = _load_brainstorm_gmm()
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        best_gmm, best_k, bic_values = mod.fit_gmm_bic(X, k_range=range(2, 100))
        assert best_gmm is not None
        assert best_gmm.n_components <= len(X)
        assert best_k == len(X)
        assert set(bic_values) == {2}

    def test_larger_X_uses_full_range_within_cap(self):
        mod = _load_brainstorm_gmm()
        rng = np.random.RandomState(0)
        X = rng.rand(6, 4).astype(np.float64)
        _, best_k, bic_values = mod.fit_gmm_bic(X, k_range=range(2, 5))
        assert set(bic_values) == {2, 3, 4}
        assert best_k <= len(X)
