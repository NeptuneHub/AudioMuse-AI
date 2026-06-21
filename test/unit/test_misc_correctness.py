"""Unit tests for two correctness fixes (PR theme #13).

1. tasks/ai/providers/mistral.py: the Mistral request timeout must be derived
   from config.AI_REQUEST_TIMEOUT_SECONDS * 1000 (milliseconds), not a
   hardcoded constant. The timeout_ms kwarg passed to chat.complete must scale
   with the configured value.

2. query/brainstorm_real_gmm_080.py: fit_gmm_bic caps the GMM component count
   at len(X), so a GaussianMixture is never fit with n_components > sample
   count (which would crash) even when a larger K range is requested.

Heavy / missing deps are stubbed; no real network or DB access occurs.
"""
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, Mock, patch

import numpy as np


_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)


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
    except (ImportError, ModuleNotFoundError):
        pass
    mod = types.ModuleType('mistralai')
    mod.Mistral = MagicMock
    sys.modules['mistralai'] = mod


def _load_submodule(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO_ROOT, relpath)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_namespace_pkg('tasks', 'tasks')
_ensure_namespace_pkg('tasks.ai', 'tasks/ai')
_ensure_namespace_pkg('tasks.ai.providers', 'tasks/ai/providers')
_ensure_mistralai_stub()

mistral_mod = _load_submodule(
    'tasks.ai.providers.mistral', 'tasks/ai/providers/mistral.py'
)


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
    """Build a Mistral client mock whose chat.complete records its kwargs."""
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


# ---------------------------------------------------------------------------
# (1) Mistral timeout scales with config.AI_REQUEST_TIMEOUT_SECONDS
# ---------------------------------------------------------------------------

class TestMistralTimeoutScaling:
    def _invoke_and_capture_timeout(self, timeout_seconds):
        import config as cfg
        cfg.AI_REQUEST_TIMEOUT_SECONDS = timeout_seconds

        mock_client, mock_chat = _mock_mistral_client()
        with patch('mistralai.Mistral', return_value=mock_client), \
                patch.object(mistral_mod.time, 'sleep'):
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
        # Guard against the old hardcoded value.
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
        # The pre-fix bug hardcoded the timeout to 960; any config value other
        # than 0.96s must therefore produce a different timeout_ms.
        for seconds in (1, 10, 45, 120):
            kwargs = self._invoke_and_capture_timeout(seconds)
            assert kwargs['timeout_ms'] == seconds * 1000


# ---------------------------------------------------------------------------
# (2) GMM component count is capped at len(X)
# ---------------------------------------------------------------------------

class TestGmmComponentCap:
    def test_k_capped_at_sample_count(self):
        mod = _load_brainstorm_gmm()
        X = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        # Request many more components than samples.
        best_gmm, best_k, bic_values = mod.fit_gmm_bic(X, k_range=range(2, 50))

        assert best_gmm is not None
        assert best_gmm.n_components <= len(X)
        assert best_k <= len(X)
        # No K above the sample count may have been evaluated.
        assert all(k <= len(X) for k in bic_values)

    def test_does_not_crash_with_tiny_X(self):
        mod = _load_brainstorm_gmm()
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        best_gmm, best_k, bic_values = mod.fit_gmm_bic(X, k_range=range(2, 100))
        assert best_gmm is not None
        assert best_gmm.n_components <= len(X)
        # With exactly 2 samples and a min K of 2, only K=2 is admissible.
        assert best_k == len(X)
        assert set(bic_values) == {2}

    def test_larger_X_uses_full_range_within_cap(self):
        mod = _load_brainstorm_gmm()
        rng = np.random.RandomState(0)
        X = rng.rand(6, 4).astype(np.float64)
        _, best_k, bic_values = mod.fit_gmm_bic(X, k_range=range(2, 5))
        # cap = min(stop-1, len(X)) = min(4, 6) = 4; range start is 2.
        assert set(bic_values) == {2, 3, 4}
        assert best_k <= len(X)
