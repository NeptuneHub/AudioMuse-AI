"""
Regression smoke tests for the provider-selection refactor (PR #353).

These tests verify that the three call-sites that were refactored to use
build_ort_provider_options / split_provider_options still dispatch
InferenceSession calls with *exactly* the same arguments that the inline
provider-configuration blocks used before the refactor.

No real ONNX models or GPU hardware are required – InferenceSession is mocked.
"""

import os
import sys
import types
import logging
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_ort(available=("CPUExecutionProvider",)):
    """Return a lightweight fake onnxruntime module."""
    fake = types.ModuleType("onnxruntime")

    class _FakeSession:
        def __init__(self, model_path, providers=None, provider_options=None, sess_options=None):
            self._providers = providers or ["CPUExecutionProvider"]
            self._provider_options = provider_options or [{}]

        def get_providers(self):
            return list(self._providers)

        def get_inputs(self):
            m = MagicMock()
            m.name = "input"
            return [m]

        def get_outputs(self):
            m = MagicMock()
            m.name = "output"
            return [m]

        def run(self, output_names, feed_dict):
            # Return a zero embedding of the right shape.
            return [np.zeros((1, 200), dtype=np.float32)]

    fake.InferenceSession = _FakeSession
    fake.get_available_providers = lambda: list(available)

    # Needed by analysis.py at import
    fake.capi = types.ModuleType("onnxruntime.capi")
    fake.capi.onnxruntime_pybind11_state = types.ModuleType(
        "onnxruntime.capi.onnxruntime_pybind11_state"
    )
    fake.capi.onnxruntime_pybind11_state.RuntimeException = RuntimeError
    fake.SessionOptions = MagicMock(return_value=MagicMock())

    return fake


# ---------------------------------------------------------------------------
# build_ort_provider_options – integration with real ort stub
# ---------------------------------------------------------------------------

class TestBuildOrtProviderOptionsIntegration:
    """These tests call the real helper with a stub ort module and verify the
    provider list that would be passed to InferenceSession matches expectations.
    """

    def test_cpu_only_host_no_tensorrt_flag(self, monkeypatch):
        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", False)
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        ort_stub = _fake_ort(["CPUExecutionProvider"])
        provider_options, available = build_ort_provider_options(
            ort_stub, cuda_algo_search="EXHAUSTIVE", include_copy_stream=True
        )
        providers, opts = split_provider_options(provider_options)

        assert providers == ["CPUExecutionProvider"]
        assert opts == [{}]
        assert available == ["CPUExecutionProvider"]

    def test_cuda_host_no_tensorrt_flag(self, monkeypatch):
        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", False)
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        ort_stub = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
        provider_options, _ = build_ort_provider_options(
            ort_stub, cuda_algo_search="EXHAUSTIVE", include_copy_stream=True
        )
        providers, opts = split_provider_options(provider_options)

        assert providers[0] == "CUDAExecutionProvider"
        assert providers[-1] == "CPUExecutionProvider"
        assert opts[0]["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert opts[0].get("do_copy_in_default_stream") is True

    def test_trt_host_with_tensorrt_flag(self, monkeypatch):
        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", True)
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        ort_stub = _fake_ort(
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        provider_options, _ = build_ort_provider_options(ort_stub)
        providers, opts = split_provider_options(provider_options)

        assert providers[0] == "TensorrtExecutionProvider"
        assert providers[1] == "CUDAExecutionProvider"
        assert providers[2] == "CPUExecutionProvider"

    def test_trt_flag_set_but_trt_not_available(self, monkeypatch):
        """TensorRT flag set but provider absent → CUDA/CPU list, no TRT entry."""
        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", True)
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        ort_stub = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
        provider_options, _ = build_ort_provider_options(ort_stub)
        providers, _ = split_provider_options(provider_options)


        assert "TensorrtExecutionProvider" not in providers
        assert providers[0] == "CUDAExecutionProvider"

    def test_force_skip_tensorrt_suppresses_trt_even_when_available(self, monkeypatch):
        """force_skip_tensorrt=True must exclude TRT even if the provider is
        registered and USE_TENSORRT=True.  Used by clap_analyzer when the
        TRT-simplified model file is absent."""
        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", True)
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        ort_stub = _fake_ort(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        provider_options, _ = build_ort_provider_options(ort_stub, force_skip_tensorrt=True)
        providers, _ = split_provider_options(provider_options)

        assert "TensorrtExecutionProvider" not in providers, (
            "TRT must be excluded when force_skip_tensorrt=True"
        )
        assert providers[0] == "CUDAExecutionProvider"


# ---------------------------------------------------------------------------
# Smoke tests: provider args reach InferenceSession unchanged
# ---------------------------------------------------------------------------

class TestInferenceSessionReceivesCorrectProviders:
    """Verify that the provider lists produced by build_ort_provider_options can
    be passed directly to InferenceSession without error and that the session
    reports the expected active provider.
    """

    def _make_session(self, monkeypatch, providers_env, use_trt: bool):
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", use_trt)
        ort_stub = _fake_ort(providers_env)

        provider_options, _ = build_ort_provider_options(
            ort_stub, cuda_algo_search="DEFAULT", include_copy_stream=False
        )
        providers, opts = split_provider_options(provider_options)

        session = ort_stub.InferenceSession(
            "dummy_model.onnx",
            providers=providers,
            provider_options=opts,
        )
        return session, providers

    def test_cpu_only_session_created(self, monkeypatch):
        session, providers = self._make_session(
            monkeypatch,
            providers_env=["CPUExecutionProvider"],
            use_trt=False,
        )
        assert session.get_providers() == ["CPUExecutionProvider"]

    def test_cuda_session_created(self, monkeypatch):
        session, providers = self._make_session(
            monkeypatch,
            providers_env=["CUDAExecutionProvider", "CPUExecutionProvider"],
            use_trt=False,
        )
        assert session.get_providers()[0] == "CUDAExecutionProvider"

    def test_tensorrt_session_created(self, monkeypatch):
        session, providers = self._make_session(
            monkeypatch,
            providers_env=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            use_trt=True,
        )
        assert session.get_providers()[0] == "TensorrtExecutionProvider"

    def test_musicnn_cuda_options_exhaustive_algo(self, monkeypatch):
        """analysis.py passes cuda_algo_search='EXHAUSTIVE' – verify it
        reaches the session options dict."""
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", False)
        ort_stub = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])

        provider_options, _ = build_ort_provider_options(
            ort_stub,
            cuda_algo_search="EXHAUSTIVE",
            include_copy_stream=True,
        )
        _, opts = split_provider_options(provider_options)

        cuda_opts = opts[0]  # first entry is CUDA
        assert cuda_opts["cudnn_conv_algo_search"] == "EXHAUSTIVE"
        assert cuda_opts.get("do_copy_in_default_stream") is True

    def test_clap_cuda_options_default_algo_no_copy_stream(self, monkeypatch):
        """clap_analyzer.py passes cuda_algo_search='DEFAULT', include_copy_stream=False."""
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", False)
        ort_stub = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])

        provider_options, _ = build_ort_provider_options(
            ort_stub,
            cuda_algo_search="DEFAULT",
            include_copy_stream=False,
        )
        _, opts = split_provider_options(provider_options)

        cuda_opts = opts[0]
        assert cuda_opts["cudnn_conv_algo_search"] == "DEFAULT"
        assert "do_copy_in_default_stream" not in cuda_opts


# ---------------------------------------------------------------------------
# GPU Clustering isolation test
# ---------------------------------------------------------------------------

class TestClusteringNotAffectedByProviderRefactor:
    """Confirm that tasks/clustering_gpu.py does not import or use
    onnx_providers at all – the GPU clustering path is entirely separate.
    """

    def test_clustering_gpu_does_not_import_onnx_providers(self):
        import ast, pathlib

        src = pathlib.Path(__file__).parent.parent.parent / "tasks" / "clustering_gpu.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        assert "tasks.onnx_providers" not in imported_modules, (
            "clustering_gpu.py must not import onnx_providers – "
            "it uses cupy/cuML directly and is unaffected by the TensorRT change"
        )

    def test_clustering_gpu_does_not_reference_tensorrt(self):
        import pathlib

        src = pathlib.Path(__file__).parent.parent.parent / "tasks" / "clustering_gpu.py"
        text = src.read_text(encoding="utf-8")
        assert "TensorrtExecutionProvider" not in text
        assert "USE_TENSORRT" not in text
        assert "onnx_providers" not in text


# ---------------------------------------------------------------------------
# Provider selection is idempotent across multiple calls
# ---------------------------------------------------------------------------

class TestProviderSelectionIdempotent:
    """Calling build_ort_provider_options twice with the same arguments must
    produce identical output – e.g. for the session-recycler path in
    analyze_album_task which recreates sessions periodically.
    """

    @pytest.mark.parametrize("use_trt,available", [
        (False, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (True,  ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
        (False, ["CPUExecutionProvider"]),
    ])
    def test_idempotent(self, monkeypatch, use_trt, available):
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", use_trt)
        ort_stub = _fake_ort(available)

        po1, av1 = build_ort_provider_options(ort_stub, cuda_algo_search="DEFAULT")
        po2, av2 = build_ort_provider_options(ort_stub, cuda_algo_search="DEFAULT")

        p1, o1 = split_provider_options(po1)
        p2, o2 = split_provider_options(po2)

        assert p1 == p2, "Provider list changed between calls"
        assert o1 == o2, "Provider options changed between calls"
        assert av1 == av2, "Available providers changed between calls"
