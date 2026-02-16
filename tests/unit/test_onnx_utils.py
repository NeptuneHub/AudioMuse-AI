import pytest
from unittest.mock import patch

from tasks import onnx_utils


def test_preferred_providers_cuda_then_cpu(monkeypatch):
    monkeypatch.setattr(onnx_utils, 'ort', None)
    class DummyOrt:
        @staticmethod
        def get_available_providers():
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    monkeypatch.setattr(onnx_utils, 'ort', DummyOrt)

    opts = onnx_utils.get_preferred_onnx_provider_options()
    assert opts[0][0] == 'CUDAExecutionProvider'
    assert any(p[0] == 'CPUExecutionProvider' for p in opts)


def test_preferred_providers_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(onnx_utils, 'ort', None)
    class DummyOrt:
        @staticmethod
        def get_available_providers():
            return ['MPSExecutionProvider', 'CPUExecutionProvider']
    monkeypatch.setattr(onnx_utils, 'ort', DummyOrt)

    opts = onnx_utils.get_preferred_onnx_provider_options()
    assert opts[0][0] == 'MPSExecutionProvider'
    assert opts[-1][0] == 'CPUExecutionProvider'


def test_preferred_providers_cpu_only(monkeypatch):
    monkeypatch.setattr(onnx_utils, 'ort', None)
    class DummyOrt:
        @staticmethod
        def get_available_providers():
            return ['CPUExecutionProvider']
    monkeypatch.setattr(onnx_utils, 'ort', DummyOrt)

    opts = onnx_utils.get_preferred_onnx_provider_options()
    assert len(opts) == 1
    assert opts[0][0] == 'CPUExecutionProvider'
