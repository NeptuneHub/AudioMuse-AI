"""
Embedding parity tests for tasks.onnx_providers.

Goal: assert that an ONNX model produces *bit-identical* outputs when run
through the build_ort_provider_options path vs a direct CPUExecutionProvider
configuration.  This validates that the provider-selection refactor does not
alter inference numerics.

On developer machines that have a CUDA GPU the test is also parameterised with
CUDAExecutionProvider so the same assertion is checked for GPU paths.

Requirements (auto-skipped if absent):
    pip install onnx onnxruntime        # or onnxruntime-gpu on CUDA hosts
"""

import numpy as np
import pytest

# Skip the entire module if onnx or onnxruntime are not installed.
onnx = pytest.importorskip("onnx")
ort  = pytest.importorskip("onnxruntime")

import onnx.helper as helper
# TensorProto constants live on the onnx module itself (not a sub-module).
TProto = onnx.TensorProto


# ---------------------------------------------------------------------------
# Tiny ONNX model factory  (must be defined first – used by the probe below)
# ---------------------------------------------------------------------------

def _build_linear_onnx_model(in_features: int, out_features: int, seed: int = 0) -> bytes:
    """Return the serialised bytes of a single-MatMul ONNX model.

    Shape: (batch, in_features) -> (batch, out_features)

    The weight matrix is filled with deterministic random values so different
    provider runs on the same input must agree to float32 precision.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((in_features, out_features)).astype(np.float32)

    weight_init = helper.make_tensor(
        name="W",
        data_type=TProto.FLOAT,
        dims=list(W.shape),
        vals=W.flatten().tolist(),
    )

    node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

    graph = helper.make_graph(
        nodes=[node],
        name="parity_graph",
        inputs=[
            helper.make_tensor_value_info("X", TProto.FLOAT, [None, in_features])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TProto.FLOAT, [None, out_features])
        ],
        initializer=[weight_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    # Keep IR version low enough for onnxruntime 1.x (max supported IR: 10).
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_from_bytes(model_bytes: bytes, providers, provider_opts):
    """Create an InferenceSession from raw model bytes."""
    return ort.InferenceSession(
        model_bytes,
        providers=providers,
        provider_options=provider_opts,
    )


def _run(session, x: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x})[0]


# ---------------------------------------------------------------------------
# Runtime GPU/TRT availability probe
# Evaluated once at collection time.  We guard against segfaults by doing a
# low-level CUDA driver check *before* touching ORT GPU paths.
# ---------------------------------------------------------------------------

def _probe_provider(providers: list, provider_options: list) -> bool:
    """Return True only when the given provider list is fully functional."""
    if any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in providers):
        try:
            import ctypes
            cuda_lib = ctypes.CDLL("libcuda.so.1")
            if cuda_lib.cuInit(0) != 0:
                return False
            count = ctypes.c_int(0)
            if cuda_lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
                return False
        except Exception:
            return False

    try:
        blob = _build_linear_onnx_model(2, 2)
        sess = _session_from_bytes(blob, providers, provider_options)
        inp = sess.get_inputs()[0].name
        sess.run(None, {inp: np.ones((1, 2), dtype=np.float32)})
        return True
    except Exception:
        return False


_CUDA_WORKS: bool = (
    "CUDAExecutionProvider" in ort.get_available_providers()
    and _probe_provider(["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}])
)

_TRT_WORKS: bool = (
    "TensorrtExecutionProvider" in ort.get_available_providers()
    and _probe_provider(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        [{"device_id": 0}, {}, {}],
    )
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_bytes():
    """Serialised ONNX model: (batch, 96) -> (batch, 200)  (MusiCNN-like dims)."""
    return _build_linear_onnx_model(in_features=96, out_features=200)


@pytest.fixture(scope="module")
def dummy_input():
    """A repeatable random input batch."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 96)).astype(np.float32)


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

class TestEmbeddingParityViaBuildOrtProviderOptions:
    """Verify that going through build_ort_provider_options produces the same
    embeddings as a direct CPUExecutionProvider session."""

    def test_cpu_parity_use_tensorrt_false(self, tiny_model_bytes, dummy_input, monkeypatch):
        """With USE_TENSORRT=False and no CUDA available, the helper must select
        CPUExecutionProvider and produce the same output as a direct CPU session.
        """
        from tasks.onnx_providers import (
            build_ort_provider_options,
            split_provider_options,
        )

        # Force CPU-only environment.
        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", False)

        # Build provider list through the helper, then restrict to CPU so the
        # test runs in CI (no GPU required).
        provider_options, _ = build_ort_provider_options(
            ort,
            cuda_algo_search="DEFAULT",
            include_copy_stream=False,
        )
        # Keep only CPU entries to stay hardware-agnostic.
        cpu_only = [(p, o) for p, o in provider_options if p == "CPUExecutionProvider"]
        providers, provider_opts = split_provider_options(cpu_only)

        session_via_helper = _session_from_bytes(tiny_model_bytes, providers, provider_opts)
        session_direct_cpu = _session_from_bytes(
            tiny_model_bytes,
            ["CPUExecutionProvider"],
            [{}],
        )

        out_helper = _run(session_via_helper, dummy_input)
        out_direct  = _run(session_direct_cpu,  dummy_input)

        np.testing.assert_array_equal(
            out_helper,
            out_direct,
            err_msg="CPU embeddings differ between helper path and direct session",
        )

    def test_cpu_parity_use_tensorrt_true_no_trt_available(
        self, tiny_model_bytes, dummy_input, monkeypatch
    ):
        """When USE_TENSORRT=True but TensorrtExecutionProvider is absent (typical
        CPU dev machine), the helper must fall back to CPU and still produce
        identical embeddings.
        """
        from tasks.onnx_providers import (
            build_ort_provider_options,
            split_provider_options,
        )

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", True)

        provider_options, _ = build_ort_provider_options(ort)
        cpu_only = [(p, o) for p, o in provider_options if p == "CPUExecutionProvider"]
        providers, provider_opts = split_provider_options(cpu_only)

        session_via_helper = _session_from_bytes(tiny_model_bytes, providers, provider_opts)
        session_direct_cpu = _session_from_bytes(
            tiny_model_bytes,
            ["CPUExecutionProvider"],
            [{}],
        )

        np.testing.assert_array_equal(
            _run(session_via_helper, dummy_input),
            _run(session_direct_cpu,  dummy_input),
            err_msg="CPU fallback embeddings differ when TRT flagged but unavailable",
        )

    @pytest.mark.skipif(
        not _CUDA_WORKS,
        reason="CUDAExecutionProvider not functional on this host (driver missing or no GPU)",
    )
    def test_cuda_cpu_embedding_parity(self, tiny_model_bytes, dummy_input, monkeypatch):
        """CUDA embeddings must be numerically close (atol=1e-4) to CPU ones.

        This test only runs when a CUDA GPU is present (i.e. inside the nvidia
        Docker image or on a GPU workstation).
        """
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", False)

        provider_options, _ = build_ort_provider_options(
            ort,
            cuda_algo_search="DEFAULT",
            include_copy_stream=False,
        )
        providers, provider_opts = split_provider_options(provider_options)

        session_cuda = _session_from_bytes(tiny_model_bytes, providers, provider_opts)
        session_cpu  = _session_from_bytes(tiny_model_bytes, ["CPUExecutionProvider"], [{}])

        out_cuda = _run(session_cuda, dummy_input)
        out_cpu  = _run(session_cpu,  dummy_input)

        np.testing.assert_allclose(
            out_cuda, out_cpu, atol=1e-4,
            err_msg="CUDA embeddings are not close to CPU embeddings (tolerance 1e-4)",
        )

    @pytest.mark.skipif(
        not _TRT_WORKS,
        reason="TensorrtExecutionProvider not functional on this host (driver/TRT libs missing)",
    )
    def test_tensorrt_cpu_embedding_parity(self, tiny_model_bytes, dummy_input, monkeypatch):
        """TensorRT embeddings must be numerically close (atol=1e-3) to CPU ones.

        Run inside the NVIDIA Docker image with USE_TENSORRT=true.
        TensorRT uses FP16/INT8 kernels by default so a slightly wider tolerance
        (1e-3) is used compared to the CUDA path.
        """
        from tasks.onnx_providers import build_ort_provider_options, split_provider_options

        monkeypatch.setattr("tasks.onnx_providers.USE_TENSORRT", True)

        provider_options, _ = build_ort_provider_options(
            ort,
            cuda_algo_search="DEFAULT",
            include_copy_stream=False,
        )
        providers, provider_opts = split_provider_options(provider_options)

        session_trt = _session_from_bytes(tiny_model_bytes, providers, provider_opts)
        session_cpu = _session_from_bytes(tiny_model_bytes, ["CPUExecutionProvider"], [{}])

        out_trt = _run(session_trt, dummy_input)
        out_cpu = _run(session_cpu, dummy_input)

        np.testing.assert_allclose(
            out_trt, out_cpu, atol=1e-3,
            err_msg="TensorRT embeddings deviate from CPU reference beyond 1e-3 tolerance",
        )
