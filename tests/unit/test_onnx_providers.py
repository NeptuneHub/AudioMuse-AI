from tasks.onnx_providers import build_ort_provider_options, split_provider_options


class _FakeOrt:
    def __init__(self, providers):
        self._providers = providers

    def get_available_providers(self):
        return self._providers


def test_cuda_cpu_when_tensorrt_disabled(monkeypatch):
    monkeypatch.setattr('tasks.onnx_providers.USE_TENSORRT', False)
    fake_ort = _FakeOrt(['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    provider_options, _ = build_ort_provider_options(fake_ort, cuda_algo_search='DEFAULT')
    providers, provider_opts = split_provider_options(provider_options)

    assert providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']
    assert provider_opts[0]['cudnn_conv_algo_search'] == 'DEFAULT'


def test_tensorrt_cuda_cpu_when_tensorrt_enabled(monkeypatch):
    monkeypatch.setattr('tasks.onnx_providers.USE_TENSORRT', True)
    fake_ort = _FakeOrt(['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    provider_options, _ = build_ort_provider_options(fake_ort, cuda_algo_search='EXHAUSTIVE')
    providers, provider_opts = split_provider_options(provider_options)

    assert providers == ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    assert provider_opts[0]['device_id'] == 0
    assert provider_opts[1]['cudnn_conv_algo_search'] == 'EXHAUSTIVE'


def test_cpu_only_fallback(monkeypatch):
    monkeypatch.setattr('tasks.onnx_providers.USE_TENSORRT', True)
    fake_ort = _FakeOrt(['CPUExecutionProvider'])

    provider_options, _ = build_ort_provider_options(fake_ort)
    providers, _ = split_provider_options(provider_options)

    assert providers == ['CPUExecutionProvider']
