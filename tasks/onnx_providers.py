import os

from config import USE_TENSORRT


def build_ort_provider_options(
    ort_module,
    cuda_algo_search='EXHAUSTIVE',
    include_copy_stream=True,
):
    """Build ordered ONNX Runtime providers with options.

    Provider preference order:
    1. TensorRT (optional, via USE_TENSORRT=true)
    2. CUDA
    3. CPU
    """
    available_providers = ort_module.get_available_providers() or []

    gpu_device_id = 0
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible and cuda_visible != '-1':
        gpu_device_id = 0

    provider_options = []

    if USE_TENSORRT and 'TensorrtExecutionProvider' in available_providers:
        provider_options.append(('TensorrtExecutionProvider', {'device_id': gpu_device_id}))

    if 'CUDAExecutionProvider' in available_providers:
        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',
        }
        if cuda_algo_search:
            cuda_options['cudnn_conv_algo_search'] = cuda_algo_search
        if include_copy_stream:
            cuda_options['do_copy_in_default_stream'] = True
        provider_options.append(('CUDAExecutionProvider', cuda_options))

    provider_options.append(('CPUExecutionProvider', {}))
    return provider_options, available_providers


def split_provider_options(provider_options):
    """Split provider tuples into `providers` and `provider_options` lists."""
    providers = [provider_name for provider_name, _ in provider_options]
    provider_opts = [options for _, options in provider_options]
    return providers, provider_opts


def log_provider_selection(logger, context, provider_options, available_providers):
    """Log available and preferred ONNX Runtime execution providers."""
    preferred = [provider_name for provider_name, _ in provider_options]
    logger.info(f"{context}: available providers: {available_providers}")
    logger.info(f"{context}: preferred providers: {preferred}")
