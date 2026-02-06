from functools import cache
import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)


@cache
def get_available_providers() -> list[str]:
    """
    Filters out ONNXRuntime providers to ones supported by Audiomuse-AI
    """
    available_providers = ort.get_available_providers()
    providers = ['CPUExecutionProvider']
    if 'OpenVINOExecutionProvider' in available_providers:
        providers.insert(0, 'OpenVINOExecutionProvider')
    if 'CUDAExecutionProvider' in available_providers:
        providers.insert(0, 'CUDAExecutionProvider')
    logger.info("Providers made available: %s",
                [provider.split('ExecutionProvider')[0] for provider in available_providers])
    return providers