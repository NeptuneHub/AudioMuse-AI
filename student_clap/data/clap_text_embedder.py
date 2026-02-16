import os
import logging
import numpy as np
import onnxruntime as ort
from typing import List

logger = logging.getLogger(__name__)

class CLAPTextEmbedder:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise RuntimeError(f"CLAP text model not found: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        from tasks.onnx_utils import get_preferred_onnx_provider_options
        providers = [p[0] for p in get_preferred_onnx_provider_options()]
        top = providers[0] if providers else 'CPUExecutionProvider'
        logger.info(f"CLAP text model loaded: {model_path}")
        if top == 'CUDAExecutionProvider':
            logger.info("✅ Using CUDA for ONNX teacher text model")
        elif top in ('MPSExecutionProvider', 'CoreMLExecutionProvider'):
            logger.info("✅ Using Apple GPU (MPS/CoreML) for ONNX teacher text model")
        else:
            logger.info("✅ Using CPU for ONNX teacher text model")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        onnx_inputs = {
            'input_ids': input_ids.astype(np.int64),
            'attention_mask': attention_mask.astype(np.int64)
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs[0]  # (batch, 512)
