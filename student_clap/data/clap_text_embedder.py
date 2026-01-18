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
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info(f"CLAP text model loaded: {model_path}")
            logger.info(f"✅ Using CUDA for ONNX teacher text model")
        else:
            providers = ['CPUExecutionProvider']
            logger.info(f"CLAP text model loaded: {model_path}")
            logger.info(f"✅ Using CPU for ONNX teacher text model")
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
