import os
import logging
import numpy as np
import onnxruntime as ort
from typing import List

logger = logging.getLogger(__name__)

class CLAPTextEmbedder:
    """Text encoder wrapper that supports ONNX *and* PyTorch `.pt` teacher checkpoints.

    - If an ONNX file is supplied the original ONNXRuntime session is used.
    - If a PyTorch `.pt` checkpoint is supplied (split checkpoint from
      `split_clap_pt.py`), the text encoder is run via `laion_clap` on CPU.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise RuntimeError(f"CLAP text model not found: {model_path}")

        lower = model_path.lower()
        if lower.endswith(('.pt', '.pth')):
            try:
                import torch
                import laion_clap
                import torch.nn as nn
            except Exception as e:
                raise RuntimeError(f"PyTorch CLAP text backend requested but missing dependency: {e}")

            logger.info(f"CLAP text (PyTorch) loaded: {model_path}")
            # Load checkpoint (split checkpoint contains text_branch + text_projection)
            state = torch.load(model_path, map_location='cpu')
            clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            clap.eval()

            if isinstance(state, dict) and ('text_branch' in state or 'text_projection' in state):
                if 'text_branch' in state:
                    clap.model.text_branch.load_state_dict(state['text_branch'], strict=False)
                if 'text_projection' in state:
                    clap.model.text_projection.load_state_dict(state['text_projection'], strict=False)
            else:
                try:
                    clap.load_ckpt(model_path)
                except Exception as e:
                    raise RuntimeError(f"Unable to load PyTorch CLAP text checkpoint: {e}")

            class TextCLAPWrapper(nn.Module):
                def __init__(self, clap_model):
                    super().__init__()
                    self.text_branch = clap_model.model.text_branch
                    self.text_projection = clap_model.model.text_projection

                def forward(self, input_ids, attention_mask):
                    text_output = self.text_branch(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    text_embed = self.text_projection(text_output.pooler_output)
                    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
                    return text_embed

            self._backend = 'torch'
            self.text_wrapper = TextCLAPWrapper(clap)
            self.text_wrapper.eval()

            # Prefer CUDA -> MPS (macOS) -> CPU for PyTorch text backend
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            self._device = device
            self.text_wrapper.to(self._device)
            logger.info(f"✅ PyTorch text teacher device: {self._device}")

        else:
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
            self._backend = 'onnx'
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Encode tokenized text (supports both backends)."""
        if getattr(self, '_backend', 'onnx') == 'torch':
            import torch
            ids = torch.from_numpy(input_ids).long().to(getattr(self, '_device', torch.device('cpu')))
            att = torch.from_numpy(attention_mask).long().to(getattr(self, '_device', torch.device('cpu')))
            with torch.no_grad():
                out = self.text_wrapper(ids, att)
            return out.cpu().numpy()

        onnx_inputs = {
            'input_ids': input_ids.astype(np.int64),
            'attention_mask': attention_mask.astype(np.int64)
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs[0]  # (batch, 512)
