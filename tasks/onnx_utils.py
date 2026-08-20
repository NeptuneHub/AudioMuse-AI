# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared ONNX session factory and inference helpers.

The single home for building ONNX Runtime sessions and running inference, so
the analysis path and the lyrics/CLAP model owners call one implementation
instead of lyrics importing the analysis worker module. A session is built from
a provider chain resolved by ``resolve_providers`` (CUDA/CoreML when available,
a plugin accelerator scoped by ``label`` via MODEL_LABELS, CPU last), with a
CPU-only fallback whenever a GPU session fails to load or runs out of memory.

Main Features:
* resolve_providers: build the provider chain for one session; ``label`` names
  the model (see MODEL_LABELS) so a plugin can offer its accelerator for only
  the graphs it compiles, and ``cpu_only_default`` keeps core CPU sessions on
  CPU unless a plugin opts in by naming the label in only_models.
* create_onnx_session: load a model with the resolved chain, falling back to a
  plain CPU session on any load failure.
* run_inference / run_inference_with_oom_fallback: map feed keys and output
  names onto the session's actual input/output tensors, with an OOM path that
  unloads the GPU session and re-runs on CPU.
"""

import logging

import onnxruntime as ort

from cpu_budget import usable_cpu_count

from .memory_utils import cleanup_onnx_session, comprehensive_memory_cleanup

logger = logging.getLogger(__name__)


def _find_onnx_name(candidate, names):
    if not names:
        return None
    stripped = candidate.split(':')[0]
    for cand in (candidate, stripped, stripped.split('/')[-1], stripped.replace('/', '_')):
        if cand in names:
            return cand
    return names[0]


def run_inference(session, feed_dict, output_tensor_name=None):
    input_names = [i.name for i in session.get_inputs()]
    mapped = {}
    for k, v in feed_dict.items():
        name = _find_onnx_name(k, input_names)
        if name is None:
            logger.error(f"Could not map input '{k}' to ONNX inputs {input_names}")
            return None
        mapped[name] = v
    output_names = [o.name for o in session.get_outputs()]
    default_output = output_names[0] if output_names else None
    out = (
        _find_onnx_name(output_tensor_name, output_names)
        if output_tensor_name
        else default_output
    )
    if out is None:
        logger.error("No ONNX output name available to run inference.")
        return None
    result = session.run([out], mapped)
    return result[0] if isinstance(result, list) and len(result) > 0 else result


MODEL_LABELS = frozenset({
    'musicnn',
    'clap',
    'clap_text',
    'whisper_encoder',
    'whisper_decoder',
    'gte',
    'silero_vad',
})


def _scoped_labels(provider, key):
    value = provider.get(key)
    if not value:
        return None
    labels = [value] if isinstance(value, str) else list(value)
    unknown = [name for name in labels if name not in MODEL_LABELS]
    if unknown:
        logger.warning(
            "ONNX provider %s: unknown %s %s - known model labels are %s",
            provider.get('name'), key, unknown, sorted(MODEL_LABELS),
        )
    return labels


def _add_plugin_provider(chain, provider, entry):
    position = provider.get('position') or 'before_cpu'
    if position == 'before_cuda':
        chain.insert(0, entry)
        return
    if position != 'before_cpu':
        logger.warning(
            "ONNX provider %s: unknown position %r - using 'before_cpu'",
            provider.get('name'), position,
        )
    chain.append(entry)


def resolve_providers(allow_coreml=False, cuda_options=None, label=None,
                      cpu_only_default=False):
    available = ort.get_available_providers()
    chain = []

    if not cpu_only_default and 'CUDAExecutionProvider' in available:
        chain.append(
            (
                'CUDAExecutionProvider',
                cuda_options
                or {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                },
            )
        )

    if not cpu_only_default and allow_coreml and 'CoreMLExecutionProvider' in available:
        chain.append(
            (
                'CoreMLExecutionProvider',
                {
                    'MLComputeUnits': 'ALL',
                    'ModelFormat': 'MLProgram',
                },
            )
        )

    for provider in _plugin_onnx_providers():
        name = provider.get('name')
        if not name or name not in available or name in [p[0] for p in chain]:
            continue
        only = _scoped_labels(provider, 'only_models')
        exclude = _scoped_labels(provider, 'exclude_models')
        if only and label not in only:
            continue
        if exclude and label in exclude:
            continue
        if cpu_only_default and not only:
            continue
        _add_plugin_provider(chain, provider, (name, provider.get('options') or {}))

    chain.append(('CPUExecutionProvider', {}))
    logger.info("ONNX provider chain for %s: %s", label or 'unlabelled', [p[0] for p in chain])
    return chain


def _plugin_onnx_providers():
    try:
        from plugin.manager import plugin_manager
        return plugin_manager.get_onnx_providers()
    except Exception:
        return []


def _default_sess_options():
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    threads = usable_cpu_count()
    if threads is not None:
        opts.intra_op_num_threads = threads
    return opts


def create_onnx_session(
    model_path, provider_options=None, label="", sess_options=None, allow_coreml=False
):
    opts = provider_options or resolve_providers(allow_coreml=allow_coreml, label=label)
    if sess_options is None:
        sess_options = _default_sess_options()
    try:
        return ort.InferenceSession(
            model_path,
            providers=[p[0] for p in opts],
            provider_options=[p[1] for p in opts],
            sess_options=sess_options,
        )
    except Exception:
        logger.warning(f"Failed to load {label or model_path} with GPU - falling back to CPU")
        return ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=sess_options,
        )


def run_inference_with_oom_fallback(
    session, feed_dict, output_tensor_name, model_path, label, file_basename
):
    try:
        return run_inference(session, feed_dict, output_tensor_name), session
    except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
        if "Failed to allocate memory" not in str(e):
            raise
        logger.warning(
            f"GPU OOM for {file_basename} during {label} inference - falling back to CPU"
        )
        try:
            try:
                cleanup_onnx_session(session, label)
            except Exception:
                logger.exception("Error cleaning up OOM'd %s session before CPU fallback", label)
            try:
                comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
            except Exception:
                logger.exception("Error during memory cleanup before %s CPU fallback", label)

            cpu_session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider'],
                sess_options=_default_sess_options(),
            )
            result = run_inference(cpu_session, feed_dict, output_tensor_name)
            if result is None:
                raise RuntimeError(
                    f"CPU fallback inference returned None for {label} ({file_basename})"
                )
            logger.info(f"Successfully completed {label} inference on CPU after OOM")
            return result, cpu_session
        finally:
            del session
