#!/usr/bin/env python3
"""
AudioMuse-AI – Prepare CLAP model for TensorRT inference
=========================================================

The current DCLAP ONNX graph (`model_epoch_36.onnx`) contains a tiny dynamic
Sequence/Loop block that computes a fixed index tensor [0..167]. TensorRT 10.x
cannot parse sequence-typed tensors, so session creation fails before fallback.

This script applies deterministic graph surgery:
1) remove SequenceEmpty + Loop + CastLike + Add + Gather nodes
2) replace them with a single Reshape equivalent

The replacement is numerically equivalent for this model and keeps output parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import logging

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("prepare_clap_trt")


def _output_path(source: str) -> str:
    stem, _ = os.path.splitext(source)
    return stem + "_trt.onnx"


def _graph_surgery(source: str, output: str) -> None:
    """Patch the CLAP graph to remove sequence ops that block TensorRT.

    Original subgraph:
      n4:   SequenceEmpty -> one_seq
      n6_2: Loop(...) -> one_seq_16, indices_17
      n10_2 CastLike(indices, indices_17) -> storage_offset_cast
      n11_2 Add(indices_17, storage_offset_cast) -> indices_19
      n12_2 Gather(self_flatten, indices_19) -> as_strided

    For this model, `indices_19` is always [0..167] and `as_strided` is just
    `self_flatten` reshaped to [1,168,1,1]. We replace with:
      trt_fix_as_strided: Reshape(self_flatten, val_1005) -> as_strided
    """
    model = onnx.load(source, load_external_data=True)

    required_names = {"n4", "n6_2", "n8_2", "n10_2", "n11_2", "n12_2"}
    node_names = {n.name for n in model.graph.node}
    missing = required_names - node_names
    if missing:
        raise RuntimeError(
            f"Expected CLAP pattern nodes not found: {sorted(missing)}. "
            "Model architecture likely changed."
        )

    skip = {"n4", "n6_2", "n10_2", "n11_2", "n12_2"}
    new_nodes = []
    inserted = False

    for node in model.graph.node:
        if node.name in skip:
            continue
        new_nodes.append(node)

        if node.name == "n8_2":
            new_nodes.append(
                helper.make_node(
                    "Reshape",
                    inputs=["self_flatten", "val_1005"],
                    outputs=["as_strided"],
                    name="trt_fix_as_strided",
                )
            )
            inserted = True

    if not inserted:
        raise RuntimeError("Failed to insert replacement Reshape after n8_2")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Keep IR version compatible with ORT 1.19.x
    if model.ir_version > 10:
        model.ir_version = 10

    onnx.checker.check_model(model)
    onnx.save(model, output)

    # Sanity check: no sequence ops remain
    patched = onnx.load(output, load_external_data=True)
    op_types = {n.op_type for n in patched.graph.node}
    blockers = {"Loop", "SequenceEmpty", "SequenceInsert", "ConcatFromSequence", "SplitToSequence", "SequenceConstruct"}
    still_there = op_types & blockers
    if still_there:
        raise RuntimeError(f"Blocked ops still present after patch: {sorted(still_there)}")

    log.info(f"Saved TRT-compatible model → {output}")


def _validate_parity(source: str, patched: str) -> None:
    """Check CPU output parity between original and patched models."""
    base = ort.InferenceSession(source, providers=["CPUExecutionProvider"], provider_options=[{}])
    new = ort.InferenceSession(patched, providers=["CPUExecutionProvider"], provider_options=[{}])

    inp = base.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.zeros(shape, dtype=np.float32)

    y0 = base.run(None, {inp.name: x})[0]
    y1 = new.run(None, {inp.name: x})[0]

    max_diff = float(np.max(np.abs(y0 - y1)))
    if not np.allclose(y0, y1, atol=1e-6, rtol=0):
        raise RuntimeError(f"Patched model parity check failed: max_abs_diff={max_diff}")
    log.info(f"CPU parity check passed (max_abs_diff={max_diff:.3e})")


def _validate_trt(patched: str) -> None:
    """Create TRT session and run one smoke inference."""
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        log.info("TensorrtExecutionProvider not available here – skipping TRT validation.")
        return

    try:
        import ctypes
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            log.info("CUDA init failed – skipping TRT validation.")
            return
    except Exception:
        log.info("CUDA driver not available – skipping TRT validation.")
        return

    sess = ort.InferenceSession(
        patched,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": 0}, {"device_id": 0}, {}],
    )

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.zeros(shape, dtype=np.float32)
    y = sess.run(None, {inp.name: x})[0]
    log.info(f"TRT session OK. providers={sess.get_providers()} output_shape={y.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRT-compatible CLAP ONNX model.")
    parser.add_argument("--source", default=None, help="Source ONNX model (default: config.CLAP_AUDIO_MODEL_PATH)")
    parser.add_argument("--output", default=None, help="Output model path (default: <source_stem>_trt.onnx)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    source = args.source
    if source is None:
        try:
            import config
            source = config.CLAP_AUDIO_MODEL_PATH
        except Exception:
            source = "/app/model/model_epoch_36.onnx"

    if not os.path.exists(source):
        log.error(f"Source model not found: {source}")
        sys.exit(1)

    output = args.output or _output_path(source)
    if os.path.exists(output) and not args.force:
        log.info(f"TRT model already exists at {output}  (use --force to rebuild)")
        _validate_parity(source, output)
        _validate_trt(output)
        return

    _graph_surgery(source, output)
    _validate_parity(source, output)
    _validate_trt(output)

    log.info("Done. Set CLAP_AUDIO_MODEL_PATH_TRT=%s (or keep auto-detect).", output)


if __name__ == "__main__":
    main()
