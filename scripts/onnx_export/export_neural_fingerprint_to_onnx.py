# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Export the neural music fingerprinter checkpoint to ONNX.

Offline build tool (tensorflow + tf2onnx) that turns the triplet checkpoint of
raraz15/neural-music-fp (Araz et al., "Enhancing Neural Audio Fingerprint
Robustness to Audio Degradation for Music Identification", ISMIR 2025, the
NAFP architecture of Chang et al.) into the ``neural_fingerprint.onnx`` that
``tasks/neural_fingerprint.py`` runs through onnxruntime at analysis and search
time, so the runtime needs no TensorFlow. Companion to ``export_gte_to_onnx.py``
and ``export_whisper_to_onnx.py``; ``run_exports.sh`` clones the source,
downloads the checkpoint from Zenodo (record 15719945) and calls this script.
Tested with Python 3.11, tensorflow 2.13.1, tf2onnx 1.17.0 and onnx 1.17.0.

Main Features:
* Reads the checkpoint's own config.yaml (embedding size, normalisation, mel
  and STFT settings), rebuilds the FingerPrinter from it and restores ckpt-100
  with a full-match assertion, so a wrong or partial checkpoint fails loudly.
* Exports one input named ``mel`` of shape (batch, mels, frames, 1) at opset
  17 with the 33 frames the runtime front end produces for one 8 kHz second
  (the layer norms of this architecture bake the frame count into their
  weights, so a different --frames fails at restore time instead of silently
  producing a graph the runtime cannot feed).
* Verifies the ONNX graph against TensorFlow on random patches (max abs diff
  under --tolerance, unit-norm outputs) and prints parameters and file size.
"""

from __future__ import annotations

import argparse
import os
import re
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_FRAMES = 33
_CONFIG_KEYS = {
    'emb_sz': ('EMB_SZ', int),
    'norm': ('BN', str),
    'n_mels': ('N_MELS', int),
    'sample_rate': ('FS', int),
    'segment_seconds': ('SEGMENT_DUR', float),
    'stft_win': ('STFT_WIN', int),
    'stft_hop': ('STFT_HOP', int),
}


def default_output() -> str:
    return os.path.join(_REPO_ROOT, 'neural_fingerprint.onnx')


def parse_checkpoint_config(text: str) -> dict:
    parsed = {}
    for name, (key, cast) in _CONFIG_KEYS.items():
        match = re.search(rf'^\s*{key}:\s*(\S+)\s*$', text, flags=re.MULTILINE)
        if match is None:
            raise SystemExit(f'Checkpoint config is missing {key}')
        parsed[name] = cast(match.group(1))
    return parsed


def read_checkpoint_config(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, 'config.yaml')
    if not os.path.isfile(path):
        raise SystemExit(f'Checkpoint config not found: {path}')
    with open(path, encoding='utf-8') as handle:
        return parse_checkpoint_config(handle.read())


def build_model(source_dir: str, cfg: dict, frames: int):
    import tensorflow as tf

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from nmfp.model.nnfp import FingerPrinter

    model = FingerPrinter(
        emb_sz=cfg['emb_sz'],
        fc_unit_dim=[32, 1],
        norm=cfg['norm'],
        mixed_precision=False,
    )
    model.trainable = False
    model(tf.zeros((1, cfg['n_mels'], frames, 1), dtype=tf.float32))
    return model


def restore_checkpoint(model, checkpoint_dir: str) -> str:
    import tensorflow as tf

    prefix = tf.train.latest_checkpoint(checkpoint_dir) or os.path.join(checkpoint_dir, 'ckpt-100')
    status = tf.train.Checkpoint(model=model).restore(prefix)
    status.expect_partial()
    status.assert_existing_objects_matched()
    return prefix


def count_parameters(model) -> int:
    import numpy as np

    return int(sum(np.prod(v.shape) for v in model.trainable_variables + model.non_trainable_variables))


def export_onnx(model, cfg: dict, frames: int, output_path: str) -> None:
    import tensorflow as tf
    import tf2onnx

    signature = (tf.TensorSpec((None, cfg['n_mels'], frames, 1), tf.float32, name='mel'),)

    @tf.function(input_signature=signature)
    def serve(mel):
        return model(mel)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tf2onnx.convert.from_function(serve, input_signature=signature, opset=17, output_path=output_path)


def verify_onnx(model, cfg: dict, frames: int, output_path: str, tolerance: float) -> float:
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    sample = np.random.default_rng(0).uniform(-1, 1, size=(3, cfg['n_mels'], frames, 1)).astype(np.float32)
    reference = model(tf.constant(sample)).numpy()
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name
    if name != 'mel':
        raise SystemExit(f'Unexpected ONNX input name {name!r}, the runtime feeds "mel"')
    produced = session.run(None, {name: sample})[0]
    if produced.shape != (3, cfg['emb_sz']):
        raise SystemExit(f'Unexpected ONNX output shape {produced.shape}')
    norms = np.linalg.norm(produced, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit(f'ONNX outputs are not unit norm: {norms}')
    diff = float(np.abs(produced - reference).max())
    if diff > tolerance:
        raise SystemExit(f'ONNX output differs from TensorFlow by {diff:.2e} (tolerance {tolerance:.0e})')
    return diff


def export_neural_fingerprint_to_onnx(
    source_dir: str, checkpoint_dir: str, output_path: str, frames: int, tolerance: float
) -> None:
    if not os.path.isdir(os.path.join(source_dir, 'nmfp', 'model')):
        raise SystemExit(f'neural-music-fp source not found under {source_dir}')
    cfg = read_checkpoint_config(checkpoint_dir)
    print(f'Checkpoint config: {cfg}, exporting {frames} frames per segment', flush=True)

    model = build_model(source_dir, cfg, frames)
    prefix = restore_checkpoint(model, checkpoint_dir)
    print(f'Restored {prefix}: {count_parameters(model):,} parameters', flush=True)

    print(f'Exporting to {output_path} (opset 17)...', flush=True)
    export_onnx(model, cfg, frames, output_path)
    diff = verify_onnx(model, cfg, frames, output_path, tolerance)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Wrote {output_path} ({size_mb:.1f} MB), max abs diff vs TensorFlow {diff:.2e}', flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source', required=True, help='Path to a clone of https://github.com/raraz15/neural-music-fp.'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Directory holding the unzipped nmfp-triplet checkpoint (ckpt-100.* and config.yaml).',
    )
    parser.add_argument(
        '--output', default=default_output(), help='Destination .onnx path (default: the repository root).'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=DEFAULT_FRAMES,
        help='Mel frames per one-second segment, must equal tasks.neural_fingerprint.SEGMENT_FRAMES.',
    )
    parser.add_argument(
        '--tolerance', type=float, default=1e-4, help='Max abs difference allowed between ONNX and TensorFlow.'
    )
    args = parser.parse_args(argv)
    export_neural_fingerprint_to_onnx(args.source, args.checkpoint, args.output, args.frames, args.tolerance)
    return 0


if __name__ == '__main__':
    sys.exit(main())
