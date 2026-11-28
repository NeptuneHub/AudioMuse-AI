# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Export the Beat This! small beat and downbeat tracker to ONNX.

Offline build tool (torch + the CPJKU/beat_this source) that turns the
upstream ``small0`` checkpoint (ISMIR 2024, code and weights under the MIT
licence) into the ``model/beat_this_small.onnx`` published in the model release, which
``tasks/smart_fade_beats.py`` runs through onnxruntime inside the smart fade
transition API, so the runtime needs no torch. Tested with Python 3.12,
torch 2.15 (CPU) and onnx 1.22; the graph loads in onnxruntime 1.19 and 1.28.

Main Features:
* Loads the checkpoint with beat_this's own ``load_model`` (``--source`` is a
  clone of github.com/CPJKU/beat_this, ``--checkpoint`` a file or an upstream
  name such as small0) and prints its sha256.
* Exports one input ``spect`` of shape (1, 1500, 128), the model's 30-second
  log-mel window at 50 frames per second, and the two logit outputs ``beat``
  and ``downbeat`` of shape (1, 1500), at opset 17 with a fixed length (the
  runtime pads a shorter window with silence).
* Verifies onnxruntime against torch on random and silent windows (max abs
  diff under --tolerance) and writes the upstream MIT notice next to the
  graph as ``beat_this_small.LICENSE``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FRAMES = 1500
MELS = 128
OPSET = 17


def default_output() -> str:
    return os.path.join(_REPO_ROOT, 'model', 'beat_this_small.onnx')


def load_model(source_dir: str, checkpoint: str):
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    from beat_this.inference import load_checkpoint, load_model as upstream_load

    state = load_checkpoint(checkpoint)
    model = upstream_load(checkpoint)
    return model, state


def checkpoint_digest(checkpoint: str) -> str:
    import torch

    candidates = [checkpoint, os.path.join(torch.hub.get_dir(), 'checkpoints', f'beat_this-{checkpoint}.ckpt')]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, 'rb') as handle:
                return hashlib.sha256(handle.read()).hexdigest()
    return 'unknown'


def wrapped(model):
    import torch

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, spect):
            out = self.inner(spect)
            return out['beat'], out['downbeat']

    return Logits(model).eval()


def export_onnx(model, output_path: str) -> None:
    import torch

    dummy = torch.zeros(1, FRAMES, MELS, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapped(model), (dummy,), output_path, input_names=['spect'],
            output_names=['beat', 'downbeat'], opset_version=OPSET, dynamo=False,
        )


def verify_onnx(model, output_path: str, tolerance: float) -> float:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(7)
    windows = [np.zeros((1, FRAMES, MELS), np.float32),
               rng.uniform(0.0, 6.0, (1, FRAMES, MELS)).astype(np.float32),
               np.abs(rng.normal(2.0, 1.5, (1, FRAMES, MELS))).astype(np.float32)]
    worst = 0.0
    net = wrapped(model)
    for window in windows:
        with torch.inference_mode():
            expected = [part.numpy() for part in net(torch.from_numpy(window))]
        got = session.run(None, {'spect': window})
        worst = max(worst, *(float(np.max(np.abs(a - b))) for a, b in zip(expected, got)))
    if worst > tolerance:
        raise SystemExit(f'ONNX differs from torch by {worst:.2e} (tolerance {tolerance:.1e})')
    return worst


def write_licence(source_dir: str, output_path: str) -> str:
    target = os.path.splitext(output_path)[0] + '.LICENSE'
    shutil.copyfile(os.path.join(source_dir, 'LICENSE'), target)
    return target


def export_beat_this_to_onnx(source_dir: str, checkpoint: str, output_path: str, tolerance: float) -> None:
    model, state = load_model(source_dir, checkpoint)
    parameters = sum(p.numel() for p in model.parameters())
    export_onnx(model, output_path)
    worst = verify_onnx(model, output_path, tolerance)
    licence = write_licence(source_dir, output_path)
    print(f'checkpoint {checkpoint} sha256 {checkpoint_digest(checkpoint)}')
    print(f'hyper parameters {state.get("hyper_parameters", {})}')
    print(f'parameters {parameters:,}; {output_path} {os.path.getsize(output_path) / 1e6:.1f} MB; '
          f'max abs diff {worst:.2e}; licence {licence}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Clone of github.com/CPJKU/beat_this')
    parser.add_argument('--checkpoint', default='small0', help='Checkpoint file or upstream name')
    parser.add_argument('--output', default=default_output(), help='Where to write the ONNX graph')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Max abs logit difference allowed')
    args = parser.parse_args(argv)
    export_beat_this_to_onnx(os.path.abspath(args.source), args.checkpoint, os.path.abspath(args.output), args.tolerance)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
