# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Export openai/whisper-small to ONNX for lyrics transcription.

Offline build tool (torch + optimum) that exports ``openai/whisper-small`` to
an ``encoder_model.onnx`` + ``decoder_model.onnx`` pair plus its config and
tokenizer files, which ``lyrics/whisper_onnx.py`` then runs at inference time
via raw onnxruntime and a numpy greedy decoder, so no torch is needed at
runtime. Companion to ``export_gte_to_onnx.py``.

Main Features:
* Drives ``optimum.exporters.onnx`` to produce the encoder/decoder ONNX pair.
* Emits the config, generation/preprocessor config and tokenizer files the
  runtime decoder and mel-filterbank preprocessing require.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def export_whisper_to_onnx(model_id: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        '-m',
        'optimum.exporters.onnx',
        '--model',
        model_id,
        '--task',
        'automatic-speech-recognition',
        '--no-post-process',
        output_dir,
    ]
    print(f'$ {" ".join(cmd)}', flush=True)
    completed = subprocess.run(cmd, capture_output=False)
    if completed.returncode != 0:
        raise SystemExit(f'optimum-cli export failed (rc={completed.returncode})')

    drop = (
        'decoder_model_merged.onnx',
        'decoder_model_merged.onnx_data',
        'decoder_with_past_model.onnx',
        'decoder_with_past_model.onnx_data',
    )
    for entry in drop:
        full = os.path.join(output_dir, entry)
        if os.path.isfile(full):
            os.remove(full)

    total = 0
    for entry in os.listdir(output_dir):
        full = os.path.join(output_dir, entry)
        if os.path.isfile(full):
            total += os.path.getsize(full)
    print(f'Whisper ONNX exported to {output_dir} ({total / (1024 * 1024):.1f} MB)', flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model', default='openai/whisper-small', help='HF Hub id of the whisper model to export.'
    )
    parser.add_argument('--output', required=True, help='Destination directory for the ONNX files.')
    args = parser.parse_args(argv)
    export_whisper_to_onnx(args.model, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
