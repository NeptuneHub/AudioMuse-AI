"""Export openai/whisper-small to ONNX for the optimum runtime.

Run once at Docker build time. The resulting bundle is loaded at runtime by
``lyrics/whisper_onnx.py`` via ``optimum.onnxruntime.ORTModelForSpeechSeq2Seq``
— no torch required at runtime, but the export step itself does need torch
(which is why this lives in a separate one-shot script).

Usage
-----
    python3 export_whisper_to_onnx.py \\
        --model openai/whisper-small \\
        --output /app/model/whisper-small-onnx

The output directory will contain (everything ``ORTModelForSpeechSeq2Seq``
expects):

    encoder_model.onnx
    decoder_model_merged.onnx       ← single decoder file that handles both
                                       first-step and KV-cache-step paths
    config.json
    generation_config.json
    preprocessor_config.json         (mel filterbank parameters)
    tokenizer files (tokenizer.json + vocab.json + merges.txt + ...)

We deliberately drop the split ``decoder_model.onnx`` /
``decoder_with_past_model.onnx`` produced alongside the merge. Optimum's
runtime defaults ``use_merged=True`` when the merged file is present and
loads it instead — same behavior, half the disk footprint per model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


# See note in export_marian_to_onnx.py — we keep only the merged decoder.
_DROP_AFTER_EXPORT = (
    'decoder_model.onnx',
    'decoder_model.onnx_data',
    'decoder_with_past_model.onnx',
    'decoder_with_past_model.onnx_data',
)


def export_whisper_to_onnx(model_id: str, output_dir: str) -> None:
    """Run ``optimum-cli export onnx`` then prune to keep only the merged decoder."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'optimum.exporters.onnx',
        '--model', model_id,
        '--task', 'automatic-speech-recognition-with-past',
        output_dir,
    ]
    print(f'$ {" ".join(cmd)}', flush=True)
    completed = subprocess.run(cmd, capture_output=False)
    if completed.returncode != 0:
        raise SystemExit(f'optimum-cli export failed (rc={completed.returncode})')

    merged_path = os.path.join(output_dir, 'decoder_model_merged.onnx')
    if not os.path.isfile(merged_path):
        raise SystemExit(
            f'ERROR: {merged_path} was not produced. The merging post-process '
            f'requires the ``accelerate`` package; install it in your venv '
            f'(``pip install "accelerate>=0.26,<1.0"``) and re-run.')

    for entry in _DROP_AFTER_EXPORT:
        full = os.path.join(output_dir, entry)
        if os.path.isfile(full):
            os.remove(full)

    total = 0
    for entry in os.listdir(output_dir):
        full = os.path.join(output_dir, entry)
        if os.path.isfile(full):
            total += os.path.getsize(full)
    print(f'Whisper ONNX exported to {output_dir} ({total / (1024*1024):.1f} MB)',
          flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='openai/whisper-small',
                        help='HF Hub id of the whisper model to export.')
    parser.add_argument('--output', required=True,
                        help='Destination directory for the ONNX files.')
    args = parser.parse_args(argv)
    export_whisper_to_onnx(args.model, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
