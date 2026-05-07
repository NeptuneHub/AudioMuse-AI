"""Export Helsinki-NLP/opus-mt-mul-en to ONNX for the optimum runtime.

One multilingual Marian model that translates from ~70 source languages to
English. Run once at Docker build time; ``lyrics/translation_onnx.py`` then
loads the resulting bundle via ``optimum.onnxruntime.ORTModelForSeq2SeqLM``
at runtime — no torch required at runtime, but the export step itself does
need torch (which is why this lives in a separate one-shot script).

Usage
-----
    python3 export_marian_to_onnx.py \\
        --model Helsinki-NLP/opus-mt-mul-en \\
        --output /app/model/opus-mt-mul-en-onnx

The output directory will contain (everything ``ORTModelForSeq2SeqLM``
expects):

    encoder_model.onnx
    decoder_model_merged.onnx       ← single decoder file that handles both
                                       first-step and KV-cache-step paths
    config.json
    generation_config.json
    tokenizer files (sentencepiece + vocab)

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


# Files we drop after the export to keep only the merged decoder. Optimum's
# default post-processing produces all three (``decoder_model.onnx``,
# ``decoder_with_past_model.onnx``, ``decoder_model_merged.onnx``) plus their
# matching ``*_data`` external-weight blobs; the merged variant is the only
# one we need at runtime.
_DROP_AFTER_EXPORT = (
    'decoder_model.onnx',
    'decoder_model.onnx_data',
    'decoder_with_past_model.onnx',
    'decoder_with_past_model.onnx_data',
)


def export_marian_to_onnx(model_id: str, output_dir: str) -> None:
    """Run ``optimum-cli export onnx`` then prune to keep only the merged decoder."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'optimum.exporters.onnx',
        '--model', model_id,
        '--task', 'text2text-generation-with-past',
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
    print(f'Translator ONNX exported to {output_dir} ({total / (1024*1024):.1f} MB)',
          flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Helsinki-NLP/opus-mt-mul-en',
                        help='HF Hub id of the Marian model to export.')
    parser.add_argument('--output', required=True,
                        help='Destination directory for the ONNX files.')
    args = parser.parse_args(argv)
    export_marian_to_onnx(args.model, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
