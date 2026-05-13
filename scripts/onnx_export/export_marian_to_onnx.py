"""Export Helsinki-NLP/opus-mt-mul-en to ONNX.

One multilingual Marian model that translates from ~70 source languages to
English. Run once at Docker build time; ``lyrics/translation_onnx.py`` then
consumes the resulting ``encoder_model.onnx`` + ``decoder_model.onnx`` files
at runtime via raw onnxruntime, with a numpy greedy-decode loop — no torch
required at runtime.

Usage
-----
    python3 export_marian_to_onnx.py \
        --model Helsinki-NLP/opus-mt-mul-en \
        --output /app/model/opus-mt-mul-en-onnx

The output directory will contain:
    encoder_model.onnx
    decoder_model.onnx
    config.json
    tokenizer files (sentencepiece + vocab)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def export_marian_to_onnx(model_id: str, output_dir: str) -> None:
    """Run ``optimum-cli export onnx`` to produce the encoder/decoder ONNX pair."""
    os.makedirs(output_dir, exist_ok=True)

    # optimum-cli is the canonical path: it ships an exporter that handles the
    # encoder/decoder split, the past_key_values plumbing, etc. We deliberately
    # do NOT pass --task because optimum auto-detects it from the model config.
    # Use --no-post-process to keep the exported graphs simple (we run our own
    # decode loop, so we don't need merged decoder graphs).
    cmd = [
        sys.executable, '-m', 'optimum.exporters.onnx',
        '--model', model_id,
        '--task', 'text2text-generation',
        '--no-post-process',
        output_dir,
    ]
    print(f'$ {" ".join(cmd)}', flush=True)
    completed = subprocess.run(cmd, capture_output=False)
    if completed.returncode != 0:
        raise SystemExit(f'optimum-cli export failed (rc={completed.returncode})')

    # Trim files we don't need at runtime to keep image size down. We only need
    # encoder_model.onnx, decoder_model.onnx, config.json, and the tokenizer
    # files. The decoder_with_past / decoder_model_merged variants double the
    # disk footprint and are not used by our greedy-decode loop.
    keep_prefixes = (
        'encoder_model',
        'decoder_model.onnx',  # NB: precise filename, not the prefix
        'config.json',
        'generation_config.json',
        'special_tokens_map.json',
        'tokenizer',
        'source.spm',
        'target.spm',
        'vocab.json',
    )
    for entry in os.listdir(output_dir):
        full = os.path.join(output_dir, entry)
        if entry == 'decoder_model_merged.onnx' or entry == 'decoder_model_merged.onnx_data':
            os.remove(full)
            continue
        if entry == 'decoder_with_past_model.onnx' or entry == 'decoder_with_past_model.onnx_data':
            os.remove(full)
            continue

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
