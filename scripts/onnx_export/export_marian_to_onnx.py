"""Export Helsinki-NLP/opus-mt-mul-en to ONNX.

One multilingual Marian model that translates from ~70 source languages to
English. Run once at Docker build time; ``lyrics/translation_onnx.py`` then
consumes the resulting ``encoder_model.onnx`` + ``decoder_model_merged.onnx``
files at runtime via raw onnxruntime, with a numpy greedy-decode loop — no torch
required at runtime.

Also used for the per-language CJK translators (issue #553):
    python3 export_marian_to_onnx.py --model Helsinki-NLP/opus-mt-zh-en --output model/opus-mt-zh-en-onnx
    python3 export_marian_to_onnx.py --model Helsinki-NLP/opus-mt-ja-en --output model/opus-mt-ja-en-onnx
    python3 export_marian_to_onnx.py --model Helsinki-NLP/opus-mt-ko-en --output model/opus-mt-ko-en-onnx

Usage
-----
    python3 export_marian_to_onnx.py \
        --model Helsinki-NLP/opus-mt-mul-en \
        --output /app/model/opus-mt-mul-en-onnx

The output directory will contain:
    encoder_model.onnx
    decoder_model_merged.onnx   (use --no-merged for the legacy decoder_model.onnx)
    config.json
    tokenizer files (sentencepiece + vocab)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def export_marian_to_onnx(model_id: str, output_dir: str, merged: bool = True) -> None:
    """Run ``optimum-cli export onnx`` to produce the encoder/decoder ONNX pair.

    With ``merged=True`` (default) the merged decoder graph
    ``decoder_model_merged.onnx`` is kept — this is what
    ``lyrics/translation_onnx.py`` loads at runtime via its ``use_cache_branch``
    decode loop. With ``merged=False`` the legacy non-merged ``decoder_model.onnx``
    is kept instead.
    """
    os.makedirs(output_dir, exist_ok=True)

    task = 'text2text-generation-with-past' if merged else 'text2text-generation'
    cmd = [
        sys.executable, '-m', 'optimum.exporters.onnx',
        '--model', model_id,
        '--task', task,
    ]
    if not merged:
        cmd.append('--no-post-process')
    cmd.append(output_dir)
    print(f'$ {" ".join(cmd)}', flush=True)
    completed = subprocess.run(cmd, capture_output=False)
    if completed.returncode != 0:
        raise SystemExit(f'optimum-cli export failed (rc={completed.returncode})')

    if merged:
        drop = ('decoder_model.onnx', 'decoder_model.onnx_data',
                'decoder_with_past_model.onnx', 'decoder_with_past_model.onnx_data')
        required = 'decoder_model_merged.onnx'
    else:
        drop = ('decoder_model_merged.onnx', 'decoder_model_merged.onnx_data',
                'decoder_with_past_model.onnx', 'decoder_with_past_model.onnx_data')
        required = 'decoder_model.onnx'
    for entry in os.listdir(output_dir):
        if entry in drop:
            os.remove(os.path.join(output_dir, entry))

    if not os.path.isfile(os.path.join(output_dir, required)):
        raise SystemExit(f'expected {required} missing in {output_dir} after export')

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
    parser.add_argument('--no-merged', dest='merged', action='store_false',
                        help='Export the legacy non-merged decoder_model.onnx '
                             'instead of decoder_model_merged.onnx.')
    parser.set_defaults(merged=True)
    args = parser.parse_args(argv)
    export_marian_to_onnx(args.model, args.output, merged=args.merged)
    return 0


if __name__ == '__main__':
    sys.exit(main())
