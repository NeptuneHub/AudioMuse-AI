"""Export openai/whisper-small to ONNX.

Run once at export time (with torch + optimum installed). The resulting
``encoder_model.onnx`` + ``decoder_model.onnx`` pair is then consumed at
runtime by ``lyrics/whisper_onnx.py`` via raw onnxruntime + a numpy greedy
decoder — no torch needed at runtime.

Usage
-----
    python3 export_whisper_to_onnx.py \
        --model openai/whisper-small \
        --output /app/model/whisper-small-onnx

The output directory will contain:
    encoder_model.onnx
    decoder_model.onnx
    config.json
    generation_config.json
    preprocessor_config.json   (mel filterbank parameters)
    tokenizer.json + vocab.json + merges.txt + special_tokens_map.json + ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def export_whisper_to_onnx(model_id: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'optimum.exporters.onnx',
        '--model', model_id,
        '--task', 'automatic-speech-recognition',
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
