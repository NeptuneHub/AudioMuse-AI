"""Export the e5 lyrics embedding model to ONNX format.

Run this once at Docker build time (where torch + transformers + optimum are
installed). The resulting ``e5-base-v2.onnx`` file is then loaded at runtime
by ``lyrics/embeddings.py`` via ``onnxruntime``, which means the runtime CPU
image does not need torch.

Usage
-----
    python3 export_e5_to_onnx.py \
        --input /app/model/e5-base-v2 \
        --output /app/model/e5-base-v2.onnx

The input directory must contain the standard HuggingFace files
(``config.json``, ``tokenizer.json``, ``model.safetensors`` or
``pytorch_model.bin``). The tokenizer files are NOT copied to the output —
the runtime loads the tokenizer from the original HF directory and only the
model is exported to ONNX.
"""

from __future__ import annotations

import argparse
import os
import sys


def export_e5_to_onnx(input_dir: str, output_path: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not os.path.isdir(input_dir):
        raise SystemExit(f'Input directory not found: {input_dir}')

    print(f'Loading e5 model from {input_dir}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(input_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        input_dir, local_files_only=True, low_cpu_mem_usage=False)
    model = model.to('cpu').eval()

    # Build a dummy input that matches the runtime call shape:
    # encoded = tokenizer(text, truncation=True, padding='max_length',
    #                     max_length=128, return_tensors='pt')
    dummy = tokenizer(
        'placeholder query for onnx export',
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f'Exporting to {output_path} (opset 14)...', flush=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy['input_ids'], dummy['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids':         {0: 'batch', 1: 'seq'},
                'attention_mask':    {0: 'batch', 1: 'seq'},
                'last_hidden_state': {0: 'batch', 1: 'seq'},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Wrote {output_path} ({size_mb:.1f} MB)', flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='Path to the e5-base-v2 HuggingFace model directory.')
    parser.add_argument('--output', required=True,
                        help='Destination .onnx file path.')
    args = parser.parse_args(argv)
    export_e5_to_onnx(args.input, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
