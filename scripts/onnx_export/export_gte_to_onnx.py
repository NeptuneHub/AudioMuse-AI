"""Export the gte-multilingual-base lyrics embedding model to INT8 ONNX.

Run this once at export time (where torch + transformers + optimum + onnx
are installed). It exports ``Alibaba-NLP/gte-multilingual-base`` to ONNX and
then applies dynamic INT8 weight quantization, producing
``gte-multilingual-base-int8.onnx``. The result is loaded at runtime by
``lyrics/gte_onnx.py`` via ``onnxruntime`` + the bare ``tokenizers`` package,
so the runtime CPU image needs neither torch nor transformers.

gte-multilingual-base uses a custom architecture, so ``trust_remote_code`` is
required when loading it. The sentence embedding is the CLS token of the last
hidden state (see the model's ``1_Pooling/config.json``); pooling is done at
runtime in ``lyrics/gte_onnx.py``, so this script exports the raw encoder
(``last_hidden_state``) only.

Usage
-----
    python3 export_gte_to_onnx.py \
        --input /tmp/gte-multilingual-base \
        --output /app/model/gte-multilingual-base-int8.onnx \
        --tokenizer-out /app/model/gte-multilingual-base

The input directory must contain the standard HuggingFace files
(``config.json``, ``tokenizer.json``, ``model.safetensors`` plus the custom
``modeling`` code). The tokenizer files are copied to ``--tokenizer-out`` so
the runtime can load ``tokenizer.json`` without transformers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


_TOKENIZER_FILES = (
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'config.json', 'sentencepiece.bpe.model',
)


def export_gte_to_onnx(input_dir: str, output_path: str,
                       tokenizer_out: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not os.path.isdir(input_dir):
        raise SystemExit(f'Input directory not found: {input_dir}')

    print(f'Loading gte-multilingual-base model from {input_dir}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        input_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        input_dir, trust_remote_code=True, low_cpu_mem_usage=False)
    model = model.to('cpu').eval()

    dummy = tokenizer(
        'placeholder query for onnx export',
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fp32_path = output_path.replace('.onnx', '') + '.fp32.onnx'

    print(f'Exporting fp32 graph to {fp32_path} (opset 14)...', flush=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy['input_ids'], dummy['attention_mask']),
            fp32_path,
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

    print(f'Quantizing to INT8 → {output_path}...', flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=fp32_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    os.remove(fp32_path)

    os.makedirs(tokenizer_out, exist_ok=True)
    copied = 0
    for fname in _TOKENIZER_FILES:
        src = os.path.join(input_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tokenizer_out, fname))
            copied += 1
    print(f'Staged {copied} tokenizer files into {tokenizer_out}', flush=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Wrote {output_path} ({size_mb:.1f} MB)', flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='Path to the gte-multilingual-base HuggingFace model directory.')
    parser.add_argument('--output', required=True,
                        help='Destination INT8 .onnx file path.')
    parser.add_argument('--tokenizer-out', required=True,
                        help='Directory to copy the runtime tokenizer files into.')
    args = parser.parse_args(argv)
    export_gte_to_onnx(args.input, args.output, args.tokenizer_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
