#!/usr/bin/env bash
# One-time ONNX export for the lyrics pipeline.
#
# Run this once (e.g. on WSL) with the project venv ACTIVATED. The resulting
# files land in ./model/ and should be uploaded to a release / served as
# pre-built artifacts so the Docker build does not have to re-export them.
#
# Outputs:
#   model/e5-base-v2.onnx              (~440 MB) — lyrics e5 embedding (ONNX weights)
#   model/e5-base-v2/                  (~3 MB)   — e5 tokenizer files (no weights)
#   model/opus-mt-mul-en-onnx/         (~550 MB) — multi-source → English translator
#                                                 (encoder + decoder_model_merged)
#   model/whisper-small-onnx/          (~1.1 GB) — speech-to-text (multilingual)
#                                                 (encoder + decoder_model_merged)
#
# The merged-decoder bundle (decoder_model_merged.onnx) is a single ONNX file
# that handles both the first-step (no past KV) and KV-cache steps via an
# internal control-flow node, so we don't need to ship the split
# decoder_model.onnx + decoder_with_past_model.onnx pair. Saves ~1 GB of
# disk over the legacy split layout.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/onnx_export/run_exports.sh

set -euo pipefail

# Resolve the repo root from this script's location so it works regardless
# of the user's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate the project venv if one isn't already active. Looks for the two
# common locations; override with VENV_DIR=/path/to/venv if your venv lives
# somewhere else.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    VENV_DIR="${VENV_DIR:-}"
    if [ -z "${VENV_DIR}" ]; then
        if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
            VENV_DIR="${REPO_ROOT}/.venv"
        elif [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
            VENV_DIR="${REPO_ROOT}/venv"
        fi
    fi
    if [ -z "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; then
        echo "ERROR: no venv found. Create one (python3 -m venv .venv) or set VENV_DIR=/path/to/venv." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
echo "Using venv: ${VIRTUAL_ENV}"

E5_SRC=/tmp/e5-base-v2
MARIAN_SRC=Helsinki-NLP/opus-mt-mul-en
WHISPER_SRC=openai/whisper-small
E5_OUT=model/e5-base-v2.onnx
MARIAN_OUT=model/opus-mt-mul-en-onnx
WHISPER_OUT=model/whisper-small-onnx

mkdir -p model

# ---------------------------------------------------------------------------
# 1) Install export-time dependencies (torch + transformers + optimum + onnx).
#    Pinned to the same versions used inside the Docker libraries stage so the
#    exported graphs match what onnxruntime sees at runtime.
# ---------------------------------------------------------------------------
echo "==> Installing export dependencies..."
# NOTE: transformers is pinned <4.54 to satisfy optimum 1.27's onnxruntime
# extra. This only affects the ONE-TIME export run; the runtime image can
# (and does) ship a newer transformers version because the exported ONNX
# graphs are self-contained.
# NOTE: ``accelerate`` is required so optimum's default post-processing step
# produces ``decoder_model_merged.onnx`` (a single file that handles both the
# first-step and KV-cache-step decoder paths). Without it the merging step is
# skipped silently and we end up shipping decoder_model.onnx +
# decoder_with_past_model.onnx instead — twice the disk footprint.
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.6.0+cpu' \
    'transformers>=4.36,<4.54' \
    'huggingface_hub>=0.20,<1.0' \
    'sentencepiece==0.2.1' \
    'onnx>=1.16,<2.0' \
    'numpy>=1.24,<2.0' \
    'optimum[onnxruntime]==1.27.0' \
    'accelerate>=0.26,<1.0'

# ---------------------------------------------------------------------------
# 2) e5-base-v2 → ONNX
#    Download the HF model then call our export_e5_to_onnx.py script.
# ---------------------------------------------------------------------------
if [ ! -f "${E5_OUT}" ]; then
    echo "==> Downloading intfloat/e5-base-v2 to ${E5_SRC}..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/e5-base-v2', local_dir='${E5_SRC}')"

    echo "==> Exporting e5 → ${E5_OUT}..."
    python scripts/onnx_export/export_e5_to_onnx.py \
        --input "${E5_SRC}" \
        --output "${E5_OUT}"
else
    echo "==> ${E5_OUT} already exists, skipping e5 export."
fi

# Always (re)stage the e5 tokenizer files into model/e5-base-v2/. The runtime
# loads the tokenizer with AutoTokenizer.from_pretrained('/app/model/e5-base-v2',
# local_files_only=True), so this directory must contain config.json,
# tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.txt —
# but NOT the PyTorch / ONNX weights (those live in e5-base-v2.onnx).
mkdir -p model/e5-base-v2
python - <<'PY'
import os, shutil
src = '/tmp/e5-base-v2'
if not os.path.isdir(src):
    # Older runs may have cleaned /tmp; re-fetch tokenizer-only files from HF.
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id='intfloat/e5-base-v2',
        local_dir='/tmp/e5-base-v2-tokenizer',
        allow_patterns=['config.json', 'tokenizer*', 'vocab*', 'special_tokens_map.json'],
    )
    src = '/tmp/e5-base-v2-tokenizer'
keep = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'vocab.txt'}
copied = 0
for fname in keep:
    s = os.path.join(src, fname)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join('model', 'e5-base-v2', fname))
        copied += 1
print(f'Staged {copied}/{len(keep)} e5 tokenizer files into model/e5-base-v2/')
PY

# ---------------------------------------------------------------------------
# 3) Helsinki-NLP/opus-mt-mul-en → ONNX (encoder + decoder_model_merged)
#    Multilingual → English translator. Loaded at runtime by
#    ORTModelForSeq2SeqLM with the merged decoder.
# ---------------------------------------------------------------------------
if [ ! -f "${MARIAN_OUT}/encoder_model.onnx" ] \
        || [ ! -f "${MARIAN_OUT}/decoder_model_merged.onnx" ]; then
    echo "==> Exporting ${MARIAN_SRC} → ${MARIAN_OUT}..."
    python scripts/onnx_export/export_marian_to_onnx.py \
        --model "${MARIAN_SRC}" \
        --output "${MARIAN_OUT}"
else
    echo "==> ${MARIAN_OUT} already exists, skipping Marian export."
fi

# ---------------------------------------------------------------------------
# 4) openai/whisper-small → ONNX (encoder + decoder_model_merged)
#    Loaded at runtime by ORTModelForSpeechSeq2Seq with the merged decoder.
# ---------------------------------------------------------------------------
if [ ! -f "${WHISPER_OUT}/encoder_model.onnx" ] \
        || [ ! -f "${WHISPER_OUT}/decoder_model_merged.onnx" ]; then
    echo "==> Exporting ${WHISPER_SRC} → ${WHISPER_OUT}..."
    python scripts/onnx_export/export_whisper_to_onnx.py \
        --model "${WHISPER_SRC}" \
        --output "${WHISPER_OUT}"
else
    echo "==> ${WHISPER_OUT} already exists, skipping whisper export."
fi

# ---------------------------------------------------------------------------
# 5) Summary
# ---------------------------------------------------------------------------
echo
echo "==> Done. Artifacts:"
ls -lh "${E5_OUT}" 2>/dev/null || true
for d in "${MARIAN_OUT}" "${WHISPER_OUT}"; do
    if [ -d "${d}" ]; then
        du -sh "${d}"
        ls -lh "${d}"
    fi
done
