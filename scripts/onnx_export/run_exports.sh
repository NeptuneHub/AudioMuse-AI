#!/usr/bin/env bash
# One-time ONNX export for the lyrics pipeline.
#
# Run this once (e.g. on WSL) with the project venv ACTIVATED. The resulting
# files land in ./model/ and should be uploaded to a release / served as
# pre-built artifacts so the Docker build does not have to re-export them.
#
# Outputs:
#   model/gte-multilingual-base-int8.onnx  (~325 MB) - lyrics embedding (INT8 ONNX)
#   model/gte-multilingual-base/           (~5 MB)   - gte tokenizer files (no weights)
#   model/whisper-small-onnx/              (~1.1 GB) - speech-to-text (multilingual)
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

GTE_SRC=/tmp/gte-multilingual-base
WHISPER_SRC=openai/whisper-small
GTE_OUT=model/gte-multilingual-base-int8.onnx
GTE_TOK_OUT=model/gte-multilingual-base
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
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.6.0+cpu' \
    'transformers>=4.36,<4.54' \
    'huggingface_hub>=0.20,<1.0' \
    'sentencepiece==0.2.1' \
    'onnx>=1.16,<2.0' \
    'numpy>=1.24,<2.0' \
    'optimum[onnxruntime]==1.27.0'

# ---------------------------------------------------------------------------
# 2) Alibaba-NLP/gte-multilingual-base -> INT8 ONNX
#    Download the HF model (custom architecture -> trust_remote_code) then call
#    our export_gte_to_onnx.py script, which exports fp32 and dynamic-INT8
#    quantizes it. The runtime applies CLS pooling + L2 normalization.
# ---------------------------------------------------------------------------
if [ ! -f "${GTE_OUT}" ]; then
    echo "==> Downloading Alibaba-NLP/gte-multilingual-base to ${GTE_SRC}..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Alibaba-NLP/gte-multilingual-base', local_dir='${GTE_SRC}')"

    echo "==> Exporting gte -> ${GTE_OUT} (INT8)..."
    python scripts/onnx_export/export_gte_to_onnx.py \
        --input "${GTE_SRC}" \
        --output "${GTE_OUT}" \
        --tokenizer-out "${GTE_TOK_OUT}"
else
    echo "==> ${GTE_OUT} already exists, skipping gte export."
fi

# ---------------------------------------------------------------------------
# 3) openai/whisper-small -> ONNX (encoder + decoder, no past KV cache)
#    Used by lyrics/whisper_onnx.py with a custom mel + greedy decode loop.
# ---------------------------------------------------------------------------
if [ ! -f "${WHISPER_OUT}/encoder_model.onnx" ] || [ ! -f "${WHISPER_OUT}/decoder_model.onnx" ]; then
    echo "==> Exporting ${WHISPER_SRC} -> ${WHISPER_OUT}..."
    python scripts/onnx_export/export_whisper_to_onnx.py \
        --model "${WHISPER_SRC}" \
        --output "${WHISPER_OUT}"
else
    echo "==> ${WHISPER_OUT} already exists, skipping whisper export."
fi

# ---------------------------------------------------------------------------
# 4) Summary
# ---------------------------------------------------------------------------
echo
echo "==> Done. Artifacts:"
ls -lh "${GTE_OUT}" 2>/dev/null || true
for d in "${GTE_TOK_OUT}" "${WHISPER_OUT}"; do
    if [ -d "${d}" ]; then
        du -sh "${d}"
        ls -lh "${d}"
    fi
done
