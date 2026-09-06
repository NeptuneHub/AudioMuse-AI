#!/usr/bin/env bash
# One-time ONNX export for the lyrics pipeline and the neural fingerprint.
#
# Run this once (e.g. on WSL) with the project venv ACTIVATED. The resulting
# files land in ./model/ (and the repository root for the fingerprint encoder)
# and should be uploaded to a release / served as pre-built artifacts so the
# Docker build does not have to re-export them.
#
# Outputs:
#   model/gte-multilingual-base-int8.onnx  (~325 MB) - lyrics embedding (INT8 ONNX)
#   model/gte-multilingual-base/           (~5 MB)   - gte tokenizer files (no weights)
#   model/whisper-small-onnx/              (~1.1 GB) - speech-to-text (multilingual)
#   neural_fingerprint.onnx                (~71 MB)  - Search by Recording encoder (step 4,
#                                                      needs a python3.11 for TensorFlow 2.13)
#   neural_fingerprint_pq.npz              (~130 KB) - its 32-byte codebook (step 5, needs
#                                                      NEURAL_FP_DSN, trained ONCE)
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
# 4) raraz15/neural-music-fp triplet checkpoint -> neural_fingerprint.onnx
#    The Search by Recording fingerprint encoder run by tasks/neural_fingerprint.py.
#    Its TensorFlow 2.13 stack only ships for Python <= 3.11, so this step
#    builds its own venv from NEURAL_FP_PYTHON (default: python3.11 on PATH)
#    and is skipped with a message when no such interpreter exists. The
#    checkpoint (~173 MB zip) comes from Zenodo record 15719945.
# ---------------------------------------------------------------------------
NFP_OUT=neural_fingerprint.onnx
NFP_SRC=/tmp/neural-music-fp
NFP_COMMIT=15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc
NFP_CKPT_URL="https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1"
NFP_CKPT_ZIP=/tmp/nmfp-triplet.zip
NFP_CKPT_DIR=/tmp/nmfp-triplet
NFP_VENV="${REPO_ROOT}/.venv-tfexport"
NFP_PYTHON="${NEURAL_FP_PYTHON:-python3.11}"

if [[ -f "${NFP_OUT}" ]]; then
    echo "==> ${NFP_OUT} already exists, skipping neural fingerprint export."
elif ! command -v "${NFP_PYTHON}" >/dev/null 2>&1; then
    echo "==> ${NFP_PYTHON} not found, skipping neural fingerprint export (set NEURAL_FP_PYTHON=/path/to/python3.11)."
else
    if [[ ! -x "${NFP_VENV}/bin/python" ]]; then
        echo "==> Creating ${NFP_VENV} with ${NFP_PYTHON}..."
        "${NFP_PYTHON}" -m venv "${NFP_VENV}"
    fi
    echo "==> Installing neural fingerprint export dependencies..."
    "${NFP_VENV}/bin/pip" install \
        'tensorflow==2.13.1' \
        'tf2onnx==1.17.0' \
        'onnx==1.17.0' \
        'onnxruntime>=1.17,<2.0' \
        'numpy==1.24.3' \
        'protobuf>=4.25,<5'

    if [[ ! -d "${NFP_SRC}/.git" ]]; then
        echo "==> Cloning neural-music-fp to ${NFP_SRC}..."
        git clone --quiet https://github.com/raraz15/neural-music-fp "${NFP_SRC}"
    fi
    git -C "${NFP_SRC}" checkout --quiet "${NFP_COMMIT}"

    if [[ ! -f "${NFP_CKPT_DIR}/config.yaml" ]]; then
        echo "==> Downloading the nmfp-triplet checkpoint to ${NFP_CKPT_ZIP}..."
        curl --proto '=https' --proto-redir '=https' --tlsv1.2 -L -o "${NFP_CKPT_ZIP}" "${NFP_CKPT_URL}"
        unzip -o -q "${NFP_CKPT_ZIP}" -d "$(dirname "${NFP_CKPT_DIR}")"
    fi

    echo "==> Exporting neural-music-fp -> ${NFP_OUT}..."
    "${NFP_VENV}/bin/python" scripts/onnx_export/export_neural_fingerprint_to_onnx.py \
        --source "${NFP_SRC}" \
        --checkpoint "${NFP_CKPT_DIR}" \
        --output "${NFP_OUT}"
fi

# ---------------------------------------------------------------------------
# 5) Neural fingerprint codebook -> neural_fingerprint_pq.npz (optional)
#    The product-quantisation codebook that stores each fingerprint vector as
#    32 bytes. Trained once on the fingerprints of a library analysed with the
#    stage; every stored blob carries its checksum, so it is never regenerated
#    for a library that already holds fingerprints. Runs only when
#    NEURAL_FP_DSN points at such a database (read only), in the project venv.
# ---------------------------------------------------------------------------
NFP_PQ_OUT=neural_fingerprint_pq.npz
if [[ -f "${NFP_PQ_OUT}" ]]; then
    echo "==> ${NFP_PQ_OUT} already exists, skipping codebook training."
elif [[ -z "${NEURAL_FP_DSN:-}" ]]; then
    echo "==> NEURAL_FP_DSN not set, skipping codebook training (needs a database with neural fingerprints)."
else
    echo "==> Training the neural fingerprint codebook -> ${NFP_PQ_OUT}..."
    python scripts/onnx_export/train_neural_fingerprint_codebook.py \
        --dsn "${NEURAL_FP_DSN}" \
        --output "${NFP_PQ_OUT}"
fi

# ---------------------------------------------------------------------------
# 6) Summary
# ---------------------------------------------------------------------------
echo
echo "==> Done. Artifacts:"
ls -lh "${GTE_OUT}" "${NFP_OUT}" "${NFP_PQ_OUT}" 2>/dev/null || true
for d in "${GTE_TOK_OUT}" "${WHISPER_OUT}"; do
    if [ -d "${d}" ]; then
        du -sh "${d}"
        ls -lh "${d}"
    fi
done
