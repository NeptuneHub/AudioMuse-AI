# syntax=docker/dockerfile:1
# AudioMuse-AI Dockerfile
# Supports both CPU (ubuntu:24.04) and GPU (nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04) builds
#
# Build examples:
#   CPU:  docker build -t audiomuse-ai .
#   GPU:  docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 -t audiomuse-ai-gpu .

ARG BASE_IMAGE=ubuntu:24.04

# ============================================================================
# Stage 1: Download ML models (cached separately for faster rebuilds)
# ============================================================================
FROM ubuntu:24.04 AS models

SHELL ["/bin/bash", "-lc"]

RUN mkdir -p /app/model

# Install download tools with exponential backoff retry
RUN set -ux; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if apt-get update && apt-get install -y --no-install-recommends wget ca-certificates curl; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "apt-get attempt $n failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    rm -rf /var/lib/apt/lists/*

# Download musicnn ONNX models with diagnostics and retry logic.
# Lyrics models (Whisper / e5 / MarianMT / silero-vad) are downloaded
# in later stages from the project release tarballs.
RUN set -eux; \
    mkdir -p /app/model; \
    urls=( \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/musicnn_embedding.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/musicnn_prediction.onnx" \
    ); \
    for u in "${urls[@]}"; do \
        n=0; \
        fname="/app/model/$(basename "$u")"; \
        # Diagnostic: print server response headers (helpful when downloads return 0 bytes) \
        wget --server-response --spider --timeout=15 --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" "$u" || true; \
        until [ "$n" -ge 5 ]; do \
            # Use wget with retries. --tries and --waitretry add backoff for transient failures. \
            if wget --no-verbose --tries=3 --retry-connrefused --waitretry=5 --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" -O "$fname" "$u"; then \
                echo "Downloaded $u -> $fname"; \
                break; \
            fi; \
            n=$((n+1)); \
            echo "wget attempt $n for $u failed — retrying in $((n*n))s"; \
            sleep $((n*n)); \
        done; \
        if [ "$n" -ge 5 ]; then \
            echo "ERROR: failed to download $u after 5 attempts"; \
            ls -lah /app/model || true; \
            exit 1; \
        fi; \
    done

# NOTE: CLAP model download moved to runner stage to avoid EOF errors with large file transfers in multi-arch builds

# ============================================================================
# Stage 2a: runtime-base — RUNTIME-ONLY system libs (parent of `runner`)
# ============================================================================
# This stage holds only what the application needs at run time: shared
# libraries (.so) that Python wheels load, plus the small set of CLI tools
# the entrypoint / supervisord / debugging rely on. It deliberately omits
# compilers and -dev headers — those live in the `base` stage below, which
# is used solely to build Python wheels in the `libraries` stage and never
# becomes a parent of `runner`.
#
# `cuda-compiler` is INTENTIONALLY kept here (not moved to build-only)
# because cupy JIT-compiles CUDA kernels at runtime on GPU builds.
FROM ${BASE_IMAGE} AS runtime-base

ARG BASE_IMAGE

SHELL ["/bin/bash", "-c"]

RUN set -ux; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        # Use noninteractive frontend to avoid tzdata prompts when installing tzdata
        if DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            python3 python3-pip \
            libfftw3-double3=3.3.10-1ubuntu3 \
            libyaml-0-2=0.2.5-1build1 \
            libsamplerate0=0.2.2-4build1 \
            libsndfile1=1.2.2-1ubuntu5.24.04.1 \
            libopenblas0 \
            liblapack3=3.12.0-3build1.1 \
            libgomp1 \
            libpq5 postgresql-client \
            ffmpeg wget curl \
            supervisor procps \
            git vim redis-tools strace iputils-ping \
            "$(if [[ "$BASE_IMAGE" =~ ^nvidia/cuda:([0-9]+)\.([0-9]+).+$ ]]; then echo "cuda-compiler-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"; fi)"; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "apt-get attempt $n failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    rm -rf /var/lib/apt/lists/* && \
    apt-get remove -y python3-numpy || true && \
    apt-get autoremove -y || true && \
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# ============================================================================
# Stage 2b: base — runtime-base + compilers / -dev headers (BUILD-ONLY)
# ============================================================================
# Adds the toolchain needed to compile Python wheels (psycopg2, essentia,
# numpy/scipy fallbacks, etc.). Parent of `libraries` only — `runner`
# branches off `runtime-base`, so gcc/g++/python3-dev and the -dev headers
# never reach the final published image.
FROM runtime-base AS base

ARG BASE_IMAGE

# Copy uv for fast package management (10-100x faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN set -ux; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            python3-dev \
            libfftw3-dev \
            libyaml-dev \
            libsamplerate0-dev \
            libsndfile1-dev \
            libopenblas-dev \
            liblapack-dev=3.12.0-3build1.1 \
            libpq-dev \
            gcc g++; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "apt-get attempt $n failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 3: Libraries - Python packages installation
# ============================================================================
FROM base AS libraries

ARG BASE_IMAGE

WORKDIR /app

# Copy requirements files
COPY requirements/ /app/requirements/

# Install Python packages with uv (combined in single layer for efficiency)
# GPU builds: cupy, cuml, onnxruntime-gpu, voyager, torch (CUDA)
# CPU builds: onnxruntime (CPU only), torch (CPU)
# Note: --index-strategy unsafe-best-match resolves conflicts between pypi.nvidia.com and pypi.org
RUN if [[ "$BASE_IMAGE" =~ ^nvidia/cuda: ]]; then \
        echo "NVIDIA base image detected: installing GPU packages (cupy, cuml, onnxruntime-gpu, voyager, torch+cuda)"; \
        uv pip install --system --no-cache --index-strategy unsafe-best-match -r /app/requirements/gpu.txt -r /app/requirements/common.txt || exit 1; \
    else \
        echo "CPU base image: installing all packages together for dependency resolution"; \
        uv pip install --system --no-cache --index-strategy unsafe-best-match -r /app/requirements/cpu.txt -r /app/requirements/common.txt || exit 1; \
    fi \
    && echo "Verifying psycopg2 installation..." \
    && python3 -c "import psycopg2; print('psycopg2 OK')" \
    && find /usr/local/lib/python3.12/dist-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

RUN python3 -m wn download oewn:2024 && echo "wn / OEWN 2024 corpus downloaded"

# Download HuggingFace models (BERT, RoBERTa, BART, T5) from GitHub release
# These are the text encoders needed by laion-clap library for text embeddings
RUN set -eux; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model"; \
    hf_models="huggingface_models.tar.gz"; \
    cache_dir="/app/.cache/huggingface"; \
    echo "Downloading HuggingFace models (~985MB)..."; \
    \
    # Download with retry logic \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/tmp/$hf_models" "$base_url/$hf_models"; then \
            echo "✓ HuggingFace models downloaded"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "Download attempt $n failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: Failed to download HuggingFace models after 5 attempts"; \
        exit 1; \
    fi; \
    \
    # Extract to cache directory \
    mkdir -p "$cache_dir"; \
    echo "Extracting HuggingFace models..."; \
    tar -xzf "/tmp/$hf_models" -C "$cache_dir"; \
    \
    # Verify extraction \
    if [ ! -d "$cache_dir/hub" ]; then \
        echo "ERROR: HuggingFace models extraction failed"; \
        exit 1; \
    fi; \
    \
    # Clean up tarball \
    rm -f "/tmp/$hf_models"; \
    \
    echo "✓ HuggingFace models extracted to $cache_dir"; \
    du -sh "$cache_dir"

# ============================================================================
# Stage 4: Runner - Final production image
# ============================================================================
# IMPORTANT: extends `runtime-base` (NOT `base`). That keeps gcc/g++,
# python3-dev and the *-dev headers out of the final image, saving
# ~300-400 MB. Anything that needs compiling lives in the `libraries`
# stage and gets COPY'd in as already-built artifacts below.
FROM runtime-base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    HF_XET_DISABLE=1

# Note: bundled HuggingFace models (e5, RoBERTa, ...) load with
# local_files_only=True per call. Marian translation models download on demand
# at first use of a new source language; HF_HUB_OFFLINE is intentionally NOT set.

WORKDIR /app

# Ensure tzdata package is installed so /usr/share/zoneinfo exists and TZ can be applied
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

# Copy Python packages from libraries stage
COPY --from=libraries /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/
# Copy console entrypoints (gunicorn, etc.) from libraries stage
COPY --from=libraries /usr/local/bin/ /usr/local/bin/
# Copy HuggingFace cache (RoBERTa model) from libraries stage
COPY --from=libraries /app/.cache/huggingface/ /app/.cache/huggingface/

# Verify cache was copied correctly
RUN ls -lah /app/.cache/huggingface/ && \
    echo "HuggingFace cache contents:" && \
    du -sh /app/.cache/huggingface/* || echo "Cache directory empty!"

# Copy all downloaded/extracted models from models stage
COPY --from=models /app/model/ /app/model/

# Download CLAP ONNX models directly in runner stage
# - DCLAP audio model (~20MB + external data): Distilled student for music analysis in worker containers
# - Text model (~478MB): Original LAION CLAP text encoder for text search in Flask containers
RUN set -eux; \
    dclap_url="https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1"; \
    text_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model"; \
    arch=$(uname -m); \
    echo "Architecture detected: $arch - Downloading CLAP ONNX models..."; \
    \
    # Download DCLAP audio model (~1.2MB ONNX + ~20MB external data) \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/app/model/model_epoch_36.onnx" "$dclap_url/model_epoch_36.onnx"; then \
            echo "✓ DCLAP audio model downloaded"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "Download attempt $n for DCLAP audio model failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: Failed to download DCLAP audio model after 5 attempts"; \
        exit 1; \
    fi; \
    \
    # Download DCLAP audio model external data file \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/app/model/model_epoch_36.onnx.data" "$dclap_url/model_epoch_36.onnx.data"; then \
            echo "✓ DCLAP audio model data downloaded"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "Download attempt $n for DCLAP audio data failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: Failed to download DCLAP audio model data after 5 attempts"; \
        exit 1; \
    fi; \
    \
    # Download text model (~478MB) \
    text_model="clap_text_model.onnx"; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/app/model/$text_model" "$text_url/$text_model"; then \
            echo "✓ CLAP text model downloaded"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "Download attempt $n for text model failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: Failed to download CLAP text model after 5 attempts"; \
        exit 1; \
    fi; \
    \
    # Verify DCLAP audio model \
    if [ ! -f "/app/model/model_epoch_36.onnx" ]; then \
        echo "ERROR: DCLAP audio model file not created"; \
        exit 1; \
    fi; \
    if [ ! -f "/app/model/model_epoch_36.onnx.data" ]; then \
        echo "ERROR: DCLAP audio model data file not created"; \
        exit 1; \
    fi; \
    \
    # Verify text model \
    if [ ! -f "/app/model/$text_model" ]; then \
        echo "ERROR: CLAP text model file not created"; \
        exit 1; \
    fi; \
    file_size=$(stat -c%s "/app/model/$text_model" 2>/dev/null || stat -f%z "/app/model/$text_model" 2>/dev/null || echo "0"); \
    if [ "$file_size" -lt 450000000 ]; then \
        echo "ERROR: CLAP text model file is too small (expected ~478MB, got $file_size bytes)"; \
        exit 1; \
    fi; \
    \
    echo "✓ CLAP models downloaded successfully (arch: $arch)"; \
    ls -lh /app/model/model_epoch_36.onnx /app/model/model_epoch_36.onnx.data "/app/model/$text_model"

# Download Whisper-small ONNX bundle (~570 MB) — HuggingFace optimum export
# of openai/whisper-small (encoder_model.onnx + decoder_model_merged.onnx +
# tokenizer files + preprocessor config). Re-hosted on the project's GitHub
# release for mirror independence. Bundle ships `whisper-small-onnx/` as
# its top-level directory. Loaded at runtime by lyrics/whisper_onnx.py via
# raw onnxruntime.
RUN set -eux; \
    whisper_dir="/app/model/whisper-small-onnx"; \
    whisper_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/lyrics_model_whisper.tar.gz"; \
    whisper_dest="/tmp/lyrics_model_whisper.tar.gz"; \
    echo "Downloading Whisper-small ONNX bundle (~570 MB)..."; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "$whisper_dest" "$whisper_url"; then \
            echo "✓ whisper bundle downloaded"; break; \
        fi; \
        n=$((n+1)); \
        echo "wget attempt $n for whisper bundle failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: failed to download whisper bundle"; exit 1; \
    fi; \
    mkdir -p /app/model; \
    tar -xzf "$whisper_dest" -C /app/model; \
    rm -f "$whisper_dest"; \
    for f in encoder_model.onnx decoder_model_merged.onnx \
             tokenizer.json tokenizer_config.json \
             special_tokens_map.json preprocessor_config.json \
             config.json generation_config.json vocab.json merges.txt; do \
        if [ ! -f "$whisper_dir/$f" ]; then \
            echo "ERROR: Whisper file missing: $whisper_dir/$f"; \
            echo "Actual /app/model contents:"; \
            ls -laR /app/model | head -50; \
            exit 1; \
        fi; \
    done; \
    echo "✓ Whisper-small ONNX model ready in $whisper_dir"; \
    du -sh "$whisper_dir"

# Download silero VAD ONNX (~2 MB) — re-hosted on the project's GitHub release
# for mirror independence (original source: snakers4/silero-vad). Bundle ships
# silero_vad.onnx at archive root. Loaded by lyrics/silero_onnx.py via raw
# onnxruntime.
RUN set -eux; \
    silero_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/lyrics_model_silero_vad.tar.gz"; \
    silero_dest="/tmp/lyrics_model_silero_vad.tar.gz"; \
    silero_path="/app/model/silero_vad.onnx"; \
    echo "Downloading silero VAD ONNX bundle (~2 MB)..."; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=5 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "$silero_dest" "$silero_url"; then \
            echo "✓ silero bundle downloaded"; break; \
        fi; \
        n=$((n+1)); \
        echo "wget attempt $n for silero bundle failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: failed to download silero bundle"; exit 1; \
    fi; \
    mkdir -p /app/model; \
    tar -xzf "$silero_dest" -C /app/model; \
    rm -f "$silero_dest"; \
    if [ ! -f "$silero_path" ]; then \
        echo "ERROR: silero_vad.onnx missing after extraction"; \
        ls -laR /app/model | head -50; \
        exit 1; \
    fi; \
    ls -lh "$silero_path"

# Download e5-base-v2 ONNX bundle (~440 MB) — re-hosted on the project's
# GitHub release for mirror independence. Tarball ships the ONNX file flat
# at the archive root (`e5-base-v2.onnx`) plus a sibling `e5-base-v2/`
# directory with the tokenizer files. Loaded by lyrics/e5_onnx.py via raw
# onnxruntime + the bare `tokenizers` package.
RUN set -eux; \
    e5_onnx_path="/app/model/e5-base-v2.onnx"; \
    e5_tok_dir="/app/model/e5-base-v2"; \
    e5_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/lyrics_model_e5.tar.gz"; \
    e5_dest="/tmp/lyrics_model_e5.tar.gz"; \
    echo "Downloading e5-base-v2 ONNX bundle (~440 MB)..."; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "$e5_dest" "$e5_url"; then \
            echo "✓ e5 bundle downloaded"; break; \
        fi; \
        n=$((n+1)); \
        echo "wget attempt $n for e5 bundle failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: failed to download e5 bundle"; exit 1; \
    fi; \
    mkdir -p /app/model; \
    tar -xzf "$e5_dest" -C /app/model; \
    rm -f "$e5_dest"; \
    if [ ! -f "$e5_onnx_path" ]; then \
        echo "ERROR: e5 ONNX missing after extraction: $e5_onnx_path"; \
        ls -laR /app/model | head -50; \
        exit 1; \
    fi; \
    for f in tokenizer.json tokenizer_config.json vocab.txt config.json special_tokens_map.json; do \
        if [ ! -f "$e5_tok_dir/$f" ]; then \
            echo "ERROR: e5 tokenizer file missing: $e5_tok_dir/$f"; exit 1; \
        fi; \
    done; \
    echo "✓ e5-base-v2 ONNX ready ($e5_onnx_path + $e5_tok_dir)"; \
    du -sh "$e5_onnx_path" "$e5_tok_dir"

# Download opus-mt-mul-en ONNX bundle (~520 MB) — multilingual-to-English
# Marian translator pre-exported by this project (no official ONNX export
# exists upstream for opus-mt-mul-en). The tarball ships
# `opus-mt-mul-en-onnx/` as its top-level directory, so we extract it
# straight into /app/model and verify the resulting path.
# Loaded at runtime by lyrics/translation_onnx.py (raw onnxruntime).
RUN set -eux; \
    marian_dir="/app/model/opus-mt-mul-en-onnx"; \
    marian_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/lyrics_model_marian.tar.gz"; \
    marian_dest="/tmp/lyrics_model_marian.tar.gz"; \
    echo "Downloading opus-mt-mul-en ONNX bundle (~520 MB)..."; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "$marian_dest" "$marian_url"; then \
            echo "✓ marian bundle downloaded"; break; \
        fi; \
        n=$((n+1)); \
        echo "wget attempt $n for marian bundle failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: failed to download marian bundle"; exit 1; \
    fi; \
    mkdir -p /app/model; \
    tar -xzf "$marian_dest" -C /app/model; \
    rm -f "$marian_dest"; \
    # Bundle ships SentencePiece tokenization (source.spm / target.spm) — no
    # tokenizer.json. Loaded at runtime via transformers.MarianTokenizer.
    for f in encoder_model.onnx decoder_model_merged.onnx \
             source.spm target.spm tokenizer_config.json \
             vocab.json config.json; do \
        if [ ! -f "$marian_dir/$f" ]; then \
            echo "ERROR: marian file missing: $marian_dir/$f"; \
            echo "Actual /app/model contents:"; \
            ls -laR /app/model | head -50; \
            exit 1; \
        fi; \
    done; \
    echo "✓ opus-mt-mul-en ONNX ready in $marian_dir"; \
    du -sh "$marian_dir"

RUN set -eu; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model"; \
    for code in zh ja ko; do \
        cjk_dir="/app/model/opus-mt-${code}-en-onnx"; \
        cjk_url="${base_url}/lyrics_model_marian_${code}.tar.gz"; \
        cjk_dest="/tmp/lyrics_model_marian_${code}.tar.gz"; \
        echo "Downloading opus-mt-${code}-en ONNX bundle (best-effort)..."; \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "$cjk_dest" "$cjk_url"; then \
            mkdir -p /app/model; \
            if tar -xzf "$cjk_dest" -C /app/model; then \
                ok=1; \
                for f in encoder_model.onnx decoder_model_merged.onnx source.spm \
                         target.spm tokenizer_config.json vocab.json config.json; do \
                    if [ ! -f "$cjk_dir/$f" ]; then echo "WARNING: opus-mt-${code}-en missing $f"; ok=0; fi; \
                done; \
                if [ "$ok" = "1" ]; then echo "✓ opus-mt-${code}-en ONNX ready in $cjk_dir"; du -sh "$cjk_dir"; \
                else rm -rf "$cjk_dir"; echo "opus-mt-${code}-en incomplete — runtime falls back to opus-mt-mul-en"; fi; \
            else echo "WARNING: extract failed for opus-mt-${code}-en — runtime falls back to opus-mt-mul-en"; fi; \
            rm -f "$cjk_dest"; \
        else \
            rm -f "$cjk_dest"; \
            echo "WARNING: download failed for opus-mt-${code}-en — runtime falls back to opus-mt-mul-en"; \
        fi; \
    done

# Copy application code (last to maximize cache hits for code changes)
COPY . /app
COPY deployment/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN chmod +x /app/docker-entrypoint.sh
RUN ls -l /etc/supervisor/conf.d && test -f /etc/supervisor/conf.d/supervisord.conf

# ============================================================================
# CPU CONSISTENCY SETTINGS
# ============================================================================
# These environment variables ensure CONSISTENT behavior across different
# AVX2-capable CPUs (e.g., Intel 6th gen vs 12th gen have different FPU defaults).
# They do NOT enable non-AVX support - AVX2 is still required for x86_64 builds.
# ARM64 builds use NEON instructions and work on all ARM64 CPUs.

# oneDNN floating-point math mode: STRICT reduces non-deterministic FP optimizations
# Keeps CPU behavior deterministic across different CPU generations
ENV ONEDNN_DEFAULT_FPMATH_MODE=STRICT

# ONNX Runtime optimization settings to prevent signal 9 crashes on newer CPUs
# (Intel 12600K and similar have different optimization behavior than older CPUs)
# Similar to TF_ENABLE_ONEDNN_OPTS=0 for TensorFlow compatibility
ENV ORT_DISABLE_ALL_OPTIMIZATIONS=1 \
    ORT_ENABLE_CPU_FP16_OPS=0

# Force consistent memory allocation and precision behavior
# Prevents different memory allocation patterns and floating-point precision issues
# between Intel generations (e.g., 12600K vs i5-6500)
ENV ORT_DISABLE_AVX512=1 \
    ORT_FORCE_SHARED_PROVIDER=1

# Force consistent MKL floating-point behavior across different Intel generations
# 12600K has different FPU precision defaults than 6th gen CPUs
ENV MKL_ENABLE_INSTRUCTIONS=AVX2 \
    MKL_DYNAMIC=FALSE

# Prevent aggressive memory pre-allocation on newer CPUs
ENV ORT_DISABLE_MEMORY_PATTERN_OPTIMIZATION=1

# numba JIT cache must land in a writable directory.
# When the container runs as a non-root user the system site-packages directory
# (/usr/local/lib/python3.x/dist-packages/) is read-only, which causes librosa
# to fail with: "cannot cache function: no locator available".
# Point numba to /tmp so it always has write access (issue: NeptuneHub/AudioMuse-AI#479).
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
