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

# Lyrics ONNX models are bundled in ``lyrics_model_onnx.tar.gz`` on the v4.0.0
# release and downloaded here. The archive expands to ``model/{e5-base-v2.onnx,
# e5-base-v2/<tokenizer files>, opus-mt-mul-en-onnx/, whisper-small-onnx/}``.
# Other small ONNX models (musicnn, silero) come from their own URLs below.
RUN set -eux; \
    mkdir -p /app/model; \
    urls=( \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/musicnn_embedding.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/musicnn_prediction.onnx" \
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
    ); \
    for u in "${urls[@]}"; do \
        n=0; \
        fname="/app/model/$(basename "$u")"; \
        # Diagnostic: print server response headers (helpful when downloads return 0 bytes) \
        wget --server-response --spider --timeout=15 --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" "$u" || true; \
        until [ "$n" -ge 5 ]; do \
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
    done; \
    # ----- Lyrics ONNX bundle (e5 + Marian + Whisper, ~2 GB) ----- \
    lyrics_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model/lyrics_model_onnx.tar.gz"; \
    lyrics_dest="/tmp/lyrics_model_onnx.tar.gz"; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "$lyrics_dest" "$lyrics_url"; then \
            echo "Downloaded lyrics ONNX bundle -> $lyrics_dest"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "wget attempt $n for lyrics ONNX bundle failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: failed to download lyrics_model_onnx.tar.gz after 5 attempts"; \
        exit 1; \
    fi; \
    # The archive contains a top-level ``model/`` directory. Strip that level so \
    # the contents land directly under /app/model/. \
    echo "Extracting lyrics ONNX bundle to /app/model..."; \
    tar -xzf "$lyrics_dest" -C /app/model --strip-components=1; \
    rm -f "$lyrics_dest"; \
    # The release archive ships e5-base-v2.onnx (weights) but does NOT bundle \
    # the e5-base-v2/ tokenizer directory. Fetch the 5 small tokenizer files \
    # (~3 MB total) from HF Hub if they aren't already there. \
    if [ ! -f /app/model/e5-base-v2/tokenizer.json ]; then \
        echo "e5-base-v2 tokenizer not in bundle; fetching from HF Hub..."; \
        mkdir -p /app/model/e5-base-v2; \
        e5_base="https://huggingface.co/intfloat/e5-base-v2/resolve/main"; \
        for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt; do \
            n=0; \
            until [ "$n" -ge 5 ]; do \
                if wget --no-verbose --tries=3 --retry-connrefused --waitretry=5 \
                    --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
                    -O "/app/model/e5-base-v2/$f" "$e5_base/$f"; then \
                    echo "Downloaded e5 tokenizer file: $f"; \
                    break; \
                fi; \
                n=$((n+1)); \
                echo "wget attempt $n for e5 $f failed — retrying in $((n*n))s"; \
                sleep $((n*n)); \
            done; \
            if [ "$n" -ge 5 ]; then \
                echo "ERROR: failed to download e5 tokenizer file $f after 5 attempts"; \
                exit 1; \
            fi; \
        done; \
    fi; \
    # Verify all required lyrics artifacts ended up in the right place. \
    for required in \
        /app/model/e5-base-v2.onnx \
        /app/model/e5-base-v2/tokenizer.json \
        /app/model/opus-mt-mul-en-onnx/encoder_model.onnx \
        /app/model/opus-mt-mul-en-onnx/decoder_model.onnx \
        /app/model/whisper-small-onnx/encoder_model.onnx \
        /app/model/whisper-small-onnx/decoder_model.onnx \
    ; do \
        if [ ! -f "$required" ]; then \
            echo "ERROR: expected lyrics ONNX artifact missing after extraction: $required"; \
            ls -laR /app/model | head -100; \
            exit 1; \
        fi; \
    done; \
    echo "✓ Lyrics ONNX bundle ready at /app/model/"

# NOTE: CLAP model download moved to runner stage to avoid EOF errors with large file transfers in multi-arch builds

# ============================================================================
# Stage 2a: runtime-base — minimal RUNTIME-ONLY system libs (used by `runner`)
# ============================================================================
# This stage intentionally has NO compilers, NO -dev headers, NO cuda-compiler,
# NO git/vim/strace. It only contains the shared libraries Python wheels load
# at runtime + the small set of tools the entrypoint/supervisord need.
# Removing the build toolchain saves ~500 MB on CPU and ~2.5 GB on GPU images.
FROM ${BASE_IMAGE} AS runtime-base

ARG BASE_IMAGE

SHELL ["/bin/bash", "-c"]

RUN set -ux; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            python3 python3-pip \
            libfftw3-double3=3.3.10-1ubuntu3 \
            libyaml-0-2=0.2.5-1build1 \
            libsamplerate0=0.2.2-4build1 \
            libsndfile1=1.2.2-1ubuntu5.24.04.1 \
            libopenblas0 \
            liblapack3 \
            libgomp1 \
            libpq5 postgresql-client \
            libjemalloc2 \
            ffmpeg wget curl ca-certificates \
            supervisor procps \
            redis-tools; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "apt-get attempt $n failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    apt-get remove -y python3-numpy || true && \
    apt-get autoremove -y --purge || true && \
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
    rm -rf /var/lib/apt/lists/*

# Symlink libjemalloc.so.2 to a stable, arch-agnostic path so LD_PRELOAD
# resolves regardless of build arch (x86_64 vs aarch64). Without this the
# bare "libjemalloc.so.2" preload sometimes fails silently and the process
# falls back to glibc malloc.
RUN set -eux; \
    arch_dir="/usr/lib/$(uname -m)-linux-gnu"; \
    test -f "$arch_dir/libjemalloc.so.2" || { \
        echo "ERROR: libjemalloc.so.2 not found at $arch_dir"; exit 1; \
    }; \
    ln -sf "$arch_dir/libjemalloc.so.2" /usr/local/lib/libjemalloc.so.2; \
    ldconfig

# ============================================================================
# Stage 2b: build-base — runtime-base + COMPILERS and -dev headers
# Used only by the `libraries` stage to build/install Python wheels. Never
# becomes part of the final runner image, so its weight is irrelevant to
# the published image size.
# ============================================================================
FROM runtime-base AS build-base

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
            gcc g++ \
            binutils \
            "$(if [[ "$BASE_IMAGE" =~ ^nvidia/cuda:([0-9]+)\.([0-9]+).+$ ]]; then echo "cuda-compiler-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"; fi)"; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "apt-get attempt $n failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    rm -rf /var/lib/apt/lists/*

# Backwards-compatible alias: some external tooling/consumers may target the
# legacy `base` stage name. Keep it as an alias of `build-base`.
FROM build-base AS base

# ============================================================================
# Stage 3: Libraries - Python packages installation
# ============================================================================
FROM build-base AS libraries

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
    && find /usr/local/lib/python3.12/dist-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete \
    # === Aggressive size cleanup of installed Python packages ===
    # 1) Drop test/example/doc dirs that ship inside many wheels (torch, scipy,
    #    transformers, llama_cpp, sklearn, …). Keeps the runtime, removes ballast.
    && find /usr/local/lib/python3.12/dist-packages \
        -type d \( -name tests -o -name test -o -name testing \
                  -o -name examples -o -name example \
                  -o -name docs -o -name doc \
                  -o -name "*.dist-info" -prune -false \) \
        -prune -exec rm -rf {} + 2>/dev/null || true \
    # 2) Strip debug symbols from every shared object. Saves hundreds of MB on
    #    GPU images (torch, onnxruntime, cupy ship with full symbol tables).
    #    `--strip-unneeded` is safe for runtime use.
    && find /usr/local/lib/python3.12/dist-packages -type f \
            \( -name "*.so" -o -name "*.so.*" \) \
            -exec strip --strip-unneeded {} + 2>/dev/null || true \
    # 3) Drop wheel install metadata not needed at runtime.
    && find /usr/local/lib/python3.12/dist-packages -type d -name "*.dist-info" \
        -exec sh -c 'rm -f "$1/RECORD" "$1/INSTALLER" "$1/REQUESTED" "$1/direct_url.json" "$1/zip-safe"' _ {} \; 2>/dev/null || true \
    # 4) Drop torch internals only used to build C++/CUDA extensions or for dev
    #    tooling. Saves ~500 MB on CPU images and ~700 MB on GPU.
    #    - torch/include       (~200 MB of C++ headers, build-time only)
    #    - torch/share/cmake   (CMake config for downstream extensions)
    #    - torch/test          (torch's own test suite)
    #    - torch/utils/benchmark (perf-tooling, not used by the app)
    && rm -rf /usr/local/lib/python3.12/dist-packages/torch/include \
              /usr/local/lib/python3.12/dist-packages/torch/share/cmake \
              /usr/local/lib/python3.12/dist-packages/torch/test \
              /usr/local/lib/python3.12/dist-packages/torch/utils/benchmark \
              2>/dev/null || true \
    # 5) Drop static archives and C/C++ header files across all wheels.
    #    Saves ~100-150 MB; these are build-time artifacts not used at runtime.
    #    DO NOT include "*.pyi" here — librosa, scipy, networkx and other
    #    scientific packages use lazy_loader.attach_stub() which reads .pyi
    #    files as RUNTIME data to wire up their lazy-loaded public API.
    #    Removing them produces:
    #        ValueError: Cannot load imports from non-existent stub '__init__.pyi'
    && find /usr/local/lib/python3.12/dist-packages -type f \
            \( -name "*.a" -o -name "*.h" -o -name "*.hpp" \) \
            -delete 2>/dev/null || true \
    # 6) Strip uv cache leftovers
    && rm -rf /root/.cache /tmp/* /var/tmp/* 2>/dev/null || true

# Download HuggingFace models (BERT, RoBERTa, BART) from GitHub release
# These are the text encoders needed by laion-clap library for text embeddings.
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
# IMPORTANT: this stage extends `runtime-base` (NOT `build-base`). That keeps
# compilers, -dev headers and `cuda-compiler-XX-X` *out* of the final image.
# Saves ~500 MB on CPU and ~2-3 GB on GPU.
FROM runtime-base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    HF_XET_DISABLE=1 \
    LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:1000,muzzy_decay_ms:0

# Note: bundled HuggingFace models (RoBERTa, ...) load with
# local_files_only=True per call. e5 + Marian (opus-mt-mul-en) + whisper-small
# come pre-exported to ONNX from the v4 release ``lyrics_model_onnx.tar.gz``
# bundle and are loaded via raw onnxruntime — no torch is required at runtime
# on the CPU image.

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

# Copy all downloaded/extracted models from the models stage. This includes
# the lyrics_model_onnx.tar.gz extraction (e5-base-v2.onnx + e5-base-v2/
# tokenizer + opus-mt-mul-en-onnx/ + whisper-small-onnx/) plus musicnn ONNX
# and silero_vad.onnx.
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

# MuLan support has been removed; the MuQ-MuLan ONNX download block (~2.5 GB)
# that lived here is gone. CLAP + lyrics cover the same use case.

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
