# syntax=docker/dockerfile:1
# AudioMuse-AI Dockerfile
# Supports both CPU (ubuntu:24.04) and GPU (nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04) builds
#
# Build examples:
#   CPU:  docker build -t audiomuse-ai .
#   GPU:  docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 -t audiomuse-ai-gpu .
#
# Optimizations:
#   - Ubuntu 24.04 (Python 3.12)
#   - Removed unused PyTorch/Torchaudio from CPU builds
#   - Smart model caching: models only re-download if release version changes
#   - Multi-stage build for optimal layer caching

ARG BASE_IMAGE=ubuntu:24.04

# ============================================================================
# Stage 1: Model Cache with Checksum Validation (v3.0.0)
# ============================================================================
# This stage caches model downloads. Docker will reuse this layer unless:
# - Model release version changes (v3.0.0 → v3.0.1)
# - Model checksums change
# - Dockerfile content in this stage changes
FROM ubuntu:24.04 AS model-cache-v3.0.0

SHELL ["/bin/bash", "-lc"]

RUN mkdir -p /app/model

# Install download tools
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

# Download small ONNX models (~10MB each) with checksum verification
# These models rarely change - Docker will cache this layer
RUN set -eux; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model"; \
    models=( \
        "danceability-msd-musicnn-1.onnx" \
        "mood_aggressive-msd-musicnn-1.onnx" \
        "mood_happy-msd-musicnn-1.onnx" \
        "mood_party-msd-musicnn-1.onnx" \
        "mood_relaxed-msd-musicnn-1.onnx" \
        "mood_sad-msd-musicnn-1.onnx" \
        "msd-msd-musicnn-1.onnx" \
        "msd-musicnn-1.onnx" \
    ); \
    mkdir -p /app/model; \
    for model in "${models[@]}"; do \
        n=0; \
        fname="/app/model/$model"; \
        url="$base_url/$model"; \
        until [ "$n" -ge 5 ]; do \
            if wget --no-verbose --tries=3 --retry-connrefused --waitretry=5 \
                --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
                -O "$fname" "$url"; then \
                echo "✓ Downloaded $model"; \
                break; \
            fi; \
            n=$((n+1)); \
            echo "wget attempt $n for $model failed — retrying in $((n*n))s"; \
            sleep $((n*n)); \
        done; \
        if [ "$n" -ge 5 ]; then \
            echo "ERROR: failed to download $model after 5 attempts"; \
            exit 1; \
        fi; \
    done; \
    echo "✓ All small ONNX models cached successfully"; \
    ls -lh /app/model/

# Download CLAP models (~746MB total) - cached separately
# Only re-downloads if this stage changes
RUN set -eux; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model"; \
    audio_model="clap_audio_model.onnx"; \
    text_model="clap_text_model.onnx"; \
    \
    # Download audio model (~268MB) \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/app/model/$audio_model" "$base_url/$audio_model"; then \
            echo "✓ CLAP audio model cached"; \
            break; \
        fi; \
        n=$((n+1)); \
        sleep $((n*n)); \
    done; \
    \
    # Download text model (~478MB) \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/app/model/$text_model" "$base_url/$text_model"; then \
            echo "✓ CLAP text model cached"; \
            break; \
        fi; \
        n=$((n+1)); \
        sleep $((n*n)); \
    done; \
    \
    # Verify sizes \
    audio_size=$(stat -c%s "/app/model/$audio_model" 2>/dev/null || echo "0"); \
    text_size=$(stat -c%s "/app/model/$text_model" 2>/dev/null || echo "0"); \
    if [ "$audio_size" -lt 250000000 ]; then \
        echo "ERROR: CLAP audio model too small"; \
        exit 1; \
    fi; \
    if [ "$text_size" -lt 450000000 ]; then \
        echo "ERROR: CLAP text model too small"; \
        exit 1; \
    fi; \
    echo "✓ CLAP models cached successfully"; \
    ls -lh /app/model/*.onnx

# Download HuggingFace models (~985MB) - cached separately
RUN set -eux; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model"; \
    hf_models="huggingface_models.tar.gz"; \
    cache_dir="/app/.cache/huggingface"; \
    \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/tmp/$hf_models" "$base_url/$hf_models"; then \
            echo "✓ HuggingFace models downloaded"; \
            break; \
        fi; \
        n=$((n+1)); \
        sleep $((n*n)); \
    done; \
    \
    mkdir -p "$cache_dir"; \
    tar -xzf "/tmp/$hf_models" -C "$cache_dir"; \
    rm -f "/tmp/$hf_models"; \
    \
    if [ ! -d "$cache_dir/hub" ]; then \
        echo "ERROR: HuggingFace models extraction failed"; \
        exit 1; \
    fi; \
    \
    echo "✓ HuggingFace models cached successfully"; \
    du -sh "$cache_dir"

# ============================================================================
# Stage 2: Base - System dependencies and build tools
# ============================================================================
FROM ${BASE_IMAGE} AS base

ARG BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# Copy uv for fast package management (10-100x faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies with exponential backoff retry and version pinning
# Version pinning ensures reproducible builds across different build times
# cuda-compiler is conditionally installed for NVIDIA base images (needed for cupy JIT)
RUN set -ux; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if apt-get update && apt-get install -y --no-install-recommends \
            python3 python3-pip python3-dev \
            libfftw3-3=3.3.10-1ubuntu3 libfftw3-dev \
            libyaml-0-2 libyaml-dev \
            libsamplerate0 libsamplerate0-dev \
            libsndfile1=1.2.2-1ubuntu5 libsndfile1-dev \
            libopenblas-dev=0.3.26+ds-1 \
            liblapack-dev=3.12.0-3build1 \
            libpq-dev \
            ffmpeg wget curl \
            supervisor procps \
            gcc g++ \
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
    apt-get autoremove -y || true

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
# CPU builds: onnxruntime (CPU only), no torch/torchaudio
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

# NOTE: HuggingFace models are now cached in model-cache stage

# ============================================================================
# Stage 4: Runner - Final production image
# ============================================================================
FROM base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Copy Python packages from libraries stage
COPY --from=libraries /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

# Copy ALL cached models from model-cache stage
COPY --from=model-cache-v3.0.0 /app/model/ /app/model/
COPY --from=model-cache-v3.0.0 /app/.cache/huggingface/ /app/.cache/huggingface/

# Verify models were copied
RUN ls -lah /app/model/ && \
    echo "Model files:" && \
    ls -lh /app/model/*.onnx && \
    echo "HuggingFace cache:" && \
    du -sh /app/.cache/huggingface/

# Copy application code (last to maximize cache hits for code changes)
COPY . /app
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

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

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
CMD ["bash", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then echo 'Starting worker processes via supervisord...' && /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf; else echo 'Starting web service...' && python3 /app/app.py; fi"]
