# syntax=docker/dockerfile:1
# AudioMuse-AI Dockerfile
# Supports both CPU (ubuntu:22.04) and GPU (nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04) builds
#
# Build examples:
#   CPU:  docker build -t audiomuse-ai .
#   GPU:  docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 -t audiomuse-ai-gpu .

ARG BASE_IMAGE=ubuntu:22.04

# ============================================================================
# Stage 1: Download ML models (cached separately for faster rebuilds)
# ============================================================================
FROM ubuntu:22.04 AS models

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

# Download ONNX models with diagnostics and retry logic
RUN set -eux; \
    urls=( \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/danceability-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/mood_aggressive-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/mood_happy-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/mood_party-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/mood_relaxed-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/mood_sad-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/msd-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v2.0.0-model/msd-musicnn-1.onnx" \
    ); \
    mkdir -p /app/model; \
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
            libfftw3-3=3.3.8-2ubuntu8 libfftw3-dev \
            libyaml-0-2 libyaml-dev \
            libsamplerate0 libsamplerate0-dev \
            libsndfile1=1.0.31-2ubuntu0.2 libsndfile1-dev \
            libopenblas-dev=0.3.20+ds-1 \
            liblapack-dev=3.10.0-2ubuntu1 \
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
# GPU builds: cupy, cuml, onnxruntime-gpu, voyager
# CPU builds: onnxruntime (CPU only)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [[ "$BASE_IMAGE" =~ ^nvidia/cuda: ]]; then \
        echo "NVIDIA base image detected: installing GPU packages (cupy, cuml, onnxruntime-gpu, voyager)"; \
        uv pip install --system -r /app/requirements/gpu.txt -r /app/requirements/common.txt; \
    else \
        echo "CPU base image: installing onnxruntime (CPU only)"; \
        uv pip install --system -r /app/requirements/cpu.txt -r /app/requirements/common.txt; \
    fi \
    && find /usr/local/lib/python3.10/dist-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.10/dist-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# ============================================================================
# Stage 4: Runner - Final production image
# ============================================================================
FROM base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy Python packages from libraries stage
COPY --from=libraries /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Copy models from models stage
COPY --from=models /app/model/ /app/model/

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
