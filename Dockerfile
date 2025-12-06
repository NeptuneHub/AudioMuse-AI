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
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/danceability-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/mood_aggressive-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/mood_happy-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/mood_party-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/mood_relaxed-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/mood_sad-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/msd-msd-musicnn-1.onnx" \
        "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/msd-musicnn-1.onnx" \
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

# Download CLAP model from GitHub releases (split into 2 parts due to size limit)
# Model: music_audioset_epoch_15_esc_90.14.pt (2.2GB) split as clap_model_part.aa + clap_model_part.ab
RUN set -eux; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model"; \
    part_aa="clap_model_part.aa"; \
    part_ab="clap_model_part.ab"; \
    merged_file="/app/model/music_audioset_epoch_15_esc_90.14.pt"; \
    echo "Downloading CLAP model parts (~2.2GB total)..."; \
    \
    # Download part aa with retry logic \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/tmp/$part_aa" "$base_url/$part_aa"; then \
            echo "Downloaded part aa"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "Download attempt $n for part aa failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: Failed to download CLAP model part aa after 5 attempts"; \
        exit 1; \
    fi; \
    \
    # Download part ab with retry logic \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget --no-verbose --tries=3 --retry-connrefused --waitretry=10 \
            --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" \
            -O "/tmp/$part_ab" "$base_url/$part_ab"; then \
            echo "Downloaded part ab"; \
            break; \
        fi; \
        n=$((n+1)); \
        echo "Download attempt $n for part ab failed — retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    if [ "$n" -ge 5 ]; then \
        echo "ERROR: Failed to download CLAP model part ab after 5 attempts"; \
        rm -f "/tmp/$part_aa"; \
        exit 1; \
    fi; \
    \
    # Merge the parts \
    echo "Merging CLAP model parts..."; \
    cat "/tmp/$part_aa" "/tmp/$part_ab" > "$merged_file"; \
    \
    # Verify merged file exists and has reasonable size \
    if [ ! -f "$merged_file" ]; then \
        echo "ERROR: Merged CLAP model file not created"; \
        exit 1; \
    fi; \
    \
    file_size=$(stat -c%s "$merged_file" 2>/dev/null || stat -f%z "$merged_file" 2>/dev/null || echo "0"); \
    if [ "$file_size" -lt 2000000000 ]; then \
        echo "ERROR: Merged CLAP model file is too small (expected ~2.2GB, got $file_size bytes)"; \
        exit 1; \
    fi; \
    \
    # Clean up split parts to save space \
    echo "Cleaning up split parts..."; \
    rm -f "/tmp/$part_aa" "/tmp/$part_ab"; \
    \
    echo "CLAP model merged successfully -> $merged_file"; \
    ls -lh "$merged_file"

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
# Note: --index-strategy unsafe-best-match resolves conflicts between pypi.nvidia.com and pypi.org
RUN if [[ "$BASE_IMAGE" =~ ^nvidia/cuda: ]]; then \
        echo "NVIDIA base image detected: installing GPU packages (cupy, cuml, onnxruntime-gpu, voyager)"; \
        uv pip install --system --no-cache --index-strategy unsafe-best-match -r /app/requirements/gpu.txt -r /app/requirements/common.txt || exit 1; \
    else \
        echo "CPU base image: installing all packages together for dependency resolution"; \
        uv pip install --system --no-cache --index-strategy unsafe-best-match -r /app/requirements/cpu.txt -r /app/requirements/common.txt || exit 1; \
    fi \
    && echo "Verifying psycopg2 installation..." \
    && python3 -c "import psycopg2; print('psycopg2 OK')" \
    && echo "Verifying torch installation..." \
    && python3 -c "import torch; print('torch OK')" \
    && find /usr/local/lib/python3.10/dist-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.10/dist-packages -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# Download HuggingFace models (BERT, RoBERTa, BART) from GitHub release
# These are the text encoders needed by laion-clap library for text embeddings
RUN set -eux; \
    base_url="https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model"; \
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
FROM base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Copy Python packages from libraries stage
COPY --from=libraries /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Copy HuggingFace cache (RoBERTa model) from libraries stage
COPY --from=libraries /app/.cache/huggingface/ /app/.cache/huggingface/

# Verify cache was copied correctly
RUN ls -lah /app/.cache/huggingface/ && \
    echo "HuggingFace cache contents:" && \
    du -sh /app/.cache/huggingface/* || echo "Cache directory empty!"

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
