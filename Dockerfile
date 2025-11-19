# or nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 (CUDA version can go as low as CUDA 12.2 but need to check)
ARG BASE_IMAGE=ubuntu:22.04
FROM ubuntu:22.04 AS models

SHELL ["/bin/bash", "-lc"]

RUN mkdir -p /app/model

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
    # Diagnostic: print server response headers (helpful when downloads return 0 bytes)
    wget --server-response --spider --timeout=15 --header="User-Agent: AudioMuse-Docker/1.0 (+https://github.com/NeptuneHub/AudioMuse-AI)" "$u" || true; \
    until [ "$n" -ge 5 ]; do \
      # Use wget with retries. --tries and --waitretry add backoff for transient failures.
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

FROM ${BASE_IMAGE} AS base

ARG BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# Install system dependencies, including ffmpeg which is crucial for pydub
# cuda-compiler is only for libdevice.10.bc, can be extracted into another stage
RUN set -ux; \
  n=0; \
  until [ "$n" -ge 5 ]; do \
    if apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      libfftw3-3=3.3.8-2ubuntu8 libyaml-0-2 libsamplerate0 \
      libsndfile1=1.0.31-2ubuntu0.2 \
      ffmpeg wget git vim \
      redis-tools curl \
      supervisor \
      strace \
      procps \
      iputils-ping \
      libopenblas-dev=0.3.20+ds-1 \
      liblapack-dev=3.10.0-2ubuntu1 \
      libpq-dev \
      gcc \
      g++ \
      "$(if [[ "$BASE_IMAGE" =~ ^nvidia/cuda:([0-9]+)\.([0-9]+).+$ ]]; then echo "cuda-compiler-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"; fi)"; then \
      break; \
    fi; \
    n=$((n+1)); \
    echo "apt-get attempt $n failed — retrying in $((n*n))s"; \
    sleep $((n*n)); \
  done; \
  rm -rf /var/lib/apt/lists/* \
  apt-get remove -y python3-numpy || true; \
  apt-get autoremove -y || true;

#RUN test -f /usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc

FROM base AS libraries

ARG TARGETARCH
ARG BASE_IMAGE

# pydub is for audio conversion
# Pin numpy to a stable version to avoid numeric differences between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    bash -lc '\
      GPU_PKGS="flatbuffers packaging protobuf sympy"; \
      pip3 install --no-cache-dir numpy==1.26.4 || exit 1; \
      pip3 install --no-cache-dir \
        scipy==1.15.3 \
        numba==0.60.0 \
        soundfile==0.13.1 \
        Flask \
        Flask-Cors \
        redis \
        requests \
        scikit-learn==1.7.2 \
        rq \
        pyyaml \
        six \
        voyager==2.1.0 \
        rapidfuzz \
        psycopg2-binary \
        ftfy \
        flasgger \
        sqlglot \
        google-generativeai \
        mistralai \
        umap-learn \
        pydub \
        python-mpd2 \
        onnx==1.14.1 \
        resampy \
        librosa==0.11.0 || exit 1; \
      if [[ "${BASE_IMAGE}" =~ ^nvidia/cuda: ]]; then \
        echo "Detected NVIDIA base image: installing GPU-only packages, onnxruntime-gpu, and RAPIDS cuML"; \
        pip3 install --no-cache-dir $GPU_PKGS || exit 1; \
        pip3 install --no-cache-dir onnxruntime-gpu==1.19.2 || exit 1; \
        echo "Installing RAPIDS cuML for GPU-accelerated clustering..."; \
        pip3 install --no-cache-dir cupy-cuda12x || exit 1; \
        pip3 install --no-cache-dir --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.12.* || exit 1; \
      else \
        echo "CPU base image: installing onnxruntime (CPU) only"; \
        pip3 install --no-cache-dir onnxruntime==1.19.2 || exit 1; \
      fi'


FROM base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# COPY --from=libraries /install/ /usr/
COPY --from=libraries /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

COPY --from=models /app/model/ /app/model/

COPY . /app

COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# oneDNN floating-point math mode: STRICT reduces non-deterministic FP optimizations
# Keep this if you want more deterministic CPU behavior when using oneDNN-enabled runtimes
ENV ONEDNN_DEFAULT_FPMATH_MODE=STRICT

# ONNX Runtime optimization settings to prevent signal 9 crashes on newer CPUs (Intel 12600K, etc.)
# Similar to TF_ENABLE_ONEDNN_OPTS=0 for TensorFlow compatibility
ENV ORT_DISABLE_ALL_OPTIMIZATIONS=1
ENV ORT_ENABLE_CPU_FP16_OPS=0

# Force consistent memory allocation and precision behavior (Intel 12600K vs i5-6500 compatibility)
# Prevents different memory allocation patterns and floating-point precision issues
ENV ORT_DISABLE_AVX512=1
ENV ORT_FORCE_SHARED_PROVIDER=1

# Force consistent floating-point behavior across different Intel generations
# 12600K has different FPU precision defaults than 6th gen CPUs
ENV MKL_ENABLE_INSTRUCTIONS=AVX2
ENV MKL_DYNAMIC=FALSE

# Prevent aggressive memory pre-allocation on newer CPUs
ENV ORT_DISABLE_MEMORY_PATTERN_OPTIMIZATION=1

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
CMD ["bash", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then echo 'Starting worker processes via supervisord...' && /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf; else echo 'Starting web service...' && python3 /app/app.py; fi"]
