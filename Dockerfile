# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:22.04
FROM ubuntu:22.04 AS models

SHELL ["/bin/bash", "-lc"]

# Tip: If you can, host these models in a GCS/S3 bucket or single zip to speed this up.
# Existing logic kept, but ensure this stage is only invalidated if URLs change.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget ca-certificates curl

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
    fname="/app/model/$(basename "$u")"; \
    if [ -f "$fname" ]; then continue; fi; \
    wget --no-verbose --tries=3 --retry-connrefused --waitretry=5 --header="User-Agent: AudioMuse-Docker/1.0" -O "$fname" "$u" || exit 1; \
    done

FROM ${BASE_IMAGE} AS base
ARG BASE_IMAGE
SHELL ["/bin/bash", "-c"]

# Utilize caching for apt to speed up system dependency installation
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -ux; \
    apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      libfftw3-3 libyaml-0-2 libsamplerate0 \
      libsndfile1 \
      ffmpeg wget git vim \
      redis-tools curl \
      supervisor strace procps iputils-ping \
      libopenblas-dev liblapack-dev libpq-dev \
      gcc g++ \
      "$(if [[ "$BASE_IMAGE" =~ ^nvidia/cuda:([0-9]+)\.([0-9]+).+$ ]]; then echo "cuda-compiler-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"; fi)"

FROM base AS libraries
ARG BASE_IMAGE

# REMOVED --no-cache-dir to allow pip caching
# Split installation to layer dependencies

# 1. Heavy GPU dependencies (Rarely change, take long to download)
RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ "$BASE_IMAGE" =~ ^nvidia/cuda: ]]; then \
      pip3 install cupy-cuda12x onnxruntime-gpu==1.19.2; \
      pip3 install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.12.*; \
    else \
      pip3 install onnxruntime==1.19.2; \
    fi

# 2. Standard Libraries (More likely to change)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
      numpy==1.26.4 \
      scipy==1.15.3 \
      numba==0.60.0 \
      soundfile==0.13.1 \
      Flask Flask-Cors redis requests \
      scikit-learn==1.7.2 rq pyyaml six \
      voyager==2.1.0 rapidfuzz psycopg2-binary \
      ftfy flasgger sqlglot google-generativeai \
      mistralai umap-learn pydub python-mpd2 \
      onnx==1.14.1 resampy librosa==0.11.0 \
      flatbuffers packaging protobuf sympy

FROM base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy pre-installed libraries
COPY --from=libraries /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Copy models (cached from stage 1)
COPY --from=models /app/model/ /app/model/

# Copy source code LAST so it doesn't invalidate previous layers
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY . /app

# ... (Environment variables remain the same) ...

ENV ORT_DISABLE_ALL_OPTIMIZATIONS=1
ENV ORT_ENABLE_CPU_FP16_OPS=0
ENV ORT_DISABLE_AVX512=1
ENV ORT_FORCE_SHARED_PROVIDER=1
ENV MKL_ENABLE_INSTRUCTIONS=AVX2
ENV MKL_DYNAMIC=FALSE
ENV ORT_DISABLE_MEMORY_PATTERN_OPTIMIZATION=1
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
CMD ["bash", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then echo 'Starting worker processes via supervisord...' && /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf; else echo 'Starting web service...' && python3 /app/app.py; fi"]