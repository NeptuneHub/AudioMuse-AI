# or nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 (CUDA version can go as low as CUDA 12.2 but need to check)
ARG BASE_IMAGE=ubuntu:22.04

FROM ubuntu:22.04 AS models

RUN mkdir -p /app/model

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -P /app/model \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/danceability-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_aggressive-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_aggressive-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_happy-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_happy-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_party-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_party-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_relaxed-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_relaxed-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_sad-audioset-vggish-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/mood_sad-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/msd-msd-musicnn-1.pb \
    https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v1.0.0-model/msd-musicnn-1.pb

FROM ${BASE_IMAGE} AS base

ARG BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# cuda-compiler is only for libdevice.10.bc, can be extracted into another stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libfftw3-3 libyaml-0-2 libsamplerate0 \
    ffmpeg wget git vim \
    redis-tools curl \
    supervisor \
    strace \
    procps \
    iputils-ping \
    libopenblas-dev \
    liblapack-dev \
    libpq-dev \
    gcc \
    g++ \
    "$(if [[ "$BASE_IMAGE" =~ ^nvidia/cuda:([0-9]+)\.([0-9]+).+$ ]]; then echo "cuda-compiler-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"; fi)" \
    && rm -rf /var/lib/apt/lists/*

#RUN test -f /usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc

FROM base AS libraries

ARG TARGETARCH
ARG BASE_IMAGE

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --prefix=/install \
      Flask \
      Flask-Cors \
      redis \
      requests \
      scikit-learn \
      rq \
      pyyaml \
      six \
      voyager \
      rapidfuzz \
      psycopg2-binary \
      ftfy \
      flasgger \
      sqlglot \
      google-generativeai \
      "$(if [[ "$TARGETARCH" = "arm64" ]]; then \
        echo "tensorflow-aarch64==2.15.0"; \
      else \
        #echo "tensorflow[$(if [[ "$BASE_IMAGE" =~ nvidia ]]; then echo "and-cuda"; fi)]==2.20.0"; \
        echo "tensorflow==2.20.0"; \
      fi)" \
      librosa

FROM base AS runner

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY --from=libraries /install/ /usr/
COPY --from=models /app/model/ /app/model/

COPY . /app

COPY deplyment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Or it will take all available memory
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages:/app

EXPOSE 8000

WORKDIR /workspace
CMD ["bash", "-c", "if [ \"$SERVICE_TYPE\" = \"worker\" ]; then echo 'Starting worker processes via supervisord...' && /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf; else echo 'Starting web service...' && python3 /app/app.py; fi"]
