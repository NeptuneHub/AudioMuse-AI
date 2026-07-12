# AMD GPU deployment (ROCm / MIGraphX)

AMD GPU **EXPERIMENTAL** support for the analysis task in the worker process,
using the ROCm MIGraphX execution provider. This can significantly speed up
processing of tracks on AMD hardware, the same way the NVIDIA image does for
CUDA (see [GPU.md](GPU.md) for the NVIDIA path).

It ships as a **second image** built from `Dockerfile-rocm`, separate from the
CPU / NVIDIA `Dockerfile`. Tested on a Radeon RX 9070 XT (gfx1201 / RDNA4).

## What is accelerated (and what is not)

Not every model can run on MIGraphX -- ONNX Runtime dropped its ROCm execution
provider, and MIGraphX (the ROCm 7.x replacement) can't parse every graph:

- **musicnn** (mood/embedding CNN) runs on GPU via `MIGraphXExecutionProvider`.
  Main win on AMD.
- **CLAP audio analysis** (DCLAP) runs on **CPU** here. Its `Resize` op with
  `keep_aspect_ratio_policy` isn't parseable by MIGraphX. Still GPU on CUDA.
- **GPU clustering** (RAPIDS cuML) is **NVIDIA-only** -- no ROCm port. Leave
  `USE_GPU_CLUSTERING=false` on this image (CPU fallback is automatic).
- **Lyrics transcription (Whisper) does run on GPU here**, but not via ONNX:
  the decoder's dynamic `If`/KV-cache subgraphs aren't parseable by MIGraphX
  either, so this image uses **faster-whisper/CTranslate2** instead, which has
  a native ROCm HIP backend. Selected via `LYRICS_WHISPER_BACKEND=faster`
  (baked into the image); falls back to CPU/int8 if GPU load fails. NVIDIA/CPU
  images keep the default ONNX Whisper backend.

## Requirements

- AMD GPU supported by ROCm 7.x. RDNA4 (gfx1201, e.g. RX 9070 XT) is natively
  supported -- no `HSA_OVERRIDE_GFX_VERSION` needed. Older/unsupported cards may
  need that override; see AMD's ROCm compatibility matrix.
- ROCm-capable kernel driver (`amdgpu`) on the host.
- ~8 GB VRAM recommended (same guidance as the NVIDIA image; on less you may hit
  the non-blocking OutOfMemory path that falls back to CPU).
- Docker with `/dev/kfd` and `/dev/dri` available on the host.

## Quick start

From the repository root:

```bash
docker compose -f deployment/docker-compose-rocm.yaml up --build
```

This builds `Dockerfile-rocm` locally and starts Redis, PostgreSQL, the Flask
app, and a GPU-enabled worker.

### GPU device permissions (important)

The compose file mounts `/dev/kfd` and `/dev/dri` and adds the host's
device-owning groups by **numeric GID** (the base image has no `render`
group by name). Find your host's GIDs:

```bash
getent group render video
```

`render` owns `/dev/kfd` + `/dev/dri/renderD*` (compute, the one that
matters); `video` owns `/dev/dri/card*` (display).

Compose defaults to `RENDER_GID=105` and `VIDEO_GID=39`. If yours differ, set
them (e.g. in `deployment/.env` or the shell):

```bash
RENDER_GID=$(getent group render | cut -d: -f3) \
VIDEO_GID=$(getent group video | cut -d: -f3) \
docker compose -f deployment/docker-compose-rocm.yaml up --build
```

## Verify the GPU is used

```bash
# GPU visible inside the worker
docker exec audiomuse-ai-worker-rocm rocminfo | grep -i gfx

# MIGraphX provider available to ONNX Runtime
docker exec audiomuse-ai-worker-rocm /opt/venv/bin/python3 \
  -c "import onnxruntime as o; print(o.get_available_providers())"
# -> ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

When an analysis run starts, the worker logs `ONNX provider chain: ['MIGraphXExecutionProvider', ...]`.
The first inference compiles the models for MIGraphX (a one-off delay); the
MIOpen kernel cache is persisted to a Docker volume so restarts skip recompiles.

## Environment (baked into the image)

`Dockerfile-rocm` sets these so you do not have to:

- `HSA_ENABLE_INTERRUPT=1` -- interrupt-driven GPU completion instead of CPU
  busy-polling, keeping worker CPU free.
- `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR=/app/.cache/miopen` -- kernel
  compile cache dir, mounted as a volume by compose to persist across restarts.
- `LYRICS_WHISPER_BACKEND=faster` -- routes lyrics ASR to faster-whisper/
  CTranslate2 (GPU). Override to `onnx` to force CPU. Also tunable:
  `LYRICS_WHISPER_FASTER_DEVICE` (default `cuda`), `LYRICS_WHISPER_FASTER_COMPUTE_TYPE`
  (default `float16`), `LYRICS_WHISPER_FASTER_MODEL_DIR`
  (default `/app/model/faster-whisper-small`).

To pin to a specific GPU on a multi-GPU host, set `HIP_VISIBLE_DEVICES` on the
worker service (commented example in the compose file).

## Notes

- ONNX Runtime with the MIGraphX provider is supplied by the
  `rocm/onnxruntime` base image itself; the image does not re-install onnxruntime
  (doing so would replace the GPU build with a CPU-only one).
- ML models and the HuggingFace cache are copied from the published CPU image
  (`ghcr.io/neptunehub/audiomuse-ai:latest`) at build time.
- To move ROCm versions, bump `BASE_IMAGE` in `Dockerfile-rocm` to a tag from
  <https://hub.docker.com/r/rocm/onnxruntime>.
