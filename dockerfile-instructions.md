# AudioMuse-AI Docker Instructions

## Quick Start: Build Commands

### 1. GPU Build (Recommended)
Builds with NVIDIA CUDA support. Requires ~25-35 mins for the first build.

```bash
# Clean up old image
docker rmi audiomuse-ai:local-nvidia 2>/dev/null

# Build with BuildKit (Required)
DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04 \
  -t audiomuse-ai:local-nvidia .
```

### 2. CPU-Only Build
Smaller image, no GPU acceleration.

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  -t audiomuse-ai:local-cpu .
```

## Key Optimizations
- **Multi-Stage Build**: Separates build tools (compilers, headers) from the runtime image.
- **Runtime Compilation**: For NVIDIA builds, the CUDA compiler is installed in the runtime stage to support `cupy` JIT compilation (fixing `cuda_fp16.h` errors).
- **Caching**:
  - **Pip Cache**: Uses BuildKit mounts to cache pip downloads (`--mount=type=cache,target=/root/.cache/pip`).
  - **Model Cache**: Models are downloaded in a separate stage to avoid re-downloading on code changes.
- **Size Reduction**:
  - Removed unused build tools (gcc, git, vim) from runtime.
  - Cleaned up Python bytecode (`__pycache__`, `.pyc`).
  - Reduced image size from ~22GB to ~6-8GB (depending on CUDA components).

## Troubleshooting

### "catastrophic error: cannot open source file 'cuda_fp16.h'"
**Cause**: Missing CUDA headers in the runtime image.
**Fix**: Ensure you are building with the latest Dockerfile which includes the conditional installation of `cuda-compiler` in the runtime stage.

### Build is Slow
**Cause**: First-time build downloads large CUDA images and compiles Python packages.
**Fix**: Subsequent builds will be fast (<2 mins) due to caching. Ensure `DOCKER_BUILDKIT=1` is set.

### GPU Not Detected
**Check**:
1. Run `nvidia-smi` on host.
2. Ensure `nvidia-container-toolkit` is installed.
3. Run with GPU flags: `docker run --rm --gpus all ...`
