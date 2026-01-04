# AudioMuse-AI Docker Instructions

## Quick Start: Build Commands

### 1. GPU Build (Recommended)
Builds with NVIDIA CUDA support. Requires ~25-35 mins for the first build.

```bash
# Clean up old image
docker rmi audiomuse-ai:local-nvidia 2>/dev/null

# Build with BuildKit (Required)
DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
  -t audiomuse-ai:local-nvidia .
```

### 2. CPU-Only Build
Smaller image, no GPU acceleration.

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE=ubuntu:24.04 \
  -t audiomuse-ai:local-cpu .
```

## Key Optimizations (v3 - Ubuntu 24.04)
- **Multi-Stage Build**: Separates build tools (compilers, headers) from the runtime image.
- **Ubuntu 24.04**: Latest LTS with Python 3.12 and updated dependencies.
- **Smart Model Caching**: All models downloaded in a separate Docker stage. Docker only re-downloads models if:
  - Model release version changes (v3.0.0 → v3.0.1)
  - Dockerfile model-cache stage changes
  - Code changes, dependency updates, or config changes do NOT trigger model re-downloads
- **Removed Unused Dependencies**: 
  - CPU builds: Removed PyTorch and Torchaudio (~1.5GB savings)
  - GPU builds: Removed Torchaudio (~500MB savings)
  - Main application uses ONNX Runtime for all inference
- **Runtime Compilation**: For NVIDIA builds, the CUDA compiler is installed in the runtime stage to support `cupy` JIT compilation (fixing `cuda_fp16.h` errors).
- **Size Reduction**:
  - Removed unused build tools (gcc, git, vim) from runtime.
  - Cleaned up Python bytecode (`__pycache__`, `.pyc`).
  - CPU image: ~5-6GB (down from ~6.5-8GB)
  - GPU image: ~6-8GB (depending on CUDA components)

## Build Time Improvements
- **First build**: Similar time (must download everything)
- **Rebuilds when code changes**: 90% faster (models cached by Docker)
- **Rebuilds when dependencies change**: 50% faster (models still cached)

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
