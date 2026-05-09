# GPU deployment

NVidia GPU **EXPERIMENTAL** support is available for analysis task in the worker process. This can significantly speed up processing of tracks.

We suggest **8GB VRAM** on GPU, with less you can experience the NON BLOCKING OutOFMemory error (that are handled by switching to CPU). The `PER_SONG_MODEL_RELOAD` env variable, that by default is TRUE, help cleaning the memory by entirely reloading the model each time, on the other side it slow the analysis process.


**NEW:** GPU-accelerated clustering is now available using RAPIDS cuML. This can provide **10-30x speedup** for clustering tasks.

**Features:**
- GPU-accelerated KMeans, DBSCAN, and PCA using RAPIDS cuML
- Automatic fallback to CPU if GPU is unavailable or encounters errors
- Supports all existing clustering configurations and parameters
- Compatible with NVIDIA GPUs with CUDA 12.8.1+ (*)

(*) Old driver are NOT supported from the actual build but you can try on your own to build your image like in https://github.com/NeptuneHub/AudioMuse-AI/issues/265

**To enable GPU clustering:**

1. Use the NVIDIA Docker image (e.g., `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`)
2. Set environment variable in your `.env` file:
   ```
   USE_GPU_CLUSTERING=true
   ```
3. Ensure NVIDIA Container Toolkit is installed on your host
4. Use docker-compose files with GPU support (e.g., `docker-compose-nvidia.yaml` or `docker-compose-worker-nvidia.yaml`)

**Performance Impact:**
- **KMeans**: 10-50x faster than CPU
- **DBSCAN**: 5-100x faster than CPU
- **PCA**: 10-40x faster than CPU
- **Overall clustering task**: 10-30x speedup for typical workloads (5000 iterations)

**Example:** A clustering task that takes 2-4 hours on CPU may complete in 5-15 minutes on GPU.

**Notes:**
- GaussianMixture and SpectralClustering use CPU (no GPU implementation available)
- GPU clustering is disabled by default (`USE_GPU_CLUSTERING=false`)
- GPU is already used for audio analysis models (ONNX inference)

## AMD GPU (ROCm)

AMD ROCm support is available for the ONNX-based audio analysis models
(MusiCNN, CLAP, MuLan). Lyrics transcription (PyTorch / Whisper) and GPU clustering (RAPIDS cuML) remain NVIDIA-only on the ROCm image — they fall back to CPU and scikit-learn respectively.

**Verified hardware:**
- Steam Deck (gfx1033 / Van Gogh APU) with `HSA_OVERRIDE_GFX_VERSION=10.3.0`

**Build the image (no CI tag yet — local build required):**

```bash
docker build \
  --build-arg BASE_IMAGE=rocm/dev-ubuntu-24.04:6.4.2 \
  -t ghcr.io/neptunehub/audiomuse-ai:latest-amd .
```

The `BASE_IMAGE` regex in the Dockerfile auto-detects any `rocm/...` base and installs `requirements/rocm.txt` (`onnxruntime-rocm`) instead of the CUDA or CPU variants. If you target a different ROCm release, edit `requirements/rocm.txt` and pin both the index URL and `onnxruntime-rocm` version to a wheel published at [repo.radeon.com/rocm/manylinux/](https://repo.radeon.com/rocm/manylinux/).

**Run with the AMD compose file:**

```bash
cd deployment
docker compose -f docker-compose-amd.yaml up -d
```

The compose file mounts `/dev/kfd` and `/dev/dri/renderD128`, adds the
`video` and `render` groups, and sets the HSA env vars Steam Deck needs. For non-Deck AMD GPUs override `HSA_OVERRIDE_GFX_VERSION` (e.g. `11.0.0` for RDNA3) in your `.env`.

**Verify the worker actually picked ROCm:**

```bash
docker exec audiomuse-ai-worker-instance python3 -c "
import onnxruntime as ort
print(ort.get_available_providers())
"
# Expect: ['ROCMExecutionProvider', 'CPUExecutionProvider', ...]

docker logs audiomuse-ai-worker-instance | grep -i provider
# Expect: "ONNX Runtime providers available: [...]; preferred: ROCMExecutionProvider"
```

**Limitations on AMD:**
- Clustering uses scikit-learn (cuML is CUDA-only). Leave `USE_GPU_CLUSTERING=false`.
- Lyrics (Whisper / Qwen) run on CPU — PyTorch ROCm wheels are not bundled in this image.