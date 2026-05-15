# GPU deployment

NVidia GPU **EXPERIMENTAL** support is available for analysis task in the worker process. This can significantly speed up processing of tracks.

---

## AMD ROCm / MIGraphX Worker

AMD GPU workers are supported via a dedicated `Dockerfile.rocm` and `docker-compose.amd-worker.yml`. These use the MIGraphX-accelerated ONNX Runtime build in place of the standard CPU wheel.

**Tested on:** RX 7900 XTX (gfx1100 / RDNA 3) with ROCm 7.2.3. Expected to work on RDNA 2 (RX 6000 series) with the GFX override below.

**Build and run:**

```bash
# Minimal build (no optional features)
docker build -f Dockerfile.rocm -t audiomuse-ai:rocm .

# With faster-whisper lyrics backend (CTranslate2-based)
docker build -f Dockerfile.rocm --build-arg WITH_FASTER_WHISPER=1 -t audiomuse-ai:rocm .

# With Essentia MTG-Jamendo mood classifiers
docker build -f Dockerfile.rocm --build-arg WITH_ESSENTIA_MOOD_MODELS=1 -t audiomuse-ai:rocm .

# Both optional features
docker build -f Dockerfile.rocm \
  --build-arg WITH_FASTER_WHISPER=1 \
  --build-arg WITH_ESSENTIA_MOOD_MODELS=1 \
  -t audiomuse-ai:rocm .

docker compose -f docker-compose.amd-worker.yml up -d --scale audiomuse-worker=3
```

**RDNA 2 / RDNA 3 GFX version override:**

Set `HSA_OVERRIDE_GFX_VERSION` in your `.env` or `docker-compose.amd-worker.yml` if your card is not auto-detected:

| Card generation | Value |
|---|---|
| RDNA 2 (RX 6000 series) | `10.3.0` |
| RDNA 3 (RX 7000 series) | `11.0.0` |

**Required host setup:**
- ROCm 6.x or 7.x host driver (`amdgpu-install`)
- `amdgpu` group membership: `sudo usermod -aG video,render $USER`
- `/dev/kfd` and `/dev/dri` available (mapped automatically by the compose file)

**Performance (RX 7900 XTX):**

MIGraphX JIT-compiles ONNX graphs on first use and caches the result to disk. With `PER_SONG_MODEL_RELOAD=false` (album-level session caching) the JIT cost is amortised across a full album, yielding approximately 7 seconds per track end-to-end including CLAP, MusiCNN, Essentia mood, and faster-whisper lyrics.

**Notes:**
- Essentia MTG-Jamendo mood classifiers run on CPU by design (models are 81 KB each; MIGraphX JIT overhead would exceed inference time)
- MusiCNN and CLAP use `MIGraphXExecutionProvider` when available, falling back to `ROCMExecutionProvider`, then `CUDAExecutionProvider`, then CPU
- Setting `PER_SONG_MODEL_RELOAD=true` (the default) reloads sessions after every track, which is safe for memory but slower

---

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
