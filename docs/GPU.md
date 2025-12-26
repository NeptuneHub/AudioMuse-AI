# GPU deployment

NVidia GPU **EXPERIMENTAL** support is available for analysis task in the worker process. This can significantly speed up processing of tracks.

With the new CLAP model used in Text Search functionality, GPU VRAM is used more. If you experience VRAM OutOFMemory error (that are handled by switching to GPU) we suggest to have a look to `CLAP_MINI_BATCH_SIZE` and `PER_SONG_MODEL_RELOAD` env variable. 

Putting the `CLAP_MINI_BATCH_SIZE`  to `1`, and `PER_SONG_MODEL_RELOAD` to `true` will use less VRAM but will be slower. This is the default configuration.

Putting the `CLAP_MINI_BATCH_SIZE`  to `40` or more, and `PER_SONG_MODEL_RELOAD` to `false` will use more VRAM but with the risk of OutOfMemory vram error.

**NEW:** GPU-accelerated clustering is now available using RAPIDS cuML. This can provide **10-30x speedup** for clustering tasks.

**Features:**
- GPU-accelerated KMeans, DBSCAN, and PCA using RAPIDS cuML
- Automatic fallback to CPU if GPU is unavailable or encounters errors
- Supports all existing clustering configurations and parameters
- Compatible with NVIDIA GPUs with CUDA 12.8.1+

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