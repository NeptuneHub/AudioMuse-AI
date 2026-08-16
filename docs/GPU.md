# GPU deployment

Nvidia GPU support is available for analysis task in the worker process. This can significantly speed up processing of tracks.

**ARM (DGX Spark / GB10) support:** the `-nvidia-arm` image adds support for NVIDIA GPUs on ARM64 hosts, such as the DGX Spark and other GB10-based machines, for both analysis and clustering. This image is **EXPERIMENTAL**. Use it the same way as the regular `-nvidia` image, just pick the `-nvidia-arm` tag.

We suggest **8GB VRAM** on GPU, with less you can experience the NON BLOCKING OutOFMemory error (that are handled by switching to CPU). The `PER_SONG_MODEL_RELOAD` env variable, that by default is TRUE, help cleaning the memory by entirely reloading the model each time, on the other side it slow the analysis process.


GPU-accelerated clustering is also available through RAPIDS cuML. It can give a **10-30x speedup** on clustering tasks.

**Features:**
- GPU-accelerated KMeans, DBSCAN, and PCA using RAPIDS cuML
- Automatic fallback to CPU if the GPU is unavailable or hits an error
- Works with all existing clustering configurations and parameters
- Compatible with NVIDIA GPUs on CUDA 13 or later (*)

(*) CUDA 13 raises the minimum host driver to >=580.x (up from >=570.x for the previous CUDA 12.8 image) - older drivers are NOT supported by the published build, but you can try to build your own image as described in https://github.com/NeptuneHub/AudioMuse-AI/issues/265

**To enable GPU clustering:**

1. Use the NVIDIA image (for example `nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04`)
2. Set the value in your `.env` file, or in the Setup Wizard:
   ```
   USE_GPU_CLUSTERING=true
   ```
3. Make sure the NVIDIA Container Toolkit is installed on the host
4. Use the GPU compose file `deployment/docker-compose-nvidia.yaml`. A worker-only GPU example is kept in `deployment/test/docker-compose-nvidia-worker-test.yaml`

**Performance Impact:**
- **KMeans**: 10-50x faster than CPU
- **DBSCAN**: 5-100x faster than CPU
- **PCA**: 10-40x faster than CPU
- **Overall clustering task**: 10-30x speedup for typical workloads (5000 iterations)

**Example:** A clustering task that takes 2-4 hours on CPU may finish in 5-15 minutes on GPU.

**Notes:**
- GMM and Spectral clustering stay on CPU, there is no cuML implementation for them in this build
- GPU clustering is disabled by default (`USE_GPU_CLUSTERING=false`)
- The GPU is also used by the audio analysis models (ONNX inference)
- The index build and the similarity queries are not GPU accelerated; they are IO bound rather than compute bound, see [ALGORITHM](ALGORITHM.md#4-similarity-indexes-disk-paged-ivf)
