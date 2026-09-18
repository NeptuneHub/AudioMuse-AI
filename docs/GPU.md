# GPU deployment

Nvidia GPU support is available for analysis task in the worker process. This can significantly speed up processing of tracks.

**ARM (DGX Spark / GB10) support:** the `-nvidia-arm` image adds support for NVIDIA GPUs on ARM64 hosts, such as the DGX Spark and other GB10-based machines, for both analysis and clustering. This image is **EXPERIMENTAL**. Use it the same way as the regular `-nvidia` image, just pick the `-nvidia-arm` tag.

We suggest **8GB VRAM** on GPU, with less you can experience the NON BLOCKING OutOFMemory error (that are handled by switching to CPU). The `PER_SONG_MODEL_RELOAD` env variable, that by default is TRUE, help cleaning the memory by entirely reloading the model each time, on the other side it slow the analysis process.


GPU-accelerated clustering is also available through RAPIDS cuML. It can give a **10-30x speedup** on clustering tasks.

**Features:**
- GPU-accelerated KMeans, DBSCAN, PCA and SpectralClustering using RAPIDS cuML
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

**Notes:**
- GMM stays on CPU, there is no cuML implementation for it
- Spectral clustering runs on cuML only with `assign_labels='kmeans'` and a `nearest_neighbors` / `precomputed` affinity (what the clustering search uses); any other combination falls back to scikit-learn
- GPU clustering is disabled by default (`USE_GPU_CLUSTERING=false`)
- The GPU is also used by the audio analysis models (ONNX inference: MusiCNN, CLAP and the neural fingerprint encoder of Search by Recording)
- The index build and the similarity queries are not GPU accelerated; they are IO bound rather than compute bound, see [ALGORITHM](ALGORITHM.md#4-similarity-indexes-disk-paged-ivf)

## Several workers on one GPU

Analysis is one album per job, so extra worker replicas are the way to use a
big GPU - but three things stop them scaling on a single card:

1. **Whisper VRAM.** Every replica that reaches the ASR fallback loads its own
   Whisper-small pipeline (~1.5 GB + activations). Three replicas transcribing
   at once fill a 12 GB card, and a fourth OOMs. Set `LYRICS_ASR_LOCK_DIR` to a
   directory mounted into every replica (the `asr-locks` volume in
   `docker-compose-nvidia.yaml`) and `LYRICS_ASR_LOCK_SLOTS` to how many
   replicas may transcribe at the same time. Replicas beyond that wait for a
   slot; the wait is not charged against the ASR timeout, and a pipeline is
   unloaded before its slot is released. Worker count is then bounded by
   CLAP/MusiCNN memory (well under 1 GB per replica) instead of by Whisper.
2. **CPU thread pools.** Without a cgroup CPU limit `cpu_budget.py` sees the
   whole host, so every replica opens ONNX thread pools as wide as the machine
   (six replicas on a 20-thread host means 600+ threads and a load average of
   ~50). Give each replica a quota (`cpus: "3"` in compose, or a Kubernetes
   CPU limit) and the pools are sized to it.
3. **gte on CPU.** The lyrics-embedding model runs on CPU by default. With
   `LYRICS_GTE_USE_GPU=true` it runs on CUDA, which on a shared GPU is cheaper
   than the CPU cores it frees for audio decoding.

Measured on an RTX 5070 (12 GB), i7-12700K (20 threads), Navidrome library
of 5.7k tracks, lyrics fallback hitting Whisper on ~25% of them:

| replicas | Whisper slots | cpus/replica | tracks/min | VRAM | load avg |
| --- | --- | --- | --- | --- | --- |
| 3 | unlimited | none | 10 | 11.5 GB (ceiling) | 13 |
| 6 | 3 | none | 19 | 6 GB | 49 |
| 6 | 3 | 3 | 29 | 7.6 GB | 8 |
| 8 | 3 | 3 | 38 | 9 GB | 8 |

Adding replicas needs RAM too (~2 GB each); watch the host before going past
what fits.
