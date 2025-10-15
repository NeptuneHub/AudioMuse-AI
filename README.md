![GitHub license](https://img.shields.io/github/license/neptunehub/AudioMuse-AI.svg)
![Latest Tag](https://img.shields.io/github/v/tag/neptunehub/AudioMuse-AI?label=latest-tag)
![Media Server Support: Jellyfin 10.10.7, Navidrome 0.58.0, LMS v3.69.0, Lyrion 9.0.2](https://img.shields.io/badge/Media%20Server-Jellyfin%2010.10.7%2C%20Navidrome%200.58.0%2C%20LMS%20v3.69.0%2C%20Lyrion%209.0.2-blue?style=flat-square&logo=server&logoColor=white)


# **AudioMuse-AI - Let the choice happen, the open-source way** 

<p align="center">
  <img src="https://github.com/NeptuneHub/AudioMuse-AI/blob/main/screenshot/audiomuseai.png?raw=true" alt="AudioMuse-AI Logo" width="480">
</p>


AudioMuse-AI is an Open Source Dockerized environment that brings **automatic playlist generation** to your self-hosted music library. Using powerful tools like [Librosa](https://github.com/librosa/librosa) and [ONNX](https://onnx.ai/), it performs **sonic analysis** on your audio files locally, allowing you to curate the perfect playlist for any mood or occasion without relying on external APIs.

Deploy it easily on your local machine with Docker Compose/Podman or scale it up in your Kubernetes cluster, with the support of **AMD64** and **ARM64** architecture. It integrate with API the main Music server like [Jellyfin](https://jellyfin.org), [Navidrome](https://www.navidrome.org/), [LMS](https://github.com/epoupon/lms/tree/master), [Lyrion](https://lyrion.org/) and many mores will come in the future.

AudioMuse-AI lets you explore your music library in innovative ways, just **start with an initial analysis**, and you’ll unlock features like:
* **Clustering**: Automatically groups sonically similar songs, creating genre-defying playlists based on the music's actual sound.
* **Playlist from Similar Songs**: Pick a track you love, and AudioMuse-AI will find all the songs in your library that share its sonic signature, creating a new discovery playlist.
* **Song Paths**: Create a seamless listening journey between two or more songs. AudioMuse-AI finds the perfect tracks to bridge the sonic gap.
* **Instant Playlists**: Simply tell the AI what you want to hear—like "high-tempo, low-energy workout music"—and it will instantly generate a playlist for you.
* **Sonic Fingerprint**: Generates playlists based on your listening habits, finding tracks similar to what you've been playing most often.

> 👉 Experience the **Sonic Analysis potentiality** with the [AudioMuse-AI-MusicServer Demo](https://github.com/NeptuneHub/AudioMuse-AI/issues/94)!  
> _Available for a limited time ⌛_

**The full list or AudioMuse-AI related repository are:** 
  > * [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI): the core application, it run Flask and Worker containers to actually run all the feature;
  > * [AudioMuse-AI Helm Chart](https://github.com/NeptuneHub/AudioMuse-AI-helm): helm chart for easy installation on Kubernetes;
  > * [AudioMuse-AI Plugin for Jellyfin](https://github.com/NeptuneHub/audiomuse-ai-plugin): Jellyfin Plugin;
  > * [AudioMuse-AI MusicServer](https://github.com/NeptuneHub/AudioMuse-AI-MusicServer): **Experimental** Open Subosnic like Music Sever with integrated sonic functionality.

And now just some **NEWS:**
> * Version 0.7.0-beta remove Tensorflow and use instead ONXX. This new version should have better result on different CPU and be also more stable between update. **IMPORTANT:** this new version will require a new analysis of the entire library to work correctly.
> * Version 0.6.9-beta introduce the support to Lyrion Music Server.

## Disclaimer

**Important:** Despite the similar name, this project (**AudioMuse-AI**) is an independent, community-driven effort. It has no official connection to the website audiomuse.ai.

We are **not affiliated with, endorsed by, or sponsored by** the owners of `audiomuse.ai`.

## **Table of Contents**

- [Quick Start Deployment on K3S WITH HELM](#quick-start-deployment-on-k3s-with-helm)
- [Quick Start Deployment on K3S](#quick-start-deployment-on-k3s)
- [Local Deployment with Docker Compose](#local-deployment-with-docker-compose)
- [Local Deployment with Podman Quadlets](#local-deployment-with-podman-quadlets)
- [Hardware Requirements](#hardware-requirements)
- [Configuration Parameters](#configuration-parameters)
- [Docker Image Tagging Strategy](#docker-image-tagging-strategy)
- [Key Technologies](#key-technologies)
- [How To Contribute](#how-to-contribute)
- [Star History](#star-history)

## **Quick Start Deployment on K3S WITH HELM**

The best way to install AudioMuse-AI on K3S (kubernetes) is by [AudioMuse-AI Helm Chart repository](https://github.com/NeptuneHub/AudioMuse-AI-helm)

*  **Prerequisites:**
    *   A running `K3S cluster`.
    *   `kubectl` configured to interact with your cluster.
    *   `helm` installed.
    *   `Jellyfin` or `Navidrome` or `Lyrion` installed.
    *   Respect the HW requirements (look the specific chapter)

You can directly check the Helm Chart repo for more details and deployments examples.

How to find jellyfin **userid**:
* Log into Jellyfin from your browser as an admin
* Go to Dashboard > “admin panel” > Users.
* Click on the user’s name that you are interested
* The User ID is visible in the URL (is the part just after = ):
  * http://your-jellyfin-server/web/index.html#!/useredit.html?userId=xxxxx

How to create an the **jellyfin's API token**:
* The API Token, still as admin you can go to Dashboard > “Admin panel” > API Key and create a new one.

## **Quick Start Deployment on K3S**

This section provides a minimal guide to deploy AudioMuse-AI on a K3S (Kubernetes) cluster by direct use of `deployment`

* **Prerequisites:**
    *   A running K3S cluster.
    *   `kubectl` configured to interact with your cluster.
    *   `Jellyfin` or `Navidrome` or `Lyrion` installed.
    *   Respect the HW requirements (look the specific chapter)

*  **Jellyfin Configuration:**
    *   Navigate to the `deployment/` directory.
    *   Edit `deployment.yaml` to configure mandatory parameters:
        *   **Secrets:**
            *   `jellyfin-credentials`: Update `api_token` and `user_id`.
            *   `postgres-credentials`: Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
            *   `gemini-api-credentials` (if using Gemini for AI Naming): Update `GEMINI_API_KEY`.
            *   `mistral-api-credentials` (if using Mistral for AI Naming): Update `MISTRAL_API_KEY`.
        *   **ConfigMap (`audiomuse-ai-config`):**
            *   Update `JELLYFIN_URL`.
            *   Ensure `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL` are correct for your setup (defaults are for in-cluster services).

*  **Navidrome/LMS (Open Subsonic API Music Server) Configuration:**
    *   Navigate to the `deployment/` directory.
    *   Edit `deployment-navidrome.yaml` to configure mandatory parameters:
        *   **Secrets:**
            *   `navidrome-credentials`: Update `NAVIDROME_USER` and `NAVIDROME_PASSWORD`.
            *   `postgres-credentials`: Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
            *   `gemini-api-credentials` (if using Gemini for AI Naming): Update `GEMINI_API_KEY`.
            *   `mistral-api-credentials` (if using Mistral for AI Naming): Update `MISTRAL_API_KEY`.
        *   **ConfigMap (`audiomuse-ai-config`):**
            *   Update `NAVIDROME_URL`.
            *   Ensure `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL` are correct for your setup (defaults are for in-cluster services).
        *   > The same instruction used for Navidrome could apply to other Mediaserver that support Subsonic API. LMS for example is supported, only remember to user the Subsonic API token instead of the password.

*  **Lyrion Configuration:**
    *   Navigate to the `deployment/` directory.
    *   Edit `deployment-lyrion.yaml` to configure mandatory parameters:
        *   **Secrets:**
            *   `postgres-credentials`: Update `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
            *   `gemini-api-credentials` (if using Gemini for AI Naming): Update `GEMINI_API_KEY`.
        *   **ConfigMap (`audiomuse-ai-config`):**
            *   Update `LYRION_URL`.
            *   Ensure `POSTGRES_HOST`, `POSTGRES_PORT`, and `REDIS_URL` are correct for your setup (defaults are for in-cluster services).
            
*  **Deploy:**
    ```bash
    kubectl apply -f deployment/deployment.yaml
    ```
*  **Access:**
    *   **Main UI:** Access at `http://<EXTERNAL-IP>:8000`
    *   **API Docs (Swagger UI):** Explore the API at `http://<EXTERNAL-IP>:8000/apidocs`
 
## **Local Deployment with Docker Compose**

AudioMuse-AI provides Docker Compose files for different media server backends:

- **Jellyfin**: Use `deployment/docker-compose.yaml`
- **Navidrome**: Use `deployment/docker-compose-navidrome.yaml`
- **Lyrion**: Use `deployment/docker-compose-lyrion.yaml`

Choose the appropriate file based on your media server setup.

For a quick local setup or for users not using Kubernetes, a `docker-compose.yaml` file is provided in the `deployment/` directory for interacting with **Jellyfin**. `docker-compose-navidrome.yaml` is instead pre-compiled to interact with **Navidrome** or other Subsonic API based Mediaserver. Finally `docker-compose-lyrion.yaml` is precompiled for Lyrion.

**Prerequisites:**
*   Docker and Docker Compose installed.
*   `Jellyfin` or `Navidrome` or `Lyrion` installed.
*   Respect the [hardware requirements](#hardware-requirements)

**Steps:**
1.  **Navigate to the `deployment` directory:**
    ```bash
    cd deployment
    ```
2.  **Review and Customize:**
    The `docker-compose.yaml`, `docker-compose-navidrome.yaml` and `docker-compose-lyrion.yaml` files are pre-configured with default credentials and settings suitable for local testing. You can edit environment variables within this file directly (e.g., `JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN` for **Jellyfin** or `NAVIDROME_URL`, `NAVIDROME_USER` and `NAVIDROME_PASSWORD` for **Navidrome**,  `LYRION_URL` for Lyrion that doesn't require any passwords).
3.  **Start the Services:**
    ```bash
    docker compose up -d
    ```
    This command starts all services (Flask app, RQ workers, Redis, PostgreSQL) in detached mode (`-d`).
4.  **Access the Application:**
    Once the containers are up, you can access the web UI at `http://localhost:8000`.
5.  **Stopping the Services:**
    ```bash
    docker compose down
    ```
**Note:**
  > If you use LMS instead of the password you need to create and use the Subsonic API token. Additional Subsonic API based Mediaserver could require it in place of the password.

## **Local Deployment with Podman Quadlets**

For an alternative local setup, [Podman Quadlet](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html) files are provided in the `deployment/podman-quadlets` directory for interacting with **Navidrome**. The unit files can  be edited for use with **Jellyfin**. 

These files are configured to automatically update AudioMuse-AI using the [latest](#docker-image-tagging-strategy) stable release and should perform an automatic rollback if the updated image fails to start.     

**Prerequisites:**
*   Podman and systemd.
*   `Jellyfin` or `Navidrome` installed.
*   Respect the [hardware requirements](#hardware-requirements)

**Steps:**
1.  **Navigate to the `deployment/podman-quadlets` directory:**
    ```bash
    cd deployment/podman-quadlets
    ```
2.  **Review and Customize:**

    The `audiomuse-ai-postgres.container` and `audiomuse-redis.container` files are pre-configured with default credentials and settings suitable for local testing. <BR>
    You will need to edit environment variables within `audiomuse-ai-worker.container` and `audiomuse-ai-flask.container` files to reflect your personal credentials and environment.
    * For **Navidrome**, update `NAVIDROME_URL`, `NAVIDROME_USER` and `NAVIDROME_PASSWORD` with your real credentials.  
    * For **Jellyfin** replace these variables with `JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN`; add your real credentials; and change the `MEDIASERVER_TYPE` to `jellyfin`. 

    Once you've customized the unit files, you will need to copy all of them into a systemd container directory, such as `/etc/containers/systemd/user/`.<BR>

3.  **Start the Services:**
    ```bash
    systemctl --user daemon-reload
    systemctl --user start audiomuse-pod
    ```
    The first command reloads systemd (generating the systemd service files) and the second command starts all AudioMuse services (Flask app, RQ worker, Redis, PostgreSQL).
4.  **Access the Application:**
    Once the containers are up, you can access the web UI at `http://localhost:8000`.
5.  **Stopping the Services:**
    ```bash
    systemctl --user stop audiomuse-pod
    ```
      
## **Hardware Requirements**

AudioMuse-Ai is actually tested on:
* **INTEL**: HP Mini PC with Intel i5-6500, 16 GB RAM and NVME SSD
* **ARM**: Raspberry Pi 5 8GB RAM and NVME SSD

The **suggested requirements** are: 4core INTEL or ARM CPU (Producted from 2015 and above) with AVX support, 8GB ram and an SSD.

It can most probably run on older CPU (from 3rd gen and above) and with less ram (maybe 4GB) but I never tested.

Intel I7 CPU of first gen or older **DON'T WORK** because Tensorflow require AVX supprt.

If you tested with CPU older than the suggested requirements, please track this in an issue ticket reporting your feedback.

You can check the [Tested Hardware and Configuration](HARDWARE.md) notes to look which one was already tested till now.

**IMPORTANT** From `0.7.0-beta` ONNX replace Tensorflow. So there is the possibility that CPU not supported till can now work.

### (Optional) Experimental Nvidia Support

NVidia GPU support is available for the worker process. This can significantly speed up processing of tracks. 

This has been tested with an NVidia RX 3060 running CUDA 12.9 and Driver V575.64.05. During testing, the worker used up to 10GiB of VRAM but your mileage may vary.

## **Configuration Parameters**

These are the parameters accepted for this script. You can pass them as environment variables using, for example, /deployment/deployment.yaml in this repository.

The **mandatory** parameter that you need to change from the example are this:

| Parameter            | Description                                                             | Default Value                     |
|----------------------|-------------------------------------------------------------------------|-----------------------------------|
| `JELLYFIN_URL`       | (Required) Your Jellyfin server's full URL                              | `http://YOUR_JELLYFIN_IP:8096`    |
| `JELLYFIN_USER_ID`   | (Required) Jellyfin User ID.                                            | *(N/A - from Secret)* |
| `JELLYFIN_TOKEN`     | (Required) Jellyfin API Token.                                          | *(N/A - from Secret)* |
| `NAVIDROME_URL`      | (Required) Your Navidrome server's full URL                             | `http://YOUR_JELLYFIN_IP:4553`    |
| `NAVIDROME_USER`     | (Required) Navidrome User ID.                                           | *(N/A - from Secret)* |
| `NAVIDROME_PASSWORD` | (Required) Navidrome user Password.                                     | *(N/A - from Secret)* |
| `LYRION_URL`         | (Required) Your Lyrion server's full URL                                | `http://YOUR_LYRION_IP:9000`      |
| `POSTGRES_USER`      | (Required) PostgreSQL username.                                         | *(N/A - from Secret)* |
| `POSTGRES_PASSWORD`  | (Required) PostgreSQL password.                                         | *(N/A - from Secret)* |
| `POSTGRES_DB`        | (Required) PostgreSQL database name.                                    | *(N/A - from Secret)* |
| `POSTGRES_HOST`      | (Required) PostgreSQL host.                                             | `postgres-service.playlist`       |
| `POSTGRES_PORT`      | (Required) PostgreSQL port.                                             | `5432`                            |
| `REDIS_URL`          | (Required) URL for Redis.                                               | `redis://localhost:6379/0`        |
| `GEMINI_API_KEY`     | (Required if `AI_MODEL_PROVIDER` is GEMINI) Your Google Gemini API Key. | *(N/A - from Secret)* |
| `MISTRAL_API_KEY`    | (Required if `AI_MODEL_PROVIDER` is MISTRAL) Your Mistral API Key.      | *(N/A - from Secret)* |

These parameter could be leave as it is:

| Parameter               | Description                                  | Default Value     |
|-------------------------|----------------------------------------------|-------------------|
| `TEMP_DIR`              | Temp directory for audio files              | `/app/temp_audio` |
| `CLEANING_SAFETY_LIMIT` | Max number of albums deleted during cleaning | `100`             |
| `MUSIC_LIBRARIES`       | Comma-separated list of music libraries/folders for analysis. If empty, all libraries/folders are scanned. For Lyrion: Use folder paths like "/music/myfolder". For Jellyfin/Navidrome: Use library/folder names. | `""` (empty - scan all) |

This are the default parameters on wich the analysis or clustering task will be lunched. You will be able to change them to another value directly in the front-end:

| Parameter                                   | Description                                                                                                                | Default Value   |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------|
| **Analysis General**                        |                                                                                                                            |                 |
| `NUM_RECENT_ALBUMS`                         | Number of recent albums to scan (0 for all).                                                                              | `0`             |
| `TOP_N_MOODS`                               | Number of top moods per track for feature vector.                                                                         | `5`             |
| **Clustering General**                      |                                                                                                                            |                 |
| `ENABLE_CLUSTERING_EMBEDDINGS`              | Whether to use audio embeddings (True) or score-based features (False) for clustering.                                    | `true`          |
| `CLUSTER_ALGORITHM`                         | Default clustering: `kmeans`, `dbscan`, `gmm`, `spectral`.                                                                | `kmeans`        |
| `MAX_SONGS_PER_CLUSTER`                     | Max songs per generated playlist segment.                                                                                 | `0`             |
| `MAX_SONGS_PER_ARTIST`                      | Max songs from one artist per cluster.                                                                                    | `3`             |
| `MAX_DISTANCE`                              | Normalized distance threshold for tracks in a cluster.                                                                    | `0.5`           |
| `CLUSTERING_RUNS`                           | Iterations for Monte Carlo evolutionary search.                                                                           | `5000`          |
| `TOP_N_PLAYLISTS`                           | POST Clustering it keep only the top N diverse playlist.                                                                  | `8`             |
| **Similarity General**                      |                                                                                                                            |                 |
| `INDEX_NAME`                                | Name of the index, no need to change.                                                                                     | `music_library` |
| `VOYAGER_EF_CONSTRUCTION`                   | Number of element analyzed to create the neighbor list in the index.                                                      | `1024`          |
| `VOYAGER_M`                                 | Number of neighbore More = higher accuracy.                                                                               | `64`            |
| `VOYAGER_QUERY_EF`                          | Number neighbor analyzed during the query.                                                                                | `1024`          |
| `VOYAGER_METRIC`                            | Different tipe of distance metrics: `angular`, `euclidean`,`dot`                                                          | `angular`       |
| `SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT`   | It enable the possibility of use the `MAX_SONGS_PER_ARTIST` also in similar song                                          | `true`          |
| **Sonic Fingerprint General**               |                                                                                                                            |                 |
| `SONIC_FINGERPRINT_NEIGHBORS`               | Default number of track for the sonic fingerprint                                                                         | `100`           |
| **Similar Song and Song Path Duplicate filtering General** |                                                                                                            |                 |
| `DUPLICATE_DISTANCE_THRESHOLD_COSINE`       | Less than this cosine distance the track is a duplicate.                                                                  | `0.01`          |
| `DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN`    | Less than this euclidean distance the track is a duplicate.                                                               | `0.15`          |
| `DUPLICATE_DISTANCE_CHECK_LOOKBACK`         | How many previous song need to be checked for duplicate.                                                                  | `1`             |
| `MOOD_SIMILARITY_THRESHOLD`                 | Maximum normalized distance for mood similarity filtering. Lower value will give more importance to mood                  | `0.15`          |
| **Song Path General**                       |                                                                                                                            |                 |
| `PATH_DISTANCE_METRIC`                      | The distance metric to use for pathfinding. Options: 'angular', 'euclidean'                                               | `angular`       |
| `PATH_DEFAULT_LENGTH`                       | Default number of songs in the path if not specified in the API request                                                   | `25`            |
| `PATH_AVG_JUMP_SAMPLE_SIZE`                 | Number of random songs to sample for calculating the average jump distance                                                | `200`           |
| `PATH_CANDIDATES_PER_STEP`                  | Number of candidate songs to retrieve from Voyager for each step in the path                                              | `25`            |
| `PATH_LCORE_MULTIPLIER`                     | It multiply the number of centroid created based on the distance. Higher is better for distant song and worst for nearest. | `3`             |
| **Evolutionary Clustering & Scoring**      |                                                                                            |                                        |
| `ITERATIONS_PER_BATCH_JOB`                  | Number of clustering iterations processed per RQ batch job.                                | `20`                                   |
| `MAX_CONCURRENT_BATCH_JOBS`                 | Maximum number of clustering batch jobs to run simultaneously.                             | `10`                                   |
| `CLUSTERING_BATCH_TIMEOUT_MINUTES`          | Max time a batch can run before being considered failed (prevents infinite hangs).        | `60`                                   |
| `CLUSTERING_MAX_FAILED_BATCHES`             | Max number of failed batches before stopping new launches and forcing completion.         | `10`                                   |
| `CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS`   | How often to check batch status for timeout detection.                                    | `30`                                   |
| `TOP_K_MOODS_FOR_PURITY_CALCULATION`        | Number of centroid's top moods to consider when calculating playlist purity.              | `3`                                    |
| `EXPLOITATION_START_FRACTION`               | Fraction of runs before starting to use elites.                                           | `0.2`                                  |
| `EXPLOITATION_PROBABILITY_CONFIG`           | Probability of mutating an elite vs. random generation.                                   | `0.7`                                  |
| `MUTATION_INT_ABS_DELTA`                    | Max absolute change for integer parameter mutation.                                        | `3`                                    |
| `MUTATION_FLOAT_ABS_DELTA`                  | Max absolute change for float parameter mutation.                                          | `0.05`                                 |
| `MUTATION_KMEANS_COORD_FRACTION`            | Fractional change for KMeans centroid coordinates.                                        | `0.05`                                 |
| **K-Means Ranges**                          |                                                                                            |                                        |
| `NUM_CLUSTERS_MIN`                          | Min $K$ for K-Means.                                                                      | `40`                                   |
| `NUM_CLUSTERS_MAX`                          | Max $K$ for K-Means.                                                                      | `100`                                  |
| `USE_MINIBATCH_KMEANS`                      | Whether to use MiniBatchKMeans (True) or standard KMeans (False) when clustering embeddings. | `false`                            |
| **DBSCAN Ranges**                           |                                                                                            |                                        |
| `DBSCAN_EPS_MIN`                            | Min epsilon for DBSCAN.                                                                   | `0.1`                                  |
| `DBSCAN_EPS_MAX`                            | Max epsilon for DBSCAN.                                                                   | `0.5`                                  |
| `DBSCAN_MIN_SAMPLES_MIN`                    | Min `min_samples` for DBSCAN.                                                             | `5`                                    |
| `DBSCAN_MIN_SAMPLES_MAX`                    | Max `min_samples` for DBSCAN.                                                             | `20`                                   |
| **GMM Ranges**                              |                                                                                            |                                        |
| `GMM_N_COMPONENTS_MIN`                      | Min components for GMM.                                                                   | `40`                                   |
| `GMM_N_COMPONENTS_MAX`                      | Max components for GMM.                                                                   | `100`                                  |
| `GMM_COVARIANCE_TYPE`                       | Covariance type for GMM (task uses `full`).                                               | `full`                                 |
| **Spectral Ranges**                         |                                                                                            |                                        |
| `SPECTRAL_N_CLUSTERS_MIN`                   | Min components for Spectral clustering.                                                   | `40`                                   |
| `SPECTRAL_N_CLUSTERS_MAX`                   | Max components for Spectral clustering.                                                   | `100`                                  |
| `SPECTRAL_N_NEIGHBORS`                      | Number of Neighbors on which do clustering. Higher is better but slower                   | `20`                                   |
| **PCA Ranges**                              |                                                                                            |                                        |
| `PCA_COMPONENTS_MIN`                        | Min PCA components (0 to disable).                                                        | `0`                                    |
| `PCA_COMPONENTS_MAX`                        | Max PCA components (e.g., `8` for feature vectors, `199` for embeddings).                 | `199`                                  |
| **AI Naming (*)**                           |                                                                                            |                                        |
| `AI_MODEL_PROVIDER`                         | AI provider: `OLLAMA`, `GEMINI`, `MISTRAL` or `NONE`.                                     | `NONE`                                 |
| `TOP_N_ELITES`                              | Number of best solutions kept as elites.                                                  | `10`                                   |
| `SAMPLING_PERCENTAGE_CHANGE_PER_RUN`        | Percentage of songs to swap out in the stratified sample between runs (0.0 to 1.0).       | `0.2`                                  |
| `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`    | Minimum number of songs to target per stratified genre during sampling.                   | `100`                                  |
| `STRATIFIED_SAMPLING_TARGET_PERCENTILE`     | Percentile of genre song counts to use for target songs per stratified genre.             | `50`                                   |
| `OLLAMA_SERVER_URL`                         | URL for your Ollama instance (if `AI_MODEL_PROVIDER` is OLLAMA).                          | `http://<your-ip>:11434/api/generate` |
| `OLLAMA_MODEL_NAME`                         | Ollama model to use (if `AI_MODEL_PROVIDER` is OLLAMA).                                   | `mistral:7b`                          |
| `GEMINI_MODEL_NAME`                         | Gemini model to use (if `AI_MODEL_PROVIDER` is GEMINI).                                   | `gemini-2.5-pro`                      |
| `MISTRAL_MODEL_NAME`                        | Mistral model to use (if `AI_MODEL_PROVIDER` is MISTRAL).                                 | `ministral-3b-latest`                  |
| **Scoring Weights**                         |                                                                                            |                                        |
| `SCORE_WEIGHT_DIVERSITY`                    | Weight for inter-playlist mood diversity.                                                 | `2.0`                                  |
| `SCORE_WEIGHT_PURITY`                       | Weight for playlist purity (intra-playlist mood consistency).                             | `1.0`                                  |
| `SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY`      | Weight for inter-playlist 'other feature' diversity.                                      | `0.0`                                  |
| `SCORE_WEIGHT_OTHER_FEATURE_PURITY`         | Weight for intra-playlist 'other feature' consistency.                                    | `0.0`                                  |
| `SCORE_WEIGHT_SILHOUETTE`                   | Weight for Silhouette Score (cluster separation).                                         | `0.0`                                  |
| `SCORE_WEIGHT_DAVIES_BOULDIN`               | Weight for Davies-Bouldin Index (cluster separation).                                     | `0.0`                                  |
| `SCORE_WEIGHT_CALINSKI_HARABASZ`            | Weight for Calinski-Harabasz Index (cluster separation).                                  | `0.0`                                  |



**(*)** For using GEMINI API you need to have a Google account, a free account can be used if needed. Same goes for Mistral. Instead if you want to self-host Ollama here you can find a deployment example:

* https://github.com/NeptuneHub/k3s-supreme-waffle/tree/main/ollama

## **Docker Image Tagging Strategy**

Our GitHub Actions workflow automatically builds and pushes Docker images. Here's how our tags work:

* :**latest**  
  * Builds from the **main branch**.  
  * Represents the latest stable release.  
  * **Recommended for most users.**  
* :**devel**  
  * Builds from the **devel branch**.  
  * Contains features still in development, not fully tested, and they could not work.  
  * **Use only for development.**  
* :**vX.Y.Z** (e.g., :v0.1.4-alpha, :v1.0.0)  
  * Immutable tags created from **specific Git releases/tags**.  
  * Ensures you're running a precise, versioned build.  
  * **Use for reproducible deployments or locking to a specific version.**
 
**IMPORTANT** the `-nvidia` image are **experimantal** image. Try it if you want to help us to improve BUT we suggest to don't use it for normal daily use for now. 

## **Key Technologies**

AudioMuse AI is built upon a robust stack of open-source technologies:

* [**Flask:**](https://flask.palletsprojects.com/) Provides the lightweight web interface for user interaction and API endpoints.  
* [**Redis Queue (RQ):**](https://redis.io/glossary/redis-queue/) A simple Python library for queueing jobs and processing them in the background with Redis.
* [**Supervisord:**](https://supervisord.org/) Supervisor is a client/server system that allows its users to monitor and control a number of processes on UNIX-like operating systems.
* [**Essentia-tensorflow**](https://essentia.upf.edu/) An open-source library for audio analysis, feature extraction, and music information retrieval. (used only until version v0.5.0-beta)
* [**MusicNN Tensorflow Audio Models from Essentia**](https://essentia.upf.edu/models.html) Leverages pre-trained MusicNN models for feature extraction and prediction. More details and models.
* [**Librosa**](https://github.com/librosa/librosa) Library for audio analysis, feature extraction, and music information retrieval. (used from version v0.6.0-beta)
* [**ONNX**](https://onnx.ai/) Open Neural Network Exchange format and [ONNX Runtime](https://onnxruntime.ai/) for fast, portable, cross-platform model inference. **(Used from v0.7.0-beta, replaces TensorFlow)**
* [**Tensorflow**](https://www.tensorflow.org/) Platform developed by Google for building, training, and deploying machine learning and deep learning models. **(Used only in versions before v0.7.0-beta)**
* [**scikit-learn**](https://scikit-learn.org/) Utilized for machine learning algorithms:
* [**voyager**](https://github.com/spotify/voyager) Approximate Nearest Neighbors used for the /similarity interface. Used from v0.6.3-beta
* [**PostgreSQL:**](https://www.postgresql.org/) A powerful, open-source relational database used for persisting:  
* [**Ollama**](https://ollama.com/) Enables self-hosting of various open-source Large Language Models (LLMs) for tasks like intelligent playlist naming.
* [**Docker / OCI-compatible Containers**](https://www.docker.com/) – The entire application is packaged as a container, ensuring consistent and portable deployment across environments.

## **How To Contribute**

Contributions, issues, and feature requests are welcome\!  
This is a BETA early release, so expect bugs or functions that are still not implemented.

For more details on how to contribute please follow the [Contributing Guidelines](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/CONTRIBUTING.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NeptuneHub/AudioMuse-AI&type=Timeline)](https://www.star-history.com/#NeptuneHub/AudioMuse-AI&Timeline)
