# Configuration Parameters

These are the parameters accepted for this script. From `v1.0.0`, only PostgreSQL, Redis, and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as lending page and is also available later from the menu under Administration > Setup Wizard.

How to find jellyfin **userid**:
* Log into Jellyfin from your browser as an admin
* Go to Dashboard > “admin panel” > Users.
* Click on the user’s name that you are interested
* The User ID is visible in the URL (is the part just after = ):
  * http://your-jellyfin-server/web/index.html#!/useredit.html?userId=xxxxx

How to create an the **jellyfin's API token**:
* The API Token, still as admin you can go to Dashboard > “Admin panel” > API Key and create a new one.


The **mandatory** parameter that you need to change from the example are this:

| Parameter            | Description                                                             | Default Value                     |
|----------------------|-------------------------------------------------------------------------|-----------------------------------|
| **Mediaserver General**                        |                                                                 |                 |
| `JELLYFIN_URL`       | (Required) Your Jellyfin server's full URL                              | `http://YOUR_JELLYFIN_IP:8096`    |
| `JELLYFIN_USER_ID`   | (Required) Jellyfin User ID.                                            | *(N/A - from Secret)* |
| `JELLYFIN_TOKEN`     | (Required) Jellyfin API Token.                                          | *(N/A - from Secret)* |
| `EMBY_URL`           | (Required) Your Emby server's full URL                                  | `http://YOUR_EMBY_IP:8096`    |
| `EMBY_USER_ID`       | (Required) Emby User ID.                                                | *(N/A - from Secret)* |
| `EMBY_TOKEN`         | (Required) Emby API Token.                                              | *(N/A - from Secret)* |
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
| `OPENAI_API_KEY`     | (Required if `AI_MODEL_PROVIDER` is OPENAI) Your OpenAI / OpenRouter API Key. | *(N/A - from Secret)* |
| **AudioMuse-AI Authentication**                        |                                                                 |                 |
| `AUTH_ENABLED`     | Enable the AudioMuse-AI authentication layer | `true`|
| `AUDIOMUSE_USER`    | Username for web UI login     | *(N/A - from Secret)* |
| `AUDIOMUSE_PASSWORD`     | Password for web UI login   | *(N/A - from Secret)* |
| `API_TOKEN`     | Bearer token for API/worker requests | *(N/A - from Secret)* |
| `JWT_SECRET`     | HMAC key used to sign session JWTs | *from Secret OR automatically created if blank* |


These parameters can be left as-is:

| Parameter               | Description                                  | Default Value     |
|-------------------------|----------------------------------------------|-------------------|
| `CLEANING_SAFETY_LIMIT` | Max number of albums deleted during cleaning | `100`             |
| `MUSIC_LIBRARIES`       | Comma-separated list of music libraries/folders for analysis. If empty, all libraries/folders are scanned. For Lyrion: Use folder paths like "/music/myfolder". For Jellyfin/Navidrome: Use library/folder names. | `""` (empty - scan all) |
| `ENABLE_PROXY_FIX` | Enable Proxy Fix for Flask when behind a reverse proxy. Example Nginx configuration: [config.py](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/config.py#L346) | `false` |
| `WORKER_URL` | This is the Url your worker instance runs on. The server instance uses this parameter to call the worker. Make sure to include /worker at the end of the url (e.g. http://worker.example.com:8029/worker) | `false` |
| `WORKER_POSTGRES_HOST` | This is the Url of your the postgres service on your server. The worker uses this to connect the postgres service the flask app uses too. Make sure to not include a protocol (like "http") (e.g. 100.000.00.00) | `false` |
| `WORKER_REDIS_URL` | This is the Url of your the redis service on your server. The worker uses this to connect to the redis service the flask app uses too. Make sure to include the protocol "redis://" and the dbindex "/0" (e.g. redis://100.000.00.00:6379/0)   | `false` |
| `TZ`     | Set the time zone of Flask and worker container | `UTC` |

These are the default parameters used when launching analysis or clustering tasks. You can change them directly in the front-end.

| Parameter                                   | Description                                                                                                                | Default Value   |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------|
| **CLAP - TEXT SEARCH AND MUSICNN MODEL**    |                                                                                                                            |                 |
| `CLAP_ENABLED`                              | If false disable CLAP model during the analysis and the use of Text Search functionality.                                  | `true` |
| `CLAP_PYTHON_MULTITHREADS`                  | CPU threading for CLAP analysis. False (default) = Use ONNX internal threading (recommended). True = Use Python ThreadPoolExecutor  | `false`         |
| `PER_SONG_MODEL_RELOAD`                     | Model reloading strategy. true (default) = Unload MusiCNN and CLAP after each song (stable VRAM, slower). false = MusiCNN reloads every 20 songs, CLAP at album end (faster but may accumulate VRAM) | `true`          |
| **Analysis General**                        |                                                                                                                            |                 |
| `NUM_RECENT_ALBUMS`                         | Number of recent albums to scan (0 for all).                                                                              | `0`             |
| `TOP_N_MOODS`                               | Number of top moods per track for feature vector.                                                                         | `5`             |
| `CLAP_ENABLED`                              | Enable or disable CLAP model for text-to-audio search capabilities.                                                       | `true`          |
| `CLAP_PYTHON_MULTITHREADS`                  | CPU threading for CLAP analysis. False (default) = Use ONNX internal threading (recommended). True = Use Python ThreadPoolExecutor  | `false`         |
| **Clustering General**                      |                                                                                                                            |                 |
| `ENABLE_CLUSTERING_EMBEDDINGS`              | Whether to use audio embeddings (True) or score-based features (False) for clustering.                                    | `true`          |
| `CLUSTER_ALGORITHM`                         | Default clustering: `kmeans`, `dbscan`, `gmm`, `spectral`.                                                                | `kmeans`        |
| `MAX_SONGS_PER_CLUSTER`                     | Max songs per generated playlist segment.                                                                                 | `0`             |
| `MAX_SONGS_PER_ARTIST`                      | Max songs from one artist per cluster.                                                                                    | `3`             |
| `MAX_DISTANCE`                              | Normalized distance threshold for tracks in a cluster.                                                                    | `0.5`           |
| `CLUSTERING_RUNS`                           | Iterations for Monte Carlo evolutionary search.                                                                           | `1000`          |
| `TOP_N_PLAYLISTS`                           | POST Clustering it keep only the top N diverse playlist.                                                                  | `8`             |
| `USE_GPU_CLUSTERING`                        | When true enalbe the use of GPU on K-Means, DBSCAN and PCA                                                                | `false`         |
| **Similarity General**                      |                                                                                                                           |                 |
| `IVF_METRIC`                                | Distance metric used by the similarity index: `angular` (cosine), `euclidean`, or `dot` (inner product). Changing it requires an index rebuild.                                                                                            | `angular`       |
| **Disk-Paged IVF Similarity Index**         |                                                                                                                            |                 |
| `IVF_NPROBE`                                | Number of nearest IVF cells probed per query — the dominant recall/latency knob. Higher = better recall + slower queries.   | `1024`          |
| `IVF_NLIST_MAX`                             | Upper cap on the number of IVF cells (coarse centroids) created at build time. Requires an index rebuild after change.      | `8192`          |
| `IVF_TRAIN_POINTS_PER_CELL`                 | Target training vectors per cell; the training sample is this × nlist, capped at the library size (FAISS floor ~39). Requires an index rebuild after change. | `50`            |
| `IVF_MAX_CELL_MB`                           | Oversized cells are split at build time so no single stored cell exceeds this many MB. Requires an index rebuild after change. | `12`            |
| `IVF_MAX_PART_SIZE_MB`                      | Hard cap (MB) on every stored BYTEA value (cells and directory parts) in Postgres. Requires an index rebuild after change.  | `50`            |
| `IVF_QUERY_CACHE_MB`                        | Hard cap (MB) on the per-request decoded-vector cache.                                                                      | `128`           |
| `IVF_READ_BATCH_CELLS`                      | Number of cells fetched per database round-trip during a query.                                                            | `16`            |
| `IVF_GLOBAL_CACHE_MB`                       | Hard cap (MB) on the process-wide, cross-request decoded-cell cache shared by all indexes. `0` disables it.                 | `1024`          |
| `IVF_PRELOAD_ALL`                           | When `true`, stream every cell into the global cache at load time (fully in-memory IVF), still bounded by `IVF_GLOBAL_CACHE_MB`. | `false`         |
| `IVF_GLOBAL_CACHE_IDLE_SECONDS`             | Drop the whole global cell cache after this many seconds with no access (frees idle RAM). `0` = never drop.                 | `300`           |
| `IVF_RESULT_CACHE_SECONDS`                  | TTL (seconds) for cached similar-song / max-distance results so repeated identical queries are instant. `0` disables it.    | `300`           |
| `IVF_RESULT_CACHE_MAX`                      | Maximum number of distinct cached query results per result cache.                                                          | `2048`          |
| `IVF_MAX_DISTANCE_NPROBE`                   | Farthest cells probed when computing the max-distance display value (reverse-IVF). `0` or a value ≥ nlist forces an exact full scan. | `256`           |
| `IVF_DISK_CACHE_ENABLED`                    | When `true`, export each index's cells to a local file at load and serve queries via mmap (OS page cache) instead of reading from Postgres per query. `false` = read from Postgres. | `true`          |
| `IVF_DISK_CACHE_IDLE_SECONDS`               | Drop the resident (RSS) pages of every disk-cache mmap after this many seconds with no query (mapping stays; the next query re-faults from disk). Frees idle RAM. `0` = never drop. | `300`           |
| `SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT`   | It enable the possibility of use the `MAX_SONGS_PER_ARTIST` also in similar song                                          | `true`          |
| `SIMILARITY_RADIUS_DEFAULT`                 | Default behavior for radius similarity mode. When `true`, similarity results may be re-ordered using the radius (bucketed) algorithm for better listening paths. | `true`          |
| **Sonic Fingerprint General**               |                                                                                                                            |                 |
| `SONIC_FINGERPRINT_NEIGHBORS`               | Default number of track for the sonic fingerprint                                                                         | `100`           |
| `SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM`     | **Navidrome only.** Max tracks a single album may contribute to the fingerprint seed pool, so one large album (e.g. a 100+ track DJ mix) cannot dominate. Other media servers fetch top songs directly and ignore this. | `3`             |
| **Song Alchemy General**                     |                                                                                                                            |                 |
| `ALCHEMY_DEFAULT_N_RESULTS`                  | Number of similar songs to return when creating the Alchemy result (default).                                              | `100`           |
| `ALCHEMY_MAX_N_RESULTS`                      | Maximum number of similar songs to return for Alchemy results.                                                             | `200`           |
| `ALCHEMY_TEMPERATURE`                        | Temperature for probabilistic sampling in Song Alchemy (softmax temperature). Use `0.0` for deterministic selection.       | `1.0`           |
| **Similar Song and Song Path Duplicate filtering General** |                                                                                                            |                 |
| `DUPLICATE_DISTANCE_THRESHOLD_COSINE`       | Less than this cosine distance the track is a duplicate.                                                                  | `0.01`          |
| `DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN`    | Less than this euclidean distance the track is a duplicate.                                                               | `0.15`          |
| `DUPLICATE_DISTANCE_CHECK_LOOKBACK`         | How many previous song need to be checked for duplicate.                                                                  | `1`             |
| `MOOD_SIMILARITY_THRESHOLD`                 | Maximum normalized distance for mood similarity filtering. Lower value will give more importance to mood                  | `0.15`          |
| **Song Path General**                       |                                                                                                                            |                 |
| `PATH_DISTANCE_METRIC`                      | The distance metric to use for pathfinding. Options: 'angular', 'euclidean'                                               | `angular`       |
| `PATH_DEFAULT_LENGTH`                       | Default number of songs in the path if not specified in the API request                                                   | `25`            |
| `PATH_AVG_JUMP_SAMPLE_SIZE`                 | Number of random songs to sample for calculating the average jump distance                                                | `200`           |
| `PATH_CANDIDATES_PER_STEP`                  | Number of candidate songs to retrieve from the IVF index for each step in the path                                              | `25`            |
| `PATH_LCORE_MULTIPLIER`                     | It multiply the number of centroid created based on the distance. Higher is better for distant song and worst for nearest. | `3`             |
| `PATH_FIX_SIZE`                             | When `true`, path generation will attempt to produce exactly the requested path length using centroid merging and backfilling. When `false`, the algorithm will perform a single best pick per centroid and may return a shorter path. Can be overridden per-request via the `path_fix_size` query parameter. | `false`         |
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
| `AI_MODEL_PROVIDER`                         | AI provider: `OLLAMA`, `GEMINI`, `MISTRAL`, `OpenAI` or `NONE`.                           | `NONE`                                 |
| `AI_REQUEST_TIMEOUT_SECONDS`                | Timeout (in seconds) for AI API requests. Increase for slower hardware or larger models.  | `300`                                  |
| `TOP_N_ELITES`                              | Number of best solutions kept as elites.                                                  | `10`                                   |
| `SAMPLING_PERCENTAGE_CHANGE_PER_RUN`        | Percentage of songs to swap out in the stratified sample between runs (0.0 to 1.0).       | `0.2`                                  |
| `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`    | Minimum number of songs to target per stratified genre during sampling.                   | `100`                                  |
| `STRATIFIED_SAMPLING_TARGET_PERCENTILE`     | Percentile of genre song counts to use for target songs per stratified genre.             | `50`                                   |
| `OLLAMA_SERVER_URL`                         | URL for your Ollama instance (if `AI_MODEL_PROVIDER` is OLLAMA).                          | `http://<your-ip>:11434/api/generate` |
| `OLLAMA_MODEL_NAME`                         | Ollama model to use (if `AI_MODEL_PROVIDER` is OLLAMA).                                   | `qwen3.5:9b`                          |
| `GEMINI_MODEL_NAME`                         | Gemini model to use (if `AI_MODEL_PROVIDER` is GEMINI).                                   | `gemini-2.5-pro`                      |
| `MISTRAL_MODEL_NAME`                        | Mistral model to use (if `AI_MODEL_PROVIDER` is MISTRAL).                                 | `ministral-3b-latest`                  |
| `OPENAI_MODEL_NAME`                         | OpenAI or OpenRouter model to use (if `AI_MODEL_PROVIDER` is OPENAI). Falls back to `OLLAMA_MODEL_NAME` if unset. | `llama3.1:8b` |
| `OPENAI_SERVER_URL`                         | URL for OpenAI / OpenRouter (if `AI_MODEL_PROVIDER` is OPENAI). Falls back to `OLLAMA_SERVER_URL` if unset. | `http://<your-ip>:11434/api/generate` |
| **Scoring Weights**                         |                                                                                            |                                        |
| `SCORE_WEIGHT_DIVERSITY`                    | Weight for inter-playlist mood diversity.                                                 | `2.0`                                  |
| `SCORE_WEIGHT_PURITY`                       | Weight for playlist purity (intra-playlist mood consistency).                             | `1.0`                                  |
| `SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY`      | Weight for inter-playlist 'other feature' diversity.                                      | `0.0`                                  |
| `SCORE_WEIGHT_OTHER_FEATURE_PURITY`         | Weight for intra-playlist 'other feature' consistency.                                    | `0.0`                                  |
| `SCORE_WEIGHT_SILHOUETTE`                   | Weight for Silhouette Score (cluster separation).                                         | `0.0`                                  |
| `SCORE_WEIGHT_DAVIES_BOULDIN`               | Weight for Davies-Bouldin Index (cluster separation).                                     | `0.0`                                  |
| `SCORE_WEIGHT_CALINSKI_HARABASZ`            | Weight for Calinski-Harabasz Index (cluster separation).                                  | `0.0`                                  |
| **Lyrics & SemGrove (Semantic + Groove) Search** |                                                                                      |                                        |
| `MUSICSERVER_LYRICS_TIMEOUT`                | Timeout (seconds) for fetching embedded lyrics from the media server (Jellyfin / Emby / Navidrome / Lyrion). Increase if your server fetches lyrics on-the-fly via plugins that may take several seconds to respond. | `2.5` |
| `LYRICS_ENABLED`                            | When `false`, the lyrics transcription/embedding step is skipped entirely during analysis. | `true`                                |
| `LYRICS_API_ENABLE`                         | When `true`, fetches lyrics from external APIs (slots 1 & 2) before falling back to Whisper-small ASR transcription. | `true`               |
| `LYRICS_ASR_ENABLE`                         | When `false`, skips the Whisper-small ASR transcription stage entirely. Tracks with no media-server lyrics and no external-API lyrics are marked as instrumental (sentinel embedding) instead of being transcribed. | `true` |
| `LYRICS_MUSICNN_SKIP`                       | When `true`, a MusicNN `instrumental` tag short-circuits lyrics analysis straight to the instrumental sentinel (no media-server / API / ASR lookup). Set `false` to ignore the MusicNN tag and run the full lyrics pipeline on every track regardless. | `true` |
| `LYRICS_API_1_URL_TEMPLATE`                 | URL template for lyrics API slot 1. Use `{artist_param}`, `{title_param}` placeholders. e.g. `https://example.com/api/get?{artist_param}={artist}&{title_param}={title}` | `""` |
| `LYRICS_API_1_ARTIST_PARAM`                 | Query parameter name for the artist in API slot 1.                                        | `artist_name`                          |
| `LYRICS_API_1_TITLE_PARAM`                  | Query parameter name for the track title in API slot 1.                                   | `track_name`                           |
| `LYRICS_API_1_LYRICS_FIELD`                 | JSON field name containing the lyrics text in the API slot 1 response.                    | `plainLyrics`                          |
| `LYRICS_API_1_APIKEY_PARAM`                 | Query parameter name for the API key in slot 1 (leave empty if no key needed).            | `""`                                   |
| `LYRICS_API_1_APIKEY_VALUE`                 | API key value for slot 1.                                                                  | `""`                                   |
| `LYRICS_API_1_TIMEOUT`                      | HTTP timeout in seconds for API slot 1.                                                    | `5.0`                                  |
| `LYRICS_API_2_URL_TEMPLATE`                 | URL template for lyrics API slot 2 (fallback after slot 1).                               | `""`                                   |
| `LYRICS_API_2_ARTIST_PARAM`                 | Query parameter name for the artist in API slot 2.                                        | `artist`                               |
| `LYRICS_API_2_TITLE_PARAM`                  | Query parameter name for the track title in API slot 2.                                   | `title`                                |
| `LYRICS_API_2_LYRICS_FIELD`                 | JSON field name containing the lyrics text in the API slot 2 response.                    | `lyrics`                               |
| `LYRICS_API_2_APIKEY_PARAM`                 | Query parameter name for the API key in slot 2.                                            | `""`                                   |
| `LYRICS_API_2_APIKEY_VALUE`                 | API key value for slot 2.                                                                  | `""`                                   |
| `LYRICS_API_2_TIMEOUT`                      | HTTP timeout in seconds for API slot 2.                                                    | `5.0`                                  |
| `VAD_VOICE_RECOGNITION`                     | Minimum seconds of voiced audio Silero VAD must detect before a track is sent to the Whisper-small ASR engine for lyric transcription. Tracks below this threshold are treated as instrumental/ambient and skip ASR entirely (the instrumental embedding sentinel is used instead). Use this knob to fine-tune instrumental/ambient song recognition in the lyrics analysis pipeline. Setting it very high (e.g. `1000`) effectively disables ASR transcription for every track, since no song can reach that much voiced audio within the 4-minute analysis clip. | `25` |
| `LYRICS_ASR_BEAM_SIZE`                      | Beam search width for the Whisper-small ASR decoder. 1 = pure greedy (fastest, most error-prone), 2 = sweet spot (catches stuck-loop attractors at ~2× greedy cost), 5 = Whisper-upstream default (max quality, ~5× cost). Each extra beam adds one extra decoder.run per generated token plus its own KV cache (~30-80 MB at a full 30 s chunk). | `5` |
| `LYRICS_ASR_MIN_AVG_LOGPROB`                | General avg_logprob floor for ASR output. Whisper-small's per-chunk avg_logprob is averaged over the track; if the result is below this threshold the transcript is dropped as likely hallucination and the track is treated as instrumental. Values are negative — closer to `0` is stricter (rejects more), more negative is looser (accepts more). `-1.0` is a permissive global floor that catches only truly degenerate transcriptions. | `-1.0` |
| `LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB`        | Additional avg_logprob floor applied only when Whisper reports a non-English language. Whisper-small is English-biased, so legitimate non-English transcriptions (CJK, Cyrillic, Arabic, etc.) naturally score lower in the `-0.5` to `-0.8` range; set this looser than the English floor (more negative) to avoid dropping valid foreign-language lyrics. Raise toward `-0.5` if you see garbage non-English transcriptions slipping through. | `-0.85` |
| `LYRICS_TEXT_MAX_COMPRESSION_RATIO`         | Compression ratio (zlib) used to filter out text that is not real lyrics. Highly repetitive content compresses far more than real lyrics, so text above this ratio is dropped before embedding. Set to `0` to disable the gate. | `15.0` |
| `SEM_GROVE_WEIGHT_LYRICS`                   | Contribution of the lyrics embedding to the merged SemGrove cosine similarity (squared scale factor, [0.0–1.0]). Requires index rebuild after change. | `0.75` |
| `SEM_GROVE_WEIGHT_AUDIO`                    | Contribution of the MusicNN audio embedding to the merged SemGrove cosine similarity (squared scale factor, [0.0–1.0]). Requires index rebuild after change. | `0.25` |


> ⚠️ **The only officially supported model is `qwen3.5:9b` or `qwen3.5:4b` for faster one**. Compatibility testing is done exclusively against it. Other models below were tested and may work, but **use them at your own risk** — issues opened for untested or arbitrary models could be closed. Different models behave differently and outputs vary between runs.

> ℹ️ **The models listed below were tested in the past and will not be retested going forward.** They are documented for reference only.

**Self-hosted (Ollama):** `gemma3:4b`, `ministral-3:3b` (fastest), plus: llama3.1:8b, llama3.2:1b/3b, gemma3:1b, qwen3:0.6b/1.7b, qwen2.5:1.5b, qwen3.5:0.8b/2b, deepseek-r1:1.5b, phi4-mini:3.8b, lfm2.5-thinking:1.2b.

**Cloud, tested March 2026:** `claude-sonnet-4.6` (best), `claude-haiku-4.5`, `gemini-3-flash-preview`. Earlier: mistral:7b, llama3.1:8b, gemini-2.5-pro, gemini-1.5-flash-latest.

You can use either an external AI API or self-host with Ollama — deployment example here:

* https://github.com/NeptuneHub/k3s-supreme-waffle/tree/main/ollama

## OpenAI-compatible hosted providers

AudioMuse-AI can use hosted services that expose an OpenAI-compatible chat completions API through the existing `OPENAI` provider. [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=AudioMuse-AI) is one example: point `OPENAI_SERVER_URL` at its OpenAI-compatible endpoint and keep using a model that has been validated for AudioMuse-AI unless you have tested another model with your library.

Example Atlas Cloud configuration:

```env
AI_MODEL_PROVIDER=OPENAI
OPENAI_SERVER_URL=https://api.atlascloud.ai/v1/chat/completions
OPENAI_MODEL_NAME=qwen3.5:9b
OPENAI_API_KEY=<atlas-key>
```
