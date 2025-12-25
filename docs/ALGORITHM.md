# **Algorithm description**

This document provides a detailed functional (high-level) and technical (algorithm-level) breakdown of the core features of the AudioMuse-AI application. It covers the data ingestion and analysis process, the playlist generation and clustering process, the similarity-based playlist creation process, the song pathfinding process, the vector-based song manipulation (alchemy) process, the interactive music map visualization, the personalized sonic fingerprint generation, and the AI-driven instant playlist creation via chat.

## **Table of Contents**

0. [Architectural Design](#0-architectural-design)  
   * [0.1. Functional Analysis (High-Level)](#01-functional-analysis-high-level)  
   * [0.2. Technical Analysis (Algorithm-Level)](#02-technical-analysis-algorithm-level)  
   * [0.3. Environment Variable Configuration](#03-environment-variable-configuration)  
   * [0.4. Concurrency Algorithm Deep Dive](#04-concurrency-algorithm-deep-dive)  

1. [Song Analysis](#1-song-analysis)  
   * [1.1. Functional Analysis (High-Level)](#11-functional-analysis-high-level)  
   * [1.2. Technical Analysis (Algorithm-Level)](#12-technical-analysis-algorithm-level)  
   * [1.3. Environment Variable Configuration](#13-environment-variable-configuration)  
2. [Song Clustering](#2-song-clustering)  
   * [2.1. Functional Analysis (High-Level)](#21-functional-analysis-high-level)  
   * [2.2. Technical Analysis (Algorithm-Level)](#22-technical-analysis-algorithm-level)  
   * [2.3. Clustering Deep Dive (Advanced Details)](#23-clustering-deep-dive-advanced-details)  
   * [2.4. Environment Variable Configuration](#24-environment-variable-configuration)  
3. [Playlist from Similar Song](#3-playlist-from-similar-song)  
   * [3.1. Functional Analysis (High-Level)](#31-functional-analysis-high-level)  
   * [3.2. Technical Analysis (Algorithm-Level)](#32-technical-analysis-algorithm-level)  
   * [3.3. Environment Variable Configuration](#33-environment-variable-configuration)  
4. [Song Path](#4-song-path)  
   * [4.1. Functional Analysis (High-Level)](#41-functional-analysis-high-level)  
   * [4.2. Technical Analysis (Algorithm-Level)](#42-technical-analysis-algorithm-level)  
   * [4.3. Environment Variable Configuration](#43-environment-variable-configuration)  
5. [Song Alchemy](#5-song-alchemy)  
   * [5.1. Functional Analysis (High-Level)](#51-functional-analysis-high-level)  
   * [5.2. Technical Analysis (Algorithm-Level)](#52-technical-analysis-algorithm-level)  
   * [5.3. Environment Variable Configuration](#53-environment-variable-configuration)  
6. [Music Map](#6-music-map)  
   * [6.1. Functional Analysis (High-Level)](#61-functional-analysis-high-level)  
   * [6.2. Technical Analysis (Algorithm-Level)](#62-technical-analysis-algorithm-level)  
   * [6.3. Environment Variable Configuration](#63-environment-variable-configuration)  
7. [Sonic Fingerprint](#7-sonic-fingerprint)  
   * [7.1. Functional Analysis (High-Level)](#71-functional-analysis-high-level)  
   * [7.2. Technical Analysis (Algorithm-Level)](#72-technical-analysis-algorithm-level)  
   * [7.3. Environment Variable Configuration](#73-environment-variable-configuration)  
8. [Instant Playlist (Chat)](#8-instant-playlist-chat)  
   * [8.1. Functional Analysis (High-Level)](#81-functional-analysis-high-level)  
   * [8.2. Technical Analysis (Algorithm-Level)](#82-technical-analysis-algorithm-level)  
   * [8.3. Environment Variable Configuration](#83-environment-variable-configuration)  
9. [Database Cleaning](#9-database-cleaning)  
   * [9.1. Functional Analysis (High-Level)](#91-functional-analysis-high-level)  
   * [9.2. Technical Analysis (Algorithm-Level)](#92-technical-analysis-algorithm-level)  
   * [9.3. Environment Variable Configuration](#93-environment-variable-configuration)  
10. [Scheduled Tasks (Cron)](#10-scheduled-tasks-cron)  
   * [10.1. Functional Analysis (High-Level)](#101-functional-analysis-high-level)  
   * [10.2. Technical Analysis (Algorithm-Level)](#102-technical-analysis-algorithm-level)  
   * [10.3. Environment Variable Configuration](#103-environment-variable-configuration)

## **0. Architectural Design**

This chapter describes the overall system architecture of AudioMuse-AI: the runtime components, data flows, deployment model, and operational considerations that tie together the web UI, background workers, vector index, and model artifacts.

### **0.1. Functional Analysis (High-Level)**

From a user's and operator's perspective, the system provides three broad capabilities:

- Interactive UI and APIs: a Flask application that serves the web UI (blueprints) and a REST API for user actions (analysis, clustering, similarity, alchemy, map, cron, cleaning, chat). The web process handles short requests, status polling, and serving static assets.
- Background processing: long-running CPU/IO-heavy jobs (analysis, clustering, cleaning, indexing, map projection) are executed by RQ workers connected to Redis. Jobs are enqueued by the web process and surfaced to the UI via the task status table.
- Fast similarity/search: an in-memory vector index (Voyager) built from stored embeddings provides sub-second nearest-neighbor queries for similarity, pathfinding, and alchemy features.

High-level flows:

- Analysis flow: UI -> POST /api/analysis/start -> enqueue `tasks.analysis.run_analysis_task` -> worker downloads audio, runs ONNX inference, writes `score` and `embedding` rows -> occasional `build_and_store_voyager_index` runs update the DB-stored index and publish a reload message.
- Clustering flow: UI -> POST /api/clustering/start -> enqueue evolutionary clustering batches -> worker returns best solutions -> post-processing writes playlists to media server adapters.
- Instant Playlist (Chat): UI -> /chat/api/chatPlaylist -> call AI providers (Ollama/Gemini/Mistral) via `ai.py` -> validate/sanitize structured query -> execute read-only query against PostgreSQL -> return results for optional playlist creation.

### **0.2. Technical Analysis (Algorithm-Level)**

Core components and responsibilities:

- Web App (Flask): Registers blueprints (chat, analysis, clustering, voyager, alchemy, map, cron, cleaning, sonic fingerprint, path, collection, external) and starts lightweight background threads such as `listen_for_index_reloads` and the cron manager. The web app persists task metadata to the `task_status` table and exposes endpoints for task control and status.

- Background Workers (RQ): Workers run RQ jobs defined under `tasks/` (e.g., `tasks.analysis`, `tasks.clustering`, `tasks.cleaning`, `tasks.song_alchemy`). Workers fetch jobs from Redis queues (`rq_queue_high`, `rq_queue_default`) and write progress to the database and job meta for UI consumption. RQ handles retries and job lifecycle.

- Redis: Used for RQ queueing, pub/sub notifications (index reloads), and short-lived job coordination.

- PostgreSQL: Source-of-truth for persistent data: `score`, `embedding`, `voyager_index_data`, `task_status`, `cron`, and playlist metadata. Jobs write analysis results and index binaries to the DB.

- Voyager Index: Built in workers by reading embeddings from PostgreSQL, then serialized and saved to the `voyager_index_data` table. The web process loads the binary index into memory at startup and listens for a pub/sub `reload` to replace it live.

- ONNX Models & Audio Stack: Analysis uses ONNX Runtime to run embedding and prediction models. Audio loading uses `librosa` with a `pydub`/ffmpeg fallback for resilient decoding. The Docker image pre-fetches ONNX model files and pins runtime libs to ensure consistent behavior across environments.

- Media Server Adapters: `mediaserver.py` provides adapters for Jellyfin, Navidrome, Emby, etc., enabling playlist creation and reading play-history for the Sonic Fingerprint feature.

Deployment considerations (informed by `Dockerfile`):

- Multi-stage Docker build: separate stage to download model artifacts and a final runtime stage that pins OS and Python dependencies. The container sets ONNX/CPU environment flags to provide deterministic runtime behavior across CPUs (ORT_DISABLE_ALL_OPTIMIZATIONS, ORT_DISABLE_AVX512, MKL flags).
- Supervisor or process manager: the image supports running the web server or RQ worker processes under Supervisor based on `SERVICE_TYPE` so both web and worker roles are reproducible.

Scalability & safety:

- Scale workers horizontally by running multiple worker containers pointed at the same Redis and DB. Only the web process should run the index-loading thread / cron manager to avoid duplicate cron enqueues unless leader election is used.
- Safety limits (e.g., `CLEANING_SAFETY_LIMIT`, `ALCHEMY_MAX_N_RESULTS`) prevent destructive or excessively large operations triggered via AI or scheduled jobs.

### **0.3. Environment Variable Configuration**

Key environment variables that shape architecture and operational behavior (non-exhaustive):

- Core infra:
   * `REDIS_URL` — connection string for Redis used by RQ and pub/sub.
   * `DATABASE_URL` — PostgreSQL connection string for persistent data.
   * `TEMP_DIR` — path where audio files are downloaded/processed.

- Models & runtime tuning:
   * `EMBEDDING_MODEL_PATH`, `PREDICTION_MODEL_PATH` — filesystem paths to ONNX model files (when not downloaded by Docker).
   * `ORT_DISABLE_ALL_OPTIMIZATIONS`, `ORT_DISABLE_AVX512`, `ORT_FORCE_SHARED_PROVIDER` — runtime flags set in Docker to stabilize ONNX behavior across CPUs.

- Job & queue limits:
   * `MAX_QUEUED_ANALYSIS_JOBS`, `MAX_CONCURRENT_BATCH_JOBS`, `ITERATIONS_PER_BATCH_JOB` — control parallelism and batch sizes for analysis/clustering.
   * `REBUILD_INDEX_BATCH_SIZE` — controls how often index rebuilds occur during large analysis runs.

Additional DB & deployment knobs (explicit)

* `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` — Individual components used to construct `DATABASE_URL` when an explicit `DATABASE_URL` is not provided. Useful in containerized or Kubernetes setups where secrets are mounted per-value.
* `AI_CHAT_DB_USER_NAME`, `AI_CHAT_DB_USER_PASSWORD` — Optional credentials for a restricted, read-only database role used when executing AI-generated SQL from the Instant Playlist (Chat) feature. The application creates/uses a low-privilege role to run SELECT-only queries when the chat flow is enabled; document these values in the Instant Playlist section as well.

- AI & provider settings:
   * `AI_MODEL_PROVIDER`, `OLLAMA_SERVER_URL`, `OLLAMA_MODEL_NAME`, `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `MISTRAL_API_KEY`, `MISTRAL_MODEL_NAME` — control how AI naming/chat is performed.

- Safety & result caps:
   * `ALCHEMY_MAX_N_RESULTS`, `ALCHEMY_DEFAULT_N_RESULTS`, `CLEANING_SAFETY_LIMIT`, `MAX_SONGS_PER_ARTIST` — enforce limits on returned/affected rows.

Operational notes:

- Logging & observability: application logs should be captured by the container runtime and RQ job meta status is stored in `task_status`. Keep API keys out of logs.
- Index reloads: the worker performing `build_and_store_voyager_index` must publish `index-updates` to notify the web process; the web process must run `listen_for_index_reloads` so it can reload the binary index in memory without restart.

### **0.4. Concurrency Algorithm Deep Dive**

This section expands on the runtime concurrency and orchestration patterns used across analysis, clustering, cleaning, and other long-running tasks. It focuses on implementation details that help operators understand batching, cancellation, monitoring, and scaling decisions.

Hierarchy & Batching

- Parent/child task model: Long jobs (e.g., full-library analysis, evolutionary clustering) are implemented as a *parent* RQ task that enumerates work and enqueues *child* tasks. This reduces queue churn and improves observability: one parent job corresponds to a manageable set of medium-sized child jobs instead of thousands of tiny tasks.
- Batch sizing: Tasks are grouped into batches (configured via `ITERATIONS_PER_BATCH_JOB` for clustering, and `REBUILD_INDEX_BATCH_SIZE` for analysis). Batches are sized to balance per-task overhead and worker throughput. Smaller batches increase responsiveness (faster progress updates, earlier index availability) but add queue and DB overhead.

Efficient Parallel Execution

- Concurrency limits: The parent enforcer limits how many child tasks are active/pending (`MAX_QUEUED_ANALYSIS_JOBS`, `MAX_CONCURRENT_BATCH_JOBS`) to avoid overloading DB/IO and to keep memory/CPU consumption predictable across worker nodes.
- Worker sizing guidance: For most medium libraries, 2–4 worker processes provide good throughput. For very large libraries or high-throughput clusters, increase workers but maintain a single web process (or leader) responsible for index reloads and cron scheduling to avoid duplicate enqueues.

Cooperative Cancellation & Robustness

- Cooperative cancellation: All long-running tasks periodically check their status and the status of their parent task in the `task_status` table. If revoked, they gracefully stop, clean temporary files, and update status. This allows preempting jobs mid-execution rather than only preventing new jobs from starting.
- Stuck-batch handling: Batch tasks implement timeouts and heartbeats. The orchestrator enforces `CLUSTERING_BATCH_TIMEOUT_MINUTES` and will kill or mark batches as failed if they exceed allowed runtime. `CLUSTERING_MAX_FAILED_BATCHES` controls when the main job gives up after repeated failures.

Monitoring & Observability

- Status propagation: Tasks write progress and logs to the `task_status` table and to RQ job meta; the web UI surfaces these updates so operators can see overall progress and per-batch details. Key status fields include STARTED/PROGRESS/SUCCESS/FAILURE/REVOKED and free-text logs.
- Batch-level artifacts: Each batch returns its best-scoring result to the orchestrator. The orchestrator maintains an in-memory list of elite solutions (TOP_N_ELITES) and uses them as seeds for exploitation mutations in later batches.

Implementation pointers (current code locations)

- Orchestration & monitoring: `tasks/clustering.py` (run_clustering_task, run_clustering_batch_task, _monitor_and_process_batches).
- Analysis batching: `tasks/analysis.py` (run_analysis_task, analyze_album_task, build_and_store_voyager_index).
- Cancellation & status: `tasks/commons.py` and `tasks/__init__.py` helpers used by multiple tasks to read/update `task_status` and check revocation.

Relevant environment variables and tuning knobs

- `ITERATIONS_PER_BATCH_JOB`, `MAX_CONCURRENT_BATCH_JOBS`, `CLUSTERING_BATCH_TIMEOUT_MINUTES`, `CLUSTERING_MAX_FAILED_BATCHES`, `CLUSTERING_BATCH_CHECK_INTERVAL_SECONDS` — tuning these directly affects parallelism, latency to first results, and fault tolerance.
- `MAX_QUEUED_ANALYSIS_JOBS`, `REBUILD_INDEX_BATCH_SIZE` — control analysis job parallelism and how often the Voyager index is rebuilt during a large analysis run.

Operational recommendations

- Reserve one web process as the "leader" for index reloads and cron to avoid duplicate scheduled enqueues when running multiple web replicas.
- Start with conservative batch sizes and worker counts; increase iteratively while monitoring DB load and queue saturation.
- Configure sensible timeouts and a small `CLUSTERING_MAX_FAILED_BATCHES` during early tuning runs to fail fast on misconfiguration.


## **1\. Song Analysis**

This section details the functional (high-level) and technical (algorithm-level) processes for the "Song Analysis" feature.

### **1.1. Functional Analysis (High-Level)**

From a user's perspective, the "Song Analysis" feature is the core data-gathering process of the application. It is the necessary first step to populate the database with the audio features required for all other functionalities, such as clustering, playlist generation, and similarity searches.

#### **Key User Interactions & Workflow**

1. **Initiation:** The user navigates to the "Analysis and Clustering" page (the application's root, index.html).  
2. **Configuration:** The user can set parameters before starting the task.  
   * **Basic View:** The primary option is "Number of Recent Albums." Setting this to 0 or a negative number instructs the system to scan the *entire* media server library, not just recent additions.  
   * **Advanced View:** The user can also configure "Top N Moods," which defines how many of the top-scoring moods are saved per track (this is primarily for database efficiency, as the full mood vector is not stored).  
3. **Execution:** The user clicks the **"Start Analysis"** button. This action is asynchronous, meaning it starts a long-running job on the server, and the UI does not freeze.  
4. **Monitoring & Feedback:** The "Task Status" panel immediately updates to show the newly created "main\_analysis" task. The user can monitor its progress in real-time:  
   * **Task ID:** A unique identifier for the entire analysis job.  
   * **Running Time:** A live-updating timer.  
   * **Status:** Shows the current state (e.g., STARTED, PROGRESS, SUCCESS, FAILURE).  
   * **Progress:** A percentage bar indicating overall completion.  
   * **Details / Log:** Provides human-readable logs, such as which album is currently being processed or how many albums have been launched, skipped, or completed.  
5. **Control:** The user can click the **"Cancel Current Task"** button at any time to stop the main analysis task and any in-progress album analysis tasks it has spawned.  
6. **Outcome:** Once the task is complete, the application's database is populated with detailed audio features and vector embeddings for all new or updated songs. The system also builds (or rebuilds) a fast-search vector index (Voyager) in the background. This analyzed data is now ready to be used by the "Start Clustering," "Song Path," and other features.

#### **Core Purpose**

The functional purpose is to **"scan, analyze, and index"** the music library. It intelligently skips music that has already been analyzed, ensuring that running the analysis again only processes new or changed albums, making subsequent runs much faster.

### **1.2. Technical Analysis (Algorithm-Level)**

The technical process is a distributed, multi-stage pipeline orchestrated by a main task that spawns child tasks for parallel processing. The entire process relies heavily on environment variables (see section 1.3) to connect to media servers, databases, and configure task behavior.

#### **Stage 1: API Call & Task Enqueueing (The "Spark")**

1. **Route:** The "Start Analysis" button sends a POST request to the /api/analysis/start endpoint (defined in app\_analysis.py).  
2. **Payload:** The request body contains num\_recent\_albums and top\_n\_moods. If not provided, the task will use defaults from NUM\_RECENT\_ALBUMS and TOP\_N\_MOODS environment variables.  
3. **Job Creation:**  
   * A unique job\_id (UUID) is generated.  
   * The endpoint enqueues a new high-priority job in the **RQ (Redis Queue)**, configured via REDIS\_URL.  
   * The function enqueued is tasks.analysis.run\_analysis\_task.  
   * The task\_id is returned to the frontend.

#### **Stage 2: Main Orchestration Task (run\_analysis\_task)**

This function, running in an RQ worker, acts as the "orchestrator."

1. **Get Albums:** It calls get\_recent\_albums(num\_recent\_albums). This function's behavior is determined by MEDIASERVER\_TYPE (e.g., "jellyfin", "navidrome") and uses the corresponding credentials (e.g., JELLYFIN\_URL, JELLYFIN\_TOKEN, etc.) and MUSIC\_LIBRARIES to filter the scan.  
2. **Iterate & Check:** It loops through every album, checking against the PostgreSQL database (configured via DATABASE\_URL) to see if all tracks already exist in the score and embedding tables.  
3. **Spawn Child Tasks:** For new or incomplete albums, it enqueues a tasks.analysis.analyze\_album\_task child task.  
4. **Manage Parallelism:** The main task monitors the number of active\_jobs and limits concurrent enqueued album tasks to MAX\_QUEUED\_ANALYSIS\_JOBS.  
5. **Batch Indexing (Voyager Index Generation):** After every REBUILD\_INDEX\_BATCH\_SIZE completed child tasks, it triggers build\_and\_store\_voyager\_index (from voyager\_manager.py). This function:  
   * Fetches all item\_id and embedding vectors from the PostgreSQL database (embedding table).  
   * Builds a new Voyager index in memory using these vectors. Configuration parameters like VOYAGER\_METRIC, VOYAGER\_EF\_CONSTRUCTION, and VOYAGER\_M are used here.  
   * Saves the built index binary data and an ID map (Voyager internal ID \-\> item\_id) back to the database (voyager\_index\_data table).  
   * Publishes a reload message via Redis (redis\_conn.publish('index-updates', 'reload')) to notify the web server process(es) to load the new index into memory.  
6. **Finalize:** After all albums, it runs a final index build and publishes the reload message.

#### **Stage 3: Album-Level Task (analyze\_album\_task)**

This function runs in parallel for each album.

1. **Fetch Tracks:** It calls get\_tracks\_from\_album(album\_id) (again using MEDIASERVER\_TYPE and credentials).  
2. **Download:** It calls download\_track(TEMP\_DIR, item) to download the audio file to the temporary directory specified by the TEMP\_DIR variable.  
3. **Analyze:** It calls the core analyze\_track function.  
4. **Save to DB:** It calls save\_track\_analysis\_and\_embedding to write results to the PostgreSQL database.

#### **Stage 4: Core Audio Analysis (analyze\_track)**

This is the heart of the analysis, using librosa and onnxruntime.

1. **Audio Loading:**  
   * It uses robust\_load\_audio\_with\_fallback, which attempts to load the audio file. This process is constrained by the AUDIO\_LOAD\_TIMEOUT variable to prevent stuck jobs on corrupt files.  
2. **Basic Feature Extraction:** Extracts Tempo, Energy, and Key/Scale.  
3. **Spectrogram Generation:** Converts the audio into Mel Spectrogram patches compatible with the ONNX models.  
4. **ONNX Model Inference (Main Models):**  
   * **Embedding:** Loads the model from EMBEDDING\_MODEL\_PATH to generate embeddings\_per\_patch.  
   * **Mood Prediction:** Loads the model from PREDICTION\_MODEL\_PATH and feeds it the embeddings\_per\_patch to get mood probabilities.  
5. **ONNX Model Inference (Secondary Models):**  
   * It loops through other models (e.g., DANCEABILITY\_MODEL\_PATH, AGGRESSIVE\_MODEL\_PATH, etc.) and feeds them the *same* embeddings\_per\_patch to get scores for other features.  
6. **Final Output:**  
   * Returns an analysis dictionary (with scalars like tempo, energy, and mood scores) and a single averaged processed\_embeddings vector for the entire track. This vector is what gets stored in the embedding table and indexed by Voyager.

### **1.3. Environment Variable Configuration**

The Song Analysis functionality is configured by the following environment variables (from config.py):

#### **Core Infrastructure**

* REDIS\_URL: **(Required)** The connection string for the Redis server, used for task queueing (RQ) and status management.  
* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** The connection string for the PostgreSQL database where all analysis results are stored and the Voyager index data is saved.

#### **Media Server**

* MEDIASERVER\_TYPE: **(Required)** Specifies the media server to connect to. (e.g., jellyfin, navidrome, emby, lyrion, mpd).  
* MUSIC\_LIBRARIES: (Optional) A comma-separated list of library names to scan. If empty, all music libraries are scanned.  
* JELLYFIN\_URL, JELLYFIN\_USER\_ID, JELLYFIN\_TOKEN: Credentials for Jellyfin (if MEDIASERVER\_TYPE="jellyfin").  
* EMBY\_URL, EMBY\_USER\_ID, EMBY\_TOKEN: Credentials for Emby (if MEDIASERVER\_TYPE="emby").  
* NAVIDROME\_URL, NAVIDROME\_USER, NAVIDROME\_PASSWORD: Credentials for Navidrome (if MEDIASERVER\_TYPE="navidrome").  
* LYRION\_URL: Credentials for Lyrion (if MEDIASERVER\_TYPE="lyrion").  
* MPD\_HOST, MPD\_PORT, MPD\_PASSWORD, MPD\_MUSIC\_DIRECTORY: Credentials for MPD (if MEDIASERVER\_TYPE="mpd").

#### **Task & Performance Tuning**

* TEMP\_DIR: The directory where audio files are temporarily downloaded for analysis (e.g., /app/temp\_audio).  
* NUM\_RECENT\_ALBUMS: The default number of recent albums to scan if the user doesn't specify. 0 means scan all albums.  
* AUDIO\_LOAD\_TIMEOUT: The maximum time (in seconds) the system will spend trying to load a single audio file before giving up. Prevents corrupt files from stalling the entire analysis.  
* MAX\_QUEUED\_ANALYSIS\_JOBS: Controls the parallelism of the main analysis task by limiting how many album-analysis child jobs can be in the queue at one time.  
* REBUILD\_INDEX\_BATCH\_SIZE: Controls how often the Voyager (vector search) index is rebuilt during a large analysis run. A smaller number means new songs are searchable faster, but with more overhead.

#### **Model & Analysis Parameters**

* TOP\_N\_MOODS: The default number of top-scoring moods to save in the score.mood\_vector column.  
* EMBEDDING\_MODEL\_PATH: Filesystem path to the main msd-musicnn-1.onnx model (generates embeddings).  
* PREDICTION\_MODEL\_PATH: Filesystem path to the msd-msd-musicnn-1.onnx model (generates mood predictions from embeddings).  
* DANCEABILITY\_MODEL\_PATH, AGGRESSIVE\_MODEL\_PATH, HAPPY\_MODEL\_PATH, PARTY\_MODEL\_PATH, RELAXED\_MODEL\_PATH, SAD\_MODEL\_PATH: Paths to the secondary models that generate "other features".

Additional analysis tuning & normalization constants

* `ENERGY_MIN`, `ENERGY_MAX` — Range used to normalize energy values (BPM- and loudness-derived) into a common scale used across scoring and UI visualizations.
* `TEMPO_MIN_BPM`, `TEMPO_MAX_BPM` — Tempo normalization bounds in beats-per-minute used when extracting and scaling tempo for score vectors.
* `DB_FETCH_CHUNK_SIZE` — Chunk size used by batch jobs when fetching large numbers of tracks from the DB. Useful to tune for memory/IO tradeoffs during clustering/analysis rebuilds.

#### **Voyager Index Building (Used during Analysis)**

* INDEX\_NAME: The name used to store the index in the database (e.g., music\_library).  
* VOYAGER\_METRIC: The distance metric used for building the index (angular or euclidean).  
* VOYAGER\_EF\_CONSTRUCTION: Voyager build-time parameter affecting index quality and build speed.  
* VOYAGER\_M: Voyager build-time parameter affecting index quality and memory usage.

## **2\. Song Clustering**

This section details the "Song Clustering" feature, which uses the data generated by the "Song Analysis" process to create automatic, thematic playlists.

### **2.1. Functional Analysis (High-Level)**

From a user's perspective, "Song Clustering" is the primary *creative* feature of the application. It takes the entire analyzed music library and intelligently groups songs into new, discoverable playlists based on their audio characteristics.

#### **Key User Interactions & Workflow**

1. **Prerequisite:** The user must have already run the "Start Analysis" task at least once.  
2. **Initiation:** The user navigates to the "Analysis and Clustering" page (index.html).  
3. **Configuration (Basic View):**  
   * **Clustering Algorithm:** Defaults to "K-Means".  
   * **TOP Playlist Number:** The user specifies how many playlists they want (e.g., 8, 20).  
   * **Clustering Runs:** Sets the number of "attempts" the system will make to find the best playlists. A higher number (e.g., 5000\) takes longer but produces better results.  
   * **K-Means Specific:** The user can set a *range* for the number of clusters (e.g., Min 40, Max 100). The system will automatically find the best number within this range.  
4. **Configuration** (Advanced **View):**  
   * **Clustering Algorithm:** The user can select more advanced algorithms like **DBSCAN**, **GMM**, or **Spectral**. Each selection reveals its own specific parameters (e.g., DBSCAN Epsilon, GMM Components).  
   * **Data Source:** The user can check "Use Embeddings for Clustering." If checked, clustering is performed on the raw 200-dimension audio embedding. If unchecked, it uses the human-readable "score vector" (tempo, energy, moods, etc.).  
   * **Scoring Weights:** The user can fine-tune the definition of a "good" playlist by adjusting weights for Diversity, Purity, Silhouette, etc.  
   * **AI Playlist Naming:** The user can select an AI Provider (Ollama, Gemini) to automatically generate creative names for the resulting playlists. If set to "None," playlists will have names based on their audio features (e.g., "Rock\_Fast\_Aggressive").  
5. **Execution:** The user clicks the **"Start Clustering"** button. This is a long-running, asynchronous job. The system *prevents* a new clustering task from starting if one is already running.  
6. **Monitoring & Feedback:** The "Task Status" panel updates to show the "main\_clustering" task. The user can monitor its progress:  
   * **Status:** Shows STARTED, PROGRESS, etc.  
   * **Progress:** The percentage bar fills as the clustering\_runs are completed.  
   * **Details / Log:** Provides real-time updates like "Progress: 100/5000 runs. Active batches: 10\. Best score: 4.52". This shows the evolutionary search in action.  
7. **Outcome:** When the task state becomes SUCCESS:  
   * Old \_automatic playlists on the media server are deleted.  
   * The new, optimized playlists are created on the media server (e.g., Jellyfin).  
   * The "Generated Playlists" section on the UI, when fetched (or on next page load), will be populated with the new playlists.

### **2.2. Technical Analysis (Algorithm-Level)**

The clustering process is a sophisticated, distributed, and evolutionary computing task. It is designed to search a massive parameter space to find the "best" possible set of playlists according to the user's weighted scoring preferences.

#### **Stage 1: API Call & Task Enqueueing**

1. **Route:** The "Start Clustering" button (script.js) sends a POST request to /api/clustering/start (defined in app\_clustering.py).  
2. **Payload:** The request body contains *all* parameters from the "Clustering Parameters" and "AI Playlist Naming" fieldsets.  
3. **Conflict Check:** The endpoint first queries the database (via DATABASE\_URL) to see if any main\_clustering task is already in a non-terminal state.  
4. **Job Creation:**  
   * A unique job\_id is generated.  
   * The endpoint enqueues a new high-priority job in the **RQ (Redis Queue)** (via REDIS\_URL).  
   * The function enqueued is tasks.clustering.run\_clustering\_task.  
   * All parameters are passed as kwargs, using configured defaults (e.g., CLUSTER\_ALGORITHM, CLUSTERING\_RUNS) if not provided by the user.

#### **Stage 2: Main Orchestration Task (run\_clustering\_task)**

This "orchestrator" function (in clustering.py) manages the evolutionary search.

1. **Data Preparation (Stratified Sampling):**  
   * It fetches all tracks from the score table.  
   * It calls \_prepare\_genre\_map to group tracks by their primary genre, using the list of genres defined in STRATIFIED\_GENRES.  
   * It calls \_calculate\_target\_songs\_per\_genre to determine a target number per genre, using STRATIFIED\_SAMPLING\_TARGET\_PERCENTILE and MIN\_SONGS\_PER\_GENRE\_FOR\_STRATIFICATION.  
2. **Batch Orchestration (The "Evolutionary Loop"):**  
   * The task runs num\_clustering\_runs (from CLUSTERING\_RUNS) times.  
   * It divides the total runs into *batches* (e.g., 250 batches of 20 runs each, defined by ITERATIONS\_PER\_BATCH\_JOB).  
   * It maintains a list of elite\_solutions: the TOP\_N\_ELITES best-scoring clustering results found so far.  
   * It loops, launching child tasks (run\_clustering\_batch\_task) up to MAX\_CONCURRENT\_BATCH\_JOBS at a time.  
3. **Monitoring & Elitism:**  
   * In its main loop, it calls \_monitor\_and\_process\_batches.  
   * This function checks for completed batch jobs. When a batch finishes, its best result is added to the elite\_solutions list.  
   * The list of elites is passed to new batches, which can "exploit" them (mutate) based on EXPLOITATION\_PROBABILITY\_CONFIG or "explore" (random guess).  
   * This monitoring function also enforces CLUSTERING\_BATCH\_TIMEOUT\_MINUTES to kill stuck batches and CLUSTERING\_MAX\_FAILED\_BATCHES to stop the job if it's failing.

#### **Stage 3: Batch Iteration Task (run\_clustering\_batch\_task)**

This is the parallel "worker" task.

1. **Data Sampling:** It calls \_get\_stratified\_song\_subset, which perturbs the previous song subset by SAMPLING\_PERCENTAGE\_CHANGE\_PER\_RUN.  
2. **Parameter Generation:** It calls \_generate\_evolutionary\_parameters, which either mutates an elite (using MUTATION\_\* variables) or generates random parameters within the user-defined ranges (e.g., NUM\_CLUSTERS\_MIN, NUM\_CLUSTERS\_MAX, PCA\_COMPONENTS\_MIN, PCA\_COMPONENTS\_MAX).  
3. **Data Selection & Scaling:** It calls \_prepare\_and\_scale\_data, which selects either the raw embedding\_vector or the score\_vector based on the ENABLE\_CLUSTERING\_EMBEDDINGS flag and applies StandardScaler.  
4. **Clustering:** It calls \_apply\_clustering\_model, which applies PCA and then fits the chosen model (KMeans, DBSCAN, etc.).  
5. **Scoring (The "Fitness Function"):**  
   * It calls \_format\_and\_score\_iteration\_result. This is the **most critical** part of the algorithm.  
   * It calculates internal metrics (silhouette, davies\_bouldin, etc.) and custom metrics (mood\_diversity, mood\_purity).  
   * The scores are normalized using pre-calculated statistics (e.g., LN\_MOOD\_DIVERSITY\_STATS, LN\_MOOD\_PURITY\_EMBEDING\_STATS).  
   * It combines these metrics using the user-defined weights (e.g., SCORE\_WEIGHT\_DIVERSITY, SCORE\_WEIGHT\_PURITY, SCORE\_WEIGHT\_SILHOUETTE) to produce a single **fitness\_score**.  
6. **Return:** The batch task completes its runs and returns the *single best result* to the main orchestrator.

#### **Stage 4: Finalization & Post-Processing (run\_clustering\_task)**

Once all batches are complete, the orchestrator has the single best-scoring result.

1. **Post-Processing Pipeline:** It runs this result through a cleanup pipeline (in clustering\_postprocessing.py):  
   * apply\_duplicate\_filtering\_to\_clustering\_result: Removes duplicate songs within playlists, using DUPLICATE\_DISTANCE\_THRESHOLD\_COSINE and DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK.  
   * apply\_minimum\_size\_filter\_to\_clustering\_result: Deletes playlists smaller than MIN\_PLAYLIST\_SIZE\_FOR\_TOP\_N.  
   * select\_top\_n\_diverse\_playlists: If top\_n\_playlists was set, this selects the N most different playlists.  
2. **Playlist Naming:** It calls \_name\_and\_prepare\_playlists.  
   * If AI\_MODEL\_PROVIDER is "None," it generates names from features.  
   * If an AI provider is set, it calls the AI (e.g., using OLLAMA\_SERVER\_URL or GEMINI\_API\_KEY) to get a creative name.  
   * It applies a final Fisher-Yates shuffle to every playlist.  
3. **Final Commit:**  
   * Calls delete\_automatic\_playlists and create\_playlist (using MEDIASERVER\_TYPE and credentials).  
   * Calls update\_playlist\_table to save the final playlists to the PostgreSQL database.  
   * Marks the main\_clustering task as SUCCESS.

### **2.3. Clustering Deep Dive (Advanced Details)**

This subsection pulls in the more detailed, implementation-aligned explanation of the evolutionary search, scoring mechanics, and AI-driven playlist naming. It intentionally focuses on the parts that are not already described in the higher-level sections above: the exact purity/diversity computations, a practical comparison of metrics, and the AI naming workflow and sanitization rules.

#### Purity & Diversity (Concrete Calculation Notes)

These two custom metrics are central to the fitness function used by the evolutionary clustering process. They are label-aware (they operate on mood/genre labels) and are therefore complementary to geometric metrics like silhouette.

Purity (intra-playlist consistency):

- For each cluster (playlist) we form a profile. If clustering used interpretable score vectors the centroid itself is the profile; if embeddings were used, we compute the average of the original score vectors for the playlist members.
- From the profile we take the top K moods (configurable via TOP_K_MOODS_FOR_PURITY_CALCULATION).
- For every song in the playlist, we take the intersection between the song's active moods and the playlist's top K moods. From that intersection we keep the maximum mood score for that song (or skip the song if no intersection).
- Sum these per-song maximums across the playlist to get a raw purity value for that playlist.
- The raw purity value is transformed with log1p and then normalized using precomputed statistics (LN_MOOD_PURITY_STATS or the embedding-specific stats) so it fits into the composite scoring range used by the evolutionary algorithm.

Example (illustrative):

- Playlist top moods: pop:0.6, indie:0.4, vocal:0.35 (top K = 3)
- Song A moods: indie:0.3, rock:0.7, vocal:0.6 → used moods: indie, vocal → song score = max(0.3, 0.6) = 0.6
- Song B moods: indie:0.4, rock:0.45, vocal:0.3 → used moods: indie, vocal → song score = 0.4
- Raw purity = 0.6 + 0.4 = 1.0 → transformed via log1p and normalized for combination.

Diversity (inter-playlist variety):

- For each playlist, extract its dominant mood (the single highest-scoring mood in its profile) and its score.
- Keep only unique dominant moods across all playlists and sum their scores to get a raw diversity value.
- Transform with log1p and normalize using LN_MOOD_DIVERSITY_STATS (or embedding-specific stats) before combining.

Example (illustrative):

- P1: dominant indie = 0.6
- P2: dominant pop = 0.5
- P3: dominant vocal = 0.55
- Raw diversity = 0.6 + 0.5 + 0.55 = 1.65 → transformed and normalized.

Why both? Purity rewards tight, theme-consistent playlists. Diversity rewards breadth across playlists so the final set is not a set of near-duplicates. The evolutionary fitness function combines them using configurable weights (SCORE_WEIGHT_PURITY, SCORE_WEIGHT_DIVERSITY).

#### Metric Comparison (Practical Notes)

When evaluating clustering outcomes the system uses both label-aware metrics (purity/diversity) and standard geometric metrics (silhouette, Davies-Bouldin, Calinski-Harabasz). Each has strengths:

- Purity/Diversity: fast (linear-ish in songs), interpretable for music use-cases, directly measures musical semantics (moods/genres). Best for guiding the curator-style objective of playlists.
- Silhouette / Davies-Bouldin / Calinski-Harabasz: measure geometric separation and cohesion; important for structure but blind to label meaning.

Quick comparison:

- Purity: label-aware, O(N·K) where K is top moods per playlist, high interpretability for music.
- Diversity: label-aware, O(C·M) where C is number of playlists and M is number of mood labels.
- Silhouette: cluster-aware and geometric, O(N^2) in naive implementations (often approximated), lower semantic interpretability for playlists.

The evolutionary algorithm uses normalized combinations of these metrics so you can tune the final behavior via the score weights in the UI.

#### Monte Carlo / Evolutionary Notes (Implementation Details)

Brief reminders of implementation details that are still current and important when reasoning about tuning:

- Stratified sampling is applied when creating the per-run subset: STRATIFIED_GENRES and MIN_SONGS_PER_GENRE_FOR_STRATIFICATION influence how many items of selected genres are forced into each sample. The target is computed using STRATIFIED_SAMPLING_TARGET_PERCENTILE applied to the distribution of counts.
- Between successive runs a percentage (SAMPLING_PERCENTAGE_CHANGE_PER_RUN) of the sampled items is replaced to introduce controlled perturbation while keeping continuity.
- Elite solutions (TOP_N_ELITES) are remembered. With probability EXPLOITATION_PROBABILITY_CONFIG the algorithm will select an elite and mutate it (MUTATION_INT_ABS_DELTA / MUTATION_FLOAT_ABS_DELTA) instead of sampling random parameters.

These mechanisms balance exploration and exploitation and are implemented inside the `tasks.clustering` orchestration and batch workers.

#### AI Playlist Naming (Practical & Sanitization Steps)

When AI is enabled (AI_MODEL_PROVIDER set), the naming step follows a small pipeline to ensure stable, safe, and deterministic output:

1. The naming prompt includes: the playlist's top moods and tempo range, a short sample list of songs from the playlist, and explicit format rules.
2. Prompt engineering instructs the model to return a single name (15–35 ASCII characters) with no extra commentary and a short list of reserved words or forbidden characters.
3. The backend calls the provider-specific helper in `ai.py` (e.g., `get_ollama_playlist_name`, `get_gemini_playlist_name`, `get_mistral_playlist_name`).
4. Returned text is cleaned: strip Markdown fences, normalize Unicode to ASCII, enforce character whitelist (printable ASCII letters, numbers, spaces, dash/underscore), truncate to length limits, and append `_automatic` (or `_instant` for chat-generated lists) as needed.
5. If the AI output can't be sanitized into a valid name, the system falls back to a deterministic feature-based name (e.g., `Rock_Fast_Aggressive`).

These steps are implemented in `tasks/clustering_postprocessing.py` and `ai.py` and are still current as of the code in this repository.

---

### **2.4. Environment Variable Configuration**

The Song Clustering functionality is configured by the following environment variables (from config.py):

#### **Core Infrastructure**

* REDIS\_URL: **(Required)** The connection string for the Redis server, used for task queueing and status management.  
* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** The connection string for the PostgreSQL database where all analysis results are read from.  
* MEDIASERVER\_TYPE, JELLYFIN\_URL, etc.: **(Required)** Media server credentials are used at the *end* of the process to delete old playlists and create the new ones.

#### **Main Clustering Configuration**

* CLUSTER\_ALGORITHM: The default algorithm to use if not specified (e.g., kmeans, dbscan).  
* ENABLE\_CLUSTERING\_EMBEDDINGS: (Boolean) Default for whether to cluster on raw embeddings (True) or "score vectors" (False).  
* CLUSTERING\_RUNS: The default number of evolutionary iterations to perform. Higher is slower but generally better.  
* TOP\_N\_PLAYLISTS: The default number of diverse playlists to select at the end.  
* MAX\_DISTANCE: Used during iteration to filter out songs that are too far from their cluster's center.  
* MAX\_SONGS\_PER\_CLUSTER: The maximum number of songs a playlist can have. 0 means unlimited.  
* MAX\_SONGS\_PER\_ARTIST: The maximum number of songs from a single artist allowed in one playlist.

#### **Algorithm Parameter Ranges (for Evolutionary Search)**

* NUM\_CLUSTERS\_MIN, NUM\_CLUSTERS\_MAX: Default range for K-Means.  
* DBSCAN\_EPS\_MIN, DBSCAN\_EPS\_MAX, DBSCAN\_MIN\_SAMPLES\_MIN, DBSCAN\_MIN\_SAMPLES\_MAX: Default ranges for DBSCAN.  
* GMM\_N\_COMPONENTS\_MIN, GMM\_N\_COMPONENTS\_MAX: Default range for GMM.  
* SPECTRAL\_N\_CLUSTERS\_MIN, SPECTRAL\_N\_CLUSTERS\_MAX: Default range for Spectral Clustering.  
* PCA\_COMPONENTS\_MIN, PCA\_COMPONENTS\_MAX: Default range for PCA components.  
* GMM\_COVARIANCE\_TYPE: A specific technical parameter for GMM (e.g., full).  
* SPECTRAL\_N\_NEIGHBORS: A specific technical parameter for Spectral Clustering.

#### **Task & Performance Tuning**

* ITERATIONS\_PER\_BATCH\_JOB: How many clustering runs to perform in a single parallel child task.  
* MAX\_CONCURRENT\_BATCH\_JOBS: The maximum number of batch tasks to run in parallel.  
* CLUSTERING\_BATCH\_TIMEOUT\_MINUTES: The maximum time a single batch can run before being killed. Prevents stuck jobs.  
* CLUSTERING\_MAX\_FAILED\_BATCHES: The number of failed batches allowed before the main clustering job gives up.  
* CLUSTERING\_BATCH\_CHECK\_INTERVAL\_SECONDS: How often the main task checks the status of its children.
* DB_FETCH_CHUNK_SIZE: Chunk size for fetching full track/embedding rows from the DB in batch jobs; tune to balance memory usage and query overhead during large runs.

#### **Evolutionary Algorithm Tuning**

* TOP\_N\_ELITES: The number of "best-so-far" solutions to keep in memory.  
* EXPLOITATION\_START\_FRACTION: The percentage of total runs to complete before the algorithm starts "exploiting" (mutating) elite solutions.  
* EXPLOITATION\_PROBABILITY\_CONFIG: The probability (0.0 to 1.0) of mutating an elite solution vs. trying a new random one.  
* MUTATION\_INT\_ABS\_DELTA, MUTATION\_FLOAT\_ABS\_DELTA: The "size" of the mutation for integer and float parameters.
* MUTATION\_KMEANS\_COORD\_FRACTION: Fractional coordinate mutation size used when mutating KMeans centroids (tunable to control the magnitude of centroid perturbations).

#### **Fitness Score (Weighting)**

* SCORE\_WEIGHT\_DIVERSITY, SCORE\_WEIGHT\_PURITY: Weights for the custom mood diversity/purity scores.  
* SCORE\_WEIGHT\_OTHER\_FEATURE\_DIVERSITY, SCORE\_WEIGHT\_OTHER\_FEATURE\_PURITY: Weights for the custom "other feature" (e.g., danceability) scores.  
* SCORE\_WEIGHT\_SILHOUETTE, SCORE\_WEIGHT\_DAVIES\_BOULDIN, SCORE\_WEIGHT\_CALINSKI\_HARABASZ: Weights for standard internal clustering validation metrics.

#### **Fitness Score (Normalization Stats)**

* LN\_MOOD\_DIVERSITY\_STATS, LN\_MOOD\_PURITY\_STATS, LN\_MOOD\_DIVERSITY\_EMBEDING\_STATS, LN\_MOOD\_PURITY\_EMBEDING\_STATS, LN\_OTHER\_FEATURES\_DIVERSITY\_STATS, LN\_OTHER\_FEATURES\_PURITY\_STATS: Pre-calculated statistical values (min, max, mean, std) used to normalize the raw scores into a comparable range.  
* TOP\_K\_MOODS\_FOR\_PURITY\_CALCULATION: How many of a cluster's top moods to use when calculating its purity score.  
* OTHER\_FEATURE\_PREDOMINANCE\_THRESHOLD\_FOR\_PURITY: The minimum score a feature (like "danceable") must have to be considered "predominant" for purity calculations.

#### **AI Playlist Naming**

* AI\_MODEL\_PROVIDER: Default AI provider (OLLAMA, GEMINI, MISTRAL, NONE).  
* OLLAMA\_SERVER\_URL, OLLAMA\_MODEL\_NAME: Configuration for Ollama.  
* GEMINI\_API\_KEY, GEMINI\_MODEL\_NAME: Configuration for Google Gemini.  
* MISTRAL\_API\_KEY, MISTRAL\_MODEL\_NAME: Configuration for Mistral.

#### **Data Sampling**

* STRATIFIED\_GENRES: The list of genres to use for stratified (balanced) sampling.  
* MIN\_SONGS\_PER\_GENRE\_FOR\_STRATIFICATION: The minimum target number of songs for each genre in the sample.  
* STRATIFIED\_SAMPLING\_TARGET\_PERCENTILE: The percentile of genre counts used to dynamically set the target number.  
* SAMPLING\_PERCENTAGE\_CHANGE\_PER\_RUN: The percentage of songs to "swap out" between clustering iterations to ensure data diversity.

#### **Post-Processing**

* MIN\_PLAYLIST\_SIZE\_FOR\_TOP\_N: The minimum number of songs a playlist must have to be considered in the final "Top N" selection.  
* DUPLICATE\_DISTANCE\_THRESHOLD\_COSINE, DUPLICATE\_DISTANCE\_THRESHOLD\_EUCLIDEAN: The distance threshold for considering two songs a "duplicate" during post-processing.  
* DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK: A performance optimization for duplicate checking.

## **3\. Playlist from Similar Song**

This section details the feature allowing users to find songs similar to a selected track and generate a playlist from the results. It relies heavily on the pre-built Voyager vector index generated during the Song Analysis phase.

### **3.1. Functional Analysis (High-Level)**

This feature provides a quick way to create a focused playlist based on the sonic similarity to a single "seed" song.

#### **Key User Interactions & Workflow**

1. **Initiation:** The user navigates to the "Playlist from Similar Song" page (similarity.html).  
2. **Seed Song Selection:** The user types into the "Artist" and/or "Track Title" fields. An autocomplete dropdown appears (autocomplete-results), suggesting matching songs from the library. The user clicks on a suggestion to select the seed song.  
3. **Configuration:**  
   * **Number of results:** The user specifies how many similar songs (n) they want in the final playlist.  
   * **Limit** songs per **artist:** A checkbox (eliminate\_duplicates) allows the user to cap the number of songs by the same artist in the results (uses MAX\_SONGS\_PER\_ARTIST from config).  
   * **Radius Similarity:** A checkbox (radius\_similarity) toggles between two different modes for finding and ordering similar songs (explained below).  
4. **Execution:** The user clicks the **"Find Similar Tracks"** button. This sends a request to the backend.  
5. **Results Display:** The backend searches the vector index and returns a list of similar songs. These are displayed in a table (results-table-wrapper) showing Title, Artist, and Distance (how similar they are to the seed song).  
6. **Playlist Creation:** If results are found, a "Create Playlist" section appears. The user can optionally edit the suggested playlist name (defaults to "Similar to$$Seed Song Title$$  
   ") and click the **"Create Playlist on Media Server"** button.  
7. **Outcome:** A new playlist containing the seed song and the found similar songs is created on the configured media server (e.g., Jellyfin).

### **3.2. Technical Analysis (Algorithm-Level)**

This feature primarily interacts with the pre-built Voyager index for fast similarity searches.

#### **Stage 1: Index Loading (Application Startup)**

* When the Flask application starts (app.py), it calls load\_voyager\_index\_for\_querying (from voyager\_manager.py).  
* This function reads the pre-built index data (binary) and ID map (JSON) from the PostgreSQL database (voyager\_index\_data table, populated during analysis) into global variables (voyager\_index, id\_map, reverse\_id\_map) in the web server's memory. This allows for very fast lookups.  
* A background thread listens on Redis (listen\_for\_index\_reloads) for reload messages (published by the analysis task) and triggers load\_voyager\_index\_for\_querying(force\_reload=True) to update the in-memory index without restarting the server.

#### **Stage 2: Autocomplete Search**

1. **Route:** Typing in the search boxes (similarity.html) triggers JavaScript (handleSearchInput) which sends GET requests to /api/search\_tracks (app\_voyager.py).  
2. **Backend Logic:** The endpoint calls search\_tracks\_by\_title\_and\_artist (voyager\_manager.py), which performs a simple SQL ILIKE query against the score table in PostgreSQL to find matching tracks.

#### **Stage 3: Finding Similar Tracks**

1. **Route:** Clicking "Find Similar Tracks" sends a GET request to /api/similar\_tracks (app\_voyager.py). Parameters include item\_id (of the seed song), n, eliminate\_duplicates, and radius\_similarity.  
2. **Backend Logic (find\_nearest\_neighbors\_by\_id in voyager\_manager.py):**  
   * **Vector Lookup:** It retrieves the embedding vector for the target\_item\_id from the in-memory Voyager index (voyager\_index.get\_vector).  
   * **Determine Query Size (k):** It calculates how many neighbors (num\_to\_query) to initially retrieve from Voyager. This is *more* than n requested by the user, especially if radius\_similarity or eliminate\_duplicates is true, to provide a larger pool for filtering.  
   * **Voyager Query:** It calls voyager\_index.query(query\_vector, k=num\_to\_query) to get the k nearest neighbors (internal Voyager IDs and distances).  
   * **ID Mapping:** It converts the internal Voyager IDs back to media server item\_id strings using the id\_map.  
   * **Radius** Similarity **Branching:**  
     * **If radius\_similarity is True:**  
       * Calls \_radius\_walk\_get\_candidates to prepare the initial pool. This involves pre-filtering candidates using \_filter\_by\_distance (removes songs extremely close to the *anchor* based on DUPLICATE\_DISTANCE\_THRESHOLD\_\*), \_deduplicate\_and\_filter\_neighbors (removes exact name/artist matches, including the anchor itself), and potentially \_filter\_by\_mood\_similarity (if enabled by MOOD\_SIMILARITY\_ENABLE). It also pre-fetches vectors and calculates distances to the anchor.  
       * Calls \_execute\_radius\_walk. This function implements a **bucketed greedy walk**:  
         * Candidates are sorted by distance to the anchor and grouped into buckets.  
         * It starts with the closest valid candidate.  
         * It iteratively selects the *next* song by evaluating candidates within a limited number of nearby buckets (BUCKETS\_TO\_SCAN).  
         * The selection criteria prioritize songs that are close to the *previously selected song* while also considering distance to the *original anchor song* (a 70/30 weighting).  
         * **Crucially,** the artist cap (MAX\_SONGS\_PER\_ARTIST if eliminate\_duplicates is true) is applied *during* the **walk**, preventing too many songs from the same artist being selected early on. A rule preventing the same artist appearing in more than two different "bucket subpaths" is also applied until the global cap is reached.  
       * The final list is ordered according to the **path taken by the walk**, aiming for smoother transitions between songs rather than just raw similarity to the anchor. It returns exactly n songs.  
     * **If radius\_similarity is False (Standard Logic):**  
       * It applies a sequence of filters to the initial Voyager results:  
         1. \_filter\_by\_distance: Removes songs too close in vector space to recently kept songs (uses DUPLICATE\_DISTANCE\_THRESHOLD\_\* and DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK). This prevents near-identical tracks (e.g., slightly different masters).  
         2. \_deduplicate\_and\_filter\_neighbors: Removes songs with the exact same Title/Artist as the anchor or already kept songs.  
         3. \_filter\_by\_mood\_similarity: (Optional, based on MOOD\_SIMILARITY\_ENABLE or user request) Removes songs whose "other features" (danceable, aggressive, etc.) differ too much from the anchor song, based on MOOD\_SIMILARITY\_THRESHOLD.  
         4. **Artist Cap:** If eliminate\_duplicates is true, it iterates through the remaining songs and keeps only up to MAX\_SONGS\_PER\_ARTIST songs per unique artist.  
       * The final list contains the top n remaining songs, **sorted by their original distance** to the anchor song.  
   * **Details Fetch:** It fetches the Title and Artist for the final list of item\_ids from the score table.  
   * **Return:** Returns the list of similar songs (including item\_id, title, author, distance) to the frontend.

#### **Stage 4: Playlist Creation**

1. **Route:** Clicking "Create Playlist" (similarity.html) sends a POST request to /api/create\_playlist (app\_voyager.py). The payload contains the desired playlist\_name and the list of track\_ids (seed song \+ similar songs).  
2. **Backend Logic:** The endpoint calls create\_playlist\_from\_ids (voyager\_manager.py), which in turn calls create\_instant\_playlist (mediaserver.py). This function uses the configured MEDIASERVER\_TYPE and credentials to interact with the media server's API and create the actual playlist.

### **3.3. Environment Variable Configuration**

The "Playlist from Similar Song" feature relies on the Voyager index built during Analysis and uses these environment variables:

#### **Core Infrastructure**

* REDIS\_URL: **(Required)** Used by the background listener thread that reloads the Voyager index.  
* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** Used to load the index data, fetch track details (Title/Artist), and perform autocomplete searches.  
* MEDIASERVER\_TYPE, JELLYFIN\_URL, etc.: **(Required)** Used at the final stage to create the playlist on the media server.

#### **Voyager Index Querying (Used during Similarity Search)**

* INDEX\_NAME: The name used to load the correct index from the database.  
* VOYAGER\_QUERY\_EF: Voyager query-time parameter affecting search speed and accuracy. Higher values are slower but potentially more accurate.  
* EMBEDDING\_DIMENSION: Used to verify the loaded index matches the expected vector size.

#### **Similarity Search Behavior & Filtering**

* MAX\_SONGS\_PER\_ARTIST: The maximum number of songs allowed per artist in the results if the "Limit songs per artist" checkbox (eliminate\_duplicates=true) is checked.  
* DUPLICATE\_DISTANCE\_THRESHOLD\_COSINE, DUPLICATE\_DISTANCE\_THRESHOLD\_EUCLIDEAN: Defines how close two songs' vectors must be (using the metric specified by VOYAGER\_METRIC) to be considered duplicates by the \_filter\_by\_distance function.  
* DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK: Performance optimization for distance filtering; checks only against the last N kept songs. 1 is usually sufficient if songs are pre-sorted or processed sequentially.  
* MOOD\_SIMILARITY\_THRESHOLD: The maximum allowed normalized distance between the "other features" (danceable, aggressive, etc.) of the seed song and a candidate for the candidate to be kept by the \_filter\_by\_mood\_similarity function.  
* MOOD\_SIMILARITY\_ENABLE: (Boolean) Global switch to enable (True) or disable (False) the mood similarity filter by default if the user doesn't specify it in the API call.  
* SIMILARITY\_ELIMINATE\_DUPLICATES\_DEFAULT: (Boolean) Default state for the "Limit songs per artist" checkbox (eliminate\_duplicates parameter) if not specified by the user.  
* SIMILARITY\_RADIUS\_DEFAULT: (Boolean) Default state for the "Radius Similarity" checkbox (radius\_similarity parameter) if not specified by the user. This determines the default search mode (Walk vs. Standard).  
* VOYAGER\_METRIC: Although primarily a build-time parameter, the query logic in \_filter\_by\_distance uses this to select the correct distance function (\_get\_direct\_cosine\_distance or \_get\_direct\_euclidean\_distance) and threshold (DUPLICATE\_DISTANCE\_THRESHOLD\_COSINE or \_EUCLIDEAN).

## **4\. Song Path**

This section details the "Song Path" feature, which aims to generate a smooth, transitional playlist between two user-selected songs.

### **4.1. Functional Analysis (High-Level)**

The "Song Path" feature allows users to explore the sonic space between two potentially different songs. It generates a playlist that starts with the first song, gradually transitions through similar-sounding tracks, and ends with the second song.

#### **Key User Interactions & Workflow**

1. **Initiation:** The user navigates to the "Song Path" page (path.html).  
2. **Song Selection:** The user selects a "Start Song" and an "End Song" using autocomplete fields similar to the "Playlist from Similar Song" feature.  
3. **Configuration:**  
   * **Songs in path:** The user specifies the desired total number of songs (max\_steps) in the path, including the start and end songs.  
   * **Keep path size exact:** A checkbox (path\_fix\_size) controls whether the algorithm should prioritize reaching the exact requested path length, potentially using a more complex centroid-merging strategy if needed. If unchecked, the path might be shorter if the algorithm cannot find suitable non-duplicate songs at each step.  
4. **Execution:** The user clicks the **"Find Path"** button. This sends a request to the backend.  
5. **Results Display:**  
   * The backend calculates the path using the vector embeddings.  
   * A table (results-table-wrapper) lists the songs in the generated path order.  
   * A chart (path-graph) visualizes the path's progression, showing distances between steps and distances to the start/end points. Different chart views (Progression, Feature Range, Feature Difference, 2D Path) can be selected.  
6. **Playlist Creation:** If a path is found, a "Create Playlist" section appears. The user can edit the suggested name and click **"Create Playlist on Media Server"** to save the path as a playlist.  
7. **Outcome:** A new playlist containing the sequence of songs forming the path is created on the media server.

### **4.2. Technical Analysis (Algorithm-Level)**

The Song Path algorithm constructs a sequence of songs by interpolating between the start and end song vectors and finding the nearest neighbors to these intermediate points.

#### **Stage 1: API Call & Parameter Handling**

1. **Route:** Clicking "Find Path" sends a GET request to /api/find\_path (app\_path.py). Parameters include start\_song\_id, end\_song\_id, max\_steps, and optionally path\_fix\_size.  
2. **Validation:** The backend checks if start and end songs are provided and are different. It retrieves the default path length (PATH\_DEFAULT\_LENGTH) if max\_steps is not given. It determines the effective path\_fix\_size boolean based on the request parameter or the PATH\_FIX\_SIZE environment variable.

#### **Stage 2: Path Generation (find\_path\_between\_songs in path\_manager.py)**

1. **Vector Retrieval:** Fetches the embedding vectors for the start\_item\_id and end\_item\_id using get\_vector\_by\_id (which reads from the in-memory Voyager index cache). It also fetches initial song details from the database.  
2. **Initialization:** Sets up used\_song\_ids and used\_signatures (normalized title/artist) sets, initially containing the start and end songs to prevent duplicates. It also initializes artist\_counts to track artists for the MAX\_SONGS\_PER\_ARTIST cap.  
3. **Centroid Interpolation:**  
   * Calculates the number of intermediate songs needed (num\_intermediate \= Lreq \- 2).  
   * Calls interpolate\_centroids(start\_vector, end\_vector, num=Lreq, metric=PATH\_DISTANCE\_METRIC). This generates Lreq points (vectors) linearly or spherically interpolated between the start and end vectors, based on the PATH\_DISTANCE\_METRIC config (euclidean or angular).  
   * Extracts the intermediate\_centroids (excluding the start and end points).  
4. **Song Selection Strategy (Depends on path\_fix\_size):**  
   * **If path\_fix\_size is False (Simpler, potentially shorter path):**  
     * It iterates through each intermediate\_centroid.  
     * For each centroid, it calls \_find\_best\_songs\_for\_job with num\_to\_find=1 and a small search radius (k\_base).  
     * \_find\_best\_songs\_for\_job:  
       * Uses find\_nearest\_neighbors\_by\_vector to get k\_search candidates near the centroid.  
       * Iterates through candidates, checking against used\_song\_ids, used\_signatures, artist\_counts (using MAX\_SONGS\_PER\_ARTIST), and applying distance filtering (\_filter\_by\_distance logic using DUPLICATE\_DISTANCE\_THRESHOLD\_\* and DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK) against the *path songs found so far*.  
       * Adds the *first* valid song found to the path and updates the used sets and artist\_counts.  
     * If \_find\_best\_songs\_for\_job fails to find a valid song for a centroid, that step is skipped, potentially making the final path shorter than Lreq.  
   * **If path\_fix\_size is True (Complex, exact length prioritized):**  
     * **Initial Job Creation:** It groups the intermediate\_centroids into an initial number of jobs (heuristic based on neighbor overlap). Each job represents one or more original centroids and is assigned a target number of songs (num\_to\_find) equal to the number of centroids it represents. The search radius (k) is scaled.  
     * **Iterative Job Processing with Merging:**  
       * It processes jobs sequentially.  
       * For each job, it calls \_find\_best\_songs\_for\_job with the job's vector, k, and num\_to\_find.  
       * **If Success:** The found songs are added to the path, and it moves to the next job.  
       * **If Failure:** \_find\_best\_songs\_for\_job finds fewer songs than needed (returns \[\] after rolling back its additions). The algorithm *merges* the failed job with the *next* job in the list:  
         * A new centroid is calculated (interpolated between the original start/end points of the merged span).  
         * The search radius k is increased (summed, capped at k\_max).  
         * num\_to\_find is updated to reflect the total songs needed by both original jobs.  
         * The merged job replaces the current job, and the loop *retries the current index* (now containing the merged job).  
       * This merging continues until either all songs are found, or the last job fails and cannot be merged further (resulting in a potentially shorter path).  
5. **Final Path Construction:**  
   * Appends the end song details to the list of found intermediate songs.  
   * Extracts the final list of item\_ids.  
   * Calls \_create\_path\_from\_ids to fetch full details for the path songs from the database.  
   * Calculates the total\_path\_distance by summing the pairwise distances (get\_distance) between consecutive songs in the final path.  
6. **Return:** Returns the list of song details (final\_path\_details) and the total\_path\_distance to the API endpoint (app\_path.py), which then formats it as JSON for the frontend.

#### **Stage 3: Playlist Creation**

* Identical to Stage 4 of "Playlist from Similar Song," using the /api/create\_playlist endpoint.

### **4.3. Environment Variable Configuration**

The Song Path feature uses the Voyager index and relies on several specific configuration variables:

#### **Core Infrastructure & Index**

* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** Used to fetch song details.  
* REDIS\_URL: **(Required)** Used by the index reloading listener.  
* INDEX\_NAME, VOYAGER\_QUERY\_EF, EMBEDDING\_DIMENSION: Used for querying the Voyager index via get\_vector\_by\_id and find\_nearest\_neighbors\_by\_vector.

#### **Path Generation Parameters**

* PATH\_DISTANCE\_METRIC: **(Crucial)** Determines the distance function used for interpolation (interpolate\_centroids) and step distance calculation (get\_distance). Options: angular, euclidean.  
* PATH\_DEFAULT\_LENGTH: The default number of songs in the path if the user doesn't specify.  
* PATH\_AVG\_JUMP\_SAMPLE\_SIZE: (Not directly used in the main path logic, but potentially in related calculations like \_calculate\_local\_average\_jump\_distance).  
* PATH\_CANDIDATES\_PER\_STEP: (Used by a heuristic during job creation when PATH\_FIX\_SIZE=True). Number of neighbors sampled near start/end to estimate overlap.  
* PATH\_LCORE\_MULTIPLIER: (Seems unused in the provided find\_path\_between\_songs logic, might be for an alternative algorithm).  
* PATH\_FIX\_SIZE: (Boolean) Controls the core pathfinding strategy (single pass vs. centroid merging) as described in the technical analysis.

#### **Filtering Parameters (Shared with Similarity)**

* MAX\_SONGS\_PER\_ARTIST: Caps the number of songs per artist allowed in the generated path.  
* DUPLICATE\_DISTANCE\_THRESHOLD\_COSINE, DUPLICATE\_DISTANCE\_THRESHOLD\_EUCLIDEAN: Used by \_find\_best\_songs\_for\_job to filter out candidates too close to previously added path songs.  
* DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK: Performance optimization for the distance check within \_find\_best\_songs\_for\_job.  
* VOYAGER\_METRIC: Used implicitly by distance filters to choose the correct threshold.

#### **Media Server (for Playlist Creation)**

* MEDIASERVER\_TYPE, JELLYFIN\_URL, etc.: **(Required)** Used at the final stage to create the playlist on the media server.

## **5\. Song Alchemy**

This section details the "Song Alchemy" feature, which allows users to combine the sonic characteristics of multiple songs, optionally subtracting others, to find tracks matching the resulting blend.

### **5.1. Functional Analysis (High-Level)**

Song Alchemy offers a powerful way to discover music by defining a desired sound profile through examples. Users can select songs they want to "include" (add their sonic essence) and songs they want to "exclude" (subtract their sonic essence). The system then finds tracks that are closest to the combined "include" profile while being distant from the "exclude" profile.

#### **Key User Interactions & Workflow**

1. **Initiation:** The user navigates to the "Song Alchemy" page (alchemy.html).  
2. **Song Selection:**  
   * The page starts with two song selection cards. The user can add more using the "Add Another Song" button.  
   * For each card, the user searches for a song using autocomplete (similar to other features).  
   * Crucially, for each selected song, the user clicks either **"Include"** or **"Exclude"**. At least one "Include" song is required.  
3. **Configuration:**  
   * **Number of results:** Specifies how many matching songs (n) to return.  
   * **Sampling temperature (τ):** Controls the randomness of the selection process. Lower values (e.g., 0.1) favor songs very close to the target profile, while higher values (e.g., 1.0 or more) allow for more variety and exploration further away.  
   * **Subtract distance threshold:** A slider adjusts how sonically *different* the results must be from the "Exclude" songs. A higher value means results must be further away from the excluded profile.  
4. **Execution:** The user clicks the **"Run Alchemy"** button. This sends a request to the backend.  
5. **Results Display:**  
   * The backend calculates the combined vector profile and searches the index.  
   * A 2D scatter plot (alchemy-plot) visually represents the selected songs (Include/Exclude), the calculated target centroids (Add/Subtract), the resulting "Kept" songs, and any "Removed" songs (filtered out by the subtract threshold). \\ \* A table (results-table) lists the final "Kept" songs, ordered by relevance to the combined profile.  
6. **Playlist Creation:** If results are found, a "Create Playlist" section appears, allowing the user to save the "Kept" songs (plus the original "Include" songs) to a new playlist on the media server.  
7. **Outcome:** The user discovers songs matching their custom sonic blend, visualized for better understanding, and can save these discoveries as a playlist.

### **5.2. Technical Analysis (Algorithm-Level)**

Song Alchemy leverages vector arithmetic on the song embeddings stored in the Voyager index.

#### **Stage 1: API Call & Input Processing**

1. **Route:** Clicking "Run Alchemy" sends a POST request to /api/alchemy (app\_alchemy.py).  
2. **Payload:** Contains a list of items, each with an id and op ("ADD" or "SUBTRACT"), the desired number of results n, the temperature, and optionally an override for subtract\_distance.  
3. **Validation:** The backend requires at least one "ADD" item.  
4. **Vector Retrieval:** The core logic resides in tasks.song\_alchemy.song\_alchemy. It starts by fetching the embedding vectors for all provided add\_ids and sub\_ids using get\_vector\_by\_id (from the in-memory Voyager index cache).

#### **Stage 2: Centroid Calculation**

1. **Add Centroid:** Calculates the average vector of all "ADD" songs (add\_centroid).  
2. **Subtract Centroid:** If "SUBTRACT" songs are provided, calculates their average vector (subtract\_centroid).

#### **Stage 3: Candidate Search**

1. **Nearest Neighbors:** Performs a nearest neighbor search using find\_nearest\_neighbors\_by\_vector centered on the add\_centroid. A significantly larger number of candidates (k \= n \* 4 or more) is requested initially to provide a pool for filtering and probabilistic sampling.

#### **Stage 4: Filtering & Scoring**

1. **Duplicate Removal:** Filters the initial candidates to remove the original ADD/SUBTRACT songs and apply standard duplicate filtering (distance-based via \_filter\_by\_distance, name-based via \_deduplicate\_and\_filter\_neighbors, and artist cap via MAX\_SONGS\_PER\_ARTIST).  
2. **Subtract Filtering:** If a subtract\_centroid exists:  
   * Calculates the direct distance (using get\_distance based on VOYAGER\_METRIC) between each remaining candidate and the subtract\_centroid.  
   * Filters out candidates whose distance is *less than* the subtract\_distance threshold (obtained from the request payload or defaults like ALCHEMY\_SUBTRACT\_DISTANCE\_ANGULAR). These are stored separately as filtered\_out.  
3. **Probabilistic Scoring (Softmax):**  
   * For the remaining candidates (those close to add\_centroid and far enough from subtract\_centroid), calculates their distance to the add\_centroid.  
   * Converts these distances into similarity scores (e.g., 1 / (distance \+ epsilon)).  
   * Applies a **softmax function** using the provided temperature parameter: exp(similarity / temperature) / sum(exp(all\_similarities / temperature)).  
     * Low temperature sharpens the probability distribution, favoring the very closest songs.  
     * High temperature flattens the distribution, making selection more random among candidates.  
4. **Selection:** Selects the top n candidates based on their softmax probabilities (either deterministically taking the highest probabilities or using weighted random sampling, depending on implementation details not fully shown but implied by "temperature").

#### **Stage 5: 2D Projection (Optional Visualization)**

1. **Dimensionality Reduction:** If candidates, centroids, or filtered songs exist, the backend performs dimensionality reduction (e.g., PCA, UMAP, or a custom discriminant projection) on their high-dimensional embedding vectors to get 2D coordinates (embedding\_2d). This allows plotting the relationships visually.

#### **Stage 6: Response**

1. **Formatting:** Packages the results:  
   * results: The final list of n selected songs (with details and original distance).  
   * filtered\_out: Songs removed by the subtract filter.  
   * add\_centroid\_2d, subtract\_centroid\_2d: 2D coordinates of centroids.  
   * add\_points, sub\_points: Original input songs with their 2D coordinates.  
   * projection: The method used for 2D projection.  
2. **Return:** Sends the JSON response to the frontend (alchemy.html) for display in the plot and table.

#### **Stage 7: Playlist Creation**

* Identical to Stage 4 of "Playlist from Similar Song," using the /api/create\_playlist endpoint. The track list includes the original "ADD" songs followed by the "kept" results, potentially trimmed to the requested n.

### **5.3. Environment Variable Configuration**

Song Alchemy uses the Voyager index and several specific parameters:

#### **Core Infrastructure & Index**

* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** Used to fetch song details for results and input songs.  
* REDIS\_URL: **(Required)** Used by the index reloading listener.  
* INDEX\_NAME, VOYAGER\_QUERY\_EF, EMBEDDING\_DIMENSION: Used for querying the Voyager index.

#### **Alchemy Parameters**

* ALCHEMY\_DEFAULT\_N\_RESULTS: The default number of results (n) if not specified by the user.  
* ALCHEMY\_MAX\_N\_RESULTS: The maximum allowed number of results (n).  
* ALCHEMY\_TEMPERATURE: The default softmax temperature (τ) controlling result randomness/determinism.  
* ALCHEMY\_SUBTRACT\_DISTANCE, ALCHEMY\_SUBTRACT\_DISTANCE\_ANGULAR, ALCHEMY\_SUBTRACT\_DISTANCE\_EUCLIDEAN: Default threshold for the subtract filter. The specific value used depends on VOYAGER\_METRIC.

#### **Filtering Parameters (Shared with Similarity)**

* MAX\_SONGS\_PER\_ARTIST: Caps the number of songs per artist allowed in the final results.  
* DUPLICATE\_DISTANCE\_THRESHOLD\_COSINE, DUPLICATE\_DISTANCE\_THRESHOLD\_EUCLIDEAN: Used for initial duplicate filtering of candidates.  
* DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK: Used for initial duplicate filtering.  
* VOYAGER\_METRIC: Determines which distance function (get\_distance) and subtract threshold (ALCHEMY\_SUBTRACT\_DISTANCE\_\*) are used.

#### **Media Server (for Playlist Creation)**

* MEDIASERVER\_TYPE, JELLYFIN\_URL, etc.: **(Required)** Used at the final stage to create the playlist on the media server.

## **6\. Music Map**

This section details the "Music Map" feature, which provides an interactive 2D visualization of the analyzed music library.

### **6.1. Functional Analysis (High-Level)**

The Music Map offers users a visual way to explore their music library based on sonic similarity. Songs that sound similar are plotted closer together in a 2D space. Users can interact with the map to discover relationships, select songs, and create playlists or song paths.

#### How the map works

The analysis begins by processing a raw audio sample and passing it through a neural network. The neural network does not output human-readable attributes like tempo or energy (although these are stored separately for other purposes). Instead, it generates a vector of 200 floating-point numbers. While this vector is not interpretable by a human, it is highly meaningful to the algorithm, as it represents patterns that empower our similarity search. Consequently, we store this 200-feature vector in our Audiomuse-AI database instead of just 3-4 human-readable values.

To visualize this 200-dimensional vector on a two-dimensional map (with X and Y coordinates), we use another machine learning algorithm called UMAP. UMAP transforms the 200-dimensional data into a 2-dimensional space, inevitably involving some rescaling and approximation.

This is why a musical path on the map may not appear perfectly strict; the visualization is an approximation of the more complex, high-dimensional data. The final two numbers plotted on the map are thus the result of two distinct rounds of machine learning processing.

#### **Key User Interactions & Workflow**

1. **Prerequisite:** The "Start Analysis" task must have been run, which generates the embeddings and potentially a precomputed 2D projection.  
2. **Initiation:** The user navigates to the "Music Map" page (map.html).  
3. **Map Loading:** The page automatically loads and displays a subset (default 25%) of the analyzed songs on a 2D scatter plot using Plotly.js. Points are colored by their top mood/genre.  
4. **Exploration:**  
   * **Zoom/Pan:** Users can zoom and pan the plot using standard Plotly controls.  
   * **Hover:** Hovering over a point reveals the song's Title, Artist, and top Mood/Genre.  
   * **Map Size:** Buttons (25%, 50%, 75%, 100%) allow the user to load a larger, potentially more detailed (but slower) version of the map.  
   * **Genre Filtering:** A clickable legend allows users to hide/show songs belonging to specific top moods/genres. "Hide all" / "Show all" controls are provided.  
5. **Selection:**  
   * **Click:** Clicking on a song point adds it to a selection list displayed below the map.  
   * **Lasso/Box Select:** Users can drag to select multiple points within an area, adding them to the selection list.  
   * **Selection Management:** The selection list shows chosen songs. Users can remove individual songs or "Clear all".  
6. **Search & Highlight:**  
   * Users can search for specific songs by Artist and/or Title using autocomplete fields.  
   * Clicking "Search" highlights the selected or first found song on the map with a distinct marker and centers the view on it (highlighting can be toggled). The searched song is also added to the selection.  
7. **Actions on Selection:**  
   * **Create playlist:** Creates a playlist on the media server containing all currently selected songs.  
   * **Song Path:** If 2 to 10 songs are selected, this button becomes enabled. Clicking it computes and draws paths between consecutive selected songs directly on the map overlay and adds the path songs to the selection list.  
8. **Refresh:** A refresh button clears all selections and overlays (search highlights, paths) and reloads the map data.  
9. **Outcome:** Users gain a visual understanding of their library's sonic landscape, discover clusters of similar music, and can directly create playlists or explore transitions based on their interactions.

### **6.2. Technical Analysis (Algorithm-Level)**

The Music Map relies on precomputed or dynamically generated 2D projections of the high-dimensional song embeddings and utilizes caching for performance.

#### **Stage 1: Data Preparation & Caching (Application Startup / Analysis End)**

1. **Precomputed Projection (Optional but Recommended):** During the run\_analysis\_task finalization (in analysis.py), after the Voyager index is built, it attempts to call build\_and\_store\_map\_projection('main\_map') (from app\_helper.py). This function likely uses UMAP or PCA to generate 2D coordinates for *all* embeddings and saves them (along with the corresponding item\_id list) to the database.  
2. **Startup Cache Building:** When the Flask application starts (app.py), a background thread runs init\_map\_cache (app\_map.py), which calls build\_map\_cache.  
3. **build\_map\_cache Logic:**  
   * Fetches *all* item\_id, title, author, mood\_vector, and embedding data from the database.  
   * Attempts to load the precomputed projection using load\_map\_projection('main\_map').  
   * If precomputed coordinates exist, it uses them.  
   * If any songs lack precomputed coordinates (or the precomputation failed/was skipped), it uses dimensionality reduction helpers (imported from tasks.song\_alchemy, preferring UMAP then PCA) to compute 2D coordinates *on-the-fly* for the missing songs.  
   * It creates lightweight versions of song data (item\_id, title, artist, embedding\_2d, mood\_vector (simplified to top mood)).  
   * It generates deterministic samples of the full dataset at 100%, 75%, 50%, and 25% using \_sample\_items.  
   * For each sample size ('100', '75', '50', '25'), it serializes the song list to JSON, compresses it using gzip, and stores both the raw JSON bytes and the gzipped bytes in the in-memory MAP\_JSON\_CACHE dictionary.

#### **Stage 2: Serving Map Data (API Endpoint)**

1. **Route:** The frontend (map.html) requests map data via fetchMapParam which calls the /api/map endpoint (app\_map.py), typically with a percent parameter (e.g., ?percent=50).  
2. **Cache Lookup:** The endpoint determines the requested percentage (defaulting to '25') and looks up the corresponding entry in the MAP\_JSON\_CACHE.  
3. **Efficient Serving:**  
   * If the client accepts gzip (Accept-Encoding: gzip) and gzipped data is available in the cache, it returns the pre-compressed json\_gzip\_bytes with the appropriate Content-Encoding header.  
   * Otherwise, it returns the raw json\_bytes.  
   * **Crucially, it sets Cache-Control: no-store headers to prevent browser/proxy caching of the map data itself**, ensuring fresh data on reload. The data is served entirely from the server's RAM cache.

#### **Stage 3: Frontend Rendering & Interaction (map.html JavaScript)**

1. **Loading:** On page load, it calls loadAndPlot('25') to fetch and display the default 25% map.  
2. **Plotting:**  
   * Uses Plotly.js (scattergl type for WebGL acceleration) to render the 2D scatter plot.  
   * Assigns colors based on the top mood/genre (colorPaletteFor, topGenre).  
   * Creates a single Plotly trace containing all points, using customdata to store the item\_id for each point.  
   * Sets hovermode: 'closest' and dragmode: 'lasso' for interaction.  
   * Generates a custom clickable HTML legend for genre filtering.  
3. **Interaction Handlers (attachPlotHandlers):**  
   * plotly\_selected: Triggered by lasso/box select. Extracts item\_ids from the event points using customdata and adds them (up to a cap) to the window.\_plotSelection array. Updates the selection panel (renderSelectionPanel).  
   * plotly\_click: Extracts the item\_id of the clicked point and adds it to window.\_plotSelection. Updates the selection panel.  
4. **Selection Management:**  
   * window.\_plotSelection stores the array of selected item\_ids.  
   * renderSelectionPanel dynamically builds the HTML list of selected songs with "REMOVE" buttons.  
   * removeFromSelection removes an ID from the array and re-renders the panel.  
   * "Clear all" button clears the array and re-renders.  
5. **Genre Filtering:**  
   * Clicking a genre in the custom legend calls toggleGenre, which adds/removes the genre from window.\_hiddenGenres.  
   * applyGenreFilterAndRerender filters the *full* dataset (window.\_plotPointsFull) based on window.\_hiddenGenres, rebuilds the Plotly trace data with the filtered points, and calls Plotly.react (or Plotly.newPlot for large datasets) to update the chart efficiently, preserving zoom/pan state and overlays.  
   * applyLegendStyles updates the visual style (strikethrough) of the legend items.  
6. **Search:** Uses the same autocomplete logic (/api/search\_tracks) as other pages. Clicking "Search" calls highlightSongById, which finds the song's coordinates (findPointById), adds a distinct overlay shape (addHighlightOverlay via Plotly.relayout), and updates the status text. Searched songs are added to the selection.  
7. **Playlist Creation:** The "Create playlist" button gathers IDs from window.\_plotSelection and sends them to the /api/create\_playlist endpoint.  
8. **Song Path Integration:** The "Song Path" button (enabled for 2-10 selected songs) iterates through consecutive pairs in window.\_plotSelection, calls /api/find\_path for each pair, appends the resulting path songs to the selection (appendSongsToSelectionPanel), and draws the path segments on the map using Plotly shapes (drawPathOnMap).  
9. **Overlays:** Search highlights and path segments are added as Plotly shapes. Functions like clearMapOverlays, togglePathPoints, togglePathLine, toggleSearchHighlight manage the visibility and removal of these shapes using Plotly.relayout. State is preserved across re-renders triggered by genre filtering.

### **6.3. Environment Variable Configuration**

The Music Map relies heavily on data generated during Analysis but has fewer direct configurations itself.

#### **Core Infrastructure**

* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** Used by build\_map\_cache to fetch all song data and embeddings at startup, and potentially to load precomputed projections.  
* MEDIASERVER\_TYPE, JELLYFIN\_URL, etc.: **(Required)** Used by the "Create playlist" button functionality.

#### **Data & Visualization (Implicit)**

* **Embedding Data:** The existence and quality of the embeddings in the embedding table (generated by Song Analysis using EMBEDDING\_MODEL\_PATH) are fundamental.  
* **Mood Data:** The mood\_vector in the score table (generated by Song Analysis using PREDICTION\_MODEL\_PATH) is used for coloring points and the legend.  
* **Projection Method:** While not directly configured via a dedicated env var for the map *serving*, the method used to generate the 2D coordinates (precomputed during analysis or fallback in build\_map\_cache, potentially influenced by availability of UMAP/PCA libraries) determines the map's layout. The specific projection method used is reported in the API response and displayed below the map.

#### **Related Features (Used via Map UI)**

* PATH\_\* variables: Used when the "Song Path" button is clicked.  
* MAX\_SONGS\_PER\_ARTIST, DUPLICATE\_DISTANCE\_THRESHOLD\_\*, etc.: Used implicitly by the /api/find\_path calls triggered from the map.

## **7\. Sonic Fingerprint**

This section details the "Sonic Fingerprint" feature, designed to generate a personalized playlist reflecting a user's listening habits.

### **7.1. Functional Analysis (High-Level)**

The Sonic Fingerprint feature creates a unique playlist tailored to an individual user's listening history. It analyzes the user's most played songs, giving more weight to recently played tracks, to create a sonic profile. It then finds other songs in the library that match this profile.

#### **Key User Interactions & Workflow**

1. **Prerequisite:** The "Start Analysis" task must have been run to populate the database with embeddings. The media server must track play counts and last played times for this feature to be effective.  
2. **Initiation:** The user navigates to the "Sonic Fingerprint" page (sonic\_fingerprint.html).  
3. **Credential Entry:** The user needs to provide their specific media server credentials (Username/User ID and potentially an API Token or Password). This is necessary because listening history is user-specific. Default credentials from the server config might pre-populate some fields, but the user must confirm or enter their own if different.  
4. **Configuration:**  
   * **Number of results:** The user specifies the desired total number of songs (n) for the final playlist.  
5. **Execution:** The user clicks the **"Generate My Sonic Fingerprint"** button.  
6. **Results Display:**  
   * The backend fetches the user's listening history, calculates the weighted fingerprint vector, and searches the Voyager index.  
   * The resulting recommended songs (a mix of the user's top played songs and newly discovered similar tracks) are displayed in a table (results-table-wrapper) showing Title, Artist, and Distance (similarity to the fingerprint).  
7. **Playlist Creation:** If results are found, a "Create Playlist" section appears. The user enters a name (default "My Sonic Fingerprint Mix") and clicks **"Create Playlist on Media Server"**.  
8. **Outcome:** A personalized playlist reflecting the user's listening taste, including both familiar top tracks and newly recommended similar songs, is created on their media server account.

### **7.2. Technical Analysis (Algorithm-Level)**

This feature combines media server interaction for user history with vector analysis and Voyager search.

#### **Stage 1: API Call & Credential Handling**

1. **Route:** Clicking "Generate My Sonic Fingerprint" sends a POST (or GET for backward compatibility) request to /api/sonic\_fingerprint/generate (app\_sonic\_fingerprint.py).  
2. **Payload/Params:** Contains the desired number of results (n) and user-specific credentials (jellyfin\_user\_identifier, jellyfin\_token or navidrome\_user, navidrome\_password).  
3. **Credential Resolution:**  
   * The backend retrieves the credentials from the request.  
   * For Jellyfin/Emby, it uses resolve\_emby\_jellyfin\_user to convert a username/identifier into the required User ID using the provided token.  
   * It packages these credentials into a user\_creds dictionary.

#### **Stage 2: Fingerprint Generation (generate\_sonic\_fingerprint in sonic\_fingerprint\_manager.py)**

1. **Fetch Top Songs:** Calls get\_top\_played\_songs (from mediaserver.py), passing the user\_creds. This function interacts with the media server API (based on MEDIASERVER\_TYPE) to retrieve the user's top SONIC\_FINGERPRINT\_TOP\_N\_SONGS most played tracks.  
2. **Fetch Embeddings:** Retrieves the embedding vectors (embedding\_vector) for these top songs from the application's PostgreSQL database (embedding table) using get\_tracks\_by\_ids.  
3. **Calculate Recency Weights:**  
   * Iterates through the top songs (for which embeddings were found).  
   * For each song, calls get\_last\_played\_time (from mediaserver.py), passing user\_creds, to get the timestamp of the last play.  
   * Calculates days\_since\_played.  
   * Applies an **exponential decay function** (weight \= exp(-decay\_rate \* days\_since\_played)) to calculate a weight. The half\_life (set to 30 days) determines how quickly the weight decreases for older plays. Songs without a valid last played time receive a fixed lower weight.  
4. **Weighted Average:** Calculates the weighted average of the embedding vectors (average\_vector). This vector represents the user's "sonic fingerprint".  
5. **Determine Target Size:** Calculates how many *new* neighbors (neighbors\_to\_find) are needed to reach the total desired playlist size (total\_desired\_size, default SONIC\_FINGERPRINT\_NEIGHBORS), accounting for the number of top songs used to create the fingerprint (num\_seed\_songs).  
6. **Voyager Search:** Calls find\_nearest\_neighbors\_by\_vector, using the calculated average\_vector as the query and requesting neighbors\_to\_find results. eliminate\_duplicates is set to True to ensure variety.  
7. **Combine Results:**  
   * Starts the final list with the original top songs used for the fingerprint (contributing\_seed\_ids).  
   * Appends similar songs found by Voyager, skipping any duplicates, until the total\_desired\_size is reached.  
8. **Return:** Returns the combined list of song results (dictionaries with item\_id and distance).

#### **Stage 3: Formatting & Response (app\_sonic\_fingerprint.py)**

1. **Fetch Details:** Retrieves Title and Artist details for the final list of item\_ids using get\_score\_data\_by\_ids.  
2. **Format:** Combines the details and distances into the final JSON structure.  
3. **Return:** Sends the list of recommended tracks to the frontend.

#### **Stage 4: Playlist Creation**

* Uses the same /api/create\_playlist endpoint as other features. Crucially, the user\_creds (collected in Stage 1\) are passed along in the request payload from the frontend (sonic\_fingerprint.html) to ensure the playlist is created for the correct user on the media server via create\_instant\_playlist.

### **7.3. Environment Variable Configuration**

The Sonic Fingerprint feature uses the following configurations:

#### **Core Infrastructure & Index**

* DATABASE\_URL (or POSTGRES\_\* variables): **(Required)** Used to fetch embeddings and song details.  
* INDEX\_NAME, VOYAGER\_QUERY\_EF, EMBEDDING\_DIMENSION: Used for querying the Voyager index.

#### **Media Server**

* MEDIASERVER\_TYPE: **(Required)** Determines how to interact with the media server API to get play history and create playlists.  
* JELLYFIN\_URL, JELLYFIN\_USER\_ID, JELLYFIN\_TOKEN: Default Jellyfin credentials (used if user doesn't provide specific ones, and for resolving usernames).  
* NAVIDROME\_URL, NAVIDROME\_USER, NAVIDROME\_PASSWORD: Default Navidrome credentials (used if user doesn't provide specific ones).  
* *(Other media server credentials)*: Used similarly based on MEDIASERVER\_TYPE.

#### **Sonic Fingerprint Parameters**

* SONIC\_FINGERPRINT\_TOP\_N\_SONGS: The number of the user's most played songs to use as the basis for the fingerprint calculation.  
* SONIC\_FINGERPRINT\_NEIGHBORS: The default total number of songs desired in the final generated playlist if the user doesn't specify n.

#### **Filtering Parameters (Used during Neighbor Search)**

* MAX\_SONGS\_PER\_ARTIST, DUPLICATE\_DISTANCE\_THRESHOLD\_\*, DUPLICATE\_DISTANCE\_CHECK\_LOOKBACK, VOYAGER\_METRIC: Used by \`find\_nearest\_neighbors\_by\_vector\` and other neighbor-search helpers to apply duplicate and artist-cap filtering.

## **8\. Instant Playlist (Chat)**

This section documents the Instant Playlist (Chat) feature: a conversational UI that accepts a user's natural-language prompt (mood, activity, description), uses an AI model to produce a playlist intent and a safe read-only query, executes the query against the analyzed music library, and optionally creates a playlist on the configured media server.

### **8.1. Functional Analysis (High-Level)**

From a user's perspective, Instant Playlist (Chat) provides a fast, natural-language-driven way to generate playlists without needing to configure clustering or sampling parameters.

Key User Interactions & Workflow

1. The user navigates to the Instant Playlist page (chat UI under the `/chat` blueprint). The page shows AI provider/model controls and a single prompt textarea (see `templates/chat.html`).
2. The user types a prompt describing mood, activity, or desired songs (e.g., "upbeat workout with electronic funk") and clicks "Get Playlist Idea".
3. The frontend gathers UI options (provider, model, optional Ollama server URL override) and posts to `/chat/api/chatPlaylist` with `{ userInput, ai_provider, ai_model, ... }`.
4. The backend constructs a structured AI prompt asking for two things: a short human-friendly message summarizing the playlist intent and a parameterized, read-only SQL (or structured query) that will return candidate songs (or, alternatively, a validated structured query object). It calls an AI provider via helper functions in `ai.py`.
5. The AI's response is validated and sanitized server-side. If valid, the server executes the query (with parameter binding) against PostgreSQL, returning the message, the executed SQL (for transparency), and the resulting songs. If invalid or unsafe, the server either rejects the AI output and returns an error or falls back to a safe templated query built from extracted entities.
6. The frontend displays:
   * Collapsible AI interaction log (full message),
   * Collapsible executed query (for debugging/transparency),
   * The resulting playlist (ordered list of songs), and
   * A "Create Playlist on Media Server" section to push the returned songs to the user's configured media server.

Outcome

The user receives an instant playlist that reflects their natural language prompt, with the option to persist it to their media server. The UI emphasizes transparency by showing the AI message and executed query.

### **8.2. Technical Analysis (Algorithm-Level)**

The backend emphasizes structured AI output, safety, and result-size limits. The implementation uses `ai.py` helpers to call Ollama/Gemini/Mistral, enforces timeouts/delays, and strictly validates any AI-generated query.

Stage 1: Request Handling

1. Endpoint: POST `/chat/api/chatPlaylist` (registered by `chat_bp` in `app_chat.py`). The payload includes `userInput`, `ai_provider`, `ai_model`, and provider-specific overrides (e.g., `ollama_server_url`).
2. Validation: Ensure `userInput` is present and non-empty. Normalize provider value (uppercase) and select server-side defaults from config if client omitted them.

Stage 2: Prompt Construction

1. Build a prompt that asks the model to return a structured JSON object with explicit keys: `message` (string), `query` (parameterized SQL string or a high-level structured query representation), and `params` (array/object of bind parameters). Example instruction: "Return JSON: {message:'..', query:'SELECT ... WHERE ... LIMIT $1', params:[50]}".
2. Include explicit constraints in the prompt: SQL must be READ-ONLY (SELECT only), no joins that access non-music tables, result size limit, and only allowed columns (item_id, title, author, album, score, feature fields). Ask the model to prefer parameterized queries and to avoid database-specific features.

Stage 3: Call AI Provider

1. Use `ai.py` to call the selected provider. For Ollama, the app may call an internal or user-provided Ollama server URL (streaming support exists). For Gemini, the server uses `google.generativeai`; for Mistral, it uses `mistralai`.
2. Apply provider-specific delays/timeouts (Gemini/Mistral use env-controlled call-delay variables) and keep a high-level timeout to avoid long waits. Log the full raw AI response for debugging (but not API keys).

Stage 4: Validate & Sanitize AI Output

1. If the AI returns JSON with `query` and `params`, parse it and inspect `query` to ensure it:
   * Is read-only (only SELECT),
   * Does not contain prohibited keywords (INSERT, UPDATE, DELETE, DROP, ALTER, COPY, TRUNCATE),
   * References only allowed table(s) (score, embedding, track-related tables) and allowed columns.
2. If the AI returns freeform SQL or text, attempt to extract intent (mood/genre/tempo/entities) and map to a safe, parameterized server-side template query (e.g., a template that selects based on mood vector similarity + tempo range + artist cap).
3. Enforce a maximum `LIMIT` (e.g., `ALCHEMY_MAX_N_RESULTS`) regardless of AI-specified limits.

Stage 5: Execute Query & Post-process Results

1. Execute the validated, parameterized query against PostgreSQL and fetch rows.
2. Post-process results: limit final result size, apply de-duplication (by title/artist), and enforce `MAX_SONGS_PER_ARTIST` if requested. Build the `query_results` array for the response.
3. Save the executed SQL and params to the response so the frontend can show the executed query in a collapsible section.

Stage 6: Optional Playlist Creation on Media Server

1. The frontend posts the playlist name and `item_ids` to an endpoint that maps internal `item_id`s to media-server-specific IDs and creates the playlist using the configured media server adapter (Jellyfin/Emby/Navidrome). These adapter functions live in `mediaserver.py`.
2. Return the media server response (success, playlist id, or error) to the frontend and display it in the UI.

Safety & Fallbacks

* If AI output fails validation, return a friendly error and either (a) allow the user to retry, (b) run a safe fallback query built from extracted entities, or (c) return an empty result with an explanation.
* Limit the number of AI calls per minute or per session to avoid abuse. Record AI call metadata (time, provider, model) for observability.

### **8.3. Environment Variable Configuration**

Instant Playlist (Chat) reuses many existing infra variables and adds a few chat-specific ones.

Core / Shared

* `REDIS_URL`, `DATABASE_URL`, `TEMP_DIR`, etc. — core infra used elsewhere in the app.

AI / Chat Specific

* `AI_MODEL_PROVIDER` — Default chat AI provider (OLLAMA, GEMINI, MISTRAL, NONE).
* `OLLAMA_SERVER_URL` — Default Ollama server (e.g., `http://localhost:11434/api/generate`) used when provider is OLLAMA. The frontend may supply an override `ollama_server_url` per-request.
* `OLLAMA_MODEL_NAME` — Default Ollama model name for playlist-related prompts.
* `GEMINI_API_KEY` — Server-side Google Gemini key used for GEMINI provider.
* `GEMINI_MODEL_NAME` — Default Gemini model (e.g., `gemini-2.5-pro`).
* `GEMINI_API_CALL_DELAY_SECONDS` — Optional delay to respect Gemini rate limits (used by `ai.py`).
* `MISTRAL_API_KEY` — Server-side Mistral key used for MISTRAL provider.
* `MISTRAL_MODEL_NAME` — Default Mistral model for playlist prompts.
* `MISTRAL_API_CALL_DELAY_SECONDS` — Optional delay for Mistral calls.

Database safety for AI chat

* `AI_CHAT_DB_USER_NAME`, `AI_CHAT_DB_USER_PASSWORD` — Credentials for a restricted, read-only DB role that the chat/AI flow uses to execute validated, parameterized SELECT queries. The application creates/uses this low-privilege user to ensure any AI-generated SQL runs without write or DDL permissions. Documented here for operators who manage DB roles and secrets.

Limits and Safety

* `ALCHEMY_DEFAULT_N_RESULTS` / `ALCHEMY_MAX_N_RESULTS` — Caps the number of songs returned by AI-generated queries.
* `CHAT_SQL_ALLOWED_READ_ONLY` — Conceptual flag: code enforces read-only SQL execution for AI results. (Enforced in code; not required to be present as an env var.)

Operational Notes

* API keys remain server-side; the frontend should never submit third-party API keys. For Ollama the UI may optionally provide an Ollama server URL if users self-host and the server is reachable from the UI.
* Log AI interactions (message, executed query, provider, model) for debugging and observability; avoid logging secrets.

## Notes

This design favors structured AI output (JSON with `message`, `query`, `params`) and strict validation to safely combine LLM creativity with controlled database queries. When a generated query cannot be validated, the server should fall back to a safe template-based query derived from the AI's intent.

```

## **9. Database Cleaning**

The Database Cleaning feature identifies and (optionally) removes tracks/albums that exist in the application's database but are no longer present on the configured media server. It runs as an asynchronous RQ task and is intended to keep the score/embedding tables in sync with the user's media server.

### **9.1. Functional Analysis (High-Level)**

From a user's perspective, Database Cleaning is accessible from the Analysis UI and provides a safe, auditable way to remove stale entries from the database.

Key User Interactions & Workflow

1. The user opens the Database Cleaning page (`/cleaning`) served by the `analysis_bp` blueprint. The page shows a summary area, Start/Clear buttons, and a status panel (see `templates/cleaning.html`).
2. The user clicks "Start Cleaning". The frontend requests `/api/cleaning/start` which enqueues a high-priority RQ job (`tasks.cleaning.identify_and_clean_orphaned_albums_task`).
3. The cleaning job performs these high-level steps:
   - Fetch all albums/tracks from the configured media server via the mediaserver adapter (Jellyfin/Navidrome/Emby).
   - Read all tracks currently present in the application's PostgreSQL database (score + embedding tables).
   - Compute the set difference to discover orphaned tracks (in DB but not on the media server).
   - Group orphaned tracks by artist/album and present a summary.
   - Apply a safety limit (configured via `CLEANING_SAFETY_LIMIT`) to avoid accidental large deletions.
   - Optionally delete the orphaned tracks and related references (embedding rows, playlist entries), and rebuild the Voyager index.
4. The UI polls the task status (same `active_tasks` / `status` APIs used elsewhere) and displays a live log, progress bar, and final summary. The user can cancel the running task via `/api/cancel/<task_id>`.

Outcome

After a successful run the database has fewer stale records, the Voyager index is rebuilt, and the UI shows a detailed summary of deleted tracks and any failures.

### **9.2. Technical Analysis (Algorithm-Level)**

The cleaning process is implemented as a fault-tolerant RQ job that emphasizes safety, logging, and transparent reporting.

Stage 1: Job Enqueueing

1. Endpoint: POST `/api/cleaning/start` (defined in `app_analysis.py`). It performs lightweight validation and enqueues the task `tasks.cleaning.identify_and_clean_orphaned_albums_task` with a UUID job id and retry policy.
2. The endpoint records a TASK_STATUS_PENDING entry in the database via `save_task_status` so the UI can immediately display the new task.

Stage 2: Media Server Enumeration

1. The cleaning task calls the mediaserver adapter's `get_recent_albums(0)` or equivalent to fetch all albums/tracks from the media server (0 indicates fetch all).
2. It iterates albums and collects all media-server track IDs. It handles transient errors per-album, logs warnings, and continues rather than failing the whole job.

Stage 3: Database Scan & Orphan Detection

1. The task queries the database for distinct `item_id` values joining `score` and `embedding` to ensure only fully analyzed tracks are considered.
2. Orphaned track IDs are computed as `database_track_ids - media_server_track_ids`.
3. Orphaned tracks are grouped by artist (and optionally album) for human-friendly presentation.

Stage 4: Safety, Presentation, and Deletion

1. Safety: If the number of orphaned albums/tracks exceeds `CLEANING_SAFETY_LIMIT`, the job limits the deletion set to the first N (and logs that the safety limit was applied).
2. Deletion: The task calls `delete_orphaned_albums_sync(orphaned_track_ids)` which removes rows from `embedding`, `score`, and any playlist/auxiliary tables referencing those tracks, handling FK constraints and committing in a transaction. Failures are collected and returned.
3. Rebuild Index: On success (or even when no orphans were found), the task rebuilds the Voyager index by calling `build_and_store_voyager_index(get_db())` to ensure the in-memory index matches the DB.

Stage 5: Logging & Task Status

1. The RQ job updates the RQ job meta and persistent `task_status` rows frequently via `save_task_status` and `current_job.meta` updates, including a `log` array and `progress` percentage.
2. The UI polls `/api/active_tasks` and displays truncated logs (server truncates to last 10 entries) and final `final_summary_details` when the job completes.

Error Handling & Resilience

* Database errors (OperationalError) are surfaced and cause the job to be retried (RQ retry policy). Transient media-server API errors are logged but do not abort the whole run unless they prevent completing essential steps.
* Deletion is transactional where possible; partial failures are captured and returned in the summary.

### **9.3. Environment Variable Configuration**

The cleaning process uses core infra variables and a few cleaning-specific ones.

Core Infra

* `REDIS_URL` — Required by RQ for job queueing and for the background listener used elsewhere.
* `DATABASE_URL` — Required to query and delete database rows, and to rebuild the Voyager index.
* `MEDIASERVER_TYPE`, `JELLYFIN_URL`, `JELLYFIN_TOKEN`, `NAVIDROME_URL`, etc. — Credentials used by the mediaserver adapter to enumerate media server albums and to resolve track IDs.

Cleaning-Specific

* `CLEANING_SAFETY_LIMIT` — Maximum number of orphaned albums (or a related cap) the automatic cleaning operation will delete in a single run to avoid catastrophic data loss.
* `MAX_QUEUED_ANALYSIS_JOBS` — Imported in the task module for coordination/limits; not directly user-adjustable for cleaning but used by task orchestration logic when needed.

Operational Notes

* The UI triggers the cleaning task via `/api/cleaning/start` and saves the returned job id to local state so it can poll and show progress. The UI also exposes a cancel button which calls `/api/cancel/<task_id>` to revoke the job.
* All actions are logged; the final summary includes counts, lists of orphaned albums (up to safety limit), deleted_count, failed_deletions, and any warnings about index rebuild failures.

```

## **10. Scheduled Tasks (Cron)**

The Scheduled Tasks feature provides a simple UI for defining cron-like schedules to automatically run recurring long-running jobs such as Analysis and Clustering. It uses a database-backed cron table and a background runner (`run_due_cron_jobs`) that enqueues jobs when the cron expression matches the current time.

### **10.1. Functional Analysis (High-Level)**

From a user's perspective, Scheduled Tasks allow configuring periodic runs without manually starting Analysis or Clustering.

Key User Interactions & Workflow

1. The user navigates to the Scheduled Tasks page (`/cron`) which displays two editable fields (Analysis and Clustering), each with a cron expression input and an Enable checkbox (see `templates/cron.html`).
2. The user enters cron expressions (for example, `0 2 * * 0-5` for nightly runs) and toggles Enable, then clicks Save.
3. The frontend POSTs the configuration to `/api/cron` which inserts or updates rows in a `cron` database table.
4. A background scheduler (the `run_due_cron_jobs` function) periodically reads enabled cron rows, evaluates whether the cron expression matches the current time, and enqueues corresponding RQ jobs when due. The function avoids duplicate runs by checking the `last_run` timestamp and skipping entries run in the last ~55 seconds.
5. Each scheduled job is enqueued like a normal user-triggered job and appears in the same task/status panels (task_status table) so users can monitor, cancel, or inspect results.

Outcome

Admins can automate regular maintenance or data-refresh tasks (analysis, clustering) using cron expressions stored in the DB. The system ensures idempotency and avoids accidental rapid re-enqueues.

### **10.2. Technical Analysis (Algorithm-Level)**

The scheduled task subsystem is lightweight and intentionally conservative: it uses a simple cron expression matcher and enqueues existing RQ tasks with default parameters.

Stage 1: CRUD UI and Persistence

1. UI: The `/cron` page (client-side JS in `cron.html`) fetches existing cron rows from `/api/cron` and populates fields for Analysis and Clustering.
2. Persistence: Saving writes to the `cron` table using `/api/cron` (POST), creating or updating rows containing `name`, `task_type`, `cron_expr`, and `enabled`.

Stage 2: Cron Matching

1. The `run_due_cron_jobs` function performs the scheduling loop. It loads enabled rows and calls `cron_matches_now(expr, ts)` to test if the cron expression matches the current timestamp.
2. The cron matcher supports `*`, single numbers, comma-separated lists, and ranges (e.g., `1-5`), and compares minute, hour, day-of-month, month, and day-of-week fields. Note: the function converts Python's `tm_wday` to cron's 0=Sun..6=Sat semantics.

Stage 3: Enqueueing Jobs

1. If a cron row matches and has not recently run, `run_due_cron_jobs` generates a UUID job id, writes a TASK_STATUS_PENDING entry (for visibility), and enqueues the appropriate RQ job:
   * For `analysis`: enqueues `tasks.analysis.run_analysis_task` with `(0, TOP_N_MOODS)` to process the whole library.
   * For `clustering`: enqueues `tasks.clustering.run_clustering_task` with a kwargs bundle derived from configuration defaults (CLUSTER_ALGORITHM, NUM_CLUSTERS_MIN/MAX, PCA ranges, scoring weights, AI naming defaults, etc.).
2. After enqueueing, it updates the `last_run` timestamp in the cron table to prevent duplicate immediate requeues.

Stage 4: Observability & Safety

1. Each cron-initiated job uses the same task-status/logging machinery as manual jobs, so progress, logs, and final summaries are available in the UI.
2. The enqueued clustering job uses conservative defaults sourced from environment/config values to avoid accidental heavy runs—these defaults can be tuned in config.

Error Handling & Resilience

* The scheduler catches exceptions per-row and logs them without stopping the whole loop. If enqueueing fails for a row, it continues to process other rows.
* The scheduler uses a small guard window (~55s) to avoid enqueuing the same job multiple times when the scheduler runs frequently.

### **10.3. Environment Variable Configuration**

Scheduled tasks reuse many core config values and a set of clustering/analysis defaults to build job parameters.

Core Infra

* `DATABASE_URL`, `REDIS_URL` — Required for reading cron rows and enqueueing RQ jobs.

Analysis & Clustering Defaults (used when enqueueing cron jobs)

* `TOP_N_MOODS` — Number of moods passed to analysis jobs when scheduled.
* `CLUSTER_ALGORITHM`, `NUM_CLUSTERS_MIN`, `NUM_CLUSTERS_MAX`, `DBSCAN_EPS_MIN`, `DBSCAN_EPS_MAX`, `DBSCAN_MIN_SAMPLES_MIN`, `DBSCAN_MIN_SAMPLES_MAX`, `GMM_N_COMPONENTS_MIN`, `GMM_N_COMPONENTS_MAX`, `SPECTRAL_N_CLUSTERS_MIN`, `SPECTRAL_N_CLUSTERS_MAX`, `PCA_COMPONENTS_MIN`, `PCA_COMPONENTS_MAX` — Default ranges used to compose clustering kwargs.
* `CLUSTERING_RUNS`, `MAX_SONGS_PER_CLUSTER`, `TOP_N_PLAYLISTS`, `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`, `STRATIFIED_SAMPLING_TARGET_PERCENTILE` — High-level clustering behavior used when cron enqueues clustering.
* `SCORE_WEIGHT_*` and other scoring weights — Defaults applied to scheduled clustering runs.
* `AI_MODEL_PROVIDER`, `OLLAMA_SERVER_URL`, `OLLAMA_MODEL_NAME`, `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `MISTRAL_API_KEY`, `MISTRAL_MODEL_NAME` — AI naming defaults applied when scheduled clustering requests automatic playlist naming.

Operational Notes

* The scheduler is intended to be invoked periodically (e.g., via a thread in `app.py` or a separate process). Ensure only one scheduler instance updates `last_run` to avoid duplicate enqueues in multi-process deployments (use a leader election or central cron runner if required).
* Tuning the cron expressions and the clustering defaults is recommended to balance freshness and compute cost.

## **11. Text Search (CLAP)**

This section details the "Text Search" feature, which enables natural language music discovery using CLAP (Contrastive Language-Audio Pretraining) embeddings. Users can search their music library using plain English descriptions like "upbeat summer songs" or "relaxing instrumental music" instead of relying on traditional metadata filters.

### **11.1. Functional Analysis (High-Level)**

From a user's perspective, "Text Search" provides an intuitive, Google-like search experience for music discovery. The feature leverages pre-computed CLAP embeddings to match natural language queries against audio characteristics.

#### **Key User Interactions & Workflow**

1. **Prerequisite:** The user must have already run the "Start Analysis" task at least once with CLAP enabled. This generates and stores CLAP embeddings for all analyzed tracks.
2. **Initiation:** The user navigates to the "Text Search" page (clap_search.html).
3. **Model Warmup (Automatic):**
   * When the page loads, it automatically calls the `/api/clap/warmup` endpoint to preload the CLAP text model into memory.
   * This warmup starts a timer (default 5 minutes) that keeps the model loaded for fast subsequent searches. The timer resets with each search.
   * A status indicator shows whether the model is "ready" and displays the remaining time before automatic unload.
4. **Query Input:**
   * The user types a natural language query into the search box (minimum 3 characters required).
   * Suggested queries are displayed as clickable buttons for inspiration (e.g., "energetic rock songs", "calm acoustic guitar", "upbeat electronic dance").
5. **Search Execution:**
   * The user presses Enter or clicks the search button.
   * The query is sent to `/api/clap/search` which returns matching songs ranked by semantic similarity.
6. **Results Display:**
   * Results are displayed in a table showing Title, Artist, and a Similarity score (0.0 to 1.0).
   * Results are ordered by relevance, with the most semantically similar songs at the top.
   * The number of results defaults to 100 but can be configured (maximum 500).
7. **Playlist Creation:**
   * If results are found, a "Create Playlist" button appears.
   * The user can edit the suggested playlist name (defaults to the search query) and click **"Create Playlist on Media Server"** to save the results.
8. **Cache Management:**
   * An admin "Refresh Cache" button allows reloading CLAP embeddings from the database after new songs are analyzed.
   * Cache statistics (number of songs, memory usage) are displayed for monitoring.
9. **Outcome:** Users discover music based on sonic characteristics and mood rather than relying solely on metadata like genre tags or artist names.

#### **Core Purpose**

The functional purpose is to **"search music by audio characteristics using simple text queries"**. CLAP embeddings capture audio characteristics in a high-dimensional space aligned with natural language descriptors. The system works best with simple queries combining genre (e.g., "rock", "electronic"), instrumentation (e.g., "acoustic guitar", "piano", "synthesizer"), and mood descriptors (e.g., "energetic", "calm"). Performance is strongest for genre and instrument identification, with mood recognition being somewhat less precise.

### **11.2. Technical Analysis (Algorithm-Level)**

The Text Search system is architected around two core components: CLAP embedding generation during analysis, and fast in-memory similarity search at query time.

#### **Stage 0: CLAP Embedding Generation (During Analysis)**

CLAP embeddings are generated as part of the song analysis pipeline, running in parallel with mood/embedding extraction.

1. **Model Architecture:** The system uses split ONNX CLAP models:
   * **Audio model** (`clap_audio_model.onnx`, ~268MB): Loaded in worker containers during analysis to generate embeddings from audio files.
   * **Text model** (`clap_text_model.onnx`, ~478MB): Loaded in Flask containers during search to generate embeddings from text queries.
   * This split architecture saves memory—workers only load audio, Flask only loads text (vs. ~746MB combined).
2. **Audio Processing (analyze_track in tasks/analysis.py):**
   * After loading audio with librosa, if `CLAP_ENABLED=true`, the system calls `get_clap_audio_embedding` from `tasks.clap_analyzer`.
   * Audio is resampled to 48kHz mono and converted to a mel-spectrogram.
   * The CLAP audio model generates a 512-dimensional embedding vector.
3. **Database Storage:**
   * The embedding is serialized to binary (BYTEA) and stored in the `clap_embedding` table, linked to the track's `item_id`.
   * Table schema: `(item_id PRIMARY KEY, embedding BYTEA)` with foreign key to `score(item_id)`.
4. **Normalization:** Both audio and text embeddings are L2-normalized, enabling fast cosine similarity computation via dot product.

#### **Stage 1: In-Memory Cache Loading (Flask Startup)**

1. **Cache Initialization:** When the Flask application starts (app.py), it calls `load_clap_cache_from_db` if `CLAP_ENABLED=true`.
2. **Bulk Load:**
   * Fetches all rows from `clap_embedding` joined with `score` to get `(item_id, embedding, title, author)`.
   * Converts BYTEA embeddings to NumPy arrays and validates dimensionality (512).
   * Builds three parallel data structures:
     * `embeddings`: NumPy matrix (N × 512) for vectorized operations.
     * `metadata`: List of dicts with `{item_id, title, author}`.
     * `item_ids`: List of item IDs for index mapping.
3. **Memory Footprint:** For a typical library of 10,000 songs:
   * Embeddings: 10,000 × 512 × 4 bytes (float32) ≈ 20 MB
   * Metadata: ~2-3 MB
   * Total: ~22-23 MB in RAM
4. **Global Cache:** Stored in `_CLAP_CACHE` global dictionary with a `loaded` flag for fast status checks.

#### **Stage 2: Model Warmup & Timer Management**

1. **Lazy Text Model Loading:**
   * The CLAP text model is **not** loaded at startup to save memory (~478MB).
   * When `/api/clap/warmup` is called (page load or first search), `initialize_clap_text_model` loads the text model into `_text_session`.
2. **Warmup Timer:**
   * After loading, a background thread starts a countdown (default `CLAP_TEXT_SEARCH_WARMUP_DURATION=300` seconds).
   * Each search or warmup call resets the timer.
   * When the timer expires, `unload_clap_model` is called to free memory.
   * The audio model is never loaded in Flask containers (only in workers).
3. **Status Endpoint:** `/api/clap/warmup/status` returns `{active: bool, seconds_remaining: int}` for UI feedback.

#### **Stage 3: Text Query Search (search_by_text)**

1. **API Call:** User submits query via POST to `/api/clap/search` with JSON: `{query: "upbeat summer songs", limit: 100}`.
2. **Auto-Warmup:** `search_by_text` calls `warmup_text_search_model()` to ensure the text model is loaded and reset the timer.
3. **Text Embedding Generation:**
   * Query text is tokenized using a RoBERTa tokenizer (loaded from `transformers` library).
   * The CLAP text model (`_text_session`) generates a 512-dimensional embedding via ONNX Runtime inference.
   * The embedding is L2-normalized.
4. **Vectorized Similarity Computation:**
   * Computes cosine similarity between the query embedding and all cached song embeddings using a single matrix multiplication:
     ```python
     similarities = _CLAP_CACHE['embeddings'] @ text_embedding  # (N,) array
     ```
   * This vectorized operation is extremely fast (~1-2ms for 10,000 songs on CPU).
5. **Top-K Selection:**
   * Uses `np.argsort(similarities)[::-1][:limit]` to get indices of top results.
   * Maps indices back to metadata to construct result dicts with `{item_id, title, author, similarity}`.
6. **Response:** Returns JSON with `{query, results, count}` to the frontend.

#### **Stage 4: Top Queries (Precomputation)**

1. **Background Generation:** At startup (if cache loaded), a background thread may compute "top queries" by:
   * Loading category-weighted terms from `query.json` (Genre, Mood, Instrumentation, etc.).
   * Generating thousands of random 3-term queries (e.g., "energetic rock guitar").
   * Scoring each query by embedding diversity and result count.
   * Selecting top N (default 50) diverse queries for UI suggestions.
2. **Caching:** Stored in `_TOP_QUERIES_CACHE` global with a `ready` flag.
3. **Endpoint:** `/api/clap/top_queries` returns precomputed queries as inspiration buttons.

#### **Stage 5: Playlist Creation**

* Identical to Stage 4 of "Playlist from Similar Song," using the `/api/create_playlist` endpoint.
* The frontend passes the list of `item_id`s from search results to create the playlist.

### **11.3. Environment Variable Configuration**

The Text Search functionality is configured by the following environment variables (from config.py):

#### **Core Infrastructure**

* `REDIS_URL`: **(Required)** Used by RQ workers and pub/sub.
* `DATABASE_URL` (or `POSTGRES_*` variables): **(Required)** Used to fetch CLAP embeddings and song metadata.
* `MEDIASERVER_TYPE`, `JELLYFIN_URL`, etc.: **(Required)** Used to create playlists from search results.

#### **CLAP Feature Toggle**

* `CLAP_ENABLED`: (Boolean, default `true`) Master switch for CLAP functionality. If `false`, CLAP embeddings are not generated during analysis and text search is disabled.

#### **CLAP Model Paths**

* `CLAP_AUDIO_MODEL_PATH`: Filesystem path to the audio ONNX model (default `/app/model/clap_audio_model.onnx`). Used by workers during analysis.
* `CLAP_TEXT_MODEL_PATH`: Filesystem path to the text ONNX model (default `/app/model/clap_text_model.onnx`). Used by Flask for search queries.
* `CLAP_MODEL_PATH`: Legacy path for combined model (not used with split architecture, kept for backward compatibility).
* `CLAP_EMBEDDING_DIMENSION`: Expected embedding vector size (fixed at 512 for LAION CLAP models). Used for validation.

#### **ONNX Runtime Configuration**

* `CLAP_PYTHON_MULTITHREADS`: (Boolean, default `false`)
  * `false`: ONNX Runtime uses automatic thread management (recommended).
  * `true`: ONNX single-threaded, expects Python-level `ThreadPoolExecutor` for parallelism.

#### **Text Search Behavior**

* `CLAP_TEXT_SEARCH_WARMUP_DURATION`: Duration in seconds to keep the text model loaded after last use (default `300` = 5 minutes). Balances memory usage vs. search responsiveness.

#### **Top Queries Generation**

* `CLAP_TOP_QUERIES_COUNT`: Number of random queries to generate and score when computing top suggestions (default `1000`).
* `CLAP_CATEGORY_WEIGHTS`: JSON dict mapping query categories to sampling weights. Categories with higher weights appear more frequently in generated queries. Default:
  ```json
  {
    "Genre": 1.5,
    "Mood": 1.5,
    "Energy": 1.2,
    "Tempo": 1.0,
    "Instrumentation": 1.0,
    "Voice_Type": 0.8,
    "Production": 0.7,
    "Era": 0.5
  }
  ```

#### **Operational Notes**

* **Memory Management:** The split-model architecture is crucial for efficient resource usage:
  * Worker containers: Load only audio model (~268MB) during analysis.
  * Flask containers: Load only text model (~478MB) during searches, with automatic unloading after inactivity.
  * Cache is always in memory (~20-30MB) but models are lazy-loaded.
* **GPU Support:** CLAP models support CUDA acceleration if available (configured via `CUDA_VISIBLE_DEVICES`). Text model typically runs on CPU in Flask; audio model benefits from GPU in workers.
* **Cache Refresh:** After running analysis on new albums, use `/api/clap/cache/refresh` to reload embeddings without restarting Flask.
* **Query Quality:** CLAP performs best with simple queries (2-4 words) combining genre, instrument, and mood descriptors. Genre and instrument queries yield the most accurate results, while mood-based queries are less precise. Examples: "energetic rock guitar", "calm piano instrumental", "electronic dance synthesizer".
* **Tokenizer:** Uses HuggingFace `transformers` library's RoBERTa tokenizer. The `TRANSFORMERS_NO_ADVISORY_WARNINGS` env var suppresses PyTorch/TensorFlow warnings since only the tokenizer is used.
