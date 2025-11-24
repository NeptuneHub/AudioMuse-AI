# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioMuse-AI is a self-hosted music analysis and playlist generation system that uses machine learning to analyze audio files and create intelligent playlists. It integrates with multiple media servers (Jellyfin, Navidrome, Lyrion, Emby) and uses audio analysis (Librosa, ONNX models) to generate sonic fingerprints and cluster similar songs.

## Architecture

### Core Components

**Flask Web Server** (`app.py`):
- Main web application serving API endpoints and UI
- Uses Flask blueprints for modular organization (each `app_*.py` is a separate blueprint)
- Background thread listens to Redis pub/sub for index reload notifications
- Cron manager thread runs scheduled tasks (analysis/clustering)
- ProxyFix middleware support for reverse proxy deployments (when `ENABLE_PROXY_FIX=true`)

**RQ Workers** (`rq_worker.py`, `rq_worker_high_priority.py`):
- Process background jobs from Redis queues (`default` and `high` priority)
- Workers execute long-running tasks like audio analysis and clustering
- Each worker maintains its own database and Redis connections

**Task Queue System**:
- Redis Queue (RQ) manages background job execution
- Two priority queues: `high` (for real-time user requests) and `default` (for batch processing)
- Parent-child task hierarchy tracked in PostgreSQL `task_status` table
- Task cancellation propagates recursively through children

**Database** (PostgreSQL):
- `tracks`: Analyzed song data with embeddings (200-dim vectors stored as bytea)
- `playlist`: Generated playlists and song associations
- `task_status`: Task tracking with parent-child relationships
- `voyager_index`: Serialized Voyager HNSW index for similarity search
- `artist_similarity_index`: GMM-based artist clustering index
- `map_projection`: 2D UMAP/PCA projections for visualization
- `cron_jobs`: Scheduled task definitions

**In-Memory Indexes**:
- Voyager HNSW index loaded at startup for fast similarity queries
- Artist GMM index for artist-based similarity
- Map projections cached for visualization endpoints

### Blueprint Organization

Each `app_*.py` file is a Flask blueprint handling a specific feature:
- `app_analysis.py`: Audio analysis endpoints
- `app_clustering.py`: Clustering and playlist generation
- `app_voyager.py`: Similarity search (formerly Annoy, now Voyager)
- `app_sonic_fingerprint.py`: User listening history analysis
- `app_path.py`: Song path generation between tracks
- `app_alchemy.py`: Blend songs by adding/subtracting characteristics
- `app_map.py`: 2D visualization of music library
- `app_artist_similarity.py`: Artist-based recommendations
- `app_extend_playlist.py`: Playlist extension with smart filters
- `app_collection.py`: Collection and queue management
- `app_chat.py`: AI chat interface for querying library
- `app_cron.py`: Scheduled task management
- `app_external.py`: External API integrations
- `app_waveform.py`: Waveform visualization

### Background Tasks (`tasks/` directory)

**Analysis** (`tasks/analysis.py`):
- Main analysis workflow: fetches albums, downloads audio, extracts features
- Uses ONNX models for inference (replaced TensorFlow in v0.7.0+)
- Generates 200-dimensional embeddings using MusicNN models
- Extracts mood predictions, energy, tempo, and "other features" (danceability, aggressive, happy, party, relaxed, sad)
- Normalizes and creates feature vectors for clustering
- Rebuilds Voyager index in batches (`REBUILD_INDEX_BATCH_SIZE`)

**Clustering** (`tasks/clustering.py`, `tasks/clustering_gpu.py`):
- Evolutionary Monte Carlo search for optimal clustering parameters
- Supports KMeans, DBSCAN, GMM, and Spectral clustering algorithms
- Optional GPU acceleration with RAPIDS cuML (`USE_GPU_CLUSTERING=true`)
- Batched execution: spawns multiple RQ jobs to parallelize runs
- Elite-based exploitation strategy: top solutions guide parameter mutation
- Scores using diversity, purity, and optional internal validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)

**Media Server Integration** (`tasks/mediaserver*.py`):
- Abstraction layer in `tasks/mediaserver.py` dispatches to server-specific implementations
- Each server module (Jellyfin, Navidrome, Emby, Lyrion, MPD) implements: `get_recent_albums()`, `get_tracks_from_album()`, `download_track()`, `create_playlist_on_server()`
- Server type determined by `MEDIASERVER_TYPE` environment variable

**Voyager Index Manager** (`tasks/voyager_manager.py`):
- Builds and stores Spotify's Voyager HNSW index for approximate nearest neighbor search
- Supports angular (cosine), euclidean, and dot product distance metrics
- Index serialized to PostgreSQL as bytea, loaded into memory on startup
- Redis pub/sub triggers index reloads across workers

**Path Manager** (`tasks/path_manager.py`):
- Generates musical journeys between two songs
- Uses centroid-based pathfinding with optional duplicate filtering
- `PATH_FIX_SIZE` controls whether to pad paths to exact length

**Song Alchemy** (`tasks/song_alchemy.py`):
- Blends song characteristics by adding/subtracting embeddings
- Supports temperature-based probabilistic sampling

## Development Commands

### Running Locally with Docker Compose

```bash
# Start all services (Flask app, workers, Redis, PostgreSQL)
docker compose -f deployment/docker-compose.yaml up -d

# View logs
docker compose -f deployment/docker-compose.yaml logs -f

# Stop services
docker compose -f deployment/docker-compose.yaml down
```

### Environment Configuration

Create `.env` file in `deployment/` directory from example:
```bash
cp deployment/.env.example deployment/.env
```

Edit `.env` with your media server credentials and API keys.

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_analysis.py
```

### Database Migrations

No formal migration tool is used. Schema changes are applied via `init_db()` in `app_helper.py`, which runs `CREATE TABLE IF NOT EXISTS` statements on startup.

### Worker Management

```bash
# Start worker manually for debugging
python rq_worker.py

# Start high-priority worker
python rq_worker_high_priority.py

# View RQ dashboard (if rq-dashboard installed)
rq-dashboard --redis-url redis://localhost:6379/0
```

## Key Configuration Parameters

**Analysis**:
- `NUM_RECENT_ALBUMS`: Number of recent albums to analyze (0 = all)
- `TOP_N_MOODS`: Number of top mood tags to include in feature vector (default: 5)
- `MUSIC_LIBRARIES`: Comma-separated list of libraries/folders to scan (empty = all)
- `REBUILD_INDEX_BATCH_SIZE`: Rebuild Voyager index after this many albums (default: 100)
- `AUDIO_LOAD_TIMEOUT`: Timeout in seconds for loading audio files (default: 600)

**Clustering**:
- `CLUSTER_ALGORITHM`: Algorithm to use (`kmeans`, `dbscan`, `gmm`, `spectral`)
- `ENABLE_CLUSTERING_EMBEDDINGS`: Use embeddings (true) or feature scores (false)
- `USE_GPU_CLUSTERING`: Enable GPU acceleration with RAPIDS cuML (experimental)
- `CLUSTERING_RUNS`: Number of evolutionary search iterations (default: 5000)
- `ITERATIONS_PER_BATCH_JOB`: Iterations per RQ batch job (default: 20)
- `MAX_CONCURRENT_BATCH_JOBS`: Max parallel batch jobs (default: 10)
- `TOP_N_PLAYLISTS`: Keep top N diverse playlists after clustering (default: 8)

**Scoring Weights**:
- `SCORE_WEIGHT_DIVERSITY`: Inter-playlist mood diversity weight (default: 2.0)
- `SCORE_WEIGHT_PURITY`: Intra-playlist mood consistency weight (default: 1.0)
- `SCORE_WEIGHT_SILHOUETTE`: Silhouette score weight (default: 0.0)

**Similarity Search**:
- `VOYAGER_METRIC`: Distance metric (`angular`, `euclidean`, `dot`)
- `VOYAGER_EF_CONSTRUCTION`: HNSW construction parameter (default: 1024)
- `VOYAGER_M`: HNSW max connections per node (default: 64)
- `VOYAGER_QUERY_EF`: HNSW query parameter (default: 1024)
- `MAX_SONGS_PER_ARTIST`: Max songs per artist in results (default: 3)

## Important Implementation Details

### Task Status Management

Tasks follow a strict lifecycle: `PENDING` → `STARTED` → `PROGRESS` → `SUCCESS/FAILURE/REVOKED`

Parent tasks spawn child tasks. When canceling, the system recursively marks all descendants as `REVOKED` and attempts to cancel jobs in RQ.

Use `save_task_status()` from `app_helper.py` to update task progress. Always include:
- `progress`: 0-100 percentage
- `details`: JSON dict with `log` array for status messages

### Index Reload Pattern

When Voyager or artist similarity indexes are rebuilt:
1. Worker stores new index to PostgreSQL
2. Worker publishes `reload` message to Redis channel `index-updates`
3. Flask app's background listener thread receives message
4. Flask app calls `load_voyager_index_for_querying(force_reload=True)`
5. All in-memory indexes refresh

**Never** make HTTP calls from workers to Flask app. Use Redis pub/sub for notifications.

### Media Server Abstraction

All media server operations go through `tasks/mediaserver.py`, which dispatches to the correct implementation based on `MEDIASERVER_TYPE`. When adding new server support:
1. Create `tasks/mediaserver_<name>.py`
2. Implement required functions: `get_recent_albums()`, `get_tracks_from_album()`, `download_track()`, `create_playlist_on_server()`
3. Add dispatch case in `tasks/mediaserver.py`

### ONNX Model Inference

Models are loaded with ONNX Runtime. Execution providers prioritize GPU if available:
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
```

Always use the tensor name mappings in `DEFINED_TENSOR_NAMES` dict (in `tasks/analysis.py`).

### Database Connections

Use Flask's `g` object for request-scoped connections:
```python
from app_helper import get_db
db = get_db()  # Returns connection from g or creates new one
# Connections auto-close via teardown_appcontext
```

In RQ workers, create new connections inside task functions:
```python
import psycopg2
from config import DATABASE_URL
conn = psycopg2.connect(DATABASE_URL)
```

### Embedding Storage

Embeddings are stored as PostgreSQL `bytea`:
```python
# Store
embedding_bytes = embedding_array.astype(np.float32).tobytes()
cur.execute("INSERT INTO tracks (embedding) VALUES (%s)", (embedding_bytes,))

# Retrieve
row = cur.fetchone()
embedding = np.frombuffer(row['embedding'], dtype=np.float32)
```

## Common Pitfalls

1. **Don't modify task details directly in RQ job meta** - Use `save_task_status()` to persist to database, which is the source of truth.

2. **Always use app context in workers** - Wrap worker code accessing Flask app resources:
   ```python
   from app import app
   with app.app_context():
       # Your code here
   ```

3. **Check terminal statuses before canceling** - Tasks in `SUCCESS`, `FAILURE`, or `REVOKED` states cannot be canceled.

4. **Batched index rebuilds** - Don't rebuild Voyager index after every track. Use `REBUILD_INDEX_BATCH_SIZE` to batch updates.

5. **GPU memory management** - When using GPU clustering, ensure tasks don't leak GPU memory. Use `del` and `gc.collect()` for large arrays.

6. **Stratified sampling** - Clustering uses stratified sampling by genre to ensure balanced representation. Configured via `STRATIFIED_GENRES` and `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`.

7. **GUI memory management** - The web GUI can crash when displaying large task logs. All API endpoints (`/api/active_tasks`, `/api/last_task`, `/api/task/<id>`) prune large arrays (`checked_album_ids`, `clustering_run_job_ids`) and truncate logs to 20 entries. The frontend JavaScript also limits JSON rendering to 50KB and performs periodic memory cleanup.

## API Documentation

Access interactive API docs at `http://localhost:8000/apidocs` (Swagger UI) when Flask app is running.

## Version Information

Current version is tracked in `config.py` as `APP_VERSION`. Version is injected into all templates via context processor and logged on startup.

From v0.7.0-beta onwards, ONNX replaced TensorFlow for model inference. Libraries analyzed before v0.7.0 must be re-analyzed for compatibility.
