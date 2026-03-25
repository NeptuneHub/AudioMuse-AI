# Architecture

AudioMuse-AI follows a distributed architecture with separate containers for web interface, task processing, and data storage.

An easy choice is to deploy everything on a single machine. Anyway, for performance reason, deploy multiple worker on multiple machine is also possible to speedup batch task like analysis and clustering. Worker can be then shutdown when not needed for this tasks.

## System Architecture

```mermaid
graph TB
    User[Browser/User<br/>Port :8000] ---|HTTP Requests| Flask[Flask Container<br/>Front-end + API]
    
    Flask ---|Enqueue Tasks| Redis[Redis Queue<br/>:6379]
    Flask ---|Read/Write| PostgreSQL[(PostgreSQL DB<br/>:5432)]
    
    Redis ---|Dequeue Tasks| Worker[Worker Container<br/>Analysis + Clustering]
    PostgreSQL ---|Read/Write| Worker
    
    MediaServer[Media Server<br/>Jellyfin/Navidrome<br/>Lyrion/Emby/LocalFiles] -.-|Fetch Music| Flask
    MediaServer -.-|Fetch Audio Files| Worker
    
    style User fill:#607D8B
    style Flask fill:#4CAF50
    style Redis fill:#FF5722
    style PostgreSQL fill:#9C27B0
    style Worker fill:#2196F3
    style MediaServer fill:#FF9800
```

## Component Responsibilities

### Flask Container
- **Web Interface**: Serves the front-end UI accessible at port 8000
- **REST API**: Provides endpoints for all AudioMuse-AI features
- **Task Orchestration**: Enqueues analysis and clustering jobs to Redis
- **Data Access**: Reads track information, playlists, and results from PostgreSQL
- **Media Server Integration**: Create playlist on mediaserver

### Worker Container
- **Job Processing**: Dequeues tasks from Redis queue
- **Audio Analysis**: Performs sonic analysis using Librosa and ONNX models
- **Clustering**: Executes playlist generation algorithms (KMeans, DBSCAN, GMM, Spectral)
- **Data Persistence**: Writes analysis results and embeddings to PostgreSQL
- **Audio Fetching**: Downloads audio files from media server for processing

### Redis Queue
- **Task Queue**: Stores pending analysis and clustering jobs
- **Job Status**: Tracks running and completed tasks
- **High Priority Queue**: Separate queue for priority tasks

### PostgreSQL Database
- **Track Registry** (`track` table): Canonical track identities with `file_path_hash` (SHA-256) for cross-provider deduplication
- **Provider Links** (`provider_track` table): Maps provider-specific item IDs to canonical track IDs, enabling multi-provider ID remapping
- **Provider Config** (`provider` table): JSONB-based provider registry (type, credentials, path prefix, priority)
- **Analysis Results** (`score`, `embedding` tables): Mood scores, feature vectors, 200-dim embeddings
- **Playlists**: Generated clusters and user playlists
- **Voyager Index** (`voyager_index_data`): Serialized HNSW vector similarity search index

### Media Server
- **Music Source**: Provides access to audio library
- **Multi-Provider Support**: Jellyfin, Navidrome, Lyrion, Emby, and LocalFiles (direct filesystem)
- **Audio Streaming**: Streams audio files for analysis (LocalFiles reads directly from disk)
- **Playlist Sync**: Target for generated playlists (supports creating on multiple providers simultaneously)

## Data Flow

### Analysis Workflow
1. User triggers analysis via Flask UI
2. Flask enqueues analysis job to Redis
3. Worker dequeues job from Redis
4. Worker fetches audio from Media Server
5. Worker performs sonic analysis
6. Worker writes results to PostgreSQL
7. Flask reads results and displays to user

### Clustering Workflow
1. User starts clustering via Flask UI
2. Flask enqueues clustering job to Redis
3. Worker dequeues job from Redis
4. Worker reads track embeddings from PostgreSQL
5. Worker executes clustering algorithm
6. Worker writes generated playlists to PostgreSQL
7. Worker optionally syncs playlists to Media Server
8. Flask displays results to user

## Network Ports

| Service | Port | Protocol |
|---------|------|----------|
| Flask (Web UI + API) | 8000 | HTTP |
| Redis | 6379 | TCP |
| PostgreSQL | 5432 | TCP |
| Jellyfin | 8096 | HTTP |
| Navidrome | 4533 | HTTP |
| Lyrion | 9000 | HTTP |
| Emby | 8096 | HTTP |
| LocalFiles | N/A | Filesystem |

## Deployment Modes

### Docker Compose
All containers run on one host, communicating via Docker network. With docker is also be possible to do deployment on multiple machine.

### Kubernetes
- Flask, Worker, Redis, PostgreSQL deployed as separate pods
- Services expose internal endpoints
- Persistent volumes for database storage

### Remote Worker
- Flask + Redis + PostgreSQL on main server
- Worker on remote machine (closer to media server or with GPU)
- Worker connects using `POSTGRES_HOST` and `REDIS_URL` pointing to main server
- Copy `.env` to remote worker and update these values to reach main server

## Scalability

- **Multiple Workers**: Deploy additional worker containers for parallel processing
- **Redis Queue**: Handles job distribution across workers
- **PostgreSQL**: Single source of truth for all data
- **Stateless Flask**: Can run multiple Flask instances behind load balancer