# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioMuse-AI is a Dockerized music analysis and playlist generation platform. It analyzes audio files from self-hosted media servers (Jellyfin, Navidrome, Lyrion, Emby, LocalFiles) using Librosa and ONNX ML models, then generates playlists through clustering, similarity search, and AI-powered natural language requests.

**Current version:** v0.9.3

## Architecture

**Distributed 3-container system:**
- **Flask container** (`app.py`): Web UI + REST API on port 8000. Registers 17 blueprints (`app_*.py`).
- **Worker containers** (`rq_worker.py`, `rq_worker_high_priority.py`): RQ workers processing analysis/clustering jobs from Redis. Managed by supervisord.
- **PostgreSQL + Redis**: Data storage and job queue.

**Key architectural patterns:**
- Flask blueprints for feature modules (`app_clustering.py`, `app_alchemy.py`, `app_voyager.py`, etc.)
- Media server abstraction via dispatcher pattern in `tasks/mediaserver.py` → individual providers (`mediaserver_jellyfin.py`, `mediaserver_navidrome.py`, etc.)
- Multi-provider support with JSONB config storage, GUI setup wizard (`app_setup.py`), and cross-provider ID remapping
- Background tasks via Redis Queue (RQ) with two priority levels: `high` and `default`
- Voyager HNSW index (Spotify) for vector similarity search (`tasks/voyager_manager.py`)
- Redis Pub/Sub channel `index-updates` for signaling index reloads between worker and Flask processes
- All configuration via environment variables, centralized in `config.py`

**ML pipeline:**
- Audio analysis: Librosa feature extraction → MusiCNN ONNX models → 200-dim embeddings + 50 mood labels + 6 other features (danceable, aggressive, happy, party, relaxed, sad)
- CLAP: Split ONNX models (audio encoder + text encoder) for text-to-audio search
- MuLan: Alternative text-to-audio search using MuQ-MuLan ONNX models
- Clustering: KMeans/DBSCAN/GMM/Spectral with guided evolutionary optimization over N runs

**Instant Playlist (AI agentic loop):**
- 4 AI providers: Gemini, OpenAI, Mistral, Ollama (`ai_mcp_client.py`)
- 6 MCP tools: `song_similarity`, `text_search`, `artist_similarity`, `song_alchemy`, `ai_brainstorm`, `search_database` (`tasks/mcp_server.py`)
- Max 5 iterations, target 100 songs, proportional sampling from tool results
- System prompt built by `_build_system_prompt()`, library context from `get_library_context()`

**Database schema (main table `score`):**
- Columns: `item_id`, `title`, `author`, `album`, `album_artist`, `tempo`, `key`, `scale`, `mood_vector`, `other_features`, `energy`, `year`, `rating`, `file_path`, `track_id`
- `mood_vector` format: `"rock:0.82,pop:0.45,indie rock:0.31"` (genre:confidence pairs)
- `other_features` format: `"danceable,happy"` (threshold-filtered labels)
- Energy raw range: 0.01–0.15 (normalized to 0–1 for AI tools)

**Multi-provider tables:**
- `track`: Canonical track identity with `file_path_hash` (SHA-256) for cross-provider deduplication
- `provider_track`: Links provider-specific `item_id` to canonical `track_id`
- `provider`: JSONB config registry (type, credentials, path prefix, priority)

## Commands

### Running the application (Docker)
```bash
docker compose -f deployment/docker-compose-unified.yaml up -d
```

### Running tests
```bash
# All tests
pytest tests/

# Single test file
pytest tests/unit/test_mcp_server.py

# Single test function
pytest tests/unit/test_mcp_server.py::test_function_name

# With markers
pytest tests/ -m "not slow"
pytest tests/ -m "unit"
```

Tests use `importlib` bypass loading in `tests/conftest.py` to avoid the `tasks/__init__.py` → pydub → audioop import chain. Session-scoped fixtures provide pre-loaded modules (`mcp_server_mod`, `ai_mcp_client_mod`, `localfiles_mod`). An autouse `config_restore` fixture saves/restores mutated config attributes.

### Building Docker images
```bash
# CPU build
docker build -t audiomuse-ai .

# GPU build (NVIDIA)
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 -t audiomuse-ai-gpu .
```

### Package management
Uses `uv` (not pip) in Docker builds. Requirements split into `requirements/common.txt`, `requirements/cpu.txt`, `requirements/gpu.txt`.

## Key Conventions

- **Blueprint pattern**: Each feature is a Flask blueprint in `app_<feature>.py` with template in `templates/<feature>.html`. Register in `app.py`.
- **Media server providers**: Must implement the standard function interface (`get_all_songs`, `create_playlist`, `download_track`, etc.) and be registered in `tasks/mediaserver.py` dispatcher.
- **Config variables**: All from environment via `os.environ.get()` in `config.py`. No hardcoded values.
- **ID resolution**: `before_request` middleware in `app.py` resolves provider-specific `item_id` or `id` params to canonical `track_id` via `resolve_track_id()` (checks `provider_track` table). All APIs accept either format.
- **Genre regex matching**: Use `(^|,)\s*rock:` pattern to prevent substring false matches (e.g., "rock" vs "indie rock").
- **Energy normalization**: AI sees 0–1 range, raw values are 0.01–0.15 (`config.ENERGY_MIN/MAX`). Conversion happens in `execute_mcp_tool`.
- **ai_brainstorm matching**: Strict 2-stage (exact then normalized fuzzy) on BOTH title AND artist.
- **Gemini tool calling**: Uses `genai.types.Tool(function_declarations=...)` with ANY mode.
- **Ollama**: Prompt-based JSON output (no native tool calling).

## Pending Feature Migrations (from experimental repo)

These features exist in `AudioMuse-AI-experimental` (v0.7.12-beta) and are planned for migration:

### Playlist Builder (Phase 2)
Interactive playlist builder with weighted centroid calculation, smart filters, and include/exclude workflow.
- Source: `app_extend_playlist.py`, `templates/extend_playlist.html`

### Plex Integration (Phase 3)
Full Plex media server support with library browsing, track downloading, and playlist management.
- Source: `tasks/mediaserver_plex.py`

### Multi-Server Playlist Sync (Phase 4)
Synchronize playlists across multiple media servers using file path matching.
- Source: `app_playlist_sync.py`, `tasks/playlist_sync.py`, `tasks/playlist_manager.py`
