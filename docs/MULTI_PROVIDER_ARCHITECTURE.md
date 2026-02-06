# Multi-Provider Architecture Design

## Overview

This document outlines the architecture for supporting multiple media providers simultaneously in AudioMuse-AI without requiring re-analysis of tracks. The design ensures:

1. **No re-analysis required** when adding new providers
2. **Seamless migration** for existing installations
3. **Future-proof** extensibility for new providers
4. **Minimal schema changes** to existing tables

## Key Design Decisions

### 1. Primary Key Strategy for Local File Provider

**Decision: Use normalized file path as the stable identifier**

Rationale:
- File paths are unique within a music library
- Content hashes would require reading entire files (slow for large libraries)
- File path changes are rare and can be handled via re-scan
- Consistent with MPD provider which already uses file paths

For the local file provider:
- `item_id` = SHA-256 hash of the normalized relative file path
- This creates a stable, predictable ID that won't change unless the file moves

### 2. Linking Tracks Across Providers

**Decision: Use file path as the universal linking key**

The key insight is that most providers ultimately point to the same physical files:
- Jellyfin, Navidrome, Lyrion, Emby all index local music directories
- Local file provider scans the same directories
- The file path (relative to the music library root) is the common denominator

### 3. Database Schema Design

#### New Tables

```sql
-- Provider configuration storage
CREATE TABLE provider (
    id SERIAL PRIMARY KEY,
    provider_type VARCHAR(50) NOT NULL,  -- jellyfin, navidrome, localfiles, etc.
    name VARCHAR(255) NOT NULL,           -- User-friendly name
    config JSONB NOT NULL,                -- Provider-specific configuration
    enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 0,           -- For ordering when same track in multiple providers
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider_type, name)
);

-- Track identity table - links analysis to file paths
CREATE TABLE track (
    id SERIAL PRIMARY KEY,
    file_path_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of normalized relative path
    file_path TEXT NOT NULL,                      -- Original file path for display
    file_size BIGINT,                             -- For change detection
    file_modified TIMESTAMP,                      -- For change detection
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Links provider-specific item_ids to tracks
CREATE TABLE provider_track (
    id SERIAL PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES provider(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL REFERENCES track(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,               -- Provider's native ID
    title TEXT,                          -- Title from this provider
    artist TEXT,                         -- Artist from this provider
    album TEXT,                          -- Album from this provider
    last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider_id, item_id),
    UNIQUE(provider_id, track_id)
);
CREATE INDEX idx_provider_track_item_id ON provider_track(item_id);
CREATE INDEX idx_provider_track_track_id ON provider_track(track_id);

-- Application settings stored in database (for GUI configuration)
CREATE TABLE app_settings (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    category VARCHAR(100),               -- For UI grouping
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Modified Tables

The `score` table remains largely unchanged, but we add a foreign key to `track`:

```sql
ALTER TABLE score ADD COLUMN track_id INTEGER REFERENCES track(id);
CREATE INDEX idx_score_track_id ON score(track_id);
```

**Critical**: `item_id` remains the PRIMARY KEY for backward compatibility. The new `track_id` column provides the link to file-based identity.

### 4. Migration Strategy

**Phase 1: Schema Extension (Non-breaking)**
1. Add new tables (`provider`, `track`, `provider_track`, `app_settings`)
2. Add `track_id` column to `score` table (nullable initially)
3. Existing code continues to work unchanged

**Phase 2: Data Migration**
1. Create default provider entry for current `MEDIASERVER_TYPE`
2. For each existing `score` record:
   - Look up the track in the current provider
   - Extract file path if available (via provider API)
   - Create `track` record
   - Create `provider_track` mapping
   - Update `score.track_id`

**Phase 3: Multi-Provider Activation**
1. Enable multi-provider mode via configuration
2. New providers can be added via GUI or API
3. When scanning new providers:
   - Match tracks by file path
   - Reuse existing analysis data
   - Create new `provider_track` mappings

### 5. API Changes

#### New Endpoints

```
POST /api/setup/provider           - Add/configure a provider
GET  /api/setup/providers          - List configured providers
PUT  /api/setup/provider/{id}      - Update provider config
DELETE /api/setup/provider/{id}    - Remove provider

GET  /api/setup/settings           - Get all settings
PUT  /api/setup/settings           - Update settings
GET  /api/setup/wizard/status      - Get setup wizard state

POST /api/provider/{id}/sync       - Sync tracks from provider
GET  /api/provider/{id}/status     - Get provider sync status
```

#### Modified Endpoints

Existing endpoints remain unchanged but internally:
- `/api/analyze` - Uses active providers (priority-ordered)
- `/api/similarity` - Returns tracks with provider context
- Playlist creation - Creates in preferred provider(s)

### 6. Provider Interface

All providers must implement:

```python
class MediaProvider:
    """Base class for media providers"""

    def get_provider_type(self) -> str:
        """Return provider type identifier (jellyfin, navidrome, etc.)"""

    def test_connection(self) -> Tuple[bool, str]:
        """Test if provider is reachable, return (success, message)"""

    def get_all_songs(self) -> List[Dict]:
        """Return all songs with metadata including file_path if available"""

    def get_tracks_from_album(self, album_id: str) -> List[Dict]:
        """Return tracks for an album"""

    def download_track(self, temp_dir: str, item: Dict) -> Optional[str]:
        """Download track to temp directory, return local path"""

    def create_playlist(self, name: str, item_ids: List[str]) -> Optional[str]:
        """Create playlist, return playlist ID"""

    def get_file_path(self, item: Dict) -> Optional[str]:
        """Extract file path from item metadata (for track linking)"""
```

### 7. Local File Provider Specifics

```python
# Configuration for local file provider
{
    "provider_type": "localfiles",
    "name": "Local Music Library",
    "config": {
        "music_directory": "/path/to/music",
        "supported_formats": ["mp3", "flac", "m4a", "ogg", "wav"],
        "scan_subdirectories": true,
        "use_embedded_metadata": true
    }
}
```

Features:
- Scans directories for audio files
- Extracts metadata from ID3 tags (MP3), Vorbis comments (FLAC/OGG), etc.
- Creates item_id from file path hash
- Supports playlist creation via M3U files

### 8. Docker Compose Simplification

**Before**: 16+ docker-compose files for various scenarios
**After**: 2 main files + optional components

```
deployment/
├── docker-compose.yaml           # CPU version (default)
├── docker-compose-nvidia.yaml    # GPU/NVIDIA version
└── docker-compose-extras.yaml    # Optional: pgAdmin, monitoring, etc.
```

All provider-specific configuration moves to:
1. `.env` file for initial setup
2. Database `app_settings` table for runtime configuration
3. GUI Setup Wizard for user-friendly configuration

### 9. Setup Wizard Flow

```
1. Welcome Screen
   - Detect if first run or existing installation
   - Show version and hardware detection (GPU available?)

2. Hardware Selection
   - CPU-only or NVIDIA GPU acceleration
   - Validate GPU drivers if selected

3. Provider Configuration
   - List of available providers with descriptions
   - Multi-select enabled providers
   - For each provider: configuration form
   - Connection test for each provider

4. Music Library Paths
   - For local file provider: select directories
   - For media servers: auto-detected from provider config

5. Advanced Settings (collapsible)
   - Database settings (show defaults, allow override)
   - Analysis settings (CLAP, MuLan options)
   - AI provider settings (optional)

6. Review & Apply
   - Summary of all settings
   - Apply configuration
   - Start initial sync (optional)

7. Complete
   - Link to main dashboard
   - Quick start guide
```

### 10. Backward Compatibility

The system maintains full backward compatibility:

1. **Environment variables** still work:
   - `MEDIASERVER_TYPE` creates a default provider on first run
   - All `JELLYFIN_*`, `NAVIDROME_*` etc. variables honored

2. **Existing data preserved**:
   - `score` table unchanged except optional `track_id` column
   - `embedding`, `clap_embedding` tables unchanged
   - All indexes and projections preserved

3. **Gradual migration**:
   - Single-provider mode works exactly as before
   - Multi-provider can be enabled via GUI/API
   - No forced migration path

### 11. File Path Normalization

To ensure consistent file path matching:

```python
def normalize_file_path(path: str, base_path: str = "") -> str:
    """
    Normalize a file path for cross-provider matching.

    - Convert to POSIX style (forward slashes)
    - Make relative to music library root
    - Lowercase (optional, for case-insensitive filesystems)
    - Remove leading/trailing whitespace
    """
    import os
    from pathlib import PurePosixPath

    # Convert to Path object
    p = Path(path)

    # Make relative if absolute and base_path provided
    if base_path and p.is_absolute():
        try:
            p = p.relative_to(base_path)
        except ValueError:
            pass  # Not relative to base, keep as-is

    # Convert to POSIX style
    normalized = PurePosixPath(p).as_posix()

    return normalized.strip()


def file_path_hash(normalized_path: str) -> str:
    """Generate SHA-256 hash of normalized file path."""
    import hashlib
    return hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()
```

## Implementation Order

1. Database schema changes (migration-safe)
2. Local file provider implementation
3. Provider configuration storage
4. Multi-provider dispatcher updates
5. Setup wizard backend
6. Setup wizard frontend
7. Docker Compose simplification
8. Documentation updates
