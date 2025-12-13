# CLAP Text Search Integration

## Overview

CLAP (Contrastive Language-Audio Pretraining) text search has been added to AudioMuse-AI, enabling natural language music search alongside the existing MusiCNN-based features.

## What's New

### Natural Language Music Search
Search your music library using descriptive text queries like:
- "upbeat summer songs"
- "relaxing piano music"
- "energetic rock guitar"
- "melancholic acoustic"
- "electronic dance music"

### Dual Embedding System
- **MusiCNN (200-dim)**: Used for all existing features (similarity, clustering, alchemy, etc.)
- **CLAP (512-dim)**: Used exclusively for text-based search

## Architecture

### Database Schema
```sql
-- New table for CLAP embeddings
CREATE TABLE clap_embedding (
    item_id TEXT PRIMARY KEY,
    embedding BYTEA NOT NULL,
    FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE
);
```

### Components Added

1. **tasks/clap_analyzer.py**
   - LAION-CLAP model loader
   - Audio file analysis (512-dim embeddings)
   - Text query encoding

2. **tasks/clap_text_search.py**
   - In-memory cache for fast search (similar to external reference project)
   - Vectorized similarity computation
   - Cache management

3. **app_clap_search.py**
   - Flask blueprint with routes:
     - `/clap_search` - Web UI
     - `/api/clap/search` - Search API
     - `/api/clap/cache/refresh` - Reload cache
     - `/api/clap/stats` - Cache statistics

4. **templates/clap_search.html**
   - User-friendly search interface
   - Example queries
   - Real-time results with similarity scores

## Configuration

### Environment Variables

```bash
# Enable/disable CLAP (default: true)
CLAP_ENABLED=true

# Model path (must be downloaded separately)
CLAP_MODEL_PATH=/app/model/music_audioset_epoch_15_esc_90.14.pt
```

### Model Download

The CLAP model is not included in the repository. Download it:

```bash
# Create model directory
mkdir -p /app/model

# Download LAION-CLAP checkpoint
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt \
     -O /app/model/music_audioset_epoch_15_esc_90.14.pt
```

## Usage

### 1. Run Analysis
CLAP embeddings are automatically generated during song analysis if `CLAP_ENABLED=true`:

```python
# Analysis runs both MusiCNN AND CLAP
# - MusiCNN → embedding table
# - CLAP → clap_embedding table
```

### 2. Access Text Search
Navigate to `/clap_search` in the web UI or use the API:

```bash
# Search via API
curl -X POST http://localhost:5000/api/clap/search \
  -H "Content-Type: application/json" \
  -d '{"query": "upbeat summer songs", "limit": 50}'
```

### 3. Cache Management
The CLAP cache loads automatically at startup. To refresh:

```bash
curl -X POST http://localhost:5000/api/clap/cache/refresh
```

## Performance

### Memory Usage
- **Cache Size**: ~2MB per 1000 songs (512-dim float32 embeddings)
- **Example**: 10,000 songs ≈ 20MB RAM

### Search Speed
- **First query**: ~0.1s (model initialization)
- **Subsequent queries**: <0.01s (vectorized NumPy operations)

## Integration with Existing Features

### ✅ No Impact On
- Voyager index (still uses MusiCNN)
- Song similarity (still uses MusiCNN)
- Clustering (still uses MusiCNN)
- Song Alchemy (still uses MusiCNN)
- Artist similarity (still uses MusiCNN GMM)
- Path finding (still uses MusiCNN)

### ⚡ New Capability
- Text-based search (CLAP only)

## API Reference

### POST /api/clap/search
Search songs by text query.

**Request:**
```json
{
  "query": "upbeat summer songs",
  "limit": 100
}
```

**Response:**
```json
{
  "query": "upbeat summer songs",
  "count": 100,
  "results": [
    {
      "item_id": "12345",
      "title": "Summer Vibes",
      "author": "Artist Name",
      "similarity": 0.87
    }
  ]
}
```

### POST /api/clap/cache/refresh
Reload CLAP cache from database.

**Response:**
```json
{
  "success": true,
  "message": "CLAP cache refreshed successfully",
  "stats": {
    "loaded": true,
    "song_count": 10000,
    "embedding_dimension": 512,
    "memory_mb": 19.53
  }
}
```

### GET /api/clap/stats
Get cache statistics.

**Response:**
```json
{
  "clap_enabled": true,
  "loaded": true,
  "song_count": 10000,
  "embedding_dimension": 512,
  "memory_mb": 19.53
}
```

## Troubleshooting

### CLAP Not Working

1. **Check if enabled:**
   ```bash
   echo $CLAP_ENABLED  # Should be "true"
   ```

2. **Verify model exists:**
   ```bash
   ls -lh $CLAP_MODEL_PATH
   ```

3. **Check logs:**
   ```bash
   # Look for CLAP initialization messages
   grep -i "clap" /var/log/audiomuse.log
   ```

### No Search Results

1. **Ensure songs are analyzed:**
   - CLAP embeddings are only created during analysis
   - Re-run analysis if you enabled CLAP after initial analysis

2. **Refresh cache:**
   ```bash
   curl -X POST http://localhost:5000/api/clap/cache/refresh
   ```

### High Memory Usage

CLAP model (~500MB) + embeddings (~2MB/1k songs). For large libraries:
- Consider disabling CLAP if memory is constrained
- Use GPU if available (will be detected automatically by PyTorch)

## Technical Details

### Why Separate Tables?
- **Different dimensions**: MusiCNN (200) vs CLAP (512)
- **Different use cases**: Audio similarity vs text search
- **Optional feature**: Can disable CLAP without affecting core functionality

### Why In-Memory Cache?
Following the external reference project pattern:
- Load all CLAP embeddings into NumPy array at startup
- Vectorized similarity: `similarities = embeddings @ text_embedding`
- O(1) search time regardless of library size
- Tradeoff: RAM for speed (acceptable for <100k songs)

### Model Architecture
- **CLAP**: LAION's Contrastive Language-Audio Pretraining
- **Audio Encoder**: HTSAT-base (High-resolution Transformer)
- **Text Encoder**: RoBERTa
- **Embedding**: 512-dim normalized vectors
- **Similarity**: Cosine similarity via dot product

## Future Enhancements

Potential improvements:
- [ ] Hybrid search (combine MusiCNN + CLAP scores)
- [ ] CLAP-based clustering
- [ ] Multimodal Voyager index (both embeddings)
- [ ] Fine-tuned CLAP on user's library
- [ ] GPU acceleration for batch analysis

## Credits

- **CLAP Model**: [LAION](https://github.com/LAION-AI/CLAP)
- **Reference Implementation**: [hw25-song-search](https://github.com/gcolangiulisuse/hw25-song-search)
- **Integration**: AudioMuse-AI team

## License

Same as AudioMuse-AI main project.
