"""
Lyrics Search Manager
Provides in-memory caching and fast search for lyrics analysis results.

Mirrors the architecture of tasks/clap_text_search.py:
- Persists a voyager HNSW index over per-song lyrics embeddings
  (gte-multilingual-base, 768-dim) into the chunked ``lyrics_index_data`` table.
- Loads the index back at Flask startup and keeps it as a module-level
  singleton.
- Caches per-song axis_vector (BYTEA float32, fixed order over MUSIC_ANALYSIS_AXES)
  loaded as a separate voyager HNSW index for fast slider/radio search.
- Exposes two search entry points:
    * search_by_axes(targets, limit) for the basic axis-slider tab
    * search_by_text(query, limit) for the open free-form text tab
"""

import logging
import sys
from typing import Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# Global in-memory caches.
_LYRICS_INDEX_CACHE = {
    'index': None,            # voyager.Index
    'id_map': None,           # {voyager_int_id: item_id_str}
    'reverse_id_map': None,   # {item_id_str: voyager_int_id}
    'loaded': False,
}

_LYRICS_AXIS_CACHE = {
    'index': None,            # voyager.Index over the binary-friendly axis vectors
    'id_map': None,           # {voyager_int_id: item_id_str}
    'reverse_id_map': None,   # {item_id_str: voyager_int_id}
    'axis_columns': None,     # list[(axis_name, label)] aligned with the vector columns
    'metadata': None,         # {item_id: {title, author}}
    'loaded': False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_lyrics_metadata(item_ids: List[str]) -> Dict[str, Dict[str, str]]:
    from .commons import fetch_track_metadata_map
    return fetch_track_metadata_map(item_ids)


def _axis_columns_from_axes() -> List[tuple]:
    """Return a stable ordered list of (axis_name, label) covering every axis label."""
    from lyrics.lyrics_transcriber import axis_columns
    return list(axis_columns())


# ---------------------------------------------------------------------------
# Voyager index: build and persist
# ---------------------------------------------------------------------------

def build_and_store_lyrics_index(db_conn=None) -> bool:
    """Build a voyager index from stored lyrics embeddings and persist it."""
    from app_helper import get_db
    from config import LYRICS_ENABLED, LYRICS_EMBEDDING_DIMENSION, VOYAGER_METRIC
    from .index_build_helpers import build_and_store_index_streaming

    if not LYRICS_ENABLED:
        logger.info("Lyrics analysis is disabled; skipping lyrics index build.")
        return False

    if db_conn is None:
        db_conn = get_db()

    return build_and_store_index_streaming(
        db_conn,
        source_table="lyrics_embedding",
        source_column="embedding",
        dim=LYRICS_EMBEDDING_DIMENSION,
        target_table="lyrics_index_data",
        index_name="lyrics_index",
        metric=VOYAGER_METRIC,
        where_clause="embedding IS NOT NULL",
        label="lyrics",
    )


# ---------------------------------------------------------------------------
# Axes voyager index: build and persist (~27-dim Cosine)
# ---------------------------------------------------------------------------

def build_and_store_lyrics_axes_index(db_conn=None) -> bool:
    """Build a voyager index from the per-song axis_scores flattened to a fixed-order vector."""
    from app_helper import get_db
    from config import LYRICS_ENABLED
    from .index_build_helpers import build_and_store_index_streaming

    if not LYRICS_ENABLED:
        logger.info("Lyrics analysis is disabled; skipping lyrics axes index build.")
        return False

    if db_conn is None:
        db_conn = get_db()

    columns = _axis_columns_from_axes()
    if not columns:
        logger.warning("No axis columns defined; skipping lyrics axes index build.")
        return False
    dim = len(columns)

    return build_and_store_index_streaming(
        db_conn,
        source_table="lyrics_embedding",
        source_column="axis_vector",
        dim=dim,
        target_table="lyrics_axes_index_data",
        index_name="lyrics_axes_index",
        metric="angular",
        where_clause="axis_vector IS NOT NULL",
        label="lyrics axes",
    )


# ---------------------------------------------------------------------------
# Voyager index: load
# ---------------------------------------------------------------------------

def _load_lyrics_index_from_db() -> bool:
    """Load persisted voyager index for lyrics from the DB into the global cache."""
    from app_helper import get_db
    from config import LYRICS_EMBEDDING_DIMENSION, VOYAGER_QUERY_EF
    from .index_build_helpers import load_voyager_index_from_db

    try:
        loaded = load_voyager_index_from_db(
            get_db(), 'lyrics_index_data', 'lyrics_index',
            LYRICS_EMBEDDING_DIMENSION, VOYAGER_QUERY_EF, label='lyrics',
        )
        if loaded is None:
            return False
        loaded_index, id_map, reverse_id_map = loaded

        _LYRICS_INDEX_CACHE['index'] = loaded_index
        _LYRICS_INDEX_CACHE['id_map'] = id_map
        _LYRICS_INDEX_CACHE['reverse_id_map'] = reverse_id_map
        _LYRICS_INDEX_CACHE['loaded'] = True

        logger.info(f"Lyrics index loaded from database with {len(id_map)} items.")
        return True
    except Exception as e:
        logger.error(f"Failed to load lyrics index from DB: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Axes voyager index: load
# ---------------------------------------------------------------------------

def _load_lyrics_axes_index_from_db() -> bool:
    """Load persisted voyager index for the lyrics axis vectors."""
    from app_helper import get_db
    from config import VOYAGER_QUERY_EF
    from .index_build_helpers import load_voyager_index_from_db

    columns = _axis_columns_from_axes()
    expected_dim = len(columns)

    try:
        loaded = load_voyager_index_from_db(
            get_db(), 'lyrics_axes_index_data', 'lyrics_axes_index',
            expected_dim, VOYAGER_QUERY_EF, label='lyrics axes',
        )
        if loaded is None:
            return False
        loaded_index, id_map, reverse_id_map = loaded

        metadata_map = _fetch_lyrics_metadata(list(id_map.values()))

        _LYRICS_AXIS_CACHE['index'] = loaded_index
        _LYRICS_AXIS_CACHE['id_map'] = id_map
        _LYRICS_AXIS_CACHE['reverse_id_map'] = reverse_id_map
        _LYRICS_AXIS_CACHE['axis_columns'] = columns
        _LYRICS_AXIS_CACHE['metadata'] = metadata_map
        _LYRICS_AXIS_CACHE['loaded'] = True

        logger.info(f"Lyrics axes index loaded from database with {len(id_map)} items.")
        return True
    except Exception as e:
        logger.error(f"Failed to load lyrics axes index from DB: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public load / refresh
# ---------------------------------------------------------------------------

def load_lyrics_cache_from_db() -> bool:
    """Load both the embedding voyager index and the axes voyager index into memory."""
    from config import LYRICS_ENABLED

    if not LYRICS_ENABLED:
        logger.info("Lyrics is disabled; skipping lyrics cache load.")
        return False

    index_ok = _load_lyrics_index_from_db()
    axis_ok = _load_lyrics_axes_index_from_db()

    if not index_ok:
        _LYRICS_INDEX_CACHE['index'] = None
        _LYRICS_INDEX_CACHE['id_map'] = None
        _LYRICS_INDEX_CACHE['reverse_id_map'] = None
        _LYRICS_INDEX_CACHE['loaded'] = False

    if not axis_ok:
        _LYRICS_AXIS_CACHE['index'] = None
        _LYRICS_AXIS_CACHE['id_map'] = None
        _LYRICS_AXIS_CACHE['reverse_id_map'] = None
        _LYRICS_AXIS_CACHE['axis_columns'] = None
        _LYRICS_AXIS_CACHE['metadata'] = None
        _LYRICS_AXIS_CACHE['loaded'] = False

    return index_ok or axis_ok


def refresh_lyrics_cache() -> bool:
    old_index_count = (
        len(_LYRICS_INDEX_CACHE['id_map'])
        if _LYRICS_INDEX_CACHE['loaded'] and _LYRICS_INDEX_CACHE['id_map'] else 0
    )
    old_axis_count = (
        len(_LYRICS_AXIS_CACHE['id_map'])
        if _LYRICS_AXIS_CACHE['loaded'] and _LYRICS_AXIS_CACHE['id_map'] else 0
    )
    logger.info(f"Refreshing lyrics cache (index={old_index_count}, axes={old_axis_count})...")
    result = load_lyrics_cache_from_db()
    new_index_count = (
        len(_LYRICS_INDEX_CACHE['id_map'])
        if _LYRICS_INDEX_CACHE['loaded'] and _LYRICS_INDEX_CACHE['id_map'] else 0
    )
    new_axis_count = (
        len(_LYRICS_AXIS_CACHE['id_map'])
        if _LYRICS_AXIS_CACHE['loaded'] and _LYRICS_AXIS_CACHE['id_map'] else 0
    )
    logger.info(
        f"Lyrics cache refresh: index {old_index_count}->{new_index_count}, "
        f"axes {old_axis_count}->{new_axis_count}"
    )
    return result


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_cache_stats() -> Dict:
    index_loaded = _LYRICS_INDEX_CACHE['loaded'] and _LYRICS_INDEX_CACHE['index'] is not None
    axis_loaded = _LYRICS_AXIS_CACHE['loaded'] and _LYRICS_AXIS_CACHE['index'] is not None

    song_count = 0
    if index_loaded and _LYRICS_INDEX_CACHE['id_map']:
        song_count = len(_LYRICS_INDEX_CACHE['id_map'])
    elif axis_loaded and _LYRICS_AXIS_CACHE['id_map']:
        song_count = len(_LYRICS_AXIS_CACHE['id_map'])

    memory_bytes = 0
    if index_loaded:
        memory_bytes += sys.getsizeof(_LYRICS_INDEX_CACHE['index'])
        if _LYRICS_INDEX_CACHE['id_map']:
            memory_bytes += sys.getsizeof(_LYRICS_INDEX_CACHE['id_map'])
        if _LYRICS_INDEX_CACHE['reverse_id_map']:
            memory_bytes += sys.getsizeof(_LYRICS_INDEX_CACHE['reverse_id_map'])
    if axis_loaded:
        memory_bytes += sys.getsizeof(_LYRICS_AXIS_CACHE['index'])
        if _LYRICS_AXIS_CACHE['id_map']:
            memory_bytes += sys.getsizeof(_LYRICS_AXIS_CACHE['id_map'])

    return {
        'loaded': index_loaded or axis_loaded,
        'index_loaded': index_loaded,
        'axis_loaded': axis_loaded,
        'song_count': song_count,
        'embedding_dimension': config.LYRICS_EMBEDDING_DIMENSION,
        'memory_mb': round(memory_bytes / (1024 * 1024), 2),
    }


def get_axes_definition() -> Dict:
    """Return MUSIC_ANALYSIS_AXES as a JSON-friendly structure for the UI."""
    from lyrics.lyrics_transcriber import MUSIC_ANALYSIS_AXES
    return {
        axis_name: {
            'description': meta.get('description', ''),
            'labels': dict(meta.get('labels', {})),
        }
        for axis_name, meta in MUSIC_ANALYSIS_AXES.items()
    }


# ---------------------------------------------------------------------------
# Search: by axes (slider-based)
# ---------------------------------------------------------------------------

def search_by_axes(targets: Dict[str, str], limit: int = 50) -> List[Dict]:
    """
    Voyager nearest-neighbor search over the binary axis vector.

    targets: {axis_name: label_str} — at most ONE label per axis. Selected → 1.0,
             everything else → 0.0. Axes the user did not pick contribute 0 across
             all their labels.
    """
    from config import LYRICS_ENABLED, MAX_SONGS_PER_ARTIST

    if not LYRICS_ENABLED:
        return []
    if not _LYRICS_AXIS_CACHE['loaded'] or _LYRICS_AXIS_CACHE['index'] is None:
        logger.error("Lyrics axes voyager index not loaded.")
        return []

    columns = _LYRICS_AXIS_CACHE['axis_columns'] or []
    if not columns:
        return []
    col_index = {col: idx for idx, col in enumerate(columns)}
    dim = len(columns)

    query_vec = np.zeros(dim, dtype=np.float32)
    selected_pairs: List[tuple] = []
    for axis_name, label in (targets or {}).items():
        if not isinstance(label, str) or not label:
            continue
        j = col_index.get((axis_name, label))
        if j is None:
            continue
        query_vec[j] = 1.0
        selected_pairs.append((axis_name, label))

    if not selected_pairs:
        logger.warning("search_by_axes called with no usable selections.")
        return []

    voyager_index = _LYRICS_AXIS_CACHE['index']
    id_map = _LYRICS_AXIS_CACHE['id_map'] or {}
    metadata_map = _LYRICS_AXIS_CACHE['metadata'] or {}

    artist_cap = MAX_SONGS_PER_ARTIST if MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0 else 0
    fetch_size = (limit + max(20, limit * 4) + 1) if artist_cap else limit
    num_to_query = min(fetch_size, len(voyager_index))
    if num_to_query <= 0:
        return []

    try:
        neighbor_ids, distances = voyager_index.query(query_vec, k=num_to_query)
    except Exception as e:
        logger.error(f"Lyrics axes voyager query failed: {e}", exc_info=True)
        return []

    results: List[Dict] = []
    artist_counts: Dict[str, int] = {}
    for vid, dist in zip(neighbor_ids, distances):
        if len(results) >= limit:
            break
        item_id = id_map.get(int(vid))
        if not item_id:
            continue
        meta = metadata_map.get(item_id, {'title': '', 'author': '', 'album': ''})
        author = meta.get('author', '') or ''
        if artist_cap and author:
            an = author.strip().lower()
            if artist_counts.get(an, 0) >= artist_cap:
                continue
            artist_counts[an] = artist_counts.get(an, 0) + 1
        similarity = 1.0 - float(dist)
        results.append({
            'item_id': item_id,
            'title': meta.get('title', ''),
            'author': author,
            'album': meta.get('album', ''),
            'similarity': similarity,
        })

    logger.info(
        f"Lyrics axis search ({len(selected_pairs)} selections): {len(results)} results "
        f"(artist cap: {artist_cap or 'disabled'})"
    )
    return results


# ---------------------------------------------------------------------------
# Search: by free text
# ---------------------------------------------------------------------------

def search_by_text(query_text: str, limit: int = 50, artist_cap: Optional[int] = None) -> List[Dict]:
    """Search lyrics by embedding the query with gte-multilingual-base and querying the voyager index.

    ``artist_cap`` controls the per-artist diversity cap: ``None`` uses the global
    ``MAX_SONGS_PER_ARTIST`` (the default, for direct user-facing search); ``0``
    disables it entirely, returning the full similarity-ranked pool up to ``limit``
    (used when feeding a candidate pool that is diversity-capped downstream).
    """
    from config import LYRICS_ENABLED, MAX_SONGS_PER_ARTIST
    from lyrics.lyrics_transcriber import embed_query_text
    from tasks.gte_warm_cache import warm_lock, warmup_gte_model

    if not LYRICS_ENABLED:
        return []
    if not _LYRICS_INDEX_CACHE['loaded'] or _LYRICS_INDEX_CACHE['index'] is None:
        logger.error("Lyrics voyager index not loaded.")
        return []

    text = (query_text or '').strip()
    if not text:
        return []

    try:
        with warm_lock():
            warmup_gte_model()
            query_vec = embed_query_text(text)
        if query_vec is None or query_vec.size == 0:
            logger.error(f"Failed to embed lyrics query: {query_text!r}")
            return []

        if artist_cap is None:
            artist_cap = MAX_SONGS_PER_ARTIST
        artist_cap = artist_cap if artist_cap and artist_cap > 0 else 0
        fetch_size = (limit + max(20, limit * 4) + 1) if artist_cap else limit

        voyager_index = _LYRICS_INDEX_CACHE['index']
        id_map = _LYRICS_INDEX_CACHE['id_map'] or {}
        num_to_query = min(fetch_size, len(voyager_index))
        if num_to_query <= 0:
            return []

        neighbor_ids, distances = voyager_index.query(query_vec, k=num_to_query)
        candidate_item_ids = [id_map.get(int(v)) for v in neighbor_ids]
        candidate_item_ids = [iid for iid in candidate_item_ids if iid]
        metadata_map = _fetch_lyrics_metadata(candidate_item_ids)

        results: List[Dict] = []
        artist_counts: Dict[str, int] = {}
        for vid, dist in zip(neighbor_ids, distances):
            if len(results) >= limit:
                break
            item_id = id_map.get(int(vid))
            if not item_id:
                continue
            meta = metadata_map.get(item_id, {'title': '', 'author': '', 'album': ''})
            author = meta.get('author', '') or ''
            if artist_cap and author:
                an = author.strip().lower()
                if artist_counts.get(an, 0) >= artist_cap:
                    continue
                artist_counts[an] = artist_counts.get(an, 0) + 1
            similarity = 1.0 - float(dist)
            results.append({
                'item_id': item_id,
                'title': meta.get('title', ''),
                'author': author,
                'album': meta.get('album', ''),
                'similarity': similarity,
            })

        logger.info(
            f"Lyrics text search '{query_text}': {len(results)} results "
            f"(artist cap: {artist_cap or 'disabled'})"
        )
        return results
    except Exception as e:
        logger.error(f"Lyrics text search failed for {query_text!r}: {e}", exc_info=True)
        return []
