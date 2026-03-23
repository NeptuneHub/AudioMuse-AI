#!/usr/bin/env python3
"""
migration_track_id.py — One-time migration from v0.9.x single-provider schema
(score.item_id TEXT PK) to the new track_id canonical architecture
(score.track_id INTEGER PK → track.id).

This script:
  1. Creates infrastructure tables (track, provider, provider_track, artist_provider_mapping)
  2. Creates a provider entry from environment config
  3. Creates track entries for every score row
  4. Creates provider_track entries linking item_ids → track_ids
  5. Rebuilds score table with track_id as PK
  6. Rebuilds embedding, clap_embedding, mulan_embedding tables with track_id PK
  7. Migrates artist_mapping → artist_provider_mapping
  8. Rebuilds playlist table with track_id FK
  9. Performs atomic table swap
 10. Verifies row counts and FK integrity
 11. Drops old tables and records completion in app_settings

Usage:
    python migration_track_id.py

Environment variables:
    DATABASE_URL or POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD
    MEDIASERVER_TYPE (auto-detected if not set)
    Provider-specific vars (JELLYFIN_URL, NAVIDROME_URL, etc.)
"""

import hashlib
import json
import logging
import os
import sys
import time
from urllib.parse import unquote, quote

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("migration_track_id")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 1000
MIGRATION_KEY = "migration_track_id_canonical_done"

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_database_url():
    """Build DATABASE_URL from environment, matching config.py logic."""
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit

    user = os.environ.get("POSTGRES_USER", "audiomuse")
    password = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
    host = os.environ.get("POSTGRES_HOST", "postgres-service.playlist")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "audiomusedb")

    user_esc = quote(user, safe="")
    pass_esc = quote(password, safe="")
    return f"postgresql://{user_esc}:{pass_esc}@{host}:{port}/{db}"


def get_connection():
    import psycopg2
    url = get_database_url()
    logger.info("Connecting to database...")
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------------------
# Path normalization (deterministic, NO provider prefix lookup)
# Must match app_helper.normalize_provider_path(file_path, provider_id=None)
# ---------------------------------------------------------------------------

def normalize_path_deterministic(file_path):
    """
    Normalize file path without consulting provider config.
    This is equivalent to normalize_provider_path(file_path, provider_id=None).
    """
    if not file_path:
        return None

    normalized = file_path

    # Handle file:// URLs (Lyrion/LMS style)
    if normalized.startswith('file://'):
        normalized = normalized[7:]
        normalized = unquote(normalized)

    # Convert Windows backslashes to forward slashes
    normalized = normalized.replace('\\', '/')

    # Common mount point prefixes to strip (order matters — longer first)
    prefixes_to_strip = [
        '/media/music/',
        '/media/Media/',
        '/media/',
        '/mnt/media/music/',
        '/mnt/media/',
        '/mnt/music/',
        '/mnt/data/music/',
        '/mnt/data/',
        '/mnt/',
        '/data/music/',
        '/data/',
        '/music/',
        '/share/music/',
        '/share/',
        '/volume1/music/',
        '/volume1/',
        '/srv/music/',
        '/srv/',
        '/home/music/',
        '/storage/music/',
        '/opt/music/',
        '/nas/music/',
        '/library/music/',
    ]

    lower_normalized = normalized.lower()
    for prefix in prefixes_to_strip:
        if lower_normalized.startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
            break

    # Remove leading slashes
    normalized = normalized.lstrip('/')

    # Handle Windows absolute paths (C:\music\... → strip up to known marker)
    if len(normalized) > 1 and normalized[1] == ':':
        for marker in ['/music/', '/Music/', '/media/', '/Media/']:
            idx = normalized.find(marker)
            if idx != -1:
                normalized = normalized[idx + len(marker):]
                break

    result = normalized.lstrip('/') if normalized else None
    return result.lower() if result else None


def compute_hash(normalized_path):
    """SHA-256 of normalized path string."""
    if not normalized_path:
        return None
    return hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()


def compute_synthetic_path(title, author, album):
    """
    Generate a synthetic 'file path' for rows without a real file_path.
    Format: __synthetic__/<title>|<author>|<album>
    """
    parts = [title or "", author or "", album or ""]
    synthetic = "__synthetic__/" + "|".join(parts)
    return synthetic


# ---------------------------------------------------------------------------
# Provider detection (mirrors config.py logic)
# ---------------------------------------------------------------------------

def detect_mediaserver_type():
    explicit = os.environ.get("MEDIASERVER_TYPE")
    if explicit:
        return explicit.lower()
    if os.environ.get("JELLYFIN_URL", ""):
        return "jellyfin"
    if os.environ.get("EMBY_URL", ""):
        return "emby"
    if os.environ.get("NAVIDROME_URL", ""):
        return "navidrome"
    if os.environ.get("LYRION_URL", ""):
        return "lyrion"
    return "localfiles"


def build_provider_config(provider_type):
    """Build config JSONB from environment variables for the given provider type."""
    if provider_type == "jellyfin":
        return {
            "url": os.environ.get("JELLYFIN_URL", ""),
            "user_id": os.environ.get("JELLYFIN_USER_ID", ""),
            "token": os.environ.get("JELLYFIN_TOKEN", ""),
        }
    elif provider_type == "emby":
        return {
            "url": os.environ.get("EMBY_URL", ""),
            "user_id": os.environ.get("EMBY_USER_ID", ""),
            "token": os.environ.get("EMBY_TOKEN", ""),
        }
    elif provider_type == "navidrome":
        return {
            "url": os.environ.get("NAVIDROME_URL", ""),
            "user": os.environ.get("NAVIDROME_USER", ""),
            "password": os.environ.get("NAVIDROME_PASSWORD", ""),
        }
    elif provider_type == "lyrion":
        return {
            "url": os.environ.get("LYRION_URL", ""),
        }
    elif provider_type == "localfiles":
        return {
            "music_directory": os.environ.get("LOCALFILES_MUSIC_DIRECTORY", "/music"),
        }
    else:
        return {}


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------

def is_migration_done(cur):
    """Check if this migration has already been completed."""
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'app_settings')"
    )
    if not cur.fetchone()[0]:
        return False
    cur.execute("SELECT value FROM app_settings WHERE key = %s", (MIGRATION_KEY,))
    row = cur.fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def table_exists(cur, table_name):
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        (table_name,),
    )
    return cur.fetchone()[0]


def column_exists(cur, table_name, column_name):
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s)",
        (table_name, column_name),
    )
    return cur.fetchone()[0]


def count_rows(cur, table_name):
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608 — table names are trusted
    return cur.fetchone()[0]


# ===========================================================================
# Migration steps
# ===========================================================================

def step1_create_infrastructure(cur):
    """Step 1: Create track, provider, provider_track, artist_provider_mapping tables."""
    logger.info("=" * 70)
    logger.info("STEP 1: Create infrastructure tables")
    logger.info("=" * 70)

    # --- track ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS track (
            id SERIAL PRIMARY KEY,
            file_path_hash VARCHAR(64) NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            normalized_path TEXT,
            norm_version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_track_file_path_hash ON track(file_path_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_track_normalized_path ON track(normalized_path)")
    logger.info("  Created table: track")

    # Add norm_version column if it doesn't exist (table may already exist without it)
    if not column_exists(cur, 'track', 'norm_version'):
        cur.execute("ALTER TABLE track ADD COLUMN norm_version INTEGER DEFAULT 1")
        logger.info("  Added norm_version column to existing track table")

    # --- provider ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS provider (
            id SERIAL PRIMARY KEY,
            provider_type VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            config JSONB NOT NULL DEFAULT '{}',
            enabled BOOLEAN DEFAULT TRUE,
            priority INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider_type, name)
        )
    """)
    logger.info("  Created table: provider")

    # --- provider_track ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS provider_track (
            id SERIAL PRIMARY KEY,
            provider_id INTEGER NOT NULL REFERENCES provider(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL REFERENCES track(id) ON DELETE CASCADE,
            item_id TEXT NOT NULL,
            title TEXT,
            artist TEXT,
            album TEXT,
            last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider_id, item_id),
            UNIQUE(provider_id, track_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_provider_track_item_id ON provider_track(item_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_provider_track_track_id ON provider_track(track_id)")
    logger.info("  Created table: provider_track")

    # --- artist_provider_mapping ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS artist_provider_mapping (
            id SERIAL PRIMARY KEY,
            artist_name TEXT NOT NULL,
            provider_id INTEGER NOT NULL REFERENCES provider(id) ON DELETE CASCADE,
            provider_artist_id TEXT NOT NULL,
            is_primary BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider_id, provider_artist_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_apm_artist_name ON artist_provider_mapping(LOWER(artist_name))")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_apm_provider_artist_id ON artist_provider_mapping(provider_artist_id)")
    logger.info("  Created table: artist_provider_mapping")

    # --- app_settings (ensure it exists for migration tracking) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key VARCHAR(255) PRIMARY KEY,
            value JSONB NOT NULL,
            category VARCHAR(100),
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("  Ensured table: app_settings")


def step2_create_provider(cur):
    """Step 2: Insert provider entry from environment config."""
    logger.info("=" * 70)
    logger.info("STEP 2: Create provider entry")
    logger.info("=" * 70)

    provider_type = detect_mediaserver_type()
    provider_config = build_provider_config(provider_type)
    provider_name = provider_type  # Use type as name for the legacy/primary provider

    logger.info(f"  Provider type: {provider_type}")
    logger.info(f"  Provider name: {provider_name}")

    cur.execute("""
        INSERT INTO provider (provider_type, name, config, enabled, priority)
        VALUES (%s, %s, %s, TRUE, 100)
        ON CONFLICT (provider_type, name) DO UPDATE SET
            config = EXCLUDED.config,
            priority = GREATEST(provider.priority, EXCLUDED.priority),
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (provider_type, provider_name, json.dumps(provider_config)))

    provider_id = cur.fetchone()[0]
    logger.info(f"  Provider ID: {provider_id}")
    return provider_id


def step3_create_track_entries(cur):
    """
    Step 3: Create track entries for every score row.

    Returns a dict mapping item_id → (track_id, file_path_hash).
    """
    logger.info("=" * 70)
    logger.info("STEP 3: Create track entries for every score row")
    logger.info("=" * 70)

    # Fetch all score rows
    cur.execute("SELECT item_id, title, author, album, file_path FROM score ORDER BY item_id")
    all_rows = cur.fetchall()
    total = len(all_rows)
    logger.info(f"  Total score rows to process: {total}")

    # Build mapping: item_id → (normalized_path, file_path_hash, raw_file_path)
    item_hash_map = {}  # item_id → (file_path_hash, normalized_path, file_path_for_track)
    hash_to_track_data = {}  # file_path_hash → (file_path, normalized_path) — first seen wins

    for item_id, title, author, album, file_path in all_rows:
        if file_path:
            normalized = normalize_path_deterministic(file_path)
            fp_for_track = file_path
        else:
            # Generate synthetic path for rows without file_path
            synthetic = compute_synthetic_path(title, author, album)
            normalized = normalize_path_deterministic(synthetic)
            fp_for_track = synthetic

        if not normalized:
            # Last resort: use item_id itself
            normalized = f"__item_id__/{item_id}".lower()
            fp_for_track = f"__item_id__/{item_id}"

        fph = compute_hash(normalized)
        item_hash_map[item_id] = (fph, normalized, fp_for_track)

        if fph not in hash_to_track_data:
            hash_to_track_data[fph] = (fp_for_track, normalized)

    logger.info(f"  Unique file path hashes: {len(hash_to_track_data)}")

    # Batch insert into track table
    inserted = 0
    skipped = 0
    hash_list = list(hash_to_track_data.items())

    for batch_start in range(0, len(hash_list), BATCH_SIZE):
        batch = hash_list[batch_start:batch_start + BATCH_SIZE]
        for fph, (fp, norm) in batch:
            cur.execute("""
                INSERT INTO track (file_path_hash, file_path, normalized_path, norm_version)
                VALUES (%s, %s, %s, 1)
                ON CONFLICT (file_path_hash) DO NOTHING
            """, (fph, fp, norm))
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

        if (batch_start + BATCH_SIZE) % 5000 == 0 or batch_start + BATCH_SIZE >= len(hash_list):
            logger.info(f"  Track insert progress: {min(batch_start + BATCH_SIZE, len(hash_list))}/{len(hash_list)}")

    logger.info(f"  Track entries inserted: {inserted}, already existed: {skipped}")

    # Build item_id → track_id mapping by looking up from track table
    item_to_track = {}  # item_id → track_id

    # Batch fetch track_ids by file_path_hash
    unique_hashes = list(set(fph for fph, _, _ in item_hash_map.values()))
    hash_to_track_id = {}

    for batch_start in range(0, len(unique_hashes), BATCH_SIZE):
        batch = unique_hashes[batch_start:batch_start + BATCH_SIZE]
        cur.execute(
            "SELECT file_path_hash, id FROM track WHERE file_path_hash = ANY(%s)",
            (batch,)
        )
        for row in cur.fetchall():
            hash_to_track_id[row[0]] = row[1]

    # Map item_id → track_id
    unmapped = 0
    for item_id, (fph, _, _) in item_hash_map.items():
        track_id = hash_to_track_id.get(fph)
        if track_id:
            item_to_track[item_id] = track_id
        else:
            unmapped += 1
            logger.warning(f"  No track_id found for item_id={item_id}, hash={fph}")

    logger.info(f"  Mapped {len(item_to_track)} item_ids to track_ids ({unmapped} unmapped)")

    return item_to_track, item_hash_map


def step4_create_provider_track(cur, provider_id, item_to_track):
    """Step 4: Create provider_track entries linking item_ids to tracks."""
    logger.info("=" * 70)
    logger.info("STEP 4: Create provider_track entries")
    logger.info("=" * 70)

    # Fetch title/author/album for each score row
    cur.execute("SELECT item_id, title, author, album FROM score")
    score_meta = {row[0]: (row[1], row[2], row[3]) for row in cur.fetchall()}

    inserted = 0
    items = list(item_to_track.items())

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        for item_id, track_id in batch:
            title, artist, album = score_meta.get(item_id, (None, None, None))
            cur.execute("""
                INSERT INTO provider_track (provider_id, track_id, item_id, title, artist, album, last_synced)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (provider_id, item_id) DO NOTHING
            """, (provider_id, track_id, item_id, title, artist, album))
            if cur.rowcount > 0:
                inserted += 1

        if (batch_start + BATCH_SIZE) % 5000 == 0 or batch_start + BATCH_SIZE >= len(items):
            logger.info(f"  provider_track insert progress: {min(batch_start + BATCH_SIZE, len(items))}/{len(items)}")

    logger.info(f"  provider_track entries created: {inserted}")


def step5_create_new_score(cur, item_to_track):
    """Step 5: Create score_new table with track_id PK and populate it."""
    logger.info("=" * 70)
    logger.info("STEP 5: Create new score table (track_id PK)")
    logger.info("=" * 70)

    # Drop score_new if it exists from a previous failed attempt
    cur.execute("DROP TABLE IF EXISTS score_new CASCADE")

    cur.execute("""
        CREATE TABLE score_new (
            track_id INTEGER PRIMARY KEY REFERENCES track(id) ON DELETE CASCADE,
            title TEXT,
            author TEXT,
            album TEXT,
            album_artist TEXT,
            tempo REAL,
            key TEXT,
            scale TEXT,
            mood_vector TEXT,
            energy REAL,
            other_features TEXT,
            year INTEGER,
            rating INTEGER,
            file_path TEXT
        )
    """)

    # Add search_u generated column
    cur.execute("""
        CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text
        LANGUAGE sql IMMUTABLE AS $$ SELECT public.unaccent($1) $$
    """)
    cur.execute("""
        ALTER TABLE score_new ADD COLUMN search_u TEXT GENERATED ALWAYS AS (
            lower(immutable_unaccent(COALESCE(title, '') || ' ' || COALESCE(author, '') || ' ' || COALESCE(album, '')))
        ) STORED
    """)

    logger.info("  Created score_new table with search_u generated column")

    # Populate score_new from score using the mapping
    # When multiple item_ids map to the same track_id (duplicates from file_path collisions),
    # we pick one arbitrarily — the first one encountered.
    seen_track_ids = set()
    inserted = 0
    skipped_dup = 0
    skipped_unmapped = 0

    cur.execute("""
        SELECT item_id, title, author, album, album_artist, tempo, key, scale,
               mood_vector, energy, other_features, year, rating, file_path
        FROM score ORDER BY item_id
    """)
    all_rows = cur.fetchall()

    batch_values = []
    for row in all_rows:
        item_id = row[0]
        track_id = item_to_track.get(item_id)
        if not track_id:
            skipped_unmapped += 1
            continue
        if track_id in seen_track_ids:
            skipped_dup += 1
            continue
        seen_track_ids.add(track_id)

        # (track_id, title, author, album, album_artist, tempo, key, scale,
        #  mood_vector, energy, other_features, year, rating, file_path)
        batch_values.append((
            track_id, row[1], row[2], row[3], row[4], row[5], row[6], row[7],
            row[8], row[9], row[10], row[11], row[12], row[13]
        ))

        if len(batch_values) >= BATCH_SIZE:
            _insert_score_batch(cur, batch_values)
            inserted += len(batch_values)
            batch_values = []

    if batch_values:
        _insert_score_batch(cur, batch_values)
        inserted += len(batch_values)

    logger.info(f"  score_new rows inserted: {inserted}")
    logger.info(f"  Skipped (duplicate track_id): {skipped_dup}")
    logger.info(f"  Skipped (unmapped item_id): {skipped_unmapped}")

    # Create indexes on score_new
    cur.execute("CREATE INDEX IF NOT EXISTS idx_score_new_file_path ON score_new(file_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS score_new_search_u_trgm ON score_new USING gin (search_u gin_trgm_ops)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_score_new_author_lower ON score_new(LOWER(author))")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_score_new_title_author_lower ON score_new(LOWER(title), LOWER(author))")
    logger.info("  Created indexes on score_new")

    return inserted


def _insert_score_batch(cur, batch_values):
    """Helper: batch INSERT into score_new."""
    from psycopg2.extras import execute_values
    execute_values(
        cur,
        """INSERT INTO score_new
           (track_id, title, author, album, album_artist, tempo, key, scale,
            mood_vector, energy, other_features, year, rating, file_path)
           VALUES %s
           ON CONFLICT (track_id) DO NOTHING""",
        batch_values,
        page_size=BATCH_SIZE,
    )


def step6_create_new_embeddings(cur, item_to_track):
    """Step 6: Create new embedding tables with track_id PK."""
    logger.info("=" * 70)
    logger.info("STEP 6: Create new embedding tables (track_id PK)")
    logger.info("=" * 70)

    # --- embedding_new ---
    cur.execute("DROP TABLE IF EXISTS embedding_new CASCADE")
    cur.execute("""
        CREATE TABLE embedding_new (
            track_id INTEGER PRIMARY KEY REFERENCES score_new(track_id) ON DELETE CASCADE,
            embedding BYTEA
        )
    """)

    # Get the set of track_ids that exist in score_new (for FK validity)
    cur.execute("SELECT track_id FROM score_new")
    valid_track_ids = set(row[0] for row in cur.fetchall())

    # Populate embedding_new
    cur.execute("SELECT item_id, embedding FROM embedding")
    emb_rows = cur.fetchall()
    inserted = 0
    skipped = 0
    seen = set()

    batch_values = []
    for item_id, emb_data in emb_rows:
        track_id = item_to_track.get(item_id)
        if not track_id or track_id not in valid_track_ids or track_id in seen:
            skipped += 1
            continue
        seen.add(track_id)
        batch_values.append((track_id, emb_data))

        if len(batch_values) >= BATCH_SIZE:
            _insert_embedding_batch(cur, "embedding_new", batch_values)
            inserted += len(batch_values)
            batch_values = []

    if batch_values:
        _insert_embedding_batch(cur, "embedding_new", batch_values)
        inserted += len(batch_values)

    logger.info(f"  embedding_new rows: {inserted} (skipped: {skipped})")

    # --- clap_embedding_new ---
    cur.execute("DROP TABLE IF EXISTS clap_embedding_new CASCADE")
    cur.execute("""
        CREATE TABLE clap_embedding_new (
            track_id INTEGER PRIMARY KEY REFERENCES score_new(track_id) ON DELETE CASCADE,
            embedding BYTEA
        )
    """)

    cur.execute("SELECT item_id, embedding FROM clap_embedding")
    clap_rows = cur.fetchall()
    inserted = 0
    skipped = 0
    seen = set()

    batch_values = []
    for item_id, emb_data in clap_rows:
        track_id = item_to_track.get(item_id)
        if not track_id or track_id not in valid_track_ids or track_id in seen:
            skipped += 1
            continue
        seen.add(track_id)
        batch_values.append((track_id, emb_data))

        if len(batch_values) >= BATCH_SIZE:
            _insert_embedding_batch(cur, "clap_embedding_new", batch_values)
            inserted += len(batch_values)
            batch_values = []

    if batch_values:
        _insert_embedding_batch(cur, "clap_embedding_new", batch_values)
        inserted += len(batch_values)

    logger.info(f"  clap_embedding_new rows: {inserted} (skipped: {skipped})")

    # --- mulan_embedding_new (conditional) ---
    has_mulan = table_exists(cur, "mulan_embedding")
    if has_mulan:
        cur.execute("DROP TABLE IF EXISTS mulan_embedding_new CASCADE")
        cur.execute("""
            CREATE TABLE mulan_embedding_new (
                track_id INTEGER PRIMARY KEY REFERENCES score_new(track_id) ON DELETE CASCADE,
                embedding BYTEA
            )
        """)

        cur.execute("SELECT item_id, embedding FROM mulan_embedding")
        mulan_rows = cur.fetchall()
        inserted = 0
        skipped = 0
        seen = set()

        batch_values = []
        for item_id, emb_data in mulan_rows:
            track_id = item_to_track.get(item_id)
            if not track_id or track_id not in valid_track_ids or track_id in seen:
                skipped += 1
                continue
            seen.add(track_id)
            batch_values.append((track_id, emb_data))

            if len(batch_values) >= BATCH_SIZE:
                _insert_embedding_batch(cur, "mulan_embedding_new", batch_values)
                inserted += len(batch_values)
                batch_values = []

        if batch_values:
            _insert_embedding_batch(cur, "mulan_embedding_new", batch_values)
            inserted += len(batch_values)

        logger.info(f"  mulan_embedding_new rows: {inserted} (skipped: {skipped})")
    else:
        logger.info("  mulan_embedding table does not exist — skipping")

    return has_mulan


def _insert_embedding_batch(cur, table_name, batch_values):
    """Helper: batch INSERT into an embedding_new-style table."""
    from psycopg2.extras import execute_values
    execute_values(
        cur,
        f"INSERT INTO {table_name} (track_id, embedding) VALUES %s ON CONFLICT (track_id) DO NOTHING",
        batch_values,
        page_size=BATCH_SIZE,
    )


def step7_migrate_artists(cur, provider_id):
    """Step 7: Migrate artist_mapping → artist_provider_mapping."""
    logger.info("=" * 70)
    logger.info("STEP 7: Migrate artist tables")
    logger.info("=" * 70)

    # Migrate from artist_mapping
    if table_exists(cur, "artist_mapping"):
        cur.execute("""
            SELECT artist_name, artist_id
            FROM artist_mapping
            WHERE artist_id IS NOT NULL AND artist_id != ''
        """)
        am_rows = cur.fetchall()
        inserted = 0
        for artist_name, artist_id in am_rows:
            cur.execute("""
                INSERT INTO artist_provider_mapping
                    (artist_name, provider_id, provider_artist_id, is_primary)
                VALUES (%s, %s, %s, TRUE)
                ON CONFLICT DO NOTHING
            """, (artist_name, provider_id, artist_id))
            if cur.rowcount > 0:
                inserted += 1
        logger.info(f"  Migrated {inserted} entries from artist_mapping")
    else:
        logger.info("  artist_mapping table does not exist — skipping")

    # Migrate from artist_id_lookup (if it exists)
    if table_exists(cur, "artist_id_lookup"):
        cur.execute("""
            SELECT artist_id, artist_name
            FROM artist_id_lookup
            WHERE artist_id IS NOT NULL AND artist_name IS NOT NULL
        """)
        ail_rows = cur.fetchall()
        inserted = 0
        for artist_id, artist_name in ail_rows:
            cur.execute("""
                INSERT INTO artist_provider_mapping
                    (artist_name, provider_id, provider_artist_id, is_primary)
                VALUES (%s, %s, %s, FALSE)
                ON CONFLICT DO NOTHING
            """, (artist_name, provider_id, artist_id))
            if cur.rowcount > 0:
                inserted += 1
        logger.info(f"  Migrated {inserted} entries from artist_id_lookup")
    else:
        logger.info("  artist_id_lookup table does not exist — skipping")


def step8_migrate_playlist(cur, item_to_track):
    """Step 8: Create playlist_new table with track_id FK."""
    logger.info("=" * 70)
    logger.info("STEP 8: Migrate playlist table")
    logger.info("=" * 70)

    cur.execute("DROP TABLE IF EXISTS playlist_new CASCADE")
    cur.execute("""
        CREATE TABLE playlist_new (
            id SERIAL PRIMARY KEY,
            playlist_name TEXT,
            track_id INTEGER REFERENCES track(id),
            title TEXT,
            author TEXT,
            UNIQUE(playlist_name, track_id)
        )
    """)

    cur.execute("SELECT playlist_name, item_id, title, author FROM playlist")
    pl_rows = cur.fetchall()

    inserted = 0
    skipped = 0
    seen_pairs = set()

    for playlist_name, item_id, title, author in pl_rows:
        track_id = item_to_track.get(item_id)
        if not track_id:
            skipped += 1
            continue

        pair_key = (playlist_name, track_id)
        if pair_key in seen_pairs:
            skipped += 1
            continue
        seen_pairs.add(pair_key)

        cur.execute("""
            INSERT INTO playlist_new (playlist_name, track_id, title, author)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (playlist_name, track_id) DO NOTHING
        """, (playlist_name, track_id, title, author))
        if cur.rowcount > 0:
            inserted += 1

    logger.info(f"  playlist_new rows: {inserted} (skipped: {skipped})")


def step9_atomic_swap(cur, has_mulan):
    """Step 9: Atomic rename of old → *_old, new → current."""
    logger.info("=" * 70)
    logger.info("STEP 9: Atomic table swap")
    logger.info("=" * 70)

    # Drop any leftover _old tables from a previous failed migration
    for old_table in [
        "score_old", "embedding_old", "clap_embedding_old",
        "mulan_embedding_old", "playlist_old",
    ]:
        cur.execute(f"DROP TABLE IF EXISTS {old_table} CASCADE")

    # --- score ---
    cur.execute("ALTER TABLE score RENAME TO score_old")
    cur.execute("ALTER TABLE score_new RENAME TO score")
    logger.info("  Swapped: score_old ← score, score ← score_new")

    # Rename indexes from score_new to match expected names
    # PostgreSQL keeps old index names after table rename; rename them for clarity
    _safe_rename_index(cur, "idx_score_new_file_path", "idx_score_file_path")
    _safe_rename_index(cur, "score_new_search_u_trgm", "score_search_u_trgm")
    _safe_rename_index(cur, "idx_score_new_author_lower", "idx_score_author_lower")
    _safe_rename_index(cur, "idx_score_new_title_author_lower", "idx_score_title_author_lower")

    # --- embedding ---
    cur.execute("ALTER TABLE embedding RENAME TO embedding_old")
    cur.execute("ALTER TABLE embedding_new RENAME TO embedding")
    logger.info("  Swapped: embedding_old ← embedding, embedding ← embedding_new")

    # --- clap_embedding ---
    cur.execute("ALTER TABLE clap_embedding RENAME TO clap_embedding_old")
    cur.execute("ALTER TABLE clap_embedding_new RENAME TO clap_embedding")
    logger.info("  Swapped: clap_embedding")

    # --- mulan_embedding (conditional) ---
    if has_mulan:
        cur.execute("ALTER TABLE mulan_embedding RENAME TO mulan_embedding_old")
        cur.execute("ALTER TABLE mulan_embedding_new RENAME TO mulan_embedding")
        logger.info("  Swapped: mulan_embedding")

    # --- playlist ---
    cur.execute("ALTER TABLE playlist RENAME TO playlist_old")
    cur.execute("ALTER TABLE playlist_new RENAME TO playlist")
    logger.info("  Swapped: playlist")


def _safe_rename_index(cur, old_name, new_name):
    """Rename an index if it exists, ignoring errors if new name already taken."""
    try:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = %s)", (old_name,)
        )
        if cur.fetchone()[0]:
            # Check if target name already exists
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = %s)", (new_name,)
            )
            if not cur.fetchone()[0]:
                cur.execute(f"ALTER INDEX {old_name} RENAME TO {new_name}")
    except Exception:
        pass  # Best-effort rename


def step10_verify(cur, expected_score_count, has_mulan):
    """Step 10: Verify row counts and FK integrity."""
    logger.info("=" * 70)
    logger.info("STEP 10: Verification")
    logger.info("=" * 70)

    errors = []

    # Row counts
    new_score_count = count_rows(cur, "score")
    old_score_count = count_rows(cur, "score_old")
    logger.info(f"  score: {new_score_count} rows (old: {old_score_count})")
    # new count may be <= old due to dedup from file_path collisions
    if new_score_count == 0 and old_score_count > 0:
        errors.append(f"score table is empty but score_old has {old_score_count} rows!")

    new_emb = count_rows(cur, "embedding")
    old_emb = count_rows(cur, "embedding_old")
    logger.info(f"  embedding: {new_emb} rows (old: {old_emb})")

    new_clap = count_rows(cur, "clap_embedding")
    old_clap = count_rows(cur, "clap_embedding_old")
    logger.info(f"  clap_embedding: {new_clap} rows (old: {old_clap})")

    if has_mulan:
        new_mulan = count_rows(cur, "mulan_embedding")
        old_mulan = count_rows(cur, "mulan_embedding_old")
        logger.info(f"  mulan_embedding: {new_mulan} rows (old: {old_mulan})")

    new_playlist = count_rows(cur, "playlist")
    old_playlist = count_rows(cur, "playlist_old")
    logger.info(f"  playlist: {new_playlist} rows (old: {old_playlist})")

    track_count = count_rows(cur, "track")
    pt_count = count_rows(cur, "provider_track")
    logger.info(f"  track: {track_count} rows")
    logger.info(f"  provider_track: {pt_count} rows")

    # FK integrity: every score.track_id should exist in track
    cur.execute("""
        SELECT COUNT(*) FROM score s
        WHERE NOT EXISTS (SELECT 1 FROM track t WHERE t.id = s.track_id)
    """)
    orphan_score = cur.fetchone()[0]
    if orphan_score > 0:
        errors.append(f"{orphan_score} score rows reference non-existent track_ids!")
    logger.info(f"  Orphaned score→track references: {orphan_score}")

    # FK integrity: every embedding.track_id should exist in score
    cur.execute("""
        SELECT COUNT(*) FROM embedding e
        WHERE NOT EXISTS (SELECT 1 FROM score s WHERE s.track_id = e.track_id)
    """)
    orphan_emb = cur.fetchone()[0]
    if orphan_emb > 0:
        errors.append(f"{orphan_emb} embedding rows reference non-existent score.track_id!")
    logger.info(f"  Orphaned embedding→score references: {orphan_emb}")

    # provider_track completeness
    cur.execute("""
        SELECT COUNT(*) FROM score s
        WHERE NOT EXISTS (
            SELECT 1 FROM provider_track pt WHERE pt.track_id = s.track_id
        )
    """)
    unlinked = cur.fetchone()[0]
    logger.info(f"  Score rows without provider_track link: {unlinked}")

    if errors:
        for e in errors:
            logger.error(f"  VERIFICATION ERROR: {e}")
        return False

    logger.info("  All verifications passed!")
    return True


def step11_cleanup(cur, has_mulan):
    """Step 11: Drop old tables and record completion."""
    logger.info("=" * 70)
    logger.info("STEP 11: Cleanup")
    logger.info("=" * 70)

    # Drop old tables (order matters for FK dependencies)
    for table in ["embedding_old", "clap_embedding_old"]:
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        logger.info(f"  Dropped: {table}")

    if has_mulan:
        cur.execute("DROP TABLE IF EXISTS mulan_embedding_old CASCADE")
        logger.info("  Dropped: mulan_embedding_old")

    cur.execute("DROP TABLE IF EXISTS playlist_old CASCADE")
    logger.info("  Dropped: playlist_old")

    cur.execute("DROP TABLE IF EXISTS score_old CASCADE")
    logger.info("  Dropped: score_old")

    # Drop legacy artist tables (data is now in artist_provider_mapping)
    if table_exists(cur, "artist_mapping"):
        cur.execute("DROP TABLE IF EXISTS artist_mapping CASCADE")
        logger.info("  Dropped: artist_mapping")
    if table_exists(cur, "artist_id_lookup"):
        cur.execute("DROP TABLE IF EXISTS artist_id_lookup CASCADE")
        logger.info("  Dropped: artist_id_lookup")

    # Invalidate stale Voyager/artist/map indexes — their id_map_json contains
    # old item_id strings that don't resolve in the new track_id schema.
    # They will be rebuilt automatically on first use or next analysis run.
    for index_table in ["voyager_index_data", "artist_index_data", "map_projection_data", "artist_component_projection"]:
        if table_exists(cur, index_table):
            cur.execute(f"DELETE FROM {index_table}")
            logger.info(f"  Cleared stale index data: {index_table} (will rebuild on next use)")

    # Record migration completion
    cur.execute("""
        INSERT INTO app_settings (key, value, category, description, updated_at)
        VALUES (%s, %s::jsonb, 'system', 'Track-ID canonical migration completed', NOW())
        ON CONFLICT (key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = NOW()
    """, (MIGRATION_KEY, json.dumps({
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "1.0",
    })))
    logger.info(f"  Recorded migration completion in app_settings (key={MIGRATION_KEY})")

    # Also mark the legacy backfill migrations as done so init_db doesn't re-run them
    for legacy_key in [
        "migration_backfill_track_table_done",
        "migration_case_normalization_done",
        "migration_dedup_score_rows_done",
    ]:
        cur.execute("""
            INSERT INTO app_settings (key, value, category, description, updated_at)
            VALUES (%s, '"true"'::jsonb, 'system', 'Auto-set by track_id migration', NOW())
            ON CONFLICT (key) DO NOTHING
        """, (legacy_key,))


# ===========================================================================
# Main
# ===========================================================================

def run_migration():
    """Execute the full migration."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("AudioMuse-AI: track_id canonical migration")
    logger.info("Converting score.item_id TEXT PK → score.track_id INTEGER PK")
    logger.info("=" * 70)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Idempotency check
            if is_migration_done(cur):
                logger.info("Migration already completed (found %s in app_settings). Skipping.", MIGRATION_KEY)
                return True

            # Pre-flight: verify score table exists and has data
            if not table_exists(cur, "score"):
                logger.error("Table 'score' does not exist. Nothing to migrate.")
                return False

            if not column_exists(cur, "score", "item_id"):
                logger.error("Column 'score.item_id' does not exist. Schema may already be migrated.")
                return False

            score_count = count_rows(cur, "score")
            if score_count == 0:
                logger.warning("score table is empty. Creating infrastructure only.")

            logger.info(f"Pre-flight: {score_count} rows in score table")
            logger.info("")

            # Step 1: Infrastructure tables
            step1_create_infrastructure(cur)

            # Step 2: Provider entry
            provider_id = step2_create_provider(cur)

            if score_count == 0:
                # Nothing to migrate — just record completion
                cur.execute("""
                    INSERT INTO app_settings (key, value, category, description, updated_at)
                    VALUES (%s, %s::jsonb, 'system', 'Track-ID canonical migration completed (empty DB)', NOW())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, (MIGRATION_KEY, json.dumps({
                    "completed": True,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "version": "1.0",
                    "note": "Empty database — infrastructure tables created only",
                })))
                conn.commit()
                logger.info("Empty database — infrastructure created. Migration recorded.")
                return True

            # Step 3: Track entries
            item_to_track, item_hash_map = step3_create_track_entries(cur)

            # Step 4: Provider track entries
            step4_create_provider_track(cur, provider_id, item_to_track)

            # Step 5: New score table
            new_score_count = step5_create_new_score(cur, item_to_track)

            # Step 6: New embedding tables
            has_mulan = step6_create_new_embeddings(cur, item_to_track)

            # Step 7: Artist migration
            step7_migrate_artists(cur, provider_id)

            # Step 8: Playlist migration
            step8_migrate_playlist(cur, item_to_track)

            # Step 9: Atomic swap
            step9_atomic_swap(cur, has_mulan)

            # Step 10: Verify
            ok = step10_verify(cur, new_score_count, has_mulan)
            if not ok:
                logger.error("Verification failed! Rolling back entire migration.")
                conn.rollback()
                return False

            # Step 11: Cleanup
            step11_cleanup(cur, has_mulan)

        # Commit the entire transaction
        conn.commit()
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Migration completed successfully in {elapsed:.1f} seconds!")
        logger.info("=" * 70)
        return True

    except Exception as e:
        logger.exception(f"Migration failed with error: {e}")
        logger.info("Rolling back all changes...")
        conn.rollback()
        return False

    finally:
        conn.close()


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
