# app_helper_artist.py
"""
Helper functions for artist mapping between names and provider-specific IDs.
Uses the artist_provider_mapping table which stores per-provider artist IDs.

Separated to avoid circular imports.
"""

import logging
from app_helper import get_db
from tasks.memory_utils import sanitize_string_for_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions (new API using artist_provider_mapping)
# ---------------------------------------------------------------------------

def upsert_artist_provider_mapping(artist_name, provider_id, provider_artist_id, is_primary=False):
    """
    Stores or updates the mapping between an artist name and a provider-specific
    artist ID in the artist_provider_mapping table.

    ON CONFLICT (provider_id, provider_artist_id) → updates artist_name and is_primary.

    Args:
        artist_name: Human-readable artist name
        provider_id: Integer FK to provider.id
        provider_artist_id: The provider's native artist ID string
        is_primary: Whether this is the primary provider mapping for the artist
    """
    # Sanitize inputs
    if artist_name:
        artist_name = sanitize_string_for_db(artist_name)
        if artist_name and len(artist_name) > 500:
            artist_name = artist_name[:500]

    if provider_artist_id:
        provider_artist_id = sanitize_string_for_db(str(provider_artist_id))
        if provider_artist_id and len(provider_artist_id) > 200:
            provider_artist_id = provider_artist_id[:200]

    if not artist_name or not provider_artist_id or provider_id is None:
        return False

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO artist_provider_mapping
                    (artist_name, provider_id, provider_artist_id, is_primary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (provider_id, provider_artist_id)
                DO UPDATE SET artist_name = EXCLUDED.artist_name,
                              is_primary  = EXCLUDED.is_primary
            """, (artist_name, provider_id, provider_artist_id, is_primary))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to upsert artist provider mapping for '{artist_name}' "
                     f"(provider_id={provider_id}): {e}")
        from app_helper import reset_db_connection
        reset_db_connection()
        return False


def get_artist_ids_by_name(artist_name):
    """
    Retrieves all provider-specific artist IDs for a given artist name
    (case-insensitive).

    Returns:
        List of dicts: [{'provider_id': int, 'provider_artist_id': str, 'is_primary': bool}, ...]
        Empty list if not found or on error.
    """
    if not artist_name:
        return []

    sanitized_name = sanitize_string_for_db(artist_name)
    if not sanitized_name:
        logger.warning(f"Artist name became empty after sanitization: {repr(artist_name)}")
        return []

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT provider_id, provider_artist_id, is_primary
                FROM artist_provider_mapping
                WHERE LOWER(artist_name) = LOWER(%s)
                ORDER BY is_primary DESC, provider_id ASC
            """, (sanitized_name,))
            rows = cur.fetchall()
            return [
                {
                    'provider_id': row[0],
                    'provider_artist_id': row[1],
                    'is_primary': row[2],
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to get artist IDs for '{sanitized_name}': {e}")
        return []


def get_artist_name_by_provider_id(provider_artist_id):
    """
    Retrieves the artist_name for a given provider-specific artist ID.

    Args:
        provider_artist_id: The provider's native artist ID string

    Returns:
        artist_name string or None if not found.
    """
    if not provider_artist_id:
        return None

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT artist_name
                FROM artist_provider_mapping
                WHERE provider_artist_id = %s
                LIMIT 1
            """, (str(provider_artist_id),))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to get artist_name for provider_artist_id '{provider_artist_id}': {e}")
        return None


def resolve_artist(query):
    """
    Resolve an artist query (name or provider-specific ID) to a canonical
    artist name.

    Resolution order:
        1. Check if query matches score.author (case-insensitive) → return
           the canonical casing from the DB.
        2. Check if query matches artist_provider_mapping.provider_artist_id
           → return the mapped artist_name.
        3. Return the original query unchanged.

    Args:
        query: Artist name or provider-specific artist ID

    Returns:
        Canonical artist_name string, or original query if no resolution found.
        None if query is None/empty.
    """
    if not query:
        return None

    try:
        conn = get_db()
        with conn.cursor() as cur:
            # Step 1: Already an artist name in the score table?
            cur.execute(
                "SELECT DISTINCT author FROM score WHERE LOWER(author) = LOWER(%s) LIMIT 1",
                (query,),
            )
            row = cur.fetchone()
            if row:
                return row[0]

            # Step 2: Provider-specific artist ID → artist_name
            cur.execute(
                "SELECT artist_name FROM artist_provider_mapping WHERE provider_artist_id = %s LIMIT 1",
                (str(query),),
            )
            row = cur.fetchone()
            if row:
                return row[0]
    except Exception as e:
        logger.error(f"Failed to resolve artist for query '{query}': {e}")

    # Step 3: Return original query as-is
    return query


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (used by existing callers)
# ---------------------------------------------------------------------------

def upsert_artist_mapping(artist_name, artist_id):
    """
    Backward-compatible wrapper.

    Stores or updates the mapping between artist name and artist ID.
    Delegates to upsert_artist_provider_mapping with provider_id=1 and
    is_primary=True (assumes the primary/first provider).

    Callers that have access to the actual provider_id should migrate to
    upsert_artist_provider_mapping() directly.
    """
    if not artist_name or not artist_id:
        return

    # Default to provider_id=1 (primary provider) for backward compatibility
    provider_id = _get_default_provider_id()
    upsert_artist_provider_mapping(artist_name, provider_id, artist_id, is_primary=True)


def get_artist_id_by_name(artist_name):
    """
    Backward-compatible wrapper.

    Retrieves a single artist_id for a given artist_name.
    Returns the primary provider's artist_id if available, otherwise the first
    result's provider_artist_id. Returns None if not found.
    """
    results = get_artist_ids_by_name(artist_name)
    if not results:
        return None

    # Prefer primary mapping
    for r in results:
        if r['is_primary']:
            return r['provider_artist_id']

    # Fall back to first result
    return results[0]['provider_artist_id']


def get_artist_name_by_id(artist_id):
    """
    Backward-compatible wrapper.

    Retrieves the artist_name for a given artist_id (provider-specific).
    Delegates to get_artist_name_by_provider_id.
    """
    return get_artist_name_by_provider_id(artist_id)


def upsert_artist_mapping_secondary(artist_name, artist_id):
    """
    Backward-compatible wrapper.

    Store a secondary provider's artist_id → artist_name mapping.
    Delegates to upsert_artist_provider_mapping with provider_id=2 and
    is_primary=False.

    Callers that have access to the actual provider_id should migrate to
    upsert_artist_provider_mapping() directly.

    Returns True if stored, False otherwise.
    """
    if not artist_name or not artist_id:
        return False

    try:
        provider_id = _get_default_secondary_provider_id()
        return upsert_artist_provider_mapping(artist_name, provider_id, artist_id, is_primary=False)
    except Exception as e:
        logger.error(f"Failed to upsert secondary artist mapping for '{artist_name}': {e}")
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_default_provider_id():
    """
    Get the primary provider ID from the provider table.
    Falls back to 1 if lookup fails (assumes first registered provider).
    """
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM provider
                ORDER BY priority DESC, id ASC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return row[0]
    except Exception as e:
        logger.debug(f"Could not look up primary provider ID: {e}")
        from app_helper import reset_db_connection
        reset_db_connection()

    return 1


def _get_default_secondary_provider_id():
    """
    Get the first non-primary provider ID from the provider table.
    Falls back to 2 if lookup fails.
    """
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM provider
                ORDER BY priority ASC, id ASC
                LIMIT 1 OFFSET 1
            """)
            row = cur.fetchone()
            if row:
                return row[0]
    except Exception as e:
        logger.debug(f"Could not look up secondary provider ID: {e}")
        from app_helper import reset_db_connection
        reset_db_connection()

    return 2
