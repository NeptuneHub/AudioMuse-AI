# Versioned, ordered schema migrations.
#
# The base schema is created idempotently by database.init_db(). This module
# layers run-once, ordered migrations on top and records applied versions in the
# schema_version table, giving a real upgrade path between releases. To add a
# migration, append (version, name, fn) to MIGRATIONS with the next integer;
# never edit or renumber a migration that has already shipped.

import logging

logger = logging.getLogger(__name__)

# Everything created by init_db() is recorded as the baseline.
BASELINE_VERSION = 0


def _ensure_version_table(cur):
    cur.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "version INTEGER PRIMARY KEY, "
        "name TEXT NOT NULL, "
        "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )


def _applied_versions(cur):
    cur.execute("SELECT version FROM schema_version")
    return {row[0] for row in cur.fetchall()}


# Ordered migrations. Each fn receives a live DB cursor and applies one change.
# Example:
#   def _m1_add_score_bpm(cur):
#       cur.execute("ALTER TABLE score ADD COLUMN IF NOT EXISTS bpm REAL")
#   MIGRATIONS = [(1, "add score.bpm", _m1_add_score_bpm)]
MIGRATIONS: list = []


def run_schema_migrations(cur, migrations=None):
    # Apply every not-yet-applied migration in version order; returns the count
    # newly applied. Idempotent: a migration already in schema_version is skipped.
    migrations = MIGRATIONS if migrations is None else migrations
    _ensure_version_table(cur)
    done = _applied_versions(cur)

    if BASELINE_VERSION not in done:
        cur.execute(
            "INSERT INTO schema_version (version, name) VALUES (%s, %s) "
            "ON CONFLICT (version) DO NOTHING",
            (BASELINE_VERSION, "baseline"),
        )
        done.add(BASELINE_VERSION)

    applied = 0
    for version, name, fn in sorted(migrations, key=lambda m: m[0]):
        if version in done:
            continue
        logger.info("Applying schema migration %d: %s", version, name)
        fn(cur)
        cur.execute(
            "INSERT INTO schema_version (version, name) VALUES (%s, %s) "
            "ON CONFLICT (version) DO NOTHING",
            (version, name),
        )
        done.add(version)
        applied += 1
    return applied


def get_schema_version(cur):
    # Highest applied version, or -1 if the table is empty / absent.
    _ensure_version_table(cur)
    cur.execute("SELECT COALESCE(MAX(version), -1) FROM schema_version")
    return cur.fetchone()[0]
