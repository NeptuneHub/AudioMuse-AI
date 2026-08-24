# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Hyperbolic Explorer engine over the Poincare-projected MusiCNN embeddings.

Projects raw MusiCNN embeddings into the Poincare ball on top of
``tasks.hyperbolic_geometry``, keeps the ``poincare_embedding`` and
``hyperbolic_radius`` columns in sync (analysis time and backfill), and powers
the similarity and directory-tree endpoints while staying in canonical item_id
space (id translation to the selected server happens in the request layer).

Main Features:
* resolve_hyperbolic_scale derives and caches the projection scale s from the
  catalogue norm distribution and persists it in app_config for cross-process
  reuse by analysis workers, the web app, and the backfill job
* save_hyperbolic_projection / backfill_hyperbolic_columns keep the hyperbolic
  columns in sync with the raw embedding, skipping rows with NULL embeddings
* fetch_poincare_rows, fetch_all_poincare_rows and
  get_projected_genre_subgenres are the public reads other managers need
  (tasks.hyperbolic_journey_manager builds the geodesic journey on top of
  them), so nothing outside this module has to reach into a private helper
  the unit suite monkeypatches by name
* fetch_all_poincare_rows streams every row through a NAMED server-side
  cursor on its own side connection (the same _open_side_connection /
  itersize convention tasks.index_build_helpers uses for every other
  full-catalogue read), never the shared request/worker connection from
  database.get_db(). This matters because it is the one read that always
  covers the whole catalogue - the union Poincare IVF index rebuild
  (tasks.hyperbolic_index) calls it unconditionally with no server scope on
  every rebuild, and an unstreamed SELECT ... ORDER BY item_id ... fetchall()
  there forces Postgres to sort and buffer the entire embedding table inside
  one backend, whose RSS then does not shrink back down afterward
* hyperbolic_similar ranks candidates by exact Poincare distance through the
  disk-paged Poincare index (exact top-k, no IVF index and no cosine
  shortcut), and raises a "run analysis to build it" ValueError when that
  index is not built rather than scanning the whole catalogue - the Flask
  layer turns that into a 400 with the message, the same way the geodesic
  journey does; roots / niche instead draw their candidate pool from the
  embedding table by radius (at least radial_spread of the radial range away
  from the seed, caller-supplied and defaulting to HYPERBOLIC_RADIAL_SPREAD) so the two
  modes visibly leave the seed's radius band, then rank by exact distance. All
  modes end with the same content-dedup + MAX_SONGS_PER_ARTIST pass as the
  similar-song page
* build_hyperbolic_tree_cache does the expensive part - the genre/subgenre
  partition, the mood fallback, and the named Poincare k-means clusters - PER SERVER,
  and persists one tree per configured server as gzipped JSON blobs chunked
  into 50 MB segmented BYTEA rows in ivf_dir (the same pattern the music map
  and IVF directory use), worker-side only (analysis end).
  load_hyperbolic_tree_cache is
  the cheap counterpart: one row read per server, no reclustering, used at Flask
  startup and by the index-reload NOTIFY handler, exactly like the music map keeps
  its expensive UMAP fit in map_projection_data and only re-does cheap JSON
  assembly in Flask. The request path (build_hyperbolic_tree) is then a pure
  dict lookup against the selected server's cached tree, returning canonical
  ids for the caller to translate. Clusters are sized to
  HYPERBOLIC_TARGET_LEAF_SIZE (default 150, i.e. ~100-200 songs) with a floor of
  HYPERBOLIC_MIN_CLUSTER_SIZE (default 20): clusters below that floor are
  pruned and a subgenre left without a valid cluster is hidden; a genre whose
  subgenres all vanished lists its tracks directly instead, so the genre root
  is always shown when the genre data is usable (only genuinely unusable data
  falls back to the legacy mood partition).
  The returned node is a reference into the shared cache, not a copy - callers
  must build a new structure rather than mutate it in place (id translation
  does this already). The persisted blob carries a schema version and is
  discarded on load when it does not match, so an upgraded Flask never serves a
  stale pre-upgrade tree.
* The directory tree is a three-level semantic taxonomy over the same
  embedding space: GENRE -> SUBGENRE -> NAMED CLUSTER. The root splits by
  MAIN GENRE taken from the data-driven genre_subgenre.json centroids
  (nearest main genre at level 0), then by SUBGENRE (nearest of that genre's
  subgenres at level 1), then into named clusters for any subgenre
  still above HYPERBOLIC_TARGET_LEAF_SIZE - nothing deeper. That cluster level
  is a Poincare k-means (_fit_clusters): k-means++ seeding, assignment and
  centroid update all run in the exact hyperbolic metric, with the centroid
  being the Frechet mean from tasks.hyperbolic_geometry.karcher_mean, so no
  Euclidean or cosine step survives anywhere in the hyperbolic path. When the file is
  absent or dimensionally incompatible the tree falls back to a legacy MOOD
  partition (nearest of the precomputed mood centroids in
  mood_centroids_real_080_clap.json) followed by a main-genre partition and
  the same named-cluster level. Clusters are always the terminal level and
  are named from the nearest mood-centroid tags (or the ancestor genre path
  plus the dominant mood) instead of a bare "Cluster N" label
"""

import gzip
import json
import logging
import re
import threading
import time

import numpy as np

import config

from .idle_unload import IdleUnloadTimer

logger = logging.getLogger(__name__)

# The tree cache is a gzipped JSON blob stored as segmented BYTEA rows
# (IVF_MAX_PART_SIZE_MB-sized chunks: "hyperbolic_tree_cache_1_3", ...) in
# the same ivf_dir table the other indexes use. That keeps it under
# Postgres' row-size limit like every other index and makes it immune to the
# web-startup app_config prune (which only ever touches the app_config
# table), so analysis persists it and Flask loads it back at startup.
_TREE_CACHE_TABLE = "ivf_dir"
_TREE_CACHE_BLOB_NAME = "hyperbolic_tree_cache"
_TREE_SKELETON_BLOB_NAME = "hyperbolic_tree_skeleton"

_FALLBACK_SCALE = 1.0
_SCALE_CACHE = {"value": None}
_GENRE_CENTROID_CACHE = {"value": None}


def resolve_hyperbolic_scale(force_recalibrate=False, auto_calibrate=True):
    if _SCALE_CACHE["value"] is not None and not force_recalibrate:
        return _SCALE_CACHE["value"]
    configured = config.HYPERBOLIC_RADIUS_SCALE
    if configured and float(configured) > 0:
        value = float(configured)
    else:
        value = _load_persisted_scale()
        if value is None or force_recalibrate:
            if not auto_calibrate:
                return None
            value = _calibrate_scale_from_catalog()
            _persist_scale(value)
    _SCALE_CACHE["value"] = value
    return value


def reset_hyperbolic_scale_cache():
    _SCALE_CACHE["value"] = None
    _GENRE_CENTROID_CACHE["value"] = None


def _load_persisted_scale():
    try:
        from database import get_app_config_value

        raw = get_app_config_value("hyperbolic_radius_scale")
        if raw is None:
            return None
        value = float(raw)
        return value if value > 0 else None
    except Exception:
        logger.exception("Could not read persisted hyperbolic radius scale")
        return None


def _persist_scale(value):
    # Not swallowed, for the same reason as _persist_tree_cache_blob: a
    # failure here must reach the worker step's error handling with a real
    # traceback, not disappear behind a one-line warning with no exception
    # info while resolve_hyperbolic_scale's caller believes it succeeded.
    from database import set_app_config_value

    set_app_config_value("hyperbolic_radius_scale", repr(float(value)))


def _calibrate_scale_from_catalog():
    from tasks.hyperbolic_geometry import calibrate_scale
    from tasks.index_build_helpers import iter_embedding_batches

    percentile = float(config.HYPERBOLIC_RADIUS_PERCENTILE)
    norm_batches = []
    try:
        for batch, _ids in iter_embedding_batches(
            "embedding",
            "embedding",
            int(config.EMBEDDING_DIMENSION),
            where_clause="embedding IS NOT NULL",
        ):
            norm_batches.append(np.linalg.norm(batch.astype(np.float32), axis=1))
    except Exception:
        logger.exception("Could not sample catalogue norms for scale calibration")
        return _FALLBACK_SCALE
    if not norm_batches:
        return _FALLBACK_SCALE
    all_norms = norm_batches[0] if len(norm_batches) == 1 else np.concatenate(norm_batches)
    all_norms = all_norms[np.isfinite(all_norms)]
    if all_norms.size == 0:
        return _FALLBACK_SCALE
    return calibrate_scale(all_norms, percentile)


def compute_hyperbolic_projection(embedding, scale=None, auto_calibrate=True):
    from tasks.hyperbolic_geometry import poincare_radius, project_to_poincare

    if embedding is None:
        return None, None
    vec = np.asarray(embedding, dtype=np.float32)
    if vec.size == 0:
        return None, None
    if scale is None:
        scale = resolve_hyperbolic_scale(auto_calibrate=auto_calibrate)
        if scale is None:
            return None, None
    proj = project_to_poincare(vec, scale)
    radius = float(poincare_radius(vec, scale))
    return proj, radius


def save_hyperbolic_projection(item_id, embedding, scale=None):
    proj, radius = compute_hyperbolic_projection(embedding, scale)
    if proj is None or radius is None:
        return False
    try:
        from database import set_hyperbolic_projection

        set_hyperbolic_projection(item_id, proj, radius)
        return True
    except Exception:
        logger.exception("Could not persist hyperbolic projection for %s", item_id)
        return False


def backfill_hyperbolic_columns(scale=None):
    from tasks.hyperbolic_geometry import poincare_radius, project_to_poincare
    from tasks.index_build_helpers import iter_embedding_batches

    if scale is None:
        scale = resolve_hyperbolic_scale(force_recalibrate=True)
    total = 0
    batches = 0
    skipped = 0
    for batch, ids in iter_embedding_batches(
        "embedding",
        "embedding",
        int(config.EMBEDDING_DIMENSION),
        where_clause="embedding IS NOT NULL",
    ):
        vecs = batch.astype(np.float32)
        proj = project_to_poincare(vecs, scale)
        radii = poincare_radius(vecs, scale)
        finite = np.isfinite(radii) & np.all(np.isfinite(proj), axis=1)
        if not np.all(finite):
            skipped += int((~finite).sum())
            ids = [iid for iid, keep in zip(ids, finite) if keep]
            proj = proj[finite]
            radii = radii[finite]
        if ids:
            _bulk_upsert_hyperbolic(ids, proj, radii)
        total += len(ids)
        batches += 1
    if skipped:
        logger.warning(
            "Skipped %d track(s) with a non-finite hyperbolic projection (corrupt or "
            "zero-vector embedding); their poincare_embedding/hyperbolic_radius stay NULL.",
            skipped,
        )
    logger.info(
        "Backfilled hyperbolic projection for %d tracks across %d batches (scale=%s)",
        total,
        batches,
        scale,
    )
    return total


def _bulk_upsert_hyperbolic(item_ids, proj_vectors, radii):
    import psycopg2
    from psycopg2.extras import execute_values
    from database import get_db

    db_conn = get_db()
    try:
        with db_conn.cursor() as cur:
            rows = [
                (
                    iid,
                    psycopg2.Binary(vec.astype(np.float32).tobytes()),
                    float(radius),
                )
                for iid, vec, radius in zip(item_ids, proj_vectors, radii)
            ]
            execute_values(
                cur,
                """
                INSERT INTO embedding (item_id, poincare_embedding, hyperbolic_radius)
                VALUES %s
                ON CONFLICT (item_id) DO UPDATE SET
                    poincare_embedding = EXCLUDED.poincare_embedding,
                    hyperbolic_radius = EXCLUDED.hyperbolic_radius
                """,
                rows,
            )
        db_conn.commit()
    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            pass
        logger.exception("Bulk hyperbolic upsert failed for %d rows", len(item_ids))
        raise


def _is_finite_row(vec, radius):
    return np.isfinite(radius) and vec.size > 0 and bool(np.all(np.isfinite(vec)))


def _fetch_poincare_rows(item_ids):
    if not item_ids:
        return {}
    from database import get_db

    out = {}
    db_conn = get_db()
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT item_id, poincare_embedding, hyperbolic_radius "
                "FROM embedding WHERE item_id = ANY(%s) "
                "AND poincare_embedding IS NOT NULL AND hyperbolic_radius IS NOT NULL "
                "ORDER BY item_id",
                (list(item_ids),),
            )
            for item_id, blob, radius in cur.fetchall():
                vec = np.frombuffer(bytes(blob), dtype=np.float32)
                if _is_finite_row(vec, radius):
                    out[item_id] = (vec, float(radius))
    except Exception:
        logger.exception("Could not fetch hyperbolic rows")
    return out


def fetch_poincare_rows(item_ids):
    return _fetch_poincare_rows(item_ids)


def fetch_all_poincare_rows(server_id=None, include_legacy_default=True):
    return _fetch_all_poincare_rows(
        server_id=server_id, include_legacy_default=include_legacy_default
    )


def _stream_poincare_rows(where_sql, params, cursor_name):
    from tasks.index_build_helpers import _open_side_connection, _STREAM_ITERSIZE

    out = {}
    skipped = 0
    select_sql = (
        "SELECT e.item_id, e.poincare_embedding, e.hyperbolic_radius "
        f"FROM embedding e WHERE {where_sql} ORDER BY e.item_id"
    )
    side_conn = _open_side_connection()
    try:
        with side_conn.cursor(name=cursor_name) as sc:
            sc.itersize = _STREAM_ITERSIZE
            sc.execute(select_sql, params)
            for item_id, blob, radius in sc:
                vec = np.frombuffer(bytes(blob), dtype=np.float32)
                if _is_finite_row(vec, radius):
                    out[item_id] = (vec, float(radius))
                else:
                    skipped += 1
    finally:
        try:
            side_conn.close()
        except Exception:
            pass
    if skipped:
        logger.warning(
            "Skipped %d hyperbolic row(s) with a non-finite radius or vector while "
            "building the tree cache.",
            skipped,
        )
    return out


def _fetch_all_poincare_rows(server_id=None, include_legacy_default=True):
    try:
        if server_id is None:
            return _stream_poincare_rows(
                "e.poincare_embedding IS NOT NULL AND e.hyperbolic_radius IS NOT NULL",
                (), "hyperbolic_poincare_rows",
            )
        from tasks.mediaserver.registry import availability_sql

        where = availability_sql("e")
        out = _stream_poincare_rows(
            f"e.poincare_embedding IS NOT NULL AND e.hyperbolic_radius IS NOT NULL AND {where}",
            (server_id, bool(include_legacy_default)), "hyperbolic_poincare_rows_scoped",
        )
        if not out and include_legacy_default:
            out = _stream_poincare_rows(
                "e.poincare_embedding IS NOT NULL AND e.hyperbolic_radius IS NOT NULL",
                (), "hyperbolic_poincare_rows_fallback",
            )
        return out
    except Exception:
        logger.exception("Could not fetch hyperbolic rows")
        return {}


def _fetch_poincare_rows_in_radius(
    bound_radius, below=True, limit=100, server_id=None, include_legacy_default=True
):
    """Fetch up to ``limit`` projected tracks on one side of a radius bound.

    Used by the roots/niche modes so their candidate pool spans the radius
    range the mode promises instead of only the seed's own radius band:
    ``below=True`` returns the tracks with ``hyperbolic_radius < bound``
    ordered from the bound downward (closest to the seed first); ``below=False``
    returns ``hyperbolic_radius > bound`` ordered from the bound upward.
    Returns ``{item_id: (vec, radius)}``.
    """
    if bound_radius is None or not np.isfinite(bound_radius):
        return {}
    from database import get_db

    operator = "<" if below else ">"
    order = "DESC" if below else "ASC"
    out = {}
    db_conn = get_db()
    try:
        with db_conn.cursor() as cur:
            if server_id is None:
                sql = (
                    "SELECT item_id, poincare_embedding, hyperbolic_radius FROM embedding "
                    "WHERE poincare_embedding IS NOT NULL AND hyperbolic_radius IS NOT NULL "
                    f"AND hyperbolic_radius {operator} %s ORDER BY hyperbolic_radius {order} LIMIT %s"
                )
                cur.execute(sql, (float(bound_radius), int(limit)))
            else:
                from tasks.mediaserver.registry import availability_sql

                where = availability_sql("e")
                sql = (
                    "SELECT e.item_id, e.poincare_embedding, e.hyperbolic_radius FROM embedding e "
                    "WHERE e.poincare_embedding IS NOT NULL AND e.hyperbolic_radius IS NOT NULL "
                    f"AND e.hyperbolic_radius {operator} %s AND {where} "
                    f"ORDER BY e.hyperbolic_radius {order} LIMIT %s"
                )
                cur.execute(
                    sql,
                    (float(bound_radius), server_id, bool(include_legacy_default), int(limit)),
                )
            for item_id, blob, radius in cur.fetchall():
                vec = np.frombuffer(bytes(blob), dtype=np.float32)
                if _is_finite_row(vec, radius):
                    out[item_id] = (vec, float(radius))
    except Exception:
        logger.exception("Could not fetch hyperbolic rows in radius window")
    return out


def get_poincare_radius(item_id):
    """Return the hyperbolic radius of one track, or None when unavailable.

    Used by the API layer to expose the seed's radius so the frontend can draw
    the mode boundary (roots = inside the seed radius, niche = outside it).
    """
    if not item_id:
        return None
    row = _fetch_poincare_rows([item_id]).get(item_id)
    return row[1] if row is not None else None


def _gather_mode_candidates(target_radius, mode, radial_spread, overfetch, server_id=None):
    spread = min(max(float(radial_spread), 0.0), 0.99)
    if mode == "roots":
        bound = target_radius * (1.0 - spread)
        return _fetch_poincare_rows_in_radius(
            bound, below=True, limit=overfetch, server_id=server_id
        )
    bound = target_radius + (1.0 - target_radius) * spread
    return _fetch_poincare_rows_in_radius(
        bound, below=False, limit=overfetch, server_id=server_id
    )


def _rank_mode_results(ids, distances, cand_radii, mode, target_radius):
    if mode == "roots":
        keep = cand_radii < target_radius
    elif mode == "niche":
        keep = cand_radii > target_radius
    else:
        keep = np.ones(len(ids), dtype=bool)
    results = []
    for i, item_id in enumerate(ids):
        if not keep[i]:
            continue
        results.append(
            {
                "item_id": item_id,
                "distance": float(distances[i]),
                "hyperbolic_radius": float(cand_radii[i]),
            }
        )
    results.sort(key=lambda r: r["distance"])
    return results


def hyperbolic_similar(target_item_id, mode="similar", limit=20, radial_spread=None, server_id=None):
    from tasks.hyperbolic_geometry import hyperbolic_distances_to

    mode = (mode or "similar").strip().lower()
    if mode not in ("similar", "roots", "niche"):
        raise ValueError('mode must be one of "similar", "roots", "niche"')
    limit = max(1, min(int(limit), int(config.HYPERBOLIC_MAX_LIMIT)))
    if radial_spread is None:
        radial_spread = config.HYPERBOLIC_RADIAL_SPREAD
    target = _fetch_poincare_rows([target_item_id]).get(target_item_id)
    if target is None:
        raise ValueError(
            "Target track has no hyperbolic projection; run the hyperbolic backfill job first"
        )
    target_vec, target_radius = target
    overfetch = max(
        int(limit) * int(config.HYPERBOLIC_CANDIDATE_OVERFETCH), int(limit) + 50
    )

    if mode == "similar":
        from tasks.hyperbolic_index import hyperbolic_nearest

        nearest = hyperbolic_nearest(
            target_vec, overfetch, server_id=server_id, exclude={target_item_id}
        )
        if nearest is None:
            raise ValueError(
                "Poincare index not built yet - run analysis to build it."
            )
        if not nearest:
            return []
        ids = [item_id for item_id, _distance in nearest]
        rows = _fetch_poincare_rows(ids)
        results = []
        for item_id, distance in nearest:
            row = rows.get(item_id)
            if row is None:
                continue
            results.append(
                {
                    "item_id": item_id,
                    "distance": float(distance),
                    "hyperbolic_radius": float(row[1]),
                }
            )
        return _deduplicate_and_cap_results(results)[:limit]

    rows = _gather_mode_candidates(
        target_radius, mode, radial_spread, overfetch, server_id=server_id
    )
    if not rows:
        return []
    ids = list(rows.keys())
    cand_vecs = np.stack([rows[i][0] for i in ids]).astype(np.float32)
    cand_radii = np.array([rows[i][1] for i in ids], dtype=np.float32)
    distances = hyperbolic_distances_to(target_vec, cand_vecs)
    results = _rank_mode_results(ids, distances, cand_radii, mode, target_radius)
    results = results[:overfetch]
    return _deduplicate_and_cap_results(results)[:limit]


def _deduplicate_and_cap_results(results):
    """Mirror the similar-song page on the final result list.

    Drops content duplicates (same title + author under different item ids) and
    caps the number of tracks per artist at MAX_SONGS_PER_ARTIST, so a playlist
    built from any hyperbolic mode follows the same dedup rules as the
    similar-song page. Tracks without resolvable author metadata are skipped by
    the artist cap, matching the shared ivf_manager behaviour.
    """
    if not results:
        return results
    from database import get_score_data_by_ids
    from tasks.search_shaping import apply_artist_cap, dedup_by_content

    ids = [r["item_id"] for r in results]
    details = {d["item_id"]: d for d in get_score_data_by_ids(ids)}
    deduped = dedup_by_content(results, details)
    return apply_artist_cap(
        deduped, lambda song: (details.get(song["item_id"]) or {}).get("author")
    )


_TREE_CACHE = {
    "n_bands": None, "nodes": None, "flat_ids": None, "track_count": None,
    "servers": {},
}

_TREE_CACHE_LOCK = threading.RLock()

_TREE_STATE = {"full_loaded": False, "full_load_running": False}

_FULL_LOAD_LOCK = threading.Lock()

# Bump whenever the persisted tree schema changes (node id scheme, node kinds,
# level structure). load_hyperbolic_tree_cache discards blobs whose version
# does not match so an upgraded Flask never serves a stale pre-upgrade tree.
_TREE_CACHE_VERSION = 3


# The default server's tree is kept under the "__default__" key (requests with
# no ?server= resolve to it) and mirrored into the legacy top-level fields so
# single-server installs and tests keep working unchanged. Each configured
# secondary server gets its own tree under its server_id.
_DEFAULT_SERVER_KEY = "__default__"


def reset_hyperbolic_tree_cache():
    with _TREE_CACHE_LOCK:
        _TREE_CACHE["n_bands"] = None
        _TREE_CACHE["nodes"] = None
        _TREE_CACHE["flat_ids"] = None
        _TREE_CACHE["track_count"] = None
        _TREE_CACHE["servers"] = {}
        _TREE_STATE["full_loaded"] = False


def _resolve_default_server_id():
    from tasks.mediaserver import registry

    try:
        return registry.get_default_server_id()
    except Exception:
        return None


def _tree_build_targets():
    """The (server_key, server_id, is_default) trees to build at analysis end.

    Every configured server gets its own tree so a request scoped to that
    server only sees genres/subgenres/clusters that server can actually back
    with songs. An empty or unreadable registry falls back to one legacy tree
    over the whole catalogue (the single-server behaviour).
    """
    from tasks.mediaserver import registry

    try:
        servers = registry.list_servers()
    except Exception:
        servers = []
    if not servers:
        return [(_DEFAULT_SERVER_KEY, None, True)]
    default_id = _resolve_default_server_id()
    targets = [(_DEFAULT_SERVER_KEY, default_id, True)]
    for s in servers:
        if s["server_id"] != default_id:
            targets.append((s["server_id"], s["server_id"], False))
    return targets


def _server_scoped_blob(base_name, server_key):
    if not server_key or server_key == _DEFAULT_SERVER_KEY:
        return base_name
    return f"{base_name}__{server_key}"


def _blob_name_for(server_key):
    return _server_scoped_blob(_TREE_CACHE_BLOB_NAME, server_key)


def _skeleton_blob_name_for(server_key):
    return _server_scoped_blob(_TREE_SKELETON_BLOB_NAME, server_key)


def _skeleton_tree(tree):
    nodes = tree.get("nodes") or {}
    skeleton_nodes = {
        nid: node for nid, node in nodes.items()
        if node.get("type") == "folder" and not node.get("leaf")
    }
    return {
        "n_bands": tree.get("n_bands"),
        "nodes": skeleton_nodes,
        "flat_ids": {},
        "track_count": tree.get("track_count") or 0,
    }


def tree_for_server(server_id=None):
    """The in-memory tree dict for a request's selected server.

    ``server_id`` None resolves to the default server's tree. The default
    server's real id is dual-keyed into ``servers`` alongside the sentinel
    default key (see the load/build functions below) so looking it up by
    either name hits the same tree. Any other id that has no tree of its own
    yet (e.g. a server added after the last analysis run) returns an empty
    tree rather than silently falling back to the default server's tree -
    a selected server must never show another server's genres/subgenres.
    """
    with _TREE_CACHE_LOCK:
        if not server_id or server_id == _DEFAULT_SERVER_KEY:
            return _TREE_CACHE
        return _TREE_CACHE["servers"].get(server_id) or {}


def build_hyperbolic_tree_cache():
    targets = _tree_build_targets()
    default_track_count = 0
    for server_key, server_id, is_default in targets:
        rows = _fetch_all_poincare_rows(
            server_id=server_id, include_legacy_default=is_default
        )
        if not rows:
            tree = {"n_bands": 0, "nodes": {}, "flat_ids": {}, "track_count": 0}
        else:
            from database import get_score_data_by_ids

            score_by_id = {d["item_id"]: d for d in get_score_data_by_ids(list(rows.keys()))}
            mood_centroids = _load_projected_mood_centroids()
            genre_subgenres = _load_projected_genre_subgenres()
            nodes, flat_ids = _build_tree_nodes(
                rows, score_by_id, mood_centroids, genre_subgenres
            )
            root_count = len(nodes["root"]["items"])
            tree = {
                "n_bands": root_count, "nodes": nodes,
                "flat_ids": flat_ids, "track_count": len(rows),
            }
        _TREE_CACHE["servers"][server_key] = tree
        if is_default:
            _TREE_CACHE["n_bands"] = tree["n_bands"]
            _TREE_CACHE["nodes"] = tree["nodes"]
            _TREE_CACHE["flat_ids"] = tree["flat_ids"]
            _TREE_CACHE["track_count"] = tree["track_count"]
            default_track_count = tree["track_count"]
            if server_id and server_id != server_key:
                _TREE_CACHE["servers"][server_id] = tree
        _persist_tree_cache_blob(tree, name=_blob_name_for(server_key))
        _persist_tree_cache_blob(_skeleton_tree(tree), name=_skeleton_blob_name_for(server_key))
    _TREE_STATE["full_loaded"] = True
    logger.info(
        "Hyperbolic tree cache built and persisted: %d server tree(s) "
        "(%d tracks in the default tree).", len(targets), default_track_count,
    )
    return default_track_count


def _scan_tree_cache_blob_names(base_name):
    """Distinct per-server tree blob base names persisted in ivf_dir.

    Segmented "name_i_n" rows are folded back to their base name. Only the
    per-server blobs (the base prefix plus "__") are returned, so Flask can
    warm every secondary server tree at startup without touching the registry.
    """
    from database import get_db

    prefix = base_name + "__"
    # The trailing % must be an UNESCAPED wildcard: prefix's underscores are
    # escaped with backslashes, but the suffix separator is a real LIKE
    # wildcard so "<base>__<server_id>" blobs are discovered.
    like = prefix.replace("_", r"\_") + "%"
    try:
        db_conn = get_db()
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT name FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
                (like,),
            )
            names = set()
            for (raw,) in cur.fetchall():
                base = re.sub(r"_\d+_\d+$", "", raw)
                if base.startswith(prefix):
                    names.add(base)
            return sorted(names)
    except Exception:
        return []


def load_hyperbolic_tree_cache():
    payload = _load_tree_cache_blob()
    if payload is None:
        _set_empty_tree_cache()
        logger.info("Hyperbolic tree cache empty: nothing persisted yet (run analysis first).")
        return 0

    if payload.get("version") != _TREE_CACHE_VERSION:
        # A blob written by an older schema (radial bands, second/third genre
        # levels, ...) would serve a stale, incompatible tree after an upgrade.
        # Discard it so the next analysis run rebuilds the current structure.
        logger.warning(
            "Hyperbolic tree cache has schema version %r (current %r); discarding "
            "it - run analysis to rebuild.", payload.get("version"), _TREE_CACHE_VERSION,
        )
        _delete_tree_cache_blob()
        _set_empty_tree_cache()
        return 0

    track_count = int(payload.get("track_count") or 0)
    default_id = _resolve_default_server_id()
    with _TREE_CACHE_LOCK:
        _TREE_CACHE["n_bands"] = payload.get("n_bands")
        _TREE_CACHE["nodes"] = payload.get("nodes") or {}
        _TREE_CACHE["flat_ids"] = payload.get("flat_ids") or {}
        _TREE_CACHE["track_count"] = track_count
        _TREE_CACHE["servers"] = {_DEFAULT_SERVER_KEY: payload}
        if default_id:
            _TREE_CACHE["servers"][default_id] = payload
        _TREE_STATE["full_loaded"] = True
    for blob_name in _scan_tree_cache_blob_names(_TREE_CACHE_BLOB_NAME):
        server_id = blob_name[len(_TREE_CACHE_BLOB_NAME) + 2:]
        per_server = _load_tree_cache_blob(name=blob_name)
        if per_server and per_server.get("version") == _TREE_CACHE_VERSION:
            with _TREE_CACHE_LOCK:
                _TREE_CACHE["servers"][server_id] = per_server
    logger.info(
        "Hyperbolic tree cache loaded from ivf_dir: %d tracks across %d nodes "
        "(%d server trees).", track_count, len(_TREE_CACHE["nodes"]),
        len(_TREE_CACHE["servers"]),
    )
    return track_count


def load_hyperbolic_tree_skeleton():
    payload = _load_tree_cache_blob(name=_TREE_SKELETON_BLOB_NAME)
    if payload is None or payload.get("version") != _TREE_CACHE_VERSION:
        return False
    default_id = _resolve_default_server_id()
    with _TREE_CACHE_LOCK:
        _TREE_CACHE["n_bands"] = payload.get("n_bands")
        _TREE_CACHE["nodes"] = payload.get("nodes") or {}
        _TREE_CACHE["flat_ids"] = {}
        _TREE_CACHE["track_count"] = int(payload.get("track_count") or 0)
        _TREE_CACHE["servers"] = {_DEFAULT_SERVER_KEY: payload}
        if default_id:
            _TREE_CACHE["servers"][default_id] = payload
        _TREE_STATE["full_loaded"] = False
    for blob_name in _scan_tree_cache_blob_names(_TREE_SKELETON_BLOB_NAME):
        server_id = blob_name[len(_TREE_SKELETON_BLOB_NAME) + 2:]
        skeleton = _load_tree_cache_blob(name=blob_name)
        if skeleton and skeleton.get("version") == _TREE_CACHE_VERSION:
            with _TREE_CACHE_LOCK:
                _TREE_CACHE["servers"][server_id] = skeleton
    logger.info("Hyperbolic tree skeleton loaded: %d nodes (%d server trees).",
                len(_TREE_CACHE["nodes"]), len(_TREE_CACHE["servers"]))
    return True


def _set_empty_tree_cache():
    with _TREE_CACHE_LOCK:
        _TREE_CACHE["n_bands"] = 0
        _TREE_CACHE["nodes"] = {}
        _TREE_CACHE["flat_ids"] = {}
        _TREE_CACHE["track_count"] = 0
        _TREE_CACHE["servers"] = {}
        _TREE_STATE["full_loaded"] = False


def _delete_tree_cache_blob():
    from database import get_db

    db_conn = get_db()
    like_pattern = _TREE_CACHE_BLOB_NAME.replace("_", r"\_") + r"\_%"
    skeleton_like = _TREE_SKELETON_BLOB_NAME.replace("_", r"\_") + r"\_%"
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ivf_dir WHERE name = %s OR name LIKE %s ESCAPE '\\' "
            "OR name = %s OR name LIKE %s ESCAPE '\\'",
            (_TREE_CACHE_BLOB_NAME, like_pattern, _TREE_SKELETON_BLOB_NAME, skeleton_like),
        )
    db_conn.commit()


def _persist_tree_cache_blob(payload, name=None):
    # Deliberately not wrapped in try/except: a persist failure here must
    # propagate to the worker step (_run_all_index_builds catches it,
    # records it through error_manager, and the run continues since this
    # step is non-fatal) rather than being swallowed into a log line nobody
    # reads while the caller still reports "built and persisted" as if it
    # worked.
    from database import get_db
    from tasks.index_build_helpers import store_segmented_blob

    name = name or _TREE_CACHE_BLOB_NAME
    if payload is None:
        _delete_tree_cache_blob()
        return
    payload = dict(payload)
    payload["version"] = _TREE_CACHE_VERSION
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    db_conn = get_db()
    # store_segmented_blob clears any stale rows first, then stores the blob
    # as one row or as IVF_MAX_PART_SIZE_MB (50 MB) "name_i_n" rows.
    store_segmented_blob(db_conn, _TREE_CACHE_TABLE, name, gzip.compress(raw))
    db_conn.commit()


def _load_tree_cache_blob(name=None):
    try:
        from database import get_db
        from tasks.index_build_helpers import load_segmented_blob

        name = name or _TREE_CACHE_BLOB_NAME
        blob = load_segmented_blob(get_db(), _TREE_CACHE_TABLE, name)
        if not blob:
            return None
        return json.loads(gzip.decompress(blob).decode("utf-8"))
    except Exception:
        logger.exception("Could not load hyperbolic tree cache")
        return None


def init_hyperbolic_cache():
    try:
        load_hyperbolic_tree_cache()
    except Exception:
        logger.exception("init_hyperbolic_cache failed")


def is_hyperbolic_tree_cache_loaded() -> bool:
    return _TREE_CACHE.get("nodes") is not None


# The tree cache is a fully materialized Python object tree (one dict per
# track/folder), not disk-paged like the IVF indexes, so its RSS cost scales
# with catalogue size. It is lazy-loaded on the first /hyperbolic page open
# (or tree API call after an idle unload) and dropped again after
# HYPERBOLIC_TREE_WARMUP_DURATION seconds with no further activity - the same
# warm-cache-timer shape as tasks.gte_warm_cache and the CLAP text model.
_TREE_TIMER = IdleUnloadTimer()


def _unload_tree_expired():
    if not is_hyperbolic_tree_cache_loaded():
        return
    logger.info("Hyperbolic tree warm cache expired - unloading tree cache")
    reset_hyperbolic_tree_cache()
    try:
        from tasks.memory_utils import release_memory_to_os

        release_memory_to_os()
    except Exception:
        logger.exception("Hyperbolic tree unload: heap release to the OS failed")


def _start_background_full_load():
    with _TREE_CACHE_LOCK:
        if _TREE_STATE["full_loaded"] or _TREE_STATE["full_load_running"]:
            return
        _TREE_STATE["full_load_running"] = True

    def _load():
        try:
            from flask_app import app
            with app.app_context():
                _ensure_full_tree_loaded()
        except Exception:
            logger.exception("Background full hyperbolic tree load failed")
        finally:
            _TREE_STATE["full_load_running"] = False

    threading.Thread(target=_load, daemon=True).start()


def _ensure_full_tree_loaded():
    if _TREE_STATE["full_loaded"]:
        return
    with _FULL_LOAD_LOCK:
        if not _TREE_STATE["full_loaded"]:
            load_hyperbolic_tree_cache()


def warmup_hyperbolic_tree_cache():
    with _TREE_TIMER.lock():
        if not is_hyperbolic_tree_cache_loaded():
            logger.info("Warming up Hyperbolic Explorer tree cache...")
            if not load_hyperbolic_tree_skeleton():
                init_hyperbolic_cache()
        if is_hyperbolic_tree_cache_loaded() and not _TREE_STATE["full_loaded"]:
            _start_background_full_load()

        duration = config.HYPERBOLIC_TREE_WARMUP_DURATION
        if _TREE_TIMER.arm(duration, _unload_tree_expired):
            logger.info("Started Hyperbolic tree warm cache timer (%ss)", duration)
        else:
            logger.debug("Reset Hyperbolic tree warm cache timer (%ss)", duration)

    return {"loaded": is_hyperbolic_tree_cache_loaded(), "expiry_seconds": duration}


def get_hyperbolic_tree_warm_status():
    expiry = _TREE_TIMER.expiry()

    if expiry is None or not is_hyperbolic_tree_cache_loaded():
        return {"active": False, "seconds_remaining": 0}
    return {"active": True, "seconds_remaining": max(0, int(expiry - time.time()))}


def build_hyperbolic_tree(node_id=None, server_id=None):
    key = (node_id or "root").strip() or "root"
    with _TREE_CACHE_LOCK:
        tree = tree_for_server(server_id)
        nodes = tree.get("nodes")
        if not nodes:
            # A specific (non-default) server with no tree of its own is a
            # different situation from "nothing analyzed at all yet" - tell
            # the caller so the UI can say so, instead of silently rendering
            # an indistinguishable empty folder.
            if server_id and server_id != _DEFAULT_SERVER_KEY and server_id not in _TREE_CACHE["servers"]:
                raise ValueError(
                    "Hyperbolic Explorer tree is not available for this "
                    "server yet - run analysis to build it."
                )
            return _empty_node(node_id, "Hyperbolic Explorer"), []
        node = nodes.get(key)
        full_loaded = _TREE_STATE["full_loaded"]
    if node is None and not full_loaded:
        _ensure_full_tree_loaded()
        with _TREE_CACHE_LOCK:
            tree = tree_for_server(server_id)
            nodes = tree.get("nodes") or {}
            node = nodes.get(key)
    if node is None:
        raise ValueError(f"Unknown tree node id: {node_id}")
    return node, (tree.get("flat_ids") or {}).get(key, [])


def _build_genre_root_items(item_ids, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids):
    root_items = []
    if _genre_subgenres_usable(vec_map, genre_subgenres):
        genre_ordered = _partition_by_genre_centroids(
            item_ids, vec_map, genre_subgenres, level=0,
        )
        if genre_ordered:
            for slug, label, members in _merge_by_slug(genre_ordered):
                if not members:
                    continue
                genre_node = _materialize_genre_folder(
                    f"root.g{slug}", label, members, vec_map, radii_map,
                    score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids, level=0,
                )
                if genre_node is not None:
                    root_items.append(genre_node)
    return root_items


def _build_mood_root_items(item_ids, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids):
    root_items = []
    for mood_label, members in _partition_by_mood(item_ids, vec_map, mood_centroids, score_by_id):
        if not members:
            continue
        mood_node = _materialize_mood(
            mood_label, members, vec_map, radii_map, score_by_id,
            mood_centroids, genre_subgenres, nodes, flat_ids,
        )
        if mood_node is not None:
            root_items.append(mood_node)
    return root_items


def _build_tree_nodes(rows, score_by_id, mood_centroids, genre_subgenres):
    item_ids = list(rows.keys())
    vec_map = {iid: rows[iid][0] for iid in item_ids}
    radii_map = {iid: rows[iid][1] for iid in item_ids}

    nodes = {}
    flat_ids = {}
    root_items = _build_genre_root_items(
        item_ids, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids
    )
    if not root_items:
        root_items = _build_mood_root_items(
            item_ids, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids
        )

    nodes["root"] = {
        "id": "root",
        "name": "Hyperbolic Explorer",
        "type": "folder",
        "kind": "root",
        "children_count": len(root_items),
        "summary": {"track_count": len(item_ids)},
        "items": root_items,
    }
    flat_ids["root"] = []
    return nodes, flat_ids


def _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids, kind="cluster"):
    items = [_track_node(i, score_by_id) for i in members]
    node = {
        "id": node_id, "name": name, "type": "folder", "leaf": True, "kind": kind,
        "children_count": len(items), "summary": summary, "items": items,
    }
    nodes[node_id] = node
    flat_ids[node_id] = list(members)
    return node


def _branch_folder(node_id, name, summary, summary_items, nodes, flat_ids, kind="cluster"):
    node = {
        "id": node_id, "name": name, "type": "folder", "leaf": False, "kind": kind,
        "children_count": len(summary_items), "summary": summary, "items": summary_items,
    }
    nodes[node_id] = node
    # Non-leaf folders never carry tracks directly - nothing here needs
    # per-server id translation until the caller drills into an actual leaf.
    flat_ids[node_id] = []
    return node


def _slugify(label):
    slug = re.sub(r"[^a-z0-9]+", "-", str(label).lower()).strip("-")
    return slug or "other"


def _merge_by_slug(ordered):
    merged = {}
    for label, members in ordered:
        slug = _slugify(label)
        if slug in merged:
            merged[slug][1].extend(members)
        else:
            merged[slug] = (label, list(members))
    return [(slug, label, members) for slug, (label, members) in merged.items()]


def _parse_label_scores(raw):
    scores = {}
    if not raw or not isinstance(raw, str):
        return scores
    for part in raw.split(","):
        label, _, value = part.partition(":")
        label = label.strip()
        if not label:
            continue
        try:
            scores[label] = float(value)
        except ValueError:
            continue
    return scores


def _dominant_mood(info):
    if not info:
        return None
    scores = _parse_label_scores(info.get("other_features"))
    candidates = [m for m in config.OTHER_FEATURE_LABELS if m in scores]
    if not candidates:
        return None
    return max(candidates, key=scores.get)


def _genre_rank(info):
    if not info:
        return []
    scores = _parse_label_scores(info.get("mood_vector"))
    genres = [g for g in config.STRATIFIED_GENRES if g in scores]
    genres.sort(key=lambda g: -scores[g])
    return genres


def _partition_by_mood(item_ids, vec_map, mood_centroids, score_by_id):
    assigns = {}
    if mood_centroids:
        from tasks.hyperbolic_geometry import hyperbolic_distance_matrix

        vecs = np.stack([vec_map[i] for i in item_ids]).astype(np.float32)
        cent = np.stack([c["vec"] for c in mood_centroids]).astype(np.float32)
        dists = hyperbolic_distance_matrix(vecs, cent)
        best = np.argmin(dists, axis=1)
        for iid, idx in zip(item_ids, best):
            assigns[iid] = mood_centroids[int(idx)]["mood"]
    else:
        for iid in item_ids:
            assigns[iid] = _dominant_mood(score_by_id.get(iid)) or "general"

    used = set(assigns.values())
    ordered = [m for m in config.OTHER_FEATURE_LABELS if m in used]
    ordered += sorted(used - set(ordered))
    return [
        (mood, [iid for iid, m in assigns.items() if m == mood])
        for mood in ordered
    ]


def _genre_subgenres_usable(vec_map, genre_subgenres):
    if not genre_subgenres or not vec_map:
        return False
    ref = next(iter(vec_map.values()))
    info = next(iter(genre_subgenres.values()))
    return bool(info) and info["vec"].shape[0] == ref.shape[0]


def _partition_by_genre(members, vec_map, score_by_id, genre_subgenres, level, parent_genre=None):
    if _genre_subgenres_usable(vec_map, genre_subgenres):
        return _partition_by_genre_centroids(
            members, vec_map, genre_subgenres, level, parent_genre
        )
    if level != 0:
        # The genre-less fallback has no subgenre data: only the main-genre
        # partition exists and anything below it is the named-cluster level.
        return None
    groups = {}
    for iid in members:
        rank = _genre_rank(score_by_id.get(iid))
        label = rank[0] if rank else "Other"
        groups.setdefault(label, []).append(iid)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    if len(ordered) < 2:
        return None
    return ordered


def _partition_by_genre_centroids(members, vec_map, genre_subgenres, level, parent_genre=None):
    from tasks.hyperbolic_geometry import hyperbolic_distance_matrix

    if not members:
        return None
    ref = vec_map[members[0]]
    if level == 0:
        centroids = [{"name": g, "vec": info["vec"]}
                     for g, info in genre_subgenres.items()]
    elif level == 1 and parent_genre in genre_subgenres:
        subs = genre_subgenres[parent_genre]["subgenres"]
        centroids = [{"name": s["name"], "vec": s["vec"]} for s in subs]
    else:
        return None
    if len(centroids) < 2 or centroids[0]["vec"].shape[0] != ref.shape[0]:
        return None
    vecs = np.stack([vec_map[i] for i in members]).astype(np.float32)
    cent = np.stack([c["vec"] for c in centroids]).astype(np.float32)
    best = np.argmin(hyperbolic_distance_matrix(vecs, cent), axis=1)
    groups = {}
    for iid, idx in zip(members, best):
        groups.setdefault(centroids[int(idx)]["name"], []).append(iid)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    if len(ordered) < 2:
        return None
    return ordered


_GENRE_KINDS = ("main_genre", "subgenre")


def _materialize_mood(mood_label, members, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids):
    node_id = f"m{_slugify(mood_label)}"
    name = mood_label.title()
    radii = np.array([radii_map[i] for i in members], dtype=np.float32)
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))

    summary_items = None
    if len(members) > target_leaf:
        summary_items = _materialize_genre_level(
            node_id, members, vec_map, radii_map, score_by_id,
            mood_centroids, genre_subgenres, nodes, flat_ids, level=0,
        )
    if summary_items is None and len(members) > target_leaf:
        summary_items = _materialize_children(
            node_id, members, vec_map, radii_map, score_by_id,
            mood_centroids, nodes, flat_ids,
        )
    if summary_items is None:
        return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids, kind="mood")
    return _branch_folder(node_id, name, summary, summary_items, nodes, flat_ids, kind="mood")


def _materialize_genre_level(parent_id, members, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids, level, parent_genre=None):
    ordered = _partition_by_genre(
        members, vec_map, score_by_id, genre_subgenres, level, parent_genre
    )
    if not ordered:
        return None
    summary_items = []
    for slug, label, gmembers in _merge_by_slug(ordered):
        gid = f"{parent_id}.g{slug}"
        genre_node = _materialize_genre_folder(
            gid, label, gmembers, vec_map, radii_map, score_by_id,
            mood_centroids, genre_subgenres, nodes, flat_ids, level,
        )
        if genre_node is None:
            continue
        summary_items.append({**genre_node, "items": []})
    return summary_items or None


def _materialize_genre_folder(node_id, label, members, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids, level):
    radii = np.array([radii_map[i] for i in members], dtype=np.float32)
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }
    name = label.title()
    kind = _GENRE_KINDS[level] if 0 <= level < len(_GENRE_KINDS) else "genre"
    genre_data_usable = _genre_subgenres_usable(vec_map, genre_subgenres)

    if genre_data_usable and level == 0:
        sub_items = _materialize_genre_level(
            node_id, members, vec_map, radii_map, score_by_id,
            mood_centroids, genre_subgenres, nodes, flat_ids, level + 1,
            parent_genre=label,
        )
        if sub_items:
            return _branch_folder(node_id, name, summary, sub_items, nodes, flat_ids, kind=kind)
        # No subgenre could form a real cluster (small library): list all of
        # the genre's tracks directly under it instead of the mood fallback.
        return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids, kind=kind)

    # SUBGENRE, or a main genre in the legacy mood fallback (no usable genre
    # file): split into named clusters of ~HYPERBOLIC_TARGET_LEAF_SIZE songs.
    # Clusters below HYPERBOLIC_MIN_CLUSTER_SIZE are pruned; a subgenre left
    # with no valid cluster is hidden, and a folder in the legacy mood fallback
    # lists its tracks as a leaf so tiny single-server moods still browse.
    prefix = _genre_path_prefix(node_id)
    cluster_items = _materialize_children(
        node_id, members, vec_map, radii_map, score_by_id,
        mood_centroids, nodes, flat_ids, name_prefix=prefix or None,
    )
    if cluster_items:
        return _branch_folder(node_id, name, summary, cluster_items, nodes, flat_ids, kind=kind)
    if not genre_data_usable:
        return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids, kind=kind)
    return None


def _genre_path_prefix(node_id):
    parts = []
    for seg in str(node_id).split("."):
        if seg.startswith("g") and len(seg) > 1:
            parts.append(seg[1:].replace("-", "_").upper())
    return "_".join(parts)


def _aggregate_cluster_features(members, score_by_id, voice_vocab):
    mood = {}
    mood_presence = {}
    voice = {}
    voice_presence = {}
    for iid in members:
        info = score_by_id.get(iid)
        if not info:
            continue
        scores = _parse_label_scores(info.get("other_features"))
        for label in config.OTHER_FEATURE_LABELS:
            if label in scores:
                mood[label] = mood.get(label, 0.0) + scores[label]
                mood_presence[label] = mood_presence.get(label, 0) + 1
        for label, val in _parse_label_scores(info.get("mood_vector")).items():
            low = label.lower()
            if low in voice_vocab:
                voice[label] = voice.get(label, 0.0) + val
                voice_presence[label] = voice_presence.get(label, 0) + 1
    return mood, mood_presence, voice, voice_presence


def _cluster_descriptor(members, score_by_id):
    n = max(1, len(members))
    voice_vocab = {v.lower() for v in config.VOICE_VOCAB}
    mood, mood_presence, voice, voice_presence = _aggregate_cluster_features(
        members, score_by_id, voice_vocab
    )
    if mood_presence:
        top_mood = max(mood, key=mood.get)
        top_presence = mood_presence[top_mood]
        if top_presence >= max(2, round(0.6 * n)):
            runner_up = max(
                (v for label, v in mood.items() if label != top_mood),
                default=0.0,
            )
            if mood[top_mood] - runner_up > 1e-9:
                return top_mood.upper()
    if voice_presence:
        top_voice = max(voice, key=voice.get)
        if voice_presence[top_voice] >= max(2, round(0.6 * n)):
            return top_voice.upper().replace(" ", "_")
    return None


def _dedupe_name(base, used):
    if base not in used:
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def _materialize_children(parent_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, name_prefix=None):
    n = len(members)
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))
    min_cluster = max(1, int(config.HYPERBOLIC_MIN_CLUSTER_SIZE))
    if n < min_cluster:
        return None
    k = max(1, round(n / target_leaf))
    k = min(k, n)
    vecs = np.stack([vec_map[i] for i in members]).astype(np.float32)
    labels = _fit_clusters(vecs, k)
    clusters = {}
    for label, iid in zip(labels, members):
        clusters.setdefault(int(label), []).append(iid)
    ordered = [clusters[j] for j in sorted(clusters)]
    # A split was expected (k > 1) but k-means collapsed everything into one
    # giant cluster: the set cannot be meaningfully separated, so bail.
    if k > 1 and max(len(c) for c in ordered) > 0.95 * n:
        return None
    # Prune clusters smaller than the minimum; a folder with no surviving
    # cluster is hidden entirely rather than shown as an empty/giant node.
    kept = [c for c in ordered if len(c) >= min_cluster]
    if not kept:
        return None

    summary_items = []
    used_names = set()
    for ci, cids in enumerate(kept):
        cluster_id = f"{parent_id}.c{ci}"
        if name_prefix:
            descriptor = _cluster_descriptor(cids, score_by_id)
            base = f"{name_prefix}_{descriptor}" if descriptor else name_prefix
            name = _dedupe_name(base, used_names)
            used_names.add(name)
        else:
            name = None
        cluster_node = _materialize_cluster(
            cluster_id, cids, vec_map, radii_map, score_by_id, mood_centroids,
            nodes, flat_ids, name=name,
        )
        summary_items.append({**cluster_node, "items": []})
    return summary_items


def _materialize_cluster(node_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, name=None):
    # Clusters are always the terminal level of the tree (GENRE -> SUBGENRE ->
    # NAMED CLUSTER): every track below them is listed directly, so a cluster
    # never recurses into further folders.
    radii = np.array([radii_map[i] for i in members], dtype=np.float32)
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }
    if name is None:
        name = _cluster_name(members, vec_map, mood_centroids)
    return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids)


def _fit_clusters(vecs, k):
    from tasks.hyperbolic_geometry import poincare_kmeans

    pts = np.asarray(vecs, dtype=np.float32)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    k = max(1, min(int(k), n))
    if k == n:
        return np.arange(n, dtype=np.int32)
    _centroids, labels = poincare_kmeans(pts, k, iterations=5, seed=0)
    return labels


def _load_projected_mood_centroids():
    try:
        with open(config.MOOD_CENTROIDS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.warning(
            "Could not load mood centroids for hyperbolic tree naming from %s",
            config.MOOD_CENTROIDS_FILE,
        )
        return []

    scale = resolve_hyperbolic_scale(auto_calibrate=False)
    if not scale:
        return []

    from tasks.hyperbolic_geometry import project_to_poincare

    out = []
    for mood, info in data.items():
        for c in info.get("centroids", []):
            raw = c.get("centroid")
            tags_by_score = c.get("top_tags") or {}
            if not raw or not tags_by_score:
                continue
            ranked_tags = [t for t, _ in sorted(tags_by_score.items(), key=lambda kv: -kv[1])]
            vec = project_to_poincare(np.asarray(raw, dtype=np.float32), scale)
            out.append({"vec": vec.astype(np.float32), "tags": ranked_tags, "mood": mood})
    return out


def get_projected_genre_subgenres():
    if _GENRE_CENTROID_CACHE["value"] is None:
        _GENRE_CENTROID_CACHE["value"] = _load_projected_genre_subgenres()
    return _GENRE_CENTROID_CACHE["value"]


def _load_projected_genre_subgenres():
    try:
        with open(config.GENRE_SUBGENRE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.warning(
            "Could not load genre/subgenre centroids for hyperbolic tree from %s",
            config.GENRE_SUBGENRE_FILE,
        )
        return {}

    scale = resolve_hyperbolic_scale(auto_calibrate=False)
    if not scale:
        return {}

    from tasks.hyperbolic_geometry import project_to_poincare

    out = {}
    for genre, info in data.items():
        subgenres = info.get("subgenres") or []
        projected = []
        raw_vecs = []
        for s in subgenres:
            raw = s.get("centroid")
            name = s.get("name")
            if not raw or not name:
                continue
            arr = np.asarray(raw, dtype=np.float32)
            vec = project_to_poincare(arr, scale)
            raw_vecs.append(arr.astype(np.float32))
            projected.append({"name": name, "vec": vec.astype(np.float32)})
        if not projected:
            continue
        genre_raw = np.mean(np.stack(raw_vecs), axis=0)
        genre_vec = project_to_poincare(
            genre_raw.astype(np.float32), scale
        ).astype(np.float32)
        out[genre] = {"vec": genre_vec, "subgenres": projected}
    return out


def _cluster_name(members, vec_map, mood_centroids):
    if not mood_centroids:
        return f"Mixed ({len(members)} tracks)"

    from tasks.hyperbolic_geometry import hyperbolic_distance

    mean_vec = np.mean(np.stack([vec_map[i] for i in members]).astype(np.float32), axis=0)
    ranked = sorted(mood_centroids, key=lambda c: hyperbolic_distance(mean_vec, c["vec"]))

    # One representative tag per nearest centroid, walking outward only to
    # avoid an exact duplicate. Two centroids that agree collapse to a tight,
    # confident pairing (e.g. "Pop / Electronic"); two that disagree surface
    # the cluster's genuine specialization as a visibly mixed pairing
    # (e.g. "Jazz / Metal") - the distance ordering IS the specialization
    # signal, with no separate qualifier word needed.
    tags = []
    for c in ranked:
        top_tag = next((t for t in c["tags"] if t not in tags), None)
        if top_tag:
            tags.append(top_tag)
        if len(tags) >= 2:
            break

    label = " / ".join(t.title() for t in tags) if tags else "Mixed"
    return f"{label} ({len(members)} tracks)"


def _track_node(item_id, score_by_id):
    info = score_by_id.get(item_id)
    if info:
        title = info.get("title") or "Unknown"
        author = info.get("author") or "Unknown"
        name = f"{title} - {author}"
    else:
        name = item_id
    return {
        "id": item_id,
        "name": name,
        "type": "track",
        "children_count": 0,
        "items": [],
    }


def _empty_node(node_id, name):
    return {
        "id": node_id or "root",
        "name": name,
        "type": "folder",
        "children_count": 0,
        "summary": {"track_count": 0},
        "items": [],
    }
