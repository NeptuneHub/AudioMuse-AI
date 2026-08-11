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
* hyperbolic_similar over-fetches raw-space IVF candidates and re-ranks them
  by exact Poincare distance, with similar / roots / niche radial mode
  filtering driven by the target track's own radius R
* build_hyperbolic_tree_cache does the expensive part - recursive k-means
  over the whole catalogue - and persists the result as a gzipped JSON blob
  chunked into 50 MB segmented BYTEA rows in ivf_dir (the same pattern the
  music map and IVF directory use), worker-side only (analysis end).
  load_hyperbolic_tree_cache is
  the cheap counterpart: one row read, no reclustering, used at Flask startup
  and by the index-reload NOTIFY handler, exactly like the music map keeps
  its expensive UMAP fit in map_projection_data and only re-does cheap JSON
  assembly in Flask. The request path (build_hyperbolic_tree) is then a pure
  dict lookup against whichever of the two last populated the in-memory
  cache, returning canonical ids for the caller to translate. The returned
  node is a reference into the shared cache, not a copy - callers must build
  a new structure rather than mutate it in place (id translation does this
  already)
* Root radial band count and k-means recursion depth are both derived from
  the catalogue size at build time (_plan_band_count), not fixed, so the tree
  stays browsable whether the library has 500 or 500,000 tracks; folders keep
  splitting until they reach HYPERBOLIC_TARGET_LEAF_SIZE
* Cluster folder names are derived from the dominant genre (and, below the
  first clustering level, the dominant mood) of their member tracks instead
  of a bare "Cluster N" label
"""

import gzip
import json
import logging
import math

import numpy as np

import config

logger = logging.getLogger(__name__)

# The tree cache is a gzipped JSON blob stored as segmented BYTEA rows
# (IVF_MAX_PART_SIZE_MB-sized chunks: "hyperbolic_tree_cache_1_3", ...) in
# the same ivf_dir table the other indexes use. That keeps it under
# Postgres' row-size limit like every other index and makes it immune to the
# web-startup app_config prune (which only ever touches the app_config
# table), so analysis persists it and Flask loads it back at startup.
_TREE_CACHE_TABLE = "ivf_dir"
_TREE_CACHE_BLOB_NAME = "hyperbolic_tree_cache"

_FALLBACK_SCALE = 1.0
_SCALE_CACHE = {"value": None}


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
            norm_batches.append(np.linalg.norm(batch.astype(np.float64), axis=1))
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
    from app_helper import get_db

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
    from app_helper import get_db

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


def _fetch_all_poincare_rows():
    from app_helper import get_db

    out = {}
    skipped = 0
    db_conn = get_db()
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT item_id, poincare_embedding, hyperbolic_radius "
                "FROM embedding WHERE poincare_embedding IS NOT NULL "
                "AND hyperbolic_radius IS NOT NULL "
                "ORDER BY item_id"
            )
            for item_id, blob, radius in cur.fetchall():
                vec = np.frombuffer(bytes(blob), dtype=np.float32)
                if _is_finite_row(vec, radius):
                    out[item_id] = (vec, float(radius))
                else:
                    skipped += 1
    except Exception:
        logger.exception("Could not fetch hyperbolic rows")
    if skipped:
        logger.warning(
            "Skipped %d hyperbolic row(s) with a non-finite radius or vector while "
            "building the tree cache.",
            skipped,
        )
    return out


def hyperbolic_similar(target_item_id, mode="similar", limit=20):
    from tasks.hyperbolic_geometry import hyperbolic_distances_to
    from tasks.ivf_manager import find_nearest_neighbors_by_id

    mode = (mode or "similar").strip().lower()
    if mode not in ("similar", "roots", "niche"):
        raise ValueError('mode must be one of "similar", "roots", "niche"')
    limit = max(1, min(int(limit), int(config.HYPERBOLIC_MAX_LIMIT)))
    target = _fetch_poincare_rows([target_item_id]).get(target_item_id)
    if target is None:
        raise ValueError(
            "Target track has no hyperbolic projection; run the hyperbolic backfill job first"
        )
    target_vec, target_radius = target
    overfetch = max(
        int(limit) * int(config.HYPERBOLIC_CANDIDATE_OVERFETCH), int(limit) + 50
    )
    candidates = find_nearest_neighbors_by_id(
        target_item_id,
        n=overfetch,
        eliminate_duplicates=False,
        mood_similarity=False,
        radius_similarity=False,
    )
    cand_ids = [c["item_id"] for c in candidates if c.get("item_id")]
    rows = _fetch_poincare_rows(cand_ids)
    if not rows:
        return []
    ids = list(rows.keys())
    cand_vecs = np.stack([rows[i][0] for i in ids]).astype(np.float64)
    cand_radii = np.array([rows[i][1] for i in ids], dtype=np.float64)
    distances = hyperbolic_distances_to(target_vec, cand_vecs)
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
    return results[:limit]


_TREE_CACHE = {"n_bands": None, "nodes": None, "flat_ids": None, "track_count": None}


def reset_hyperbolic_tree_cache():
    _TREE_CACHE["n_bands"] = None
    _TREE_CACHE["nodes"] = None
    _TREE_CACHE["flat_ids"] = None
    _TREE_CACHE["track_count"] = None


def _plan_band_count(total_tracks):
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))
    ratio = max(int(total_tracks), 1) / target_leaf
    raw = round(math.log2(ratio)) if ratio > 0 else int(config.HYPERBOLIC_MIN_BANDS)
    return int(min(
        int(config.HYPERBOLIC_MAX_BANDS),
        max(int(config.HYPERBOLIC_MIN_BANDS), raw),
    ))


def build_hyperbolic_tree_cache(n_bands=None):
    rows = _fetch_all_poincare_rows()
    if not rows:
        _TREE_CACHE["n_bands"] = 0
        _TREE_CACHE["nodes"] = {}
        _TREE_CACHE["flat_ids"] = {}
        _TREE_CACHE["track_count"] = 0
        _persist_tree_cache_blob(None)
        logger.info("Hyperbolic tree cache built empty: no projected tracks yet.")
        return 0

    resolved_bands = int(n_bands) if n_bands else _plan_band_count(len(rows))
    from app_helper import get_score_data_by_ids

    score_by_id = {d["item_id"]: d for d in get_score_data_by_ids(list(rows.keys()))}
    mood_centroids = _load_projected_mood_centroids()
    nodes, flat_ids = _build_tree_nodes(rows, resolved_bands, score_by_id, mood_centroids)
    track_count = len(rows)
    _TREE_CACHE["n_bands"] = resolved_bands
    _TREE_CACHE["nodes"] = nodes
    _TREE_CACHE["flat_ids"] = flat_ids
    _TREE_CACHE["track_count"] = track_count
    _persist_tree_cache_blob({
        "n_bands": resolved_bands, "nodes": nodes, "flat_ids": flat_ids, "track_count": track_count,
    })
    logger.info(
        "Hyperbolic tree cache built and persisted: %d tracks across %d nodes (%d root bands)",
        track_count, len(nodes), resolved_bands,
    )
    return track_count


def load_hyperbolic_tree_cache():
    payload = _load_tree_cache_blob()
    if payload is None:
        _TREE_CACHE["n_bands"] = 0
        _TREE_CACHE["nodes"] = {}
        _TREE_CACHE["flat_ids"] = {}
        _TREE_CACHE["track_count"] = 0
        logger.info("Hyperbolic tree cache empty: nothing persisted yet (run analysis first).")
        return 0

    _TREE_CACHE["n_bands"] = payload.get("n_bands")
    _TREE_CACHE["nodes"] = payload.get("nodes") or {}
    _TREE_CACHE["flat_ids"] = payload.get("flat_ids") or {}
    track_count = int(payload.get("track_count") or 0)
    _TREE_CACHE["track_count"] = track_count
    logger.info(
        "Hyperbolic tree cache loaded from ivf_dir: %d tracks across %d nodes.",
        track_count, len(_TREE_CACHE["nodes"]),
    )
    return track_count


def _delete_tree_cache_blob():
    from app_helper import get_db

    db_conn = get_db()
    like_pattern = _TREE_CACHE_BLOB_NAME.replace("_", r"\_") + r"\_%\_%"
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ivf_dir WHERE name = %s OR name LIKE %s ESCAPE '\\'",
            (_TREE_CACHE_BLOB_NAME, like_pattern),
        )
    db_conn.commit()


def _persist_tree_cache_blob(payload):
    # Deliberately not wrapped in try/except: a persist failure here must
    # propagate to the worker step (_run_all_index_builds catches it,
    # records it through error_manager, and the run continues since this
    # step is non-fatal) rather than being swallowed into a log line nobody
    # reads while the caller still reports "built and persisted" as if it
    # worked.
    from app_helper import get_db
    from tasks.index_build_helpers import store_segmented_blob

    if payload is None:
        _delete_tree_cache_blob()
        return
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    db_conn = get_db()
    # store_segmented_blob clears any stale rows first, then stores the blob
    # as one row or as IVF_MAX_PART_SIZE_MB (50 MB) "name_i_n" rows.
    store_segmented_blob(db_conn, _TREE_CACHE_TABLE, _TREE_CACHE_BLOB_NAME, gzip.compress(raw))
    db_conn.commit()


def _load_tree_cache_blob():
    try:
        from app_helper import get_db
        from tasks.index_build_helpers import load_segmented_blob

        blob = load_segmented_blob(get_db(), _TREE_CACHE_TABLE, _TREE_CACHE_BLOB_NAME)
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


def build_hyperbolic_tree(node_id=None, depth=None):
    nodes = _TREE_CACHE["nodes"]
    if not nodes:
        return _empty_node(node_id, "Hyperbolic Explorer"), []
    key = (node_id or "root").strip() or "root"
    node = nodes.get(key)
    if node is None:
        raise ValueError(f"Unknown tree node id: {node_id}")
    return node, _TREE_CACHE["flat_ids"].get(key, [])


def _build_tree_nodes(rows, n_bands, score_by_id, mood_centroids):
    from tasks.hyperbolic_geometry import assign_radial_bands, split_radial_bands

    item_ids = list(rows.keys())
    radii_map = {iid: rows[iid][1] for iid in item_ids}
    vec_map = {iid: rows[iid][0] for iid in item_ids}
    radii = np.array([radii_map[i] for i in item_ids], dtype=np.float64)
    boundaries = split_radial_bands(radii, max(1, int(n_bands)))
    band_assign = assign_radial_bands(radii, boundaries)
    band_members = {bi: [] for bi in range(len(boundaries))}
    for iid, bi in zip(item_ids, band_assign):
        band_members[int(bi)].append(iid)

    nodes = {}
    flat_ids = {}
    root_items = []
    for bi in sorted(band_members):
        members = band_members[bi]
        if not members:
            continue
        band_node = _materialize_band(
            bi, boundaries, members, radii_map, vec_map, score_by_id, mood_centroids, nodes, flat_ids
        )
        root_items.append(band_node)

    nodes["root"] = {
        "id": "root",
        "name": "Hyperbolic Explorer",
        "type": "folder",
        "children_count": len(root_items),
        "summary": {"track_count": sum(len(m) for m in band_members.values())},
        "items": root_items,
    }
    flat_ids["root"] = []
    return nodes, flat_ids


def _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids):
    items = [_track_node(i, score_by_id) for i in members]
    node = {
        "id": node_id, "name": name, "type": "folder", "leaf": True,
        "children_count": len(items), "summary": summary, "items": items,
    }
    nodes[node_id] = node
    flat_ids[node_id] = list(members)
    return node


def _branch_folder(node_id, name, summary, summary_items, nodes, flat_ids):
    node = {
        "id": node_id, "name": name, "type": "folder", "leaf": False,
        "children_count": len(summary_items), "summary": summary, "items": summary_items,
    }
    nodes[node_id] = node
    # Non-leaf folders never carry tracks directly - nothing here needs
    # per-server id translation until the caller drills into an actual leaf.
    flat_ids[node_id] = []
    return node


def _materialize_band(band_index, boundaries, members, radii_map, vec_map, score_by_id, mood_centroids, nodes, flat_ids):
    lo, hi = boundaries[band_index]
    radii = np.array([radii_map[i] for i in members], dtype=np.float64)
    node_id = f"b{band_index}"
    name = f"Band {band_index + 1} (radius {lo:.3f} - {hi:.3f})"
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }

    summary_items = None
    if len(members) > int(config.HYPERBOLIC_TARGET_LEAF_SIZE):
        summary_items = _materialize_children(
            node_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level=1
        )
    if summary_items is None:
        return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids)
    return _branch_folder(node_id, name, summary, summary_items, nodes, flat_ids)


def _materialize_children(parent_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level):
    n = len(members)
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))
    branching = max(2, int(config.HYPERBOLIC_TARGET_BRANCHING))
    k = min(branching, max(2, round(n / target_leaf)))
    vecs = np.stack([vec_map[i] for i in members]).astype(np.float64)
    labels = _fit_clusters(vecs, k)
    clusters = {}
    for label, iid in zip(labels, members):
        clusters.setdefault(int(label), []).append(iid)
    ordered = [clusters[j] for j in sorted(clusters)]
    if len(ordered) < 2 or max(len(c) for c in ordered) > 0.95 * n:
        # k-means could not meaningfully separate this set (e.g. near-identical
        # embeddings) - stop here rather than recurse without making progress.
        return None

    summary_items = []
    for ci, cids in enumerate(ordered):
        cluster_id = f"{parent_id}.c{ci}"
        cluster_node = _materialize_cluster(
            cluster_id, cids, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level
        )
        summary_items.append({**cluster_node, "items": []})
    return summary_items


def _materialize_cluster(node_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level):
    radii = np.array([radii_map[i] for i in members], dtype=np.float64)
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }
    name = _cluster_name(members, vec_map, mood_centroids)

    summary_items = None
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))
    if len(members) > target_leaf and level < int(config.HYPERBOLIC_MAX_TREE_RECURSION):
        summary_items = _materialize_children(
            node_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level + 1
        )
    if summary_items is None:
        return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids)
    return _branch_folder(node_id, name, summary, summary_items, nodes, flat_ids)


def _fit_clusters(vecs, k):
    if vecs.shape[0] > 5000:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=k, batch_size=1000, n_init=3, max_iter=100, random_state=0
        )
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=0)
    return km.fit_predict(vecs)


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
            out.append({"vec": vec.astype(np.float64), "tags": ranked_tags, "mood": mood})
    return out


def _cluster_name(members, vec_map, mood_centroids):
    if not mood_centroids:
        return f"Mixed ({len(members)} tracks)"

    from tasks.hyperbolic_geometry import hyperbolic_distance

    mean_vec = np.mean(np.stack([vec_map[i] for i in members]).astype(np.float64), axis=0)
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
