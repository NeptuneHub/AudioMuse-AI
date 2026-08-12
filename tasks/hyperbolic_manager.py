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
* build_hyperbolic_tree_cache does the expensive part - the mood partition,
  the main/second/third-genre grouping, and any k-means fallback over the
  whole catalogue - and persists the result as a gzipped JSON blob chunked
  into 50 MB segmented BYTEA rows in ivf_dir (the same pattern the music map
  and IVF directory use), worker-side only (analysis end).
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
* The directory tree is a semantic taxonomy over the same embedding space:
  the root splits by MAIN GENRE taken from the data-driven genre_subgenre.json
  centroids (nearest main genre at level 0), then by SUBGENRE (nearest of that
  genre's subgenres at level 1). When the file is absent or dimensionally
  incompatible the tree falls back to a legacy MOOD partition (nearest of the
  precomputed mood centroids in mood_centroids_real_080_clap.json) followed by
  main/second/third genre from each track's mood_vector. Folders keep
  splitting until they reach HYPERBOLIC_TARGET_LEAF_SIZE, and groups with no
  further genre (or still too large after the genre levels) fall back to
  k-means sub-folders named from the nearest mood-centroid tags instead of a
  bare "Cluster N" label
"""

import gzip
import json
import logging
import math
import re

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
    # n_bands is advisory and kept for API compatibility: the root is now a
    # mood partition, so the reported root count comes from the built tree.
    rows = _fetch_all_poincare_rows()
    if not rows:
        _TREE_CACHE["n_bands"] = 0
        _TREE_CACHE["nodes"] = {}
        _TREE_CACHE["flat_ids"] = {}
        _TREE_CACHE["track_count"] = 0
        _persist_tree_cache_blob(None)
        logger.info("Hyperbolic tree cache built empty: no projected tracks yet.")
        return 0

    from app_helper import get_score_data_by_ids

    score_by_id = {d["item_id"]: d for d in get_score_data_by_ids(list(rows.keys()))}
    mood_centroids = _load_projected_mood_centroids()
    genre_subgenres = _load_projected_genre_subgenres()
    nodes, flat_ids = _build_tree_nodes(rows, score_by_id, mood_centroids, genre_subgenres)
    root_count = len(nodes["root"]["items"])
    track_count = len(rows)
    _TREE_CACHE["n_bands"] = root_count
    _TREE_CACHE["nodes"] = nodes
    _TREE_CACHE["flat_ids"] = flat_ids
    _TREE_CACHE["track_count"] = track_count
    _persist_tree_cache_blob({
        "n_bands": root_count, "nodes": nodes, "flat_ids": flat_ids, "track_count": track_count,
    })
    logger.info(
        "Hyperbolic tree cache built and persisted: %d tracks across %d nodes (%d root moods)",
        track_count, len(nodes), root_count,
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


def _build_tree_nodes(rows, score_by_id, mood_centroids, genre_subgenres):
    item_ids = list(rows.keys())
    vec_map = {iid: rows[iid][0] for iid in item_ids}
    radii_map = {iid: rows[iid][1] for iid in item_ids}

    nodes = {}
    flat_ids = {}
    root_items = []
    # The root is a MAIN GENRE partition when genre_subgenre.json centroids
    # are dimensionally usable: every track is assigned to the nearest main
    # genre, and each genre folder then splits into its subgenres. Without
    # usable genre data we fall back to the legacy mood partition (nearest
    # precomputed mood centroid, then dominant CLAP mood, then a single
    # "General" bucket) so the tree still renders.
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
                root_items.append(genre_node)
    if not root_items:
        for mood_label, members in _partition_by_mood(item_ids, vec_map, mood_centroids, score_by_id):
            if not members:
                continue
            mood_node = _materialize_mood(
                mood_label, members, vec_map, radii_map, score_by_id,
                mood_centroids, genre_subgenres, nodes, flat_ids,
            )
            root_items.append(mood_node)

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
    # Sibling labels that slugify identically ("Progressive rock" vs
    # "progressive rock", both present in the shipped genre_subgenre.json)
    # would share a node id, and the second _leaf_folder/_branch_folder write
    # would silently overwrite the first in nodes/flat_ids, orphaning its
    # tracks while the parent still counts them - merge such groups instead.
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
    # Ordered genre labels for a track (STRATIFIED_GENRES only), highest score
    # first; index 0 is the main genre, 1 the second, 2 the third.
    if not info:
        return []
    scores = _parse_label_scores(info.get("mood_vector"))
    genres = [g for g in config.STRATIFIED_GENRES if g in scores]
    genres.sort(key=lambda g: -scores[g])
    return genres


def _partition_by_mood(item_ids, vec_map, mood_centroids, score_by_id):
    assigns = {}
    if mood_centroids:
        # One vectorized distance matrix over the whole catalogue instead of
        # a per-track scalar loop; this ran over ~1M scalar calls before.
        from tasks.hyperbolic_geometry import hyperbolic_distance_matrix

        vecs = np.stack([vec_map[i] for i in item_ids]).astype(np.float64)
        cent = np.stack([c["vec"] for c in mood_centroids]).astype(np.float64)
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
    # The genre_subgenre.json centroids only drive the tree when they live in
    # the same embedding dimension as the projected vectors (a mismatch means
    # the file belongs to a different model or library).
    if not genre_subgenres or not vec_map:
        return False
    ref = next(iter(vec_map.values()))
    info = next(iter(genre_subgenres.values()))
    return bool(info) and info["vec"].shape[0] == ref.shape[0]


def _partition_by_genre(members, vec_map, score_by_id, genre_subgenres, level, parent_genre=None):
    # Data-driven genre taxonomy from genre_subgenre.json when its centroids
    # are dimensionally usable. Level 0 assigns every track to the nearest
    # main genre; level 1, within a main genre, to the nearest of that genre's
    # subgenres. There is no deeper genre data, so the caller falls back to
    # k-means for oversized groups - never to the mixed mood_vector genre
    # ranking.
    if _genre_subgenres_usable(vec_map, genre_subgenres):
        return _partition_by_genre_centroids(
            members, vec_map, genre_subgenres, level, parent_genre
        )
    # Fallback (no usable genre_subgenre.json): ordered mood_vector genre
    # labels (STRATIFIED_GENRES), highest score first, so index 0 is main,
    # 1 second, 2 third genre.
    groups = {}
    for iid in members:
        rank = _genre_rank(score_by_id.get(iid))
        label = rank[level] if level < len(rank) else "Other"
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
        # Only one genre level (and one subgenre level) is encoded in
        # genre_subgenre.json, so deeper genre splits are k-means.
        return None
    if len(centroids) < 2 or centroids[0]["vec"].shape[0] != ref.shape[0]:
        return None
    vecs = np.stack([vec_map[i] for i in members]).astype(np.float64)
    cent = np.stack([c["vec"] for c in centroids]).astype(np.float64)
    best = np.argmin(hyperbolic_distance_matrix(vecs, cent), axis=1)
    groups = {}
    for iid, idx in zip(members, best):
        groups.setdefault(centroids[int(idx)]["name"], []).append(iid)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    if len(ordered) < 2:
        return None
    return ordered


# Genre folder kinds in the data-driven tree (genre_subgenre.json): level 0
# is the main genre, level 1 the subgenre. The legacy mood_vector path still
# uses main/second/third names for its three STRATIFIED_GENRES levels.
_GENRE_KINDS = ("main_genre", "subgenre")
_LEGACY_GENRE_KINDS = ("main_genre", "second_genre", "third_genre")


def _materialize_mood(mood_label, members, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids):
    node_id = f"m{_slugify(mood_label)}"
    name = mood_label.title()
    radii = np.array([radii_map[i] for i in members], dtype=np.float64)
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
            mood_centroids, nodes, flat_ids, level=1,
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
        summary_items.append({**genre_node, "items": []})
    return summary_items


def _materialize_genre_folder(node_id, label, members, vec_map, radii_map, score_by_id, mood_centroids, genre_subgenres, nodes, flat_ids, level):
    radii = np.array([radii_map[i] for i in members], dtype=np.float64)
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }
    name = label.title()
    if _genre_subgenres_usable(vec_map, genre_subgenres):
        kinds = _GENRE_KINDS
    else:
        kinds = _LEGACY_GENRE_KINDS
    kind = kinds[level] if 0 <= level < len(kinds) else "genre"
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))
    genre_depth = max(1, int(config.HYPERBOLIC_GENRE_DEPTH))

    summary_items = None
    if len(members) > target_leaf and label != "Other" and level + 1 < genre_depth:
        summary_items = _materialize_genre_level(
            node_id, members, vec_map, radii_map, score_by_id,
            mood_centroids, genre_subgenres, nodes, flat_ids, level + 1,
            parent_genre=label,
        )
    if summary_items is None and len(members) > target_leaf:
        # Genre depth exhausted (or no meaningful genre split): fall back to
        # k-means so oversized groups stay browsable. Clusters under a
        # genre/subgenre folder are named from their ancestors plus the
        # dominant mood/voice (e.g. ROCK_PROGRESSIVE_ROCK_HAPPY), never from
        # the mood-centroid genre pairing.
        prefix = _genre_path_prefix(node_id)
        summary_items = _materialize_children(
            node_id, members, vec_map, radii_map, score_by_id,
            mood_centroids, nodes, flat_ids, level=1,
            name_prefix=prefix or None,
        )
    if summary_items is None:
        return _leaf_folder(node_id, name, members, summary, score_by_id, nodes, flat_ids, kind=kind)
    return _branch_folder(node_id, name, summary, summary_items, nodes, flat_ids, kind=kind)


def _genre_path_prefix(node_id):
    # Build the uppercase underscore parent path from a node id so a k-means
    # cluster under a genre/subgenre folder is named from its ancestors,
    # e.g. root.grock.gprogressive-rock.c0 -> "ROCK_PROGRESSIVE_ROCK".
    parts = []
    for seg in str(node_id).split("."):
        if seg.startswith("g") and len(seg) > 1:
            parts.append(seg[1:].replace("-", "_").upper())
    return "_".join(parts)


def _cluster_descriptor(members, score_by_id):
    # The mood/voice that most represents a cluster. The PRIMARY descriptor is
    # the dominant CLAP mood from OTHER_FEATURE_LABELS (danceable / aggressive /
    # happy / party / relaxed / sad) - the label with the highest accumulated
    # value across the cluster. A VOICE_VOCAB tag (female/male vocalists) is
    # appended only as a fallback, when no single mood is confident AND the
    # voice tag genuinely dominates the cluster. A label only names the folder
    # when it is truly representative - never invented; returns None so the
    # caller falls back to a numbered ancestor-path name instead.
    n = max(1, len(members))
    mood = {}
    mood_presence = {}
    voice = {}
    voice_presence = {}
    voice_vocab = {v.lower() for v in config.VOICE_VOCAB}
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
    # Dominant CLAP mood first: the label with the highest accumulated value,
    # present on a clear majority of the cluster. A tie (two moods within a
    # tiny epsilon) or a split cluster gets no fabricated label - the caller
    # falls back to a numbered ancestor-path name instead.
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
    # Voice as fallback only: a VOICE_VOCAB tag must dominate a clear majority
    # of the cluster, and there must be no confident mood to describe it with.
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


def _materialize_children(parent_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level, name_prefix=None):
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
    used_names = set()
    for ci, cids in enumerate(ordered):
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
            nodes, flat_ids, level, name=name, name_prefix=name_prefix,
        )
        summary_items.append({**cluster_node, "items": []})
    return summary_items


def _materialize_cluster(node_id, members, vec_map, radii_map, score_by_id, mood_centroids, nodes, flat_ids, level, name=None, name_prefix=None):
    radii = np.array([radii_map[i] for i in members], dtype=np.float64)
    summary = {
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "track_count": len(members),
    }
    if name is None:
        name = _cluster_name(members, vec_map, mood_centroids)

    summary_items = None
    target_leaf = max(1, int(config.HYPERBOLIC_TARGET_LEAF_SIZE))
    # In the data-driven genre tree (name_prefix set) a cluster is a terminal
    # leaf: the taxonomy is GENRE -> SUBGENRE -> CLUSTERS and never deeper. The
    # legacy mood path may keep splitting oversized clusters when it has no
    # genre data to lean on.
    if name_prefix is None and len(members) > target_leaf and level < int(config.HYPERBOLIC_MAX_TREE_RECURSION):
        summary_items = _materialize_children(
            node_id, members, vec_map, radii_map, score_by_id, mood_centroids,
            nodes, flat_ids, level + 1, name_prefix=name_prefix,
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
            raw_vecs.append(arr.astype(np.float64))
            projected.append({"name": name, "vec": vec.astype(np.float64)})
        if not projected:
            continue
        # Main genre representative: mean of the RAW subgenre centroids,
        # projected afterwards. Averaging already-projected points shrinks
        # the representative toward the origin in proportion to the genre's
        # spread, and the hyperbolic metric then keeps every large-radius
        # track away from it - the broadest genres (rock, pop) would almost
        # never win the nearest-centroid root partition.
        genre_raw = np.mean(np.stack(raw_vecs), axis=0)
        genre_vec = project_to_poincare(
            genre_raw.astype(np.float32), scale
        ).astype(np.float64)
        out[genre] = {"vec": genre_vec, "subgenres": projected}
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
