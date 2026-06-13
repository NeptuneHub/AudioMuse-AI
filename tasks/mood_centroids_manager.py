# tasks/mood_centroids_manager.py
"""Per-backend mood centroids builder.

Replaces the static ``mood_centroids_real_080_clap.json`` (frozen
200-dim MusiCNN vectors) with a build-time derivation against whichever
``SONIC_BACKEND`` is currently active. Centroids are stored in the
``mood_centroids_data`` table keyed by ``(backend, mood)`` so multiple
backends can coexist alongside their respective embeddings.

Pipeline position: invoked from
:func:`tasks.analysis._run_all_index_builds` right after the Voyager
index rebuild, before the projection builds. That ordering means
centroids always reflect the embedding distribution of the same
analysis run that produced the current Voyager index.

Reader contract: ``load_mood_centroids(backend)`` returns the same
``{mood: {best_k, bic_values, centroids: [...]}}`` shape the legacy JSON
exposed, so existing consumers in ``app_voyager``, ``app_path`` and
``tasks.song_alchemy`` are a one-line swap.

Algorithm: for each mood in ``OTHER_FEATURE_LABELS``, pull every
analyzed track whose CLAP-derived ``score.other_features`` score for
that mood crosses ``MOOD_CENTROIDS_MIN_SCORE``; fetch the
active-backend embeddings for those tracks; run KMeans with
``MOOD_CENTROIDS_K`` clusters (capped at the number of qualifying
tracks); persist one row per (backend, mood) carrying every cluster's
centroid vector, member count, average mood score, and top-tag
representative pulled from the cluster members' ``mood_vector`` strings.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_score_dict(raw: Optional[str]) -> Dict[str, float]:
    """Parse a 'label1:0.42,label2:0.55' string into ``{label: score}``."""
    out: Dict[str, float] = {}
    if not raw:
        return out
    for part in raw.split(','):
        k, _, v = part.partition(':')
        k = k.strip()
        if not k:
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _fetch_tracks_for_mood(cur, mood: str, backend: str, min_score: float):
    """Return ``[(item_id, embedding_bytes, mood_score, mood_vector)]``."""
    cur.execute(
        """
        SELECT s.item_id, e.embedding, s.other_features, s.mood_vector
        FROM score s
        JOIN embedding e ON s.item_id = e.item_id AND e.backend = %s
        WHERE s.other_features IS NOT NULL AND e.embedding IS NOT NULL
        """,
        (backend,),
    )
    out = []
    for item_id, blob, other_features, mood_vector in cur.fetchall():
        scores = _parse_score_dict(other_features)
        score = scores.get(mood, 0.0)
        if score >= min_score:
            out.append((item_id, blob, score, mood_vector))
    return out


def _top_tags(mood_vectors: List[str], top_n: int = 5) -> List[str]:
    """Return the most common top-1 mood label across a cluster's tracks."""
    counter: Counter = Counter()
    for mv in mood_vectors:
        d = _parse_score_dict(mv)
        if not d:
            continue
        # Pick the single highest-scored label in this track's mood_vector
        # as the "representative" tag, then count across the cluster.
        best = max(d.items(), key=lambda kv: kv[1])
        counter[best[0]] += 1
    return [label for label, _ in counter.most_common(top_n)]


def _cluster_one_mood(
    rows: List[tuple], embedding_dim: int, k_target: int,
) -> List[Dict[str, Any]]:
    """KMeans-cluster a single mood's embeddings into centroids."""
    if not rows:
        return []
    # Decode embeddings + drop dimension-mismatched rows defensively. The
    # rest of the pipeline (Voyager build) already does the same check,
    # but a stale row can sneak through if the Voyager rebuild hasn't
    # run yet against the current backend.
    vectors: List[np.ndarray] = []
    mood_scores: List[float] = []
    mood_vectors: List[str] = []
    item_ids: List[str] = []
    skipped_dim = 0
    for item_id, blob, score, mv in rows:
        if blob is None or len(blob) != embedding_dim * 4:
            skipped_dim += 1
            continue
        vectors.append(np.frombuffer(blob, dtype=np.float32))
        mood_scores.append(score)
        mood_vectors.append(mv or "")
        item_ids.append(item_id)
    if skipped_dim:
        logger.info("mood_centroids: skipped %d rows with mismatched embedding dim", skipped_dim)
    if not vectors:
        return []

    X = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    k = max(1, min(int(k_target), X.shape[0]))

    # sklearn KMeans is already a transitive dependency (umap-learn /
    # scikit-learn); import inline so module import is still cheap.
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_.astype(np.float32, copy=False)

    out: List[Dict[str, Any]] = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        n_songs = int(mask.sum())
        if n_songs == 0:
            continue
        cluster_mood_scores = [mood_scores[i] for i in range(len(mood_scores)) if mask[i]]
        cluster_mood_vecs = [mood_vectors[i] for i in range(len(mood_vectors)) if mask[i]]
        out.append({
            "cluster_id": cluster_id,
            "n_songs": n_songs,
            "mood_score": float(np.mean(cluster_mood_scores)) if cluster_mood_scores else 0.0,
            "centroid": centroids[cluster_id].tolist(),
            "top_tags": _top_tags(cluster_mood_vecs),
        })
    # Sort by cluster size descending so cluster #0 is always the
    # largest "core" group for the mood — the UI surfaces them in this
    # order in the centroid picker dropdown.
    out.sort(key=lambda c: -c["n_songs"])
    for new_id, entry in enumerate(out):
        entry["cluster_id"] = new_id
    return out


def build_and_store_mood_centroids(db_conn=None) -> bool:
    """Recompute and persist mood centroids for the active backend.

    Reads embeddings from the active backend, clusters per
    ``OTHER_FEATURE_LABELS``, writes one ``mood_centroids_data`` row per
    (active_backend, mood). Returns True on success even when some
    moods had no qualifying tracks (those rows are deleted so stale
    centroids don't linger).
    """
    from config import (
        EMBEDDING_DIMENSION, OTHER_FEATURE_LABELS,
        MOOD_CENTROIDS_K, MOOD_CENTROIDS_MIN_SCORE,
    )
    from .sonic_backends import active_backend_name

    backend = active_backend_name()
    if db_conn is None:
        from app_helper import get_db
        db_conn = get_db()

    logger.info(
        "Building mood centroids for backend=%s dim=%d (k=%d, min_score=%.2f)",
        backend, EMBEDDING_DIMENSION, MOOD_CENTROIDS_K, MOOD_CENTROIDS_MIN_SCORE,
    )

    with db_conn.cursor() as cur:
        for mood in OTHER_FEATURE_LABELS:
            rows = _fetch_tracks_for_mood(cur, mood, backend, MOOD_CENTROIDS_MIN_SCORE)
            centroids = _cluster_one_mood(rows, EMBEDDING_DIMENSION, MOOD_CENTROIDS_K)
            payload = {
                "best_k": len(centroids),
                "bic_values": None,  # informational; original JSON had BIC sweep
                "centroids": centroids,
            }
            if centroids:
                cur.execute(
                    """
                    INSERT INTO mood_centroids_data (backend, mood, centroids, embedding_dim)
                    VALUES (%s, %s, %s::jsonb, %s)
                    ON CONFLICT (backend, mood) DO UPDATE SET
                        centroids = EXCLUDED.centroids,
                        embedding_dim = EXCLUDED.embedding_dim,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (backend, mood, json.dumps(payload), EMBEDDING_DIMENSION),
                )
                logger.info(
                    "  ✓ mood='%s' clusters=%d (sizes %s)",
                    mood, len(centroids),
                    [c["n_songs"] for c in centroids],
                )
            else:
                cur.execute(
                    "DELETE FROM mood_centroids_data WHERE backend = %s AND mood = %s",
                    (backend, mood),
                )
                logger.info("  ✗ mood='%s' no qualifying tracks; cleared stale row", mood)
        db_conn.commit()
    return True


def load_mood_centroids(backend: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return ``{mood: {best_k, bic_values, centroids: [...]}}`` for backend.

    When the per-backend DB rows are empty AND ``backend == 'musicnn'``,
    fall back to the legacy bundled JSON so a fresh install with no
    analysis yet still produces a usable Similarity / Alchemy page.
    """
    from .sonic_backends import active_backend_name
    from app_helper import get_db

    backend = backend or active_backend_name()
    result: Dict[str, Dict[str, Any]] = {}
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT mood, centroids FROM mood_centroids_data WHERE backend = %s",
                (backend,),
            )
            for mood, payload in cur.fetchall():
                if isinstance(payload, str):
                    payload = json.loads(payload)
                result[mood] = payload
    except Exception as e:
        logger.warning("Failed to read mood_centroids_data for backend %r: %s", backend, e)

    if result:
        return result

    # First-boot fallback: ship the legacy 200-dim MusiCNN JSON so the
    # UI isn't empty before the first analysis run completes.
    if backend == "musicnn":
        try:
            from config import MOOD_CENTROIDS_FILE
            with open(MOOD_CENTROIDS_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.info("Legacy mood centroids JSON unavailable: %s", e)
    return {}
