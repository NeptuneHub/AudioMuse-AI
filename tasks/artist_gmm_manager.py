# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Artist-similarity manager: per-artist Gaussian mixture models and their index.

Models each artist as a Gaussian mixture over its tracks' embeddings, then serves
artist-to-artist similarity. Builds and persists the artist index (parallel to the
per-song IVF indexes), loads it for querying, and answers the similar-artists and
artist-search endpoints; also used by tasks.song_alchemy for artist anchors.

Main Features:
* fit_artist_gmm / select_optimal_gmm_components: fit a diagonal-covariance GMM per
  artist, auto-selecting component count within configured bounds.
* gmm_soft_chamfer_distance: soft-Chamfer distance over component means for
  artist-vs-artist scoring, with a lazily loaded, force-reloadable index cache.
* find_similar_artists / search_artists_by_name / get_artist_tracks: query surface.
"""

import logging
import hashlib
import numpy as np
import os
import threading
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture

from config import INDEX_BUILD_WORKERS

logger = logging.getLogger(__name__)

ARTIST_INDEX_NAME = 'artist_similarity_index'
GMM_N_COMPONENTS_MIN = 2
GMM_N_COMPONENTS_MAX = 10
GMM_COVARIANCE_TYPE = 'diag'
GMM_MAX_ITER = 100
GMM_N_INIT = 3
MIN_TRACKS_PER_ARTIST = 1
_FETCH_TRACKS_PER_BATCH = 20000
_MIN_ARTISTS_FOR_POOL = 32

artist_index = None
artist_map = None
reverse_artist_map = None
artist_gmm_params = None
_index_lock = threading.Lock()


def select_optimal_gmm_components(
    embeddings: np.ndarray,
    min_components: int = GMM_N_COMPONENTS_MIN,
    max_components: int = GMM_N_COMPONENTS_MAX,
) -> int:
    return fit_best_gmm(embeddings, min_components, max_components)[0]


def fit_best_gmm(
    embeddings: np.ndarray,
    min_components: int = GMM_N_COMPONENTS_MIN,
    max_components: int = GMM_N_COMPONENTS_MAX,
) -> Tuple[int, Optional[GaussianMixture]]:
    """The best-BIC component count AND the model fitted with it.

    The winner is handed back so the caller does not refit it: the sweep already
    fitted it on this data with these parameters and this random_state, so a
    refit is one more restarted EM run for numbers we have.
    """
    n_samples = len(embeddings)

    if n_samples == 1:
        return 1, None

    if n_samples <= 5:
        max_feasible = min(n_samples, max_components)
    else:
        max_feasible = max(min_components, min(max_components, n_samples // 5))

    if max_feasible < min_components:
        max_feasible = min(min_components, n_samples)

    if max_feasible < 1:
        return 1, None

    best_bic = float('inf')
    best_n_components = min(min_components, max_feasible)
    best_gmm = None

    for n_components in range(1, max_feasible + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=GMM_COVARIANCE_TYPE,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                random_state=42,
            )
            gmm.fit(embeddings)

            bic = gmm.bic(embeddings)

            if bic < best_bic:
                best_bic = bic
                best_n_components = n_components
                best_gmm = gmm

        except Exception as e:
            logger.debug(f"Failed to fit GMM with {n_components} components: {e}")
            continue

    logger.debug(
        f"Selected {best_n_components} components for {n_samples} samples (BIC: {best_bic:.2f})"
    )
    return best_n_components, best_gmm


def fit_artist_gmm(artist_name: str, track_embeddings: List[np.ndarray]) -> Optional[Dict]:
    if len(track_embeddings) < MIN_TRACKS_PER_ARTIST:
        logger.warning(
            f"Artist '{artist_name}' has only {len(track_embeddings)} tracks, need at least {MIN_TRACKS_PER_ARTIST}"
        )
        return None

    try:
        all_embeddings = np.vstack(track_embeddings)
        n_samples, n_features = all_embeddings.shape

        if n_samples < 5:
            logger.info(
                f"Artist '{artist_name}' has {n_samples} tracks - using each song as a GMM component with equal weights"
            )

            n_components = n_samples
            weights = [1.0 / n_components] * n_components
            means = all_embeddings.tolist()

            gmm_params = {
                'weights': weights,
                'means': means,
                'n_components': n_components,
                'n_features': n_features,
                'n_tracks': n_samples,
                'is_few_songs': True,
            }

            logger.info(
                f"Created {n_components}-component GMM for artist '{artist_name}' (1 component per song, equal weights)"
            )
            return gmm_params

        optimal_n_components, gmm = fit_best_gmm(all_embeddings)

        if gmm is None:
            gmm = GaussianMixture(
                n_components=optimal_n_components,
                covariance_type=GMM_COVARIANCE_TYPE,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                random_state=42,
            )
            gmm.fit(all_embeddings)

        gmm_params = {
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.tolist(),
            'n_components': optimal_n_components,
            'n_features': all_embeddings.shape[1],
            'n_tracks': len(track_embeddings),
            'is_few_songs': False,
        }

        logger.info(
            f"Fitted GMM for artist '{artist_name}' with {len(track_embeddings)} tracks, {optimal_n_components} components, {all_embeddings.shape[1]}-dim embeddings"
        )

        return gmm_params

    except Exception:
        logger.exception(f"Failed to fit GMM for artist '{artist_name}'")
        return None


def serialize_gmm_for_hnsw(gmm_params: Dict) -> np.ndarray:
    means = np.array(gmm_params['means'])
    weights = np.array(gmm_params['weights'])

    weights = weights / np.sum(weights)

    weighted_mean = np.sum(weights[:, np.newaxis] * means, axis=0)

    return weighted_mean


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[np.newaxis, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _cosine_distance_matrix(means1: np.ndarray, means2: np.ndarray) -> np.ndarray:
    m1 = _l2_normalize_rows(means1)
    m2 = _l2_normalize_rows(means2)
    return (1.0 - np.clip(m1 @ m2.T, -1.0, 1.0)).astype(np.float32)


def gmm_soft_chamfer_distance(gmm1_params: Dict, gmm2_params: Dict) -> float:
    means1 = np.asarray(gmm1_params['means'], dtype=np.float32)
    means2 = np.asarray(gmm2_params['means'], dtype=np.float32)
    if means1.ndim != 2 or means2.ndim != 2 or means1.shape[0] == 0 or means2.shape[0] == 0:
        return float('inf')
    w1 = np.asarray(gmm1_params['weights'], dtype=np.float32)
    w2 = np.asarray(gmm2_params['weights'], dtype=np.float32)
    w1 = w1 / (float(w1.sum()) + 1e-12)
    w2 = w2 / (float(w2.sum()) + 1e-12)
    dmat = _cosine_distance_matrix(means1, means2)
    forward = float(np.sum(w1 * dmat.min(axis=1)))
    backward = float(np.sum(w2 * dmat.min(axis=0)))
    return 0.5 * (forward + backward)


def _gmm_worker_count(pending: int) -> int:
    if INDEX_BUILD_WORKERS == 1 or pending < _MIN_ARTISTS_FOR_POOL:
        return 1
    if INDEX_BUILD_WORKERS > 1:
        return min(INDEX_BUILD_WORKERS, pending)
    return max(1, min(8, (os.cpu_count() or 2) // 2, pending))


def _shutdown_gmm_pool() -> None:
    """Release loky's worker processes, their semaphores and their temp folders.

    loky holds its workers open for reuse across Parallel calls, which is right
    while the fits are running and wrong afterwards: nothing else in this process
    fits GMMs, and an RQ job process that exits with them still alive leaks every
    semaphore and memmap folder they hold.
    """
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:
        logger.warning("Could not shut the artist GMM worker pool down", exc_info=True)


def _fit_artist_job(job: Tuple[str, np.ndarray, str]) -> Tuple[str, Optional[Dict]]:
    """One artist's GMM. Top level so a worker process can unpickle it."""
    artist_name, embeddings, tracks_hash = job
    params = fit_artist_gmm(artist_name, embeddings)
    if params is None:
        return artist_name, None
    params['tracks_hash'] = tracks_hash
    return artist_name, params


def _cached_gmm_params(existing_gmm_params, artist_name, tracks_hash):
    """The stored GMM for an artist whose tracks have not changed, else None."""
    if not existing_gmm_params:
        return None
    params = existing_gmm_params.get(artist_name)
    if params is not None and params.get('tracks_hash') == tracks_hash:
        return params
    return None


def _artist_batches(pending, artist_tracks):
    """Group artists so one embedding round trip covers ~_FETCH_TRACKS_PER_BATCH tracks."""
    batch = []
    tracks = 0
    for artist_name in pending:
        batch.append(artist_name)
        tracks += len(artist_tracks[artist_name])
        if tracks >= _FETCH_TRACKS_PER_BATCH:
            yield batch
            batch = []
            tracks = 0
    if batch:
        yield batch


def _artist_jobs(cur, batch, artist_tracks, artist_track_hashes):
    """Fetch one batch of artists' embeddings and pack them into fit jobs."""
    wanted = [track['item_id'] for name in batch for track in artist_tracks[name]]
    cur.execute(
        "SELECT item_id, embedding FROM embedding "
        "WHERE item_id = ANY(%s) AND embedding IS NOT NULL",
        (wanted,),
    )
    vectors = {
        str(item_id): np.frombuffer(blob, dtype=np.float32)
        for item_id, blob in cur.fetchall()
        if blob
    }
    jobs = []
    for artist_name in batch:
        rows = [
            vectors[track['item_id']]
            for track in artist_tracks[artist_name]
            if track['item_id'] in vectors
        ]
        if len(rows) < MIN_TRACKS_PER_ARTIST:
            continue
        jobs.append((artist_name, np.vstack(rows), artist_track_hashes[artist_name]))
    return jobs


def _fit_pending_artists(cur, pending, artist_tracks, artist_track_hashes):
    """{artist: gmm_params} for every artist that needs a (re)fit.

    Two things made this the slowest step of a full rebuild, and neither was the
    GMM itself: one embedding round trip PER ARTIST, and every artist fitted on a
    single core. Embeddings are now fetched a batch of artists at a time, and the
    fits - a BIC sweep of restarted EM runs, ~50 ms of pure Python each - fan out
    across PROCESSES. Threads are useless here: the EM loop holds the GIL, and
    measured on 300 artists they ran 3x SLOWER than serial while processes ran 6x
    faster.

    joblib's loky backend, NOT a forked pool: an index rebuild trains the IVF
    cells (OpenMP) before it reaches the artists, and forking a process that has
    already run OpenMP deadlocks its children - which is exactly what a fork pool
    did here. loky starts its workers fresh, so there is no inherited OpenMP state
    to hang on, and it holds them open across the batches.

    The pool is shut down before returning. loky keeps its workers alive for
    REUSE by default, and this runs inside an RQ job process that exits when the
    job does: the workers outlive it, and their semaphores and memmap folders are
    still there to be complained about at shutdown. Memmapping is off too - an
    artist's embeddings are tens of KB, not worth a /tmp file each.
    """
    workers = _gmm_worker_count(len(pending))
    logger.info(
        "Fitting %d artist GMMs across %d worker process(es)...", len(pending), workers
    )
    fitted = {}
    runner = (
        Parallel(n_jobs=workers, backend='loky', max_nbytes=None)
        if workers > 1
        else None
    )
    try:
        for batch in _artist_batches(pending, artist_tracks):
            try:
                jobs = _artist_jobs(cur, batch, artist_tracks, artist_track_hashes)
            except Exception:
                logger.exception(
                    "Failed to fetch embeddings for a batch of %d artists", len(batch)
                )
                continue
            if not jobs:
                continue
            if runner is None:
                results = [_fit_artist_job(job) for job in jobs]
            else:
                results = runner(delayed(_fit_artist_job)(job) for job in jobs)
            for artist_name, params in results:
                if params is not None:
                    fitted[artist_name] = params
    finally:
        if runner is not None:
            _shutdown_gmm_pool()
    return fitted


def build_and_store_artist_index(db_conn=None):
    if db_conn is None:
        from app_helper import get_db

        db_conn = get_db()

    logger.info("Starting to build artist similarity index using GMM + IVF...")

    cur = db_conn.cursor()

    try:
        from .index_build_helpers import load_segmented_blob, unpack_artist_metadata

        existing_gmm_params = None
        try:
            metadata_blob = load_segmented_blob(db_conn, "artist_metadata_data", "artist_metadata")
            if metadata_blob:
                _, existing_gmm_params = unpack_artist_metadata(metadata_blob)
                logger.info(
                    f"Loaded existing GMM params for {len(existing_gmm_params)} artists "
                    f"(incremental mode, from artist_metadata_data BYTEA)"
                )
        except Exception as e:
            logger.warning(f"Could not load existing GMM params, will do full rebuild: {e}")
            existing_gmm_params = None

        logger.info("Fetching artists and tracks from database...")

        cur.execute("""
            SELECT DISTINCT author, item_id, title
            FROM score
            WHERE author IS NOT NULL AND author != ''
            ORDER BY author, title
        """)

        rows = cur.fetchall()

        if not rows:
            logger.warning("No tracks found in database for artist index building")
            return

        artist_tracks = defaultdict(list)
        for author, item_id, title in rows:
            artist_tracks[author].append({'item_id': item_id, 'title': title})

        logger.info(f"Found {len(artist_tracks)} artists with tracks")

        artist_track_hashes = {}
        for artist_name, tracks in artist_tracks.items():
            sorted_ids = sorted(track['item_id'] for track in tracks)
            hash_input = ','.join(sorted_ids)
            artist_track_hashes[artist_name] = hashlib.md5(
                hash_input.encode(), usedforsecurity=False
            ).hexdigest()

        cached = {
            artist_name: _cached_gmm_params(
                existing_gmm_params, artist_name, artist_track_hashes[artist_name]
            )
            for artist_name in artist_tracks
        }
        pending = [name for name, params in cached.items() if params is None]
        fitted = _fit_pending_artists(cur, pending, artist_tracks, artist_track_hashes)

        artist_gmms = {}
        artist_names_list = []
        for artist_name in artist_tracks:
            params = cached[artist_name] or fitted.get(artist_name)
            if params is None:
                continue
            artist_gmms[artist_name] = params
            artist_names_list.append(artist_name)

        reused_count = len(artist_tracks) - len(pending)
        refitted_count = len(fitted)
        logger.info(
            f"GMM fitting complete: {refitted_count} refitted, {reused_count} reused (unchanged), {len(artist_gmms)} total"
        )

        if len(artist_gmms) == 0:
            logger.warning("No valid GMMs created, skipping index build")
            return

        first_gmm = artist_gmms[artist_names_list[0]]
        gmm_vector_dim = len(serialize_gmm_for_hnsw(first_gmm))
        from .index_build_helpers import pack_artist_metadata, store_segmented_blob
        from .paged_ivf import build_and_store_paged_ivf

        logger.info(
            "Building artist IVF index (angular) for %d artists, dim=%d ...",
            len(artist_names_list),
            gmm_vector_dim,
        )
        artist_map_dict = {vid: a for vid, a in enumerate(artist_names_list)}
        vectors = np.array(
            [serialize_gmm_for_hnsw(artist_gmms[a]) for a in artist_names_list], dtype=np.float32
        )
        metadata_blob = pack_artist_metadata(artist_map_dict, artist_gmms)
        ok = build_and_store_paged_ivf(
            db_conn, ARTIST_INDEX_NAME, vectors, list(artist_names_list), gmm_vector_dim, "angular",
            consume_vectors=True,
        )
        if not ok:
            db_conn.rollback()
            logger.warning("Artist IVF build produced no index; aborting.")
            return
        store_segmented_blob(
            db_conn, target_table="artist_metadata_data", name="artist_metadata", blob=metadata_blob
        )
        db_conn.commit()
        logger.info("Artist IVF index built and stored (%d artists).", len(artist_gmms))
        return

    except Exception:
        logger.exception("Failed to build artist index")
        db_conn.rollback()
        raise

    finally:
        cur.close()


def load_artist_index_for_querying(force_reload=False):
    global artist_index, artist_map, reverse_artist_map, artist_gmm_params

    with _index_lock:
        if artist_index is not None and not force_reload:
            logger.info("Artist index already loaded in memory")
            return

        from app_helper import get_db

        logger.info("Loading artist similarity index from database...")

        conn = get_db()
        cur = conn.cursor()

        def _reset_cache():
            global artist_index, artist_map, reverse_artist_map, artist_gmm_params
            artist_index = None
            artist_map = None
            reverse_artist_map = None
            artist_gmm_params = None

        from .index_build_helpers import load_segmented_blob, unpack_artist_metadata

        try:
            from .paged_ivf import has_paged_ivf, load_paged_ivf_index

            if not has_paged_ivf(conn, ARTIST_INDEX_NAME):
                logger.info("Artist IVF index not found; not built yet.")
                _reset_cache()
                return
            loaded = load_paged_ivf_index(
                conn, ARTIST_INDEX_NAME, None, "angular", conn_factory=get_db,
                label="artist", track_scoped=False,
            )
            if loaded is None:
                _reset_cache()
                return
            metadata_blob = load_segmented_blob(conn, "artist_metadata_data", "artist_metadata")
            if not metadata_blob:
                logger.error("Artist IVF index present but metadata blob missing; aborting load.")
                _reset_cache()
                return
            parsed_artist_map, parsed_gmm_params = unpack_artist_metadata(metadata_blob)
            artist_index = loaded[0]
            if len(artist_index) != len(parsed_artist_map):
                logger.error(
                    "Artist IVF index element count (%d) != metadata artist_map count (%d); "
                    "aborting load to avoid mapping vectors to the wrong artist.",
                    len(artist_index),
                    len(parsed_artist_map),
                )
                _reset_cache()
                return
            artist_map = parsed_artist_map
            reverse_artist_map = {v: k for k, v in artist_map.items()}
            artist_gmm_params = parsed_gmm_params
            logger.info("Artist IVF index loaded (%d artists).", len(artist_map))
            return

        except Exception:
            logger.exception("Failed to load artist index")
            artist_index = None
            artist_map = None
            reverse_artist_map = None
            artist_gmm_params = None

        finally:
            cur.close()


def get_representative_songs_for_component(
    artist_name: str, component_index: int, top_k: int = 3
) -> List[Dict]:
    from app_helper import get_db

    if artist_gmm_params is None or artist_name not in artist_gmm_params:
        logger.warning(f"No GMM found for artist '{artist_name}'")
        return []

    gmm_params = artist_gmm_params[artist_name]

    means = np.array(gmm_params['means'])
    if component_index >= len(means):
        logger.warning(f"Component index {component_index} out of range for artist '{artist_name}'")
        return []

    component_mean = means[component_index]

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT s.item_id, s.title, e.embedding
            FROM score s
            JOIN embedding e ON s.item_id = e.item_id
            WHERE s.author = %s AND e.embedding IS NOT NULL
            ORDER BY s.title
        """,
            (artist_name,),
        )

        rows = cur.fetchall()

        if not rows:
            return []

        song_distances = []
        for item_id, title, embedding_bytes in rows:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            distance = np.linalg.norm(embedding - component_mean)
            song_distances.append(
                {'item_id': item_id, 'title': title, 'distance_to_component': float(distance)}
            )

        song_distances.sort(key=lambda x: x['distance_to_component'])
        return song_distances[:top_k]

    except Exception:
        logger.exception(f"Failed to get representative songs for artist '{artist_name}'")
        return []

    finally:
        cur.close()


def compute_component_matches(
    gmm1_params: Dict, gmm2_params: Dict, artist1_name: str, artist2_name: str, top_k: int = 3
) -> List[Dict]:
    means1 = np.array(gmm1_params['means'])
    means2 = np.array(gmm2_params['means'])
    weights1 = np.array(gmm1_params['weights'])
    weights2 = np.array(gmm2_params['weights'])

    distances = _cosine_distance_matrix(means1, means2)

    flat_indices = np.argsort(distances.ravel())[:top_k]
    matches = []

    for flat_idx in flat_indices:
        comp1_idx = flat_idx // distances.shape[1]
        comp2_idx = flat_idx % distances.shape[1]
        distance = distances[comp1_idx, comp2_idx]

        artist1_songs = get_representative_songs_for_component(artist1_name, comp1_idx, top_k=3)
        artist2_songs = get_representative_songs_for_component(artist2_name, comp2_idx, top_k=3)

        matches.append(
            {
                'component1_index': int(comp1_idx),
                'component2_index': int(comp2_idx),
                'distance': float(distance),
                'component1_weight': float(weights1[comp1_idx]),
                'component2_weight': float(weights2[comp2_idx]),
                'artist1_representative_songs': artist1_songs,
                'artist2_representative_songs': artist2_songs,
            }
        )

    return matches


def _resolve_indexed_artist_name(query_artist):
    if query_artist in reverse_artist_map:
        return query_artist

    from app_helper_artist import get_artist_name_by_id

    resolved_name = get_artist_name_by_id(query_artist)
    if resolved_name:
        logger.info(f"Resolved artist ID '{query_artist}' to name '{resolved_name}'")
        return resolved_name
    return query_artist


def _score_candidate_artists(labels, query_id, query_gmm):
    scored = []
    for idx in labels:
        if idx == query_id:
            continue
        candidate_artist = artist_map.get(idx)
        if candidate_artist is None:
            continue
        candidate_gmm = artist_gmm_params.get(candidate_artist)
        if candidate_gmm is None:
            continue
        scored.append(
            (gmm_soft_chamfer_distance(query_gmm, candidate_gmm), candidate_artist, candidate_gmm)
        )
    scored.sort(key=lambda t: t[0])
    return scored


def _build_similar_artist_result(
    score, candidate_artist, candidate_gmm, query_gmm, artist_name, include_component_matches
):
    from app_helper_artist import get_artist_id_by_name

    result = {
        'artist': candidate_artist,
        'artist_id': get_artist_id_by_name(candidate_artist),
        'divergence': float(score),
    }
    if include_component_matches:
        result['component_matches'] = compute_component_matches(
            query_gmm, candidate_gmm, artist_name, candidate_artist, top_k=3
        )
        result['query_artist_components'] = query_gmm['n_components']
        result['candidate_artist_components'] = candidate_gmm['n_components']
    return result


def find_similar_artists(
    query_artist,
    n: int = 10,
    ef_search: Optional[int] = None,
    include_component_matches: bool = False,
) -> List[Dict]:
    if artist_index is None or artist_map is None or artist_gmm_params is None:
        logger.error("Artist index not loaded")
        raise RuntimeError("Artist similarity index not available")

    artist_name = _resolve_indexed_artist_name(query_artist)

    if artist_name not in reverse_artist_map:
        logger.warning(f"Artist '{artist_name}' not found in index")
        return []

    query_id = reverse_artist_map[artist_name]

    query_gmm = artist_gmm_params[artist_name]

    from .paged_ivf import begin_query

    begin_query(artist_index)

    k_candidates = min(3 * n + 1, len(artist_map))
    query_vector = serialize_gmm_for_hnsw(query_gmm)
    try:
        labels, _distances = artist_index.query(query_vector, k=k_candidates)
    except Exception:
        logger.exception(f"IVF query failed for artist '{artist_name}'")
        return []

    scored = _score_candidate_artists(labels, query_id, query_gmm)

    return [
        _build_similar_artist_result(
            score,
            candidate_artist,
            candidate_gmm,
            query_gmm,
            artist_name,
            include_component_matches,
        )
        for score, candidate_artist, candidate_gmm in scored[:n]
    ]


def search_artists_by_name(
    query: str,
    limit: int = 20,
    offset: int = 0,
    server_id: str | None = None,
    include_legacy_default: bool = False,
) -> List[Dict]:
    if not query:
        return []

    from app_helper import get_db
    from app_helper_artist import get_artist_id_by_name

    conn = get_db()
    cur = conn.cursor()

    try:
        query_pattern = f"%{query}%"

        availability = ""
        availability_params = []
        if server_id:
            from tasks.mediaserver.registry import availability_sql

            availability = " AND " + availability_sql('score')
            availability_params = [server_id, bool(include_legacy_default)]
        cur.execute(
            """
            SELECT DISTINCT author, COUNT(*) as track_count
            FROM score
            WHERE author ILIKE %s AND author IS NOT NULL AND author != ''
            """ + availability + """
            GROUP BY author
            ORDER BY track_count DESC, author
            LIMIT %s OFFSET %s
        """,
            tuple([query_pattern] + availability_params + [limit, offset]),
        )

        results = []
        for author, track_count in cur.fetchall():
            artist_id = get_artist_id_by_name(author)
            results.append({'artist': author, 'artist_id': artist_id, 'track_count': track_count})

        return results

    except Exception:
        logger.exception("Failed to search artists")
        return []

    finally:
        cur.close()


def get_artist_tracks(artist_identifier: str) -> List[Dict]:
    from app_helper import get_db
    from app_helper_artist import get_artist_name_by_id

    artist_name = artist_identifier
    if artist_identifier:
        resolved_name = get_artist_name_by_id(artist_identifier)
        if resolved_name:
            artist_name = resolved_name

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT item_id, title, author
            FROM score
            WHERE author = %s
            ORDER BY title
        """,
            (artist_name,),
        )

        results = []
        for item_id, title, author in cur.fetchall():
            results.append({'item_id': item_id, 'title': title, 'author': author})

        return results

    except Exception:
        logger.exception(f"Failed to get tracks for artist '{artist_name}'")
        return []

    finally:
        cur.close()


def cleanup_resources():
    global artist_index, artist_map, reverse_artist_map, artist_gmm_params

    with _index_lock:
        artist_index = None
        artist_map = None
        reverse_artist_map = None
        artist_gmm_params = None
        logger.info("Artist index resources cleaned up")
