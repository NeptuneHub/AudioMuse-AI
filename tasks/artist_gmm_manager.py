# tasks/artist_gmm_manager.py
"""
Artist Similarity using Gaussian Mixture Models (GMM) and IVF Index.

This module implements artist similarity by:
1. Using existing embedding vectors (same as song similarity) for all tracks per artist
2. Fitting a GMM to the embedding vectors to represent each artist's "sound profile"
3. Building an angular (cosine) IVF index over each GMM's weighted centroid for fast,
   coarse candidate retrieval
4. Reranking the over-fetched candidates by a weighted soft-Chamfer cosine distance
   between the two artists' GMM component means, so multimodality matters and not just
   the centroid

This approach is fast (no audio re-processing) and consistent with the song similarity system.
"""

import logging
import hashlib
import numpy as np
import threading
from typing import Dict, List, Optional
from collections import defaultdict

from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# --- Configuration ---
ARTIST_INDEX_NAME = 'artist_similarity_index'
GMM_N_COMPONENTS_MIN = 2  # Minimum number of GMM components
GMM_N_COMPONENTS_MAX = 10  # Maximum number of GMM components (will auto-select using BIC)
GMM_COVARIANCE_TYPE = 'diag'  # 'diag' is faster than 'full' and works well for high-dim embeddings
GMM_MAX_ITER = 100
GMM_N_INIT = 3
MIN_TRACKS_PER_ARTIST = 1  # Minimum tracks needed to build a GMM for an artist (lowered to include all artists)

# --- Global cache for the loaded artist index ---
artist_index = None  # ivf.Index object
artist_map = None  # {vec_int_id: artist_name}
reverse_artist_map = None  # {artist_name: vec_int_id}
artist_gmm_params = None  # {artist_name: gmm_params_dict}
_index_lock = threading.Lock()


def select_optimal_gmm_components(embeddings: np.ndarray, min_components: int = GMM_N_COMPONENTS_MIN, max_components: int = GMM_N_COMPONENTS_MAX) -> int:
    """
    Automatically select the optimal number of GMM components using BIC (Bayesian Information Criterion).
    Lower BIC is better - it balances model complexity with fit quality.
    
    Args:
        embeddings: Array of embedding vectors (n_samples, n_features)
        min_components: Minimum number of components to try
        max_components: Maximum number of components to try
        
    Returns: Optimal number of components
    """
    n_samples = len(embeddings)
    
    # Special case: 1 track = 1 component (no need for BIC search)
    if n_samples == 1:
        return 1
    
    # Limit max components based on data size
    # For small datasets: 1 component per track, capped at configured max
    # For larger datasets: at least 5 samples per component (relaxed from 10)
    if n_samples <= 5:
        # Very small dataset: use 1 component per track (simple representation)
        max_feasible = min(n_samples, max_components)
    else:
        # Larger dataset: at least 5 samples per component
        max_feasible = max(min_components, min(max_components, n_samples // 5))
    
    # Ensure we have at least min_components (unless dataset is tiny)
    if max_feasible < min_components:
        max_feasible = min(min_components, n_samples)
    
    # Must have at least 1 component
    if max_feasible < 1:
        return 1
    
    best_bic = float('inf')
    best_n_components = min(min_components, max_feasible)
    
    # Try different numbers of components
    for n_components in range(1, max_feasible + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=GMM_COVARIANCE_TYPE,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                random_state=42
            )
            gmm.fit(embeddings)
            
            # Compute BIC (lower is better)
            bic = gmm.bic(embeddings)
            
            if bic < best_bic:
                best_bic = bic
                best_n_components = n_components
                
        except Exception as e:
            logger.debug(f"Failed to fit GMM with {n_components} components: {e}")
            continue
    
    logger.debug(f"Selected {best_n_components} components for {n_samples} samples (BIC: {best_bic:.2f})")
    return best_n_components


def fit_artist_gmm(artist_name: str, track_embeddings: List[np.ndarray]) -> Optional[Dict]:
    """
    Fit a Gaussian Mixture Model to represent an artist's sound profile.
    Uses the existing embedding vectors (same as used for song similarity).
    
    For artists with < 5 songs: Uses each song as a GMM component with equal weights.
    This is more honest than inventing artificial clusters - it represents the actual songs.
    
    For artists with >= 5 songs: Fits a proper GMM using sklearn's GaussianMixture.
    
    Args:
        artist_name: Name of the artist
        track_embeddings: List of embedding vectors from the artist's tracks
        
    Returns: Dictionary containing GMM parameters or None if fitting fails
    """
    if len(track_embeddings) < MIN_TRACKS_PER_ARTIST:
        logger.warning(f"Artist '{artist_name}' has only {len(track_embeddings)} tracks, need at least {MIN_TRACKS_PER_ARTIST}")
        return None
    
    try:
        # Stack all embeddings into a single array
        all_embeddings = np.vstack(track_embeddings)
        n_samples, n_features = all_embeddings.shape
        
        # For artists with few songs (< 5): use each song as its own component
        # This is simpler and more honest than trying to fit a statistical model
        if n_samples < 5:
            logger.info(f"Artist '{artist_name}' has {n_samples} tracks - using each song as a GMM component with equal weights")
            
            
            # Each song becomes one component with equal weight
            n_components = n_samples
            weights = [1.0 / n_components] * n_components
            means = all_embeddings.tolist()  # Each row is a song's embedding

            gmm_params = {
                'weights': weights,
                'means': means,
                'n_components': n_components,
                'n_features': n_features,
                'n_tracks': n_samples,
                'is_few_songs': True  # Flag to identify few-song artists
            }

            logger.info(f"Created {n_components}-component GMM for artist '{artist_name}' (1 component per song, equal weights)")
            return gmm_params
        
        # Multi-song artist (>= 5 songs): fit GMM normally using sklearn
        # Automatically select optimal number of components for this artist
        optimal_n_components = select_optimal_gmm_components(all_embeddings)
        
        # Fit GMM to the embedding vectors
        gmm = GaussianMixture(
            n_components=optimal_n_components,
            covariance_type=GMM_COVARIANCE_TYPE,
            max_iter=GMM_MAX_ITER,
            n_init=GMM_N_INIT,
            random_state=42
        )
        
        gmm.fit(all_embeddings)
        
        # Extract GMM parameters. NOTE: covariances and covariance_type were
        # previously stored too, but nothing in the live query path ever read
        # them (the entire Jeffreys -> KL Monte Carlo chain was dead code).
        # Dropping them keeps the per-artist payload ~half its old size, which
        # is what gets gmm_params under PG's 1 GB MaxAllocSize cap at large
        # library scales.
        gmm_params = {
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.tolist(),
            'n_components': optimal_n_components,  # Store the actual number used
            'n_features': all_embeddings.shape[1],
            'n_tracks': len(track_embeddings),
            'is_few_songs': False
        }
        
        logger.info(f"Fitted GMM for artist '{artist_name}' with {len(track_embeddings)} tracks, {optimal_n_components} components, {all_embeddings.shape[1]}-dim embeddings")
        
        return gmm_params
        
    except Exception as e:
        logger.error(f"Failed to fit GMM for artist '{artist_name}': {e}")
        return None


def serialize_gmm_for_hnsw(gmm_params: Dict) -> np.ndarray:
    """
    Serialize GMM parameters into a fixed-size vector for IVF indexing.
    (Function name kept for backward compatibility)
    
    We compute a weighted average of the component means, weighted by the
    component weights. This gives a single representative vector for the GMM
    that captures its center of mass.
    
    Args:
        gmm_params: GMM parameters dictionary
        
    Returns: 1D numpy array (feature_dim,)
    """
    means = np.array(gmm_params['means'])
    weights = np.array(gmm_params['weights'])
    
    # Normalize weights to ensure they sum to 1 (handle numerical errors)
    weights = weights / np.sum(weights)
    
    # Compute weighted mean: sum(w_i * mean_i)
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
    """Pairwise cosine distance (1 - cos) between two sets of component means."""
    m1 = _l2_normalize_rows(means1)
    m2 = _l2_normalize_rows(means2)
    return (1.0 - np.clip(m1 @ m2.T, -1.0, 1.0)).astype(np.float32)


def gmm_soft_chamfer_distance(gmm1_params: Dict, gmm2_params: Dict) -> float:
    """Weighted soft-Chamfer cosine distance between two artists' GMMs.

    Treats each artist as its SET of component means (not one centroid): builds the
    pairwise cosine-distance matrix, then takes w_i * min_j D[i,j] for every query
    component i and symmetrizes by averaging the candidate->query direction. Lower
    means more similar; component weights make dominant sound-modes count more.
    """
    means1 = np.asarray(gmm1_params['means'], dtype=np.float32)
    means2 = np.asarray(gmm2_params['means'], dtype=np.float32)
    w1 = np.asarray(gmm1_params['weights'], dtype=np.float32)
    w2 = np.asarray(gmm2_params['weights'], dtype=np.float32)
    w1 = w1 / (float(w1.sum()) + 1e-12)
    w2 = w2 / (float(w2.sum()) + 1e-12)
    dmat = _cosine_distance_matrix(means1, means2)
    forward = float(np.sum(w1 * dmat.min(axis=1)))   # query -> nearest candidate component
    backward = float(np.sum(w2 * dmat.min(axis=0)))  # candidate -> nearest query component
    return 0.5 * (forward + backward)


def build_and_store_artist_index(db_conn=None):
    """
    Build IVF index for artist similarity using GMM representations.
    
    This function:
    1. Fetches all artists and their tracks from the database
    2. Extracts audio features for each track
    3. Fits a GMM for each artist
    4. Builds a IVF index for fast similarity search
    5. Stores the index in the database
    
    Args:
        db_conn: Database connection (if None, will acquire one)
    """
    if db_conn is None:
        from app_helper import get_db
        db_conn = get_db()
    
    logger.info("Starting to build artist similarity index using GMM + IVF...")
    
    cur = db_conn.cursor()
    
    try:
        # Step 0: Load existing GMM params for incremental rebuild from the
        # binary blob in artist_metadata_data, yielding a {artist_name:
        # gmm_params_dict} mapping that drives the tracks_hash reuse check below.
        # An absent/empty blob simply means a full rebuild.
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

        # Step 1: Get all artists and their tracks
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

        # Group tracks by artist
        artist_tracks = defaultdict(list)
        for author, item_id, title in rows:
            artist_tracks[author].append({'item_id': item_id, 'title': title})

        logger.info(f"Found {len(artist_tracks)} artists with tracks")

        # Step 1.5: Compute per-artist track hashes for change detection
        artist_track_hashes = {}
        for artist_name, tracks in artist_tracks.items():
            sorted_ids = sorted(track['item_id'] for track in tracks)
            hash_input = ','.join(sorted_ids)
            artist_track_hashes[artist_name] = hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()

        # Step 2: Fetch embeddings and fit GMMs (incremental: skip unchanged artists)
        logger.info("Fetching embeddings and fitting GMMs...")

        artist_gmms = {}
        artist_names_list = []
        reused_count = 0
        refitted_count = 0

        for idx, (artist_name, tracks) in enumerate(artist_tracks.items(), 1):
            if idx % 50 == 0:
                logger.info(f"Processing artist {idx}/{len(artist_tracks)}: {artist_name}")

            current_hash = artist_track_hashes[artist_name]

            # Check if we can reuse existing GMM (same tracks_hash means no change)
            if (existing_gmm_params is not None
                    and artist_name in existing_gmm_params
                    and existing_gmm_params[artist_name].get('tracks_hash') == current_hash):
                # Artist unchanged - reuse stored GMM
                artist_gmms[artist_name] = existing_gmm_params[artist_name]
                artist_names_list.append(artist_name)
                reused_count += 1
                continue

            # Artist is new or changed - fetch embeddings and refit GMM
            track_embeddings = []
            item_ids = [track['item_id'] for track in tracks]

            try:
                # Batch fetch embeddings from database
                cur.execute("""
                    SELECT item_id, embedding
                    FROM embedding
                    WHERE item_id = ANY(%s) AND embedding IS NOT NULL
                """, (item_ids,))

                embedding_rows = cur.fetchall()

                for item_id, embedding_bytes in embedding_rows:
                    if embedding_bytes:
                        # Deserialize embedding vector
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        track_embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Failed to fetch embeddings for artist {artist_name}: {e}")
                continue

            # Fit GMM for this artist
            if len(track_embeddings) >= MIN_TRACKS_PER_ARTIST:
                gmm_params = fit_artist_gmm(artist_name, track_embeddings)

                if gmm_params is not None:
                    # Store the tracks_hash alongside GMM params for future incremental rebuilds
                    gmm_params['tracks_hash'] = current_hash
                    artist_gmms[artist_name] = gmm_params
                    artist_names_list.append(artist_name)
                    refitted_count += 1

        logger.info(f"GMM fitting complete: {refitted_count} refitted, {reused_count} reused (unchanged), {len(artist_gmms)} total")
        
        if len(artist_gmms) == 0:
            logger.warning("No valid GMMs created, skipping index build")
            return
        
        # Build the disk-paged IVF artist index (angular/cosine) over the weighted
        # GMM centroids. Angular makes the coarse pass use the same metric as the
        # soft-Chamfer rerank, and lets the cells store as int8 like every other index.
        first_gmm = artist_gmms[artist_names_list[0]]
        gmm_vector_dim = len(serialize_gmm_for_hnsw(first_gmm))
        from .index_build_helpers import pack_artist_metadata, store_segmented_blob
        from .paged_ivf import build_and_store_paged_ivf
        logger.info("Building artist IVF index (angular) for %d artists, dim=%d ...", len(artist_names_list), gmm_vector_dim)
        artist_map_dict = {vid: a for vid, a in enumerate(artist_names_list)}
        vectors = np.array([serialize_gmm_for_hnsw(artist_gmms[a]) for a in artist_names_list], dtype=np.float32)
        metadata_blob = pack_artist_metadata(artist_map_dict, artist_gmms)
        ok = build_and_store_paged_ivf(db_conn, ARTIST_INDEX_NAME, vectors, list(artist_names_list), gmm_vector_dim, "angular")
        if not ok:
            db_conn.rollback()
            logger.warning("Artist IVF build produced no index; aborting.")
            return
        store_segmented_blob(db_conn, target_table="artist_metadata_data", name="artist_metadata", blob=metadata_blob)
        db_conn.commit()
        logger.info("Artist IVF index built and stored (%d artists).", len(artist_gmms))
        return

    except Exception as e:
        logger.error(f"Failed to build artist index: {e}", exc_info=True)
        db_conn.rollback()
        raise
    
    finally:
        cur.close()


def load_artist_index_for_querying(force_reload=False):
    """
    Load the artist IVF index from database into memory.
    
    Args:
        force_reload: If True, force reload even if already loaded
    """
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
            loaded = load_paged_ivf_index(conn, ARTIST_INDEX_NAME, None, "angular", conn_factory=get_db, label="artist")
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
                    len(artist_index), len(parsed_artist_map),
                )
                _reset_cache()
                return
            artist_map = parsed_artist_map
            reverse_artist_map = {v: k for k, v in artist_map.items()}
            artist_gmm_params = parsed_gmm_params
            logger.info("Artist IVF index loaded (%d artists).", len(artist_map))
            return

        except Exception as e:
            logger.error(f"Failed to load artist index: {e}", exc_info=True)
            artist_index = None
            artist_map = None
            reverse_artist_map = None
            artist_gmm_params = None
        
        finally:
            cur.close()


def get_representative_songs_for_component(artist_name: str, component_index: int, top_k: int = 3) -> List[Dict]:
    """
    Find the most representative songs for a specific GMM component.
    
    For few-song artists (< 5 songs): Each component IS a song, so just return that song.
    For multi-song artists: Find songs whose embeddings are closest to the component mean.
    
    Args:
        artist_name: Name of the artist
        component_index: Index of the GMM component (0-based)
        top_k: Number of representative songs to return
        
    Returns: List of song dictionaries with item_id, title, distance_to_component
    """
    from app_helper import get_db
    
    # Get GMM parameters for this artist
    if artist_gmm_params is None or artist_name not in artist_gmm_params:
        logger.warning(f"No GMM found for artist '{artist_name}'")
        return []
    
    gmm_params = artist_gmm_params[artist_name]
    
    # Get the component mean (centroid)
    means = np.array(gmm_params['means'])
    if component_index >= len(means):
        logger.warning(f"Component index {component_index} out of range for artist '{artist_name}'")
        return []
    
    component_mean = means[component_index]
    
    # Fetch all tracks and embeddings for this artist
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT s.item_id, s.title, e.embedding
            FROM score s
            JOIN embedding e ON s.item_id = e.item_id
            WHERE s.author = %s AND e.embedding IS NOT NULL
            ORDER BY s.title
        """, (artist_name,))
        
        rows = cur.fetchall()
        
        if not rows:
            return []
        
        # Compute distances from each song to the component mean
        song_distances = []
        for item_id, title, embedding_bytes in rows:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            distance = np.linalg.norm(embedding - component_mean)
            song_distances.append({
                'item_id': item_id,
                'title': title,
                'distance_to_component': float(distance)
            })
        
        # Sort by distance and return top-k closest songs
        song_distances.sort(key=lambda x: x['distance_to_component'])
        return song_distances[:top_k]
        
    except Exception as e:
        logger.error(f"Failed to get representative songs for artist '{artist_name}': {e}")
        return []
    
    finally:
        cur.close()


def compute_component_matches(gmm1_params: Dict, gmm2_params: Dict, artist1_name: str, artist2_name: str, top_k: int = 3) -> List[Dict]:
    """
    Find which GMM components match between two artists.
    Shows which "sound profiles" are similar between artists.
    
    Args:
        gmm1_params: GMM parameters for first artist
        gmm2_params: GMM parameters for second artist
        artist1_name: Name of first artist (query artist)
        artist2_name: Name of second artist (candidate)
        top_k: Number of top component matches to return
        
    Returns: List of component matches with distances and representative songs
    """
    means1 = np.array(gmm1_params['means'])  # Shape: [n_components1, n_features]
    means2 = np.array(gmm2_params['means'])  # Shape: [n_components2, n_features]
    weights1 = np.array(gmm1_params['weights'])
    weights2 = np.array(gmm2_params['weights'])

    # Pairwise cosine distance D[i,j] between component i of artist1 and component j of
    # artist2 (matches the soft-Chamfer rerank and the angular IVF coarse pass).
    distances = _cosine_distance_matrix(means1, means2)
    
    # Find top-k closest component pairs
    flat_indices = np.argsort(distances.ravel())[:top_k]
    matches = []
    
    for flat_idx in flat_indices:
        comp1_idx = flat_idx // distances.shape[1]
        comp2_idx = flat_idx % distances.shape[1]
        distance = distances[comp1_idx, comp2_idx]
        
        # Get representative songs for each component
        artist1_songs = get_representative_songs_for_component(artist1_name, comp1_idx, top_k=3)
        artist2_songs = get_representative_songs_for_component(artist2_name, comp2_idx, top_k=3)
        
        matches.append({
            'component1_index': int(comp1_idx),
            'component2_index': int(comp2_idx),
            'distance': float(distance),
            'component1_weight': float(weights1[comp1_idx]),
            'component2_weight': float(weights2[comp2_idx]),
            'artist1_representative_songs': artist1_songs,
            'artist2_representative_songs': artist2_songs
        })
    
    return matches


def find_similar_artists(query_artist, n: int = 10, ef_search: Optional[int] = None, include_component_matches: bool = False) -> List[Dict]:
    """
    Find similar artists using the IVF index.
    Accepts either artist name or artist ID.
    
    Args:
        query_artist: Name or ID of the query artist
        n: Number of similar artists to return
        ef_search: accepted for API compatibility; the disk-paged IVF backend
            controls recall via IVF_NPROBE, so this value is ignored
        include_component_matches: If True, include component-level similarity explanation
        
    Returns: List of dictionaries with 'artist', 'artist_id', 'divergence' keys
             If include_component_matches=True, also includes 'component_matches' key
    """
    if artist_index is None or artist_map is None or artist_gmm_params is None:
        logger.error("Artist index not loaded")
        raise RuntimeError("Artist similarity index not available")
    
    # Try to resolve artist ID to name if it looks like an ID (not in reverse_artist_map)
    artist_name = query_artist
    if query_artist not in reverse_artist_map:
        from app_helper_artist import get_artist_name_by_id
        resolved_name = get_artist_name_by_id(query_artist)
        if resolved_name:
            artist_name = resolved_name
            logger.info(f"Resolved artist ID '{query_artist}' to name '{artist_name}'")
    
    if artist_name not in reverse_artist_map:
        logger.warning(f"Artist '{artist_name}' not found in index")
        return []
    
    # Get query artist's IVF ID
    query_id = reverse_artist_map[artist_name]
    
    # Get query GMM parameters
    query_gmm = artist_gmm_params[artist_name]
    
    from .paged_ivf import begin_query
    begin_query(artist_index)

    # Stage 1 -- coarse retrieval: the IVF over the weighted GMM centroid is only a
    # fast prefilter, so over-fetch ~3x the requested count to give the rerank room
    # to reorder. The raw IVF order is NOT the final ranking.
    k_candidates = min(3 * n + 1, len(artist_map))  # +1 to account for self
    query_vector = serialize_gmm_for_hnsw(query_gmm)
    try:
        labels, _distances = artist_index.query(query_vector, k=k_candidates)
    except Exception as e:
        logger.error(f"IVF query failed for artist '{artist_name}': {e}", exc_info=True)
        return []

    # Stage 2 -- rerank: score each candidate against the query as a SET of GMM
    # component means via the weighted soft-Chamfer cosine distance, so two artists
    # match on shared sound-modes rather than just their centroids.
    scored = []
    for idx in labels:
        if idx == query_id:
            continue  # Skip self
        candidate_artist = artist_map.get(idx)
        if candidate_artist is None:
            continue
        candidate_gmm = artist_gmm_params.get(candidate_artist)
        if candidate_gmm is None:
            continue
        scored.append((gmm_soft_chamfer_distance(query_gmm, candidate_gmm), candidate_artist, candidate_gmm))

    scored.sort(key=lambda t: t[0])

    from app_helper_artist import get_artist_id_by_name

    results = []
    for score, candidate_artist, candidate_gmm in scored[:n]:
        result = {
            'artist': candidate_artist,
            'artist_id': get_artist_id_by_name(candidate_artist),
            'divergence': float(score)  # weighted soft-Chamfer cosine distance (lower = closer)
        }
        if include_component_matches:
            result['component_matches'] = compute_component_matches(
                query_gmm, candidate_gmm, artist_name, candidate_artist, top_k=3
            )
            result['query_artist_components'] = query_gmm['n_components']
            result['candidate_artist_components'] = candidate_gmm['n_components']
        results.append(result)

    return results


def search_artists_by_name(query: str, limit: int = 20, offset: int = 0) -> List[Dict]:
    """
    Search for artists by name (for autocomplete).
    Returns both artist name and ID if available.
    
    Args:
        query: Search query string
        limit: Maximum number of results
        offset: Number of results to skip (for pagination)
        
    Returns: List of dictionaries with artist information including artist_id
    """
    if not query:
        return []
    
    from app_helper import get_db
    from app_helper_artist import get_artist_id_by_name
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Simple case-insensitive search
        query_pattern = f"%{query}%"
        
        cur.execute("""
            SELECT DISTINCT author, COUNT(*) as track_count
            FROM score
            WHERE author ILIKE %s AND author IS NOT NULL AND author != ''
            GROUP BY author
            ORDER BY track_count DESC, author
            LIMIT %s OFFSET %s
        """, (query_pattern, limit, offset))
        
        results = []
        for author, track_count in cur.fetchall():
            artist_id = get_artist_id_by_name(author)
            results.append({
                'artist': author,
                'artist_id': artist_id,
                'track_count': track_count
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to search artists: {e}")
        return []
    
    finally:
        cur.close()


def get_artist_tracks(artist_identifier: str) -> List[Dict]:
    """
    Get all tracks for a given artist (by name or ID).
    
    Args:
        artist_identifier: Name or ID of the artist
        
    Returns: List of track dictionaries with item_id, title, author
    """
    from app_helper import get_db
    from app_helper_artist import get_artist_name_by_id
    
    # Try to resolve ID to name if needed
    artist_name = artist_identifier
    if artist_identifier:
        resolved_name = get_artist_name_by_id(artist_identifier)
        if resolved_name:
            artist_name = resolved_name
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT item_id, title, author
            FROM score
            WHERE author = %s
            ORDER BY title
        """, (artist_name,))
        
        results = []
        for item_id, title, author in cur.fetchall():
            results.append({
                'item_id': item_id,
                'title': title,
                'author': author
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get tracks for artist '{artist_name}': {e}")
        return []
    
    finally:
        cur.close()


def cleanup_resources():
    """Cleanup loaded index and release memory."""
    global artist_index, artist_map, reverse_artist_map, artist_gmm_params
    
    with _index_lock:
        artist_index = None
        artist_map = None
        reverse_artist_map = None
        artist_gmm_params = None
        logger.info("Artist index resources cleaned up")
