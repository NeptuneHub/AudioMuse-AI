# tasks/artist_gmm_manager.py
"""
Artist Similarity using Gaussian Mixture Models (GMM) and HNSW Index.

This module implements artist similarity by:
1. Using existing embedding vectors (same as song similarity) for all tracks per artist
2. Fitting a GMM to the embedding vectors to represent each artist's "sound profile"
3. Building an HNSW index for fast approximate nearest neighbor search
4. Using Jeffreys Divergence (symmetric KL divergence) to measure GMM similarity

This approach is fast (no audio re-processing) and consistent with the song similarity system.
"""

import os
import logging
import json
import pickle
import tempfile
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.mixture import GaussianMixture
import hnswlib

logger = logging.getLogger(__name__)

# --- Configuration ---
ARTIST_INDEX_NAME = 'artist_similarity_index'
GMM_N_COMPONENTS_MIN = 2  # Minimum number of GMM components
GMM_N_COMPONENTS_MAX = 10  # Maximum number of GMM components (will auto-select using BIC)
GMM_COVARIANCE_TYPE = 'diag'  # 'diag' is faster than 'full' and works well for high-dim embeddings
GMM_MAX_ITER = 100
GMM_N_INIT = 3
MIN_TRACKS_PER_ARTIST = 1  # Minimum tracks needed to build a GMM for an artist (lowered to include all artists)

# HNSW index parameters
HNSW_M = 32  # Number of bi-directional links created for every new element
HNSW_EF_CONSTRUCTION = 200  # Size of the dynamic list during index construction
HNSW_EF_SEARCH = 100  # Size of the dynamic list during search (can be adjusted at query time)

# Monte Carlo sampling for KL divergence approximation
MC_SAMPLES = 500  # Number of samples for Monte Carlo KL divergence estimation (reduced for speed)

# --- Global cache for the loaded artist index ---
artist_index = None  # hnswlib.Index object
artist_map = None  # {hnsw_int_id: artist_name}
reverse_artist_map = None  # {artist_name: hnsw_int_id}
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
        
        # Automatically select optimal number of components for this artist
        # (handles small datasets gracefully, including single-track artists)
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
        
        # Extract GMM parameters
        gmm_params = {
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.tolist(),
            'covariances': gmm.covariances_.tolist(),
            'n_components': optimal_n_components,  # Store the actual number used
            'covariance_type': GMM_COVARIANCE_TYPE,
            'n_features': all_embeddings.shape[1],
            'n_tracks': len(track_embeddings)
        }
        
        logger.info(f"Fitted GMM for artist '{artist_name}' with {len(track_embeddings)} tracks, {optimal_n_components} components, {all_embeddings.shape[1]}-dim embeddings")
        
        return gmm_params
        
    except Exception as e:
        logger.error(f"Failed to fit GMM for artist '{artist_name}': {e}")
        return None


def gmm_params_to_objects(gmm_params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert GMM parameters dictionary to numpy arrays.
    Handles both fresh numpy arrays and JSON-deserialized lists.
    
    Returns: (weights, means, covariances)
    """
    weights = np.array(gmm_params['weights'], dtype=np.float64)
    means = np.array(gmm_params['means'], dtype=np.float64)
    covariances = np.array(gmm_params['covariances'], dtype=np.float64)
    
    # Normalize weights to ensure they sum to 1 (handle numerical errors from JSON serialization)
    weights = weights / np.sum(weights)
    
    return weights, means, covariances


def compute_kl_divergence_mc(gmm1_params: Dict, gmm2_params: Dict, n_samples: int = MC_SAMPLES) -> float:
    """
    Compute KL divergence D_KL(gmm1 || gmm2) using Monte Carlo sampling.
    
    Uses the formula: D_KL(p||q) = E_p[log(p(x)) - log(q(x))]
    Approximated by sampling from gmm1 and computing the average log-density difference.
    
    Args:
        gmm1_params: GMM parameters for first distribution
        gmm2_params: GMM parameters for second distribution
        n_samples: Number of Monte Carlo samples
        
    Returns: KL divergence estimate
    """
    try:
        # Reconstruct GMM objects
        weights1, means1, covs1 = gmm_params_to_objects(gmm1_params)
        weights2, means2, covs2 = gmm_params_to_objects(gmm2_params)
        
        # Sample from gmm1
        samples = sample_from_gmm(weights1, means1, covs1, n_samples)
        
        # Compute log densities under both GMMs
        log_p = log_gmm_density(samples, weights1, means1, covs1)
        log_q = log_gmm_density(samples, weights2, means2, covs2)
        
        # KL divergence estimate
        kl_div = np.mean(log_p - log_q)
        
        return max(0.0, kl_div)  # KL divergence is non-negative
        
    except Exception as e:
        logger.error(f"Failed to compute KL divergence: {e}")
        return float('inf')


def sample_from_gmm(weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample points from a Gaussian Mixture Model.
    
    Args:
        weights: Component weights (shape: [n_components])
        means: Component means (shape: [n_components, n_features])
        covariances: Component covariances (shape depends on covariance_type)
        n_samples: Number of samples to generate
        
    Returns: Array of samples (shape: [n_samples, n_features])
    """
    n_components = len(weights)
    n_features = means.shape[1]
    
    # Normalize weights to ensure they sum to 1
    weights = weights / np.sum(weights)
    
    # Sample component indices based on weights
    component_indices = np.random.choice(n_components, size=n_samples, p=weights)
    
    # Generate samples from selected components
    samples = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        comp_idx = component_indices[i]
        mean = means[comp_idx]
        cov = covariances[comp_idx]
        
        # Add regularization for numerical stability
        cov_reg = cov + np.eye(n_features) * 1e-6
        
        try:
            samples[i] = np.random.multivariate_normal(mean, cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to diagonal covariance
            cov_diag = np.diag(np.diag(cov_reg))
            samples[i] = np.random.multivariate_normal(mean, cov_diag)
    
    return samples


def log_gmm_density(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """
    Compute log probability density of samples X under a GMM.
    
    Args:
        X: Samples (shape: [n_samples, n_features])
        weights: Component weights (shape: [n_components])
        means: Component means (shape: [n_components, n_features])
        covariances: Component covariances
        
    Returns: Log densities (shape: [n_samples])
    """
    n_samples = X.shape[0]
    n_components = len(weights)
    
    # Compute log probability for each component
    log_probs = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        mean = means[k]
        cov = covariances[k]
        
        # Multivariate Gaussian log probability
        diff = X - mean
        
        # Add small epsilon to diagonal for numerical stability
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        
        try:
            # Compute log determinant and inverse
            sign, log_det = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                log_det = 0  # Fallback
            
            cov_inv = np.linalg.inv(cov_reg)
            
            # Mahalanobis distance
            mahal = np.sum(diff @ cov_inv * diff, axis=1)
            
            # Log probability
            log_probs[:, k] = np.log(weights[k] + 1e-10) - 0.5 * (log_det + mahal + X.shape[1] * np.log(2 * np.pi))
            
        except np.linalg.LinAlgError:
            log_probs[:, k] = -np.inf
    
    # Use logsumexp for numerical stability
    from scipy.special import logsumexp
    return logsumexp(log_probs, axis=1)


def compute_jeffreys_divergence(gmm1_params: Dict, gmm2_params: Dict) -> float:
    """
    Compute Jeffreys Divergence (symmetric KL divergence) between two GMMs.
    
    Jeffreys Divergence: D_J(p||q) = D_KL(p||q) + D_KL(q||p)
    
    Args:
        gmm1_params: GMM parameters for first artist
        gmm2_params: GMM parameters for second artist
        
    Returns: Jeffreys divergence value (lower = more similar)
    """
    kl_pq = compute_kl_divergence_mc(gmm1_params, gmm2_params)
    kl_qp = compute_kl_divergence_mc(gmm2_params, gmm1_params)
    
    jeffreys_div = kl_pq + kl_qp
    
    return jeffreys_div


def serialize_gmm_for_hnsw(gmm_params: Dict) -> np.ndarray:
    """
    Serialize GMM parameters into a fixed-size vector for HNSW indexing.
    
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


def build_and_store_artist_index(db_conn=None):
    """
    Build HNSW index for artist similarity using GMM representations.
    
    This function:
    1. Fetches all artists and their tracks from the database
    2. Extracts audio features for each track
    3. Fits a GMM for each artist
    4. Builds an HNSW index for fast similarity search
    5. Stores the index in the database
    
    Args:
        db_conn: Database connection (if None, will acquire one)
    """
    if db_conn is None:
        from app_helper import get_db
        db_conn = get_db()
    
    logger.info("Starting to build artist similarity index using GMM + HNSW...")
    
    cur = db_conn.cursor()
    
    try:
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
        
        # Step 2: Fetch embeddings and fit GMMs
        logger.info("Fetching embeddings and fitting GMMs...")
        
        artist_gmms = {}
        artist_names_list = []
        
        for idx, (artist_name, tracks) in enumerate(artist_tracks.items(), 1):
            if idx % 50 == 0:
                logger.info(f"Processing artist {idx}/{len(artist_tracks)}: {artist_name}")
            
            # Fetch embeddings for all tracks by this artist
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
                    artist_gmms[artist_name] = gmm_params
                    artist_names_list.append(artist_name)
        
        logger.info(f"Successfully fitted GMMs for {len(artist_gmms)} artists")
        
        if len(artist_gmms) == 0:
            logger.warning("No valid GMMs created, skipping index build")
            return
        
        # Step 3: Build HNSW index
        logger.info("Building HNSW index...")
        
        # Determine dimensionality from first GMM
        first_gmm = artist_gmms[artist_names_list[0]]
        gmm_vector_dim = len(serialize_gmm_for_hnsw(first_gmm))
        
        logger.info(f"HNSW vector dimension: {gmm_vector_dim}")
        
        # Initialize HNSW index
        # Note: We use L2 space here, but actual similarity will use custom Jeffreys divergence
        # The HNSW index is primarily for fast approximate search structure
        index = hnswlib.Index(space='l2', dim=gmm_vector_dim)
        index.init_index(max_elements=len(artist_gmms), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        index.set_ef(HNSW_EF_SEARCH)
        
        # Create mappings
        artist_map_dict = {}
        reverse_artist_map_dict = {}
        
        # Prepare data for batch insertion
        vectors = []
        ids = []
        
        for hnsw_id, artist_name in enumerate(artist_names_list):
            gmm_params = artist_gmms[artist_name]
            
            # Serialize GMM to vector
            gmm_vector = serialize_gmm_for_hnsw(gmm_params)
            
            vectors.append(gmm_vector)
            ids.append(hnsw_id)
            
            artist_map_dict[hnsw_id] = artist_name
            reverse_artist_map_dict[artist_name] = hnsw_id
        
        # Add all vectors to index
        vectors_array = np.array(vectors)
        ids_array = np.array(ids)
        
        index.add_items(vectors_array, ids_array)
        
        logger.info(f"Added {len(artist_gmms)} artists to HNSW index")
        
        # Step 4: Serialize and store in database
        logger.info("Serializing and storing index in database...")
        
        # Serialize index to temp file then read bytes (hnswlib doesn't have save_to_bytes)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hnsw") as tmp:
                temp_file_path = tmp.name
            
            index.save_index(temp_file_path)
            
            with open(temp_file_path, 'rb') as f:
                index_bytes = f.read()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        # Serialize mappings and GMM parameters
        artist_map_json = json.dumps(artist_map_dict)
        gmm_params_json = json.dumps(artist_gmms)
        
        # Store in database (atomic update)
        cur.execute("""
            INSERT INTO artist_index_data (index_name, index_data, artist_map_json, gmm_params_json, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (index_name)
            DO UPDATE SET
                index_data = EXCLUDED.index_data,
                artist_map_json = EXCLUDED.artist_map_json,
                gmm_params_json = EXCLUDED.gmm_params_json,
                created_at = EXCLUDED.created_at
        """, (ARTIST_INDEX_NAME, index_bytes, artist_map_json, gmm_params_json))
        
        db_conn.commit()
        
        logger.info(f"Artist similarity index built and stored successfully ({len(artist_gmms)} artists)")
        
    except Exception as e:
        logger.error(f"Failed to build artist index: {e}", exc_info=True)
        db_conn.rollback()
        raise
    
    finally:
        cur.close()


def load_artist_index_for_querying(force_reload=False):
    """
    Load the artist HNSW index from database into memory.
    
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
        
        try:
            cur.execute("""
                SELECT index_data, artist_map_json, gmm_params_json
                FROM artist_index_data
                WHERE index_name = %s
            """, (ARTIST_INDEX_NAME,))
            
            row = cur.fetchone()
            
            if not row:
                logger.warning("Artist index not found in database")
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
                return
            
            index_bytes, artist_map_json, gmm_params_json = row
            
            # Deserialize mappings and GMM parameters
            artist_map_dict = json.loads(artist_map_json)
            gmm_params_dict = json.loads(gmm_params_json)
            
            # Convert string keys to integers for artist_map
            artist_map = {int(k): v for k, v in artist_map_dict.items()}
            
            # Build reverse map
            reverse_artist_map = {v: k for k, v in artist_map.items()}
            
            # Store GMM parameters
            artist_gmm_params = gmm_params_dict
            
            # Determine dimensionality
            first_artist = artist_map[0]
            first_gmm = gmm_params_dict[first_artist]
            gmm_vector_dim = len(serialize_gmm_for_hnsw(first_gmm))
            
            # Load HNSW index from bytes using temp file
            index = hnswlib.Index(space='l2', dim=gmm_vector_dim)
            
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".hnsw") as tmp:
                    temp_file_path = tmp.name
                    tmp.write(index_bytes)
                
                index.load_index(temp_file_path, max_elements=len(artist_map))
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            
            index.set_ef(HNSW_EF_SEARCH)
            
            artist_index = index
            
            logger.info(f"Artist index loaded successfully ({len(artist_map)} artists)")
            
        except Exception as e:
            logger.error(f"Failed to load artist index: {e}", exc_info=True)
            artist_index = None
            artist_map = None
            reverse_artist_map = None
            artist_gmm_params = None
        
        finally:
            cur.close()


def find_similar_artists(query_artist, n: int = 10, ef_search: Optional[int] = None) -> List[Dict]:
    """
    Find similar artists using the HNSW index.
    Accepts either artist name or artist ID.
    
    Args:
        query_artist: Name or ID of the query artist
        n: Number of similar artists to return
        ef_search: HNSW search parameter (higher = more accurate but slower)
        
    Returns: List of dictionaries with 'artist', 'artist_id', and 'divergence' keys
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
    
    # Get query artist's HNSW ID
    query_id = reverse_artist_map[artist_name]
    
    # Get query GMM parameters
    query_gmm = artist_gmm_params[artist_name]
    
    # Set ef_search if provided
    if ef_search is not None:
        artist_index.set_ef(ef_search)
    
    # HNSW search: get similar artists based on weighted mean distance
    k_candidates = min(n + 1, len(artist_map))  # +1 to account for self
    
    # Get query vector (weighted mean of GMM)
    query_vector = serialize_gmm_for_hnsw(query_gmm)
    
    # Search
    labels, distances = artist_index.knn_query(query_vector, k=k_candidates)
    
    # Build results (skip self)
    results = []
    
    from app_helper_artist import get_artist_id_by_name
    
    for idx, dist in zip(labels[0], distances[0]):
        if idx == query_id:
            continue  # Skip self
        
        candidate_artist = artist_map[idx]
        candidate_artist_id = get_artist_id_by_name(candidate_artist)
        
        results.append({
            'artist': candidate_artist,
            'artist_id': candidate_artist_id,
            'divergence': float(dist)  # Use L2 distance as similarity score
        })
    
    # Return top N (already sorted by HNSW)
    return results[:n]


def search_artists_by_name(query: str, limit: int = 15) -> List[Dict]:
    """
    Search for artists by name (for autocomplete).
    Returns both artist name and ID if available.
    
    Args:
        query: Search query string
        limit: Maximum number of results
        
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
            LIMIT %s
        """, (query_pattern, limit))
        
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
