# tasks/artist_gmm_manager.py
"""
Artist Similarity using Gaussian Mixture Models (GMM) and Voyager Index.

This module implements artist similarity by:
1. Using existing embedding vectors (same as song similarity) for all tracks per artist
2. Fitting a GMM to the embedding vectors to represent each artist's "sound profile"
3. Building a Voyager index for fast approximate nearest neighbor search
4. Using Jeffreys Divergence (symmetric KL divergence) to measure GMM similarity

This approach is fast (no audio re-processing) and consistent with the song similarity system.
"""

import os
import logging
import json
import pickle
import tempfile
import io
import re
import hashlib
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.mixture import GaussianMixture
import voyager  # type: ignore

logger = logging.getLogger(__name__)
from config import ARTIST_INDEX_MAX_PART_SIZE_MB

# --- Configuration ---
ARTIST_INDEX_NAME = 'artist_similarity_index'
GMM_N_COMPONENTS_MIN = 2  # Minimum number of GMM components
GMM_N_COMPONENTS_MAX = 10  # Maximum number of GMM components (will auto-select using BIC)
GMM_COVARIANCE_TYPE = 'diag'  # 'diag' is faster than 'full' and works well for high-dim embeddings
GMM_MAX_ITER = 100
GMM_N_INIT = 3
MIN_TRACKS_PER_ARTIST = 1  # Minimum tracks needed to build a GMM for an artist (lowered to include all artists)
ARTIST_INDEX_MAX_PART_SIZE = ARTIST_INDEX_MAX_PART_SIZE_MB * 1024 * 1024  # bytes threshold for segmented artist index storage

def _split_bytes(data: bytes, part_size: int) -> list:
    """Split `data` into byte chunks, each <= part_size."""
    return [data[i:i + part_size] for i in range(0, len(data), part_size)]

# Voyager index parameters (similar to song index)
VOYAGER_M = 32  # Number of bi-directional links created for every new element
VOYAGER_EF_CONSTRUCTION = 200  # Size of the dynamic list during index construction
VOYAGER_EF_SEARCH = 100  # Size of the dynamic list during search (can be adjusted at query time)

# Monte Carlo sampling for KL divergence approximation
MC_SAMPLES = 500  # Number of samples for Monte Carlo KL divergence estimation (reduced for speed)

# --- Global cache for the loaded artist index ---
artist_index = None  # voyager.Index object
artist_map = None  # {voyager_int_id: artist_name}
reverse_artist_map = None  # {artist_name: voyager_int_id}
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
            
            # Use a small fixed covariance for numerical stability
            # This acts like narrow Gaussians centered on each actual song
            fixed_variance = 0.01
            
            # Each song becomes one component with equal weight
            n_components = n_samples
            weights = [1.0 / n_components] * n_components
            means = all_embeddings.tolist()  # Each row is a song's embedding
            covariances = [[fixed_variance] * n_features] * n_components  # Same small variance for all
            
            gmm_params = {
                'weights': weights,
                'means': means,
                'covariances': covariances,
                'n_components': n_components,
                'covariance_type': GMM_COVARIANCE_TYPE,
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
        
        # Extract GMM parameters
        gmm_params = {
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.tolist(),
            'covariances': gmm.covariances_.tolist(),
            'n_components': optimal_n_components,  # Store the actual number used
            'covariance_type': GMM_COVARIANCE_TYPE,
            'n_features': all_embeddings.shape[1],
            'n_tracks': len(track_embeddings),
            'is_few_songs': False
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
        cov_diag = covariances[comp_idx]  # Shape: [n_features] for 'diag' type
        
        # Convert diagonal covariance to full matrix for multivariate_normal
        # Add small regularization for numerical stability
        cov_matrix = np.diag(cov_diag + 1e-6)
        
        try:
            samples[i] = np.random.multivariate_normal(mean, cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback: sample from univariate normals independently
            samples[i] = mean + np.random.randn(n_features) * np.sqrt(cov_diag + 1e-6)
    
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
        cov_diag = covariances[k]  # Shape: [n_features] for 'diag' type
        
        # Multivariate Gaussian log probability
        diff = X - mean
        
        # Add small regularization for numerical stability
        cov_diag_reg = cov_diag + 1e-6
        
        try:
            # For diagonal covariance, computations are simpler
            # Log determinant: sum of log of diagonal elements
            log_det = np.sum(np.log(cov_diag_reg))
            
            # Mahalanobis distance: sum of (diff^2 / variance) for each dimension
            mahal = np.sum(diff**2 / cov_diag_reg, axis=1)
            
            # Log probability
            log_probs[:, k] = np.log(weights[k] + 1e-10) - 0.5 * (log_det + mahal + X.shape[1] * np.log(2 * np.pi))
            
        except Exception as e:
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
    Serialize GMM parameters into a fixed-size vector for Voyager indexing.
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


def build_and_store_artist_index(db_conn=None):
    """
    Build Voyager index for artist similarity using GMM representations.
    
    This function:
    1. Fetches all artists and their tracks from the database
    2. Extracts audio features for each track
    3. Fits a GMM for each artist
    4. Builds a Voyager index for fast similarity search
    5. Stores the index in the database
    
    Args:
        db_conn: Database connection (if None, will acquire one)
    """
    if db_conn is None:
        from app_helper import get_db
        db_conn = get_db()
    
    logger.info("Starting to build artist similarity index using GMM + Voyager...")
    
    cur = db_conn.cursor()
    
    try:
        # Step 0: Load existing GMM params for incremental rebuild
        existing_gmm_params = None
        try:
            cur.execute("""
                SELECT gmm_params_json
                FROM artist_index_data
                WHERE index_name = %s
            """, (ARTIST_INDEX_NAME,))
            row = cur.fetchone()
            if row and row[0]:
                existing_gmm_params = json.loads(row[0])
                logger.info(f"Loaded existing GMM params for {len(existing_gmm_params)} artists (incremental mode)")
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
            artist_track_hashes[artist_name] = hashlib.md5(hash_input.encode()).hexdigest()

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
        
        # Step 3: Build Voyager index
        logger.info("Building Voyager index...")
        
        # Determine dimensionality from first GMM
        first_gmm = artist_gmms[artist_names_list[0]]
        gmm_vector_dim = len(serialize_gmm_for_hnsw(first_gmm))
        
        logger.info(f"Voyager vector dimension: {gmm_vector_dim}")
        
        # Adaptive parameters based on dataset size
        # For small datasets, Voyager needs smaller M and ef_construction values
        num_artists = len(artist_gmms)
        
        # M: number of bi-directional links (should be smaller for small datasets)
        # Rule of thumb: M should be at most num_elements / 2, typically 12-48
        if num_artists < 100:
            M = min(12, max(4, num_artists // 4))
        elif num_artists < 1000:
            M = 16
        else:
            M = VOYAGER_M
        
        # ef_construction: must be > M, but not too large for small datasets
        # Rule of thumb: 2*M to 10*M depending on size
        if num_artists < 100:
            ef_construction = max(M + 1, min(100, num_artists * 2))
        elif num_artists < 1000:
            ef_construction = 100
        else:
            ef_construction = VOYAGER_EF_CONSTRUCTION
        
        logger.info(f"Using adaptive Voyager parameters for {num_artists} artists: M={M}, ef_construction={ef_construction}")
        
        # Initialize Voyager index
        # Note: We use L2 (Euclidean) space here, but actual similarity will use custom Jeffreys divergence
        # The Voyager index is primarily for fast approximate search structure
        index = voyager.Index(voyager.Space.Euclidean, num_dimensions=gmm_vector_dim, M=M, ef_construction=ef_construction)
        
        # Create mappings
        artist_map_dict = {}
        reverse_artist_map_dict = {}
        
        # Prepare data for batch insertion
        vectors = []
        ids = []
        
        for voyager_id, artist_name in enumerate(artist_names_list):
            gmm_params = artist_gmms[artist_name]
            
            # Serialize GMM to vector
            gmm_vector = serialize_gmm_for_hnsw(gmm_params)
            
            vectors.append(gmm_vector)
            ids.append(voyager_id)
            
            artist_map_dict[voyager_id] = artist_name
            reverse_artist_map_dict[artist_name] = voyager_id
        
        # Add all vectors to index (batch add for better performance)
        try:
            vectors_array = np.array(vectors, dtype=np.float32)
            ids_array = np.array(ids, dtype=np.int64)
            
            logger.info(f"Adding {len(vectors)} vectors with shape {vectors_array.shape} to Voyager index...")
            index.add_items(vectors_array, ids=ids_array)
            
            logger.info(f"Successfully added {len(artist_gmms)} artists to Voyager index (num_elements={index.num_elements})")
        except Exception as e:
            logger.error(f"Failed to add items to Voyager index: {e}", exc_info=True)
            raise
        
        # Step 4: Serialize and store in database
        logger.info("Serializing and storing index in database...")
        
        # Serialize index to bytes using temp file (voyager uses save/load with files)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as tmp:
                temp_file_path = tmp.name
            
            logger.info(f"Saving Voyager index to temp file: {temp_file_path}")
            index.save(temp_file_path)
            
            with open(temp_file_path, 'rb') as f:
                index_bytes = f.read()
            
            logger.info(f"Index serialized to {len(index_bytes)} bytes")
        except Exception as e:
            logger.error(f"Failed to serialize Voyager index: {e}", exc_info=True)
            raise
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        # Serialize mappings and GMM parameters
        artist_map_json = json.dumps(artist_map_dict)
        gmm_params_json = json.dumps(artist_gmms)
        
        # Store in database (atomic update)
        try:
            # Delete any existing single or segmented rows for this logical index name
            cur.execute("DELETE FROM artist_index_data WHERE index_name = %s OR index_name LIKE %s", (ARTIST_INDEX_NAME, ARTIST_INDEX_NAME + "_%_%"))

            # Small enough to store in a single row (backwards-compatible)
            if len(index_bytes) <= ARTIST_INDEX_MAX_PART_SIZE:
                cur.execute("""
                    INSERT INTO artist_index_data (index_name, index_data, artist_map_json, gmm_params_json, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (index_name) DO UPDATE SET
                        index_data = EXCLUDED.index_data,
                        artist_map_json = EXCLUDED.artist_map_json,
                        gmm_params_json = EXCLUDED.gmm_params_json,
                        created_at = EXCLUDED.created_at
                """, (ARTIST_INDEX_NAME, index_bytes, artist_map_json, gmm_params_json))
                logger.info("Stored artist index as a single row (no segmentation required).")
            else:
                parts = _split_bytes(index_bytes, ARTIST_INDEX_MAX_PART_SIZE)
                num_parts = len(parts)
                logger.info(f"Artist index size {len(index_bytes)} exceeds {ARTIST_INDEX_MAX_PART_SIZE_MB}MB - storing as {num_parts} segmented rows.")

                insert_q = "INSERT INTO artist_index_data (index_name, index_data, artist_map_json, gmm_params_json, created_at) VALUES (%s, %s, %s, %s, NOW())"
                for idx, part in enumerate(parts, start=1):
                    part_name = f"{ARTIST_INDEX_NAME}_{idx}_{num_parts}"
                    # store full metadata only in the first part; other parts keep empty strings
                    part_artist_map_json = artist_map_json if idx == 1 else ''
                    part_gmm_params_json = gmm_params_json if idx == 1 else ''
                    cur.execute(insert_q, (part_name, part, part_artist_map_json, part_gmm_params_json))

                logger.info(f"Stored artist index in {num_parts} parts (prefix='{ARTIST_INDEX_NAME}_<part>_<total>').")

            db_conn.commit()
            logger.info(f"Artist similarity index built and stored successfully ({len(artist_gmms)} artists)")

        except Exception as e:
            try:
                db_conn.rollback()
            except Exception:
                pass
            logger.error("Failed to store segmented artist index: %s", e, exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"Failed to build artist index: {e}", exc_info=True)
        db_conn.rollback()
        raise
    
    finally:
        cur.close()


def load_artist_index_for_querying(force_reload=False):
    """
    Load the artist Voyager index from database into memory.
    
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

            if row and row[0]:
                # Single-row index found (backwards compatible)
                index_bytes, artist_map_json, gmm_params_json = row

                logger.info(f"Retrieved artist index from database: {len(index_bytes)} bytes")

                # Deserialize mappings and GMM parameters
                artist_map_dict = json.loads(artist_map_json)
                gmm_params_dict = json.loads(gmm_params_json)

                logger.info(f"Deserialized metadata: {len(artist_map_dict)} artists")

                # Convert string keys to integers for artist_map
                artist_map = {int(k): v for k, v in artist_map_dict.items()}

                # Build reverse map
                reverse_artist_map = {v: k for k, v in artist_map.items()}

                # Store GMM parameters
                artist_gmm_params = gmm_params_dict

                # Load Voyager index from bytes using BytesIO stream
                logger.info("Loading Voyager index from BytesIO stream...")
                try:
                    index_stream = io.BytesIO(index_bytes)
                    index = voyager.Index.load(index_stream)

                    artist_index = index

                    logger.info(f"Artist index loaded successfully ({len(artist_map)} artists, num_elements={artist_index.num_elements})")
                except Exception as load_error:
                    logger.error(f"Failed to load Voyager index from BytesIO: {load_error}", exc_info=True)
                    raise

                return

            # Not found as single row — try segmented parts named ARTIST_INDEX_NAME_<part>_<total>
            cur.execute("SELECT index_name, index_data, artist_map_json, gmm_params_json FROM artist_index_data WHERE index_name LIKE %s", (ARTIST_INDEX_NAME + "_%_%",))
            candidates = cur.fetchall()

            if not candidates:
                logger.warning("Artist index not found in database (single or segmented)")
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
                return

            seg_pattern = re.compile(rf"^{re.escape(ARTIST_INDEX_NAME)}_(\d+)_(\d+)$")
            parts = []
            total_expected = None
            artist_map_json_candidate = None
            gmm_params_json_candidate = None

            for row in candidates:
                name, part_data, part_artist_map_json, part_gmm_params_json = row
                m = seg_pattern.match(name)
                if not m:
                    continue
                part_no = int(m.group(1))
                total = int(m.group(2))
                if total_expected is None:
                    total_expected = total
                elif total_expected != total:
                    logger.error(f"Segment total mismatch for Artist index parts (found totals {total_expected} and {total}). Aborting load.")
                    artist_index = None
                    artist_map = None
                    reverse_artist_map = None
                    artist_gmm_params = None
                    return

                parts.append((part_no, part_data, part_artist_map_json, part_gmm_params_json))
                if part_artist_map_json and not artist_map_json_candidate:
                    artist_map_json_candidate = part_artist_map_json
                if part_gmm_params_json and not gmm_params_json_candidate:
                    gmm_params_json_candidate = part_gmm_params_json

            if not parts:
                logger.error(f"No valid segmented Artist index rows found for prefix '{ARTIST_INDEX_NAME}'.")
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
                return

            if total_expected is None or len(parts) != total_expected:
                logger.error(f"Incomplete Artist index segments: expected {total_expected}, found {len(parts)}. Aborting load to avoid corruption.")
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
                return

            parts.sort(key=lambda p: p[0])

            # Reassemble binary and pick metadata from first non-empty segment
            index_bytes = b"".join([p[1] for p in parts])
            if not index_bytes:
                logger.error("Reassembled Artist index binary is empty. Aborting load.")
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
                return

            if not artist_map_json_candidate or not gmm_params_json_candidate:
                logger.error("No non-empty artist_map_json/gmm_params_json found in segmented Artist index rows. Aborting load.")
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
                return

            try:
                index_stream = io.BytesIO(index_bytes)
                index = voyager.Index.load(index_stream)

                parsed_artist_map = {int(k): v for k, v in json.loads(artist_map_json_candidate).items()}
                # Validate element counts if possible
                try:
                    idx_count = getattr(index, 'num_elements', None)
                except Exception:
                    idx_count = None

                if idx_count is not None and idx_count != len(parsed_artist_map):
                    logger.error(f"Artist index element count mismatch after reassembly: index.num_elements={idx_count}, artist_map={len(parsed_artist_map)}. Aborting load.")
                    artist_index = None
                    artist_map = None
                    reverse_artist_map = None
                    artist_gmm_params = None
                    return

                artist_index = index
                artist_map = parsed_artist_map
                reverse_artist_map = {v: k for k, v in artist_map.items()}
                artist_gmm_params = json.loads(gmm_params_json_candidate)

                logger.info(f"Artist segmented index ({len(parts)} parts) with {len(artist_map)} artists loaded successfully into memory.")
                return
            except Exception as load_error:
                logger.error(f"Failed to load reassembled Artist index: {load_error}", exc_info=True)
                artist_index = None
                artist_map = None
                reverse_artist_map = None
                artist_gmm_params = None
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
    
    # Compute pairwise distances between all components
    # Distance[i,j] = distance between component i of artist1 and component j of artist2
    distances = np.linalg.norm(means1[:, np.newaxis, :] - means2[np.newaxis, :, :], axis=2)
    
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
    Find similar artists using the Voyager index.
    Accepts either artist name or artist ID.
    
    Args:
        query_artist: Name or ID of the query artist
        n: Number of similar artists to return
        ef_search: Voyager search parameter (higher = more accurate but slower)
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
    
    # Get query artist's Voyager ID
    query_id = reverse_artist_map[artist_name]
    
    # Get query GMM parameters
    query_gmm = artist_gmm_params[artist_name]
    
    # Set ef_search if provided (voyager uses .ef property instead of set_ef method)
    if ef_search is not None:
        artist_index.ef = ef_search
    
    # Voyager search: get similar artists based on weighted mean distance
    k_candidates = min(n + 1, len(artist_map))  # +1 to account for self
    
    # Get query vector (weighted mean of GMM)
    query_vector = serialize_gmm_for_hnsw(query_gmm)
    
    # Voyager search: get similar artists based on weighted mean distance
    try:
        labels, distances = artist_index.query(query_vector, k=k_candidates)
    except Exception as e:
        logger.error(f"Voyager query failed for artist '{artist_name}': {e}", exc_info=True)
        return []
    
    # Build results (skip self)
    results = []
    
    from app_helper_artist import get_artist_id_by_name
    
    for idx, dist in zip(labels, distances):
        if idx == query_id:
            continue  # Skip self
        
        candidate_artist = artist_map[idx]
        candidate_artist_id = get_artist_id_by_name(candidate_artist)
        candidate_gmm = artist_gmm_params[candidate_artist]
        
        result = {
            'artist': candidate_artist,
            'artist_id': candidate_artist_id,
            'divergence': float(dist)  # Use L2 distance as similarity score
        }
        
        # Add component-level matching if requested
        if include_component_matches:
            component_matches = compute_component_matches(
                query_gmm, 
                candidate_gmm, 
                artist_name, 
                candidate_artist,
                top_k=3  # Top 3 component matches
            )
            result['component_matches'] = component_matches
            result['query_artist_components'] = query_gmm['n_components']
            result['candidate_artist_components'] = candidate_gmm['n_components']
        
        results.append(result)
    
    # Return top N (already sorted by Voyager)
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
