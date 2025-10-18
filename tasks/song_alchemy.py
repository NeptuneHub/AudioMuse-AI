import logging
from typing import List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id
from app_helper import get_score_data_by_ids
import config

try:
    # sklearn is already a dependency; import lazily for environments where it's present
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
except Exception:
    PCA = None
    LogisticRegression = None

logger = logging.getLogger(__name__)


def _compute_centroid_from_ids(ids: List[str]) -> np.ndarray:
    """Fetch vectors by id and compute their centroid (mean)."""
    vectors = []
    for item_id in ids:
        vec = get_vector_by_id(item_id)
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def _project_to_2d(vectors: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Simple PCA via SVD to project a list of vectors to 2D.
    Returns a list of (x, y) tuples in the same order as input vectors.
    If there are fewer than 2 vectors, returns zeros for all.
    """
    if not vectors:
        return []
    mat = np.vstack(vectors)
    # Center
    mean = np.mean(mat, axis=0)
    mat_c = mat - mean
    # SVD
    try:
        u, s, vh = np.linalg.svd(mat_c, full_matrices=False)
    except Exception:
        # Fallback: return zeros
        return [(0.0, 0.0) for _ in vectors]
    # Take first two principal components
    pcs = vh[:2]
    proj = mat_c.dot(pcs.T)
    # Normalize projection for nicer plotting
    if proj.size == 0:
        return [(0.0, 0.0) for _ in vectors]
    # Normalize preserving aspect ratio: use a single global scale so x/y units are comparable
    # center at zero
    proj_centered = proj - proj.mean(axis=0)
    max_abs = np.max(np.abs(proj_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in vectors]
    scaled = proj_centered / max_abs
    # clamp to [-1,1] for safety
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def _project_aligned_add_sub(vectors: List[np.ndarray], add_centroid: np.ndarray, subtract_centroid: np.ndarray) -> List[Tuple[float, float]]:
    """Project vectors to 2D where the x-axis is aligned with the vector
    from add_centroid -> subtract_centroid. The y-axis is the leading
    orthogonal component (first PC of residuals).
    This emphasizes separation along the add-vs-subtract direction.
    """
    if not vectors:
        return []
    # Convert list to matrix and center relative to add_centroid
    mat = np.vstack(vectors)
    rel = mat - add_centroid
    axis = subtract_centroid - add_centroid
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        # Fallback to PCA if centroids coincide
        return _project_to_2d(vectors)
    axis_u = axis / axis_norm

    # Compute x coordinates as projection on axis
    x_coords = rel.dot(axis_u)

    # Remove axis component to get residuals for y-axis computation
    proj_on_axis = np.outer(x_coords, axis_u)
    residuals = rel - proj_on_axis

    # Find leading direction in residuals via SVD
    try:
        # If residuals are all near-zero, SVD will still succeed but produce small values
        u, s, vh = np.linalg.svd(residuals, full_matrices=False)
        if vh.shape[0] >= 1:
            y_u = vh[0]
        else:
            y_u = None
    except Exception:
        y_u = None

    if y_u is None or np.linalg.norm(y_u) == 0:
        # Create an arbitrary orthogonal vector to axis_u
        # pick an index where axis_u has smallest absolute value
        idx = int(np.argmin(np.abs(axis_u)))
        e = np.zeros_like(axis_u)
        e[idx] = 1.0
        y_u = e - np.dot(e, axis_u) * axis_u
        norm_y = np.linalg.norm(y_u)
        if norm_y == 0:
            # fallback
            return _project_to_2d(vectors)
        y_u = y_u / norm_y
    else:
        # ensure orthogonal to axis_u (numerical stability)
        y_u = y_u - np.dot(y_u, axis_u) * axis_u
        y_u_norm = np.linalg.norm(y_u)
        if y_u_norm == 0:
            return _project_to_2d(vectors)
        y_u = y_u / y_u_norm

    y_coords = residuals.dot(y_u)

    coords = np.vstack([x_coords, y_coords]).T
    # Center and scale uniformly so x and y share same units
    coords_centered = coords - coords.mean(axis=0)
    max_abs = np.max(np.abs(coords_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in vectors]
    scaled = coords_centered / max_abs
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def _project_with_umap(vectors: List[np.ndarray], n_components: int = 2) -> List[Tuple[float, float]]:
    """Try to project using UMAP if available. Raises ImportError if umap is not installed."""
    import umap
    if not vectors:
        return []
    mat = np.vstack(vectors)
    reducer = umap.UMAP(n_components=n_components, random_state=42, n_jobs=-1)
    embedding = reducer.fit_transform(mat)
    # Center and scale uniformly so x and y share same units
    emb_centered = embedding - embedding.mean(axis=0)
    max_abs = np.max(np.abs(emb_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in vectors]
    scaled = emb_centered / max_abs
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def _project_with_discriminant(add_vectors: List[np.ndarray], sub_vectors: List[np.ndarray], all_vectors: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Compute a discriminant direction separating add and sub using PCA+LogisticRegression.
    Returns 2D coords for all_vectors projected onto (discriminant axis, residual axis).
    Falls back (raises) if sklearn not available or insufficient samples.
    """
    if LogisticRegression is None or PCA is None:
        raise RuntimeError('sklearn not available')
    # Need at least one sample in each class
    if not add_vectors or not sub_vectors:
        raise RuntimeError('Insufficient classes for discriminant')

    X_train = np.vstack([np.vstack(add_vectors), np.vstack(sub_vectors)])
    y_train = np.array([1] * len(add_vectors) + [0] * len(sub_vectors))

    n_samples, n_features = X_train.shape
    # Reduce dimensionality so training is stable (components <= n_samples-1)
    max_components = min(32, n_samples - 1, n_features)
    if max_components < 1:
        raise RuntimeError('Not enough samples for discriminant PCA')

    pca = PCA(n_components=max_components, random_state=42)
    Xp = pca.fit_transform(X_train)

    # Fit logistic regression with regularization for robustness
    try:
        # Use 'saga' solver with n_jobs=-1 to leverage multiple cores
        clf = LogisticRegression(penalty='l2', C=1.0, solver='saga', max_iter=1000, n_jobs=-1)
        clf.fit(Xp, y_train)
    except Exception:
        # Fallback with less regularization if solver fails
        clf = LogisticRegression(penalty='l2', C=0.1, solver='saga', max_iter=1000, n_jobs=-1)
        clf.fit(Xp, y_train)

    # direction in PCA space
    coef = clf.coef_.ravel()
    norm = np.linalg.norm(coef)
    if norm == 0:
        raise RuntimeError('Discriminant produced zero vector')
    dir_pca = coef / norm

    # Project all vectors into PCA space then onto discriminant for x coords
    all_mat = np.vstack(all_vectors)
    all_pca = pca.transform(all_mat)
    x_coords = all_pca.dot(dir_pca)

    # Residuals in PCA space
    proj_on_dir = np.outer(x_coords, dir_pca)
    residuals = all_pca - proj_on_dir
    # y direction: leading PC of residuals
    try:
        u, s, vh = np.linalg.svd(residuals, full_matrices=False)
        if vh.shape[0] >= 1:
            y_u = vh[0]
        else:
            y_u = None
    except Exception:
        y_u = None

    if y_u is None or np.linalg.norm(y_u) == 0:
        # fallback: arbitrary orthogonal
        idx = int(np.argmin(np.abs(dir_pca)))
        e = np.zeros_like(dir_pca)
        e[idx] = 1.0
        y_u = e - np.dot(e, dir_pca) * dir_pca
        y_u = y_u / (np.linalg.norm(y_u) or 1.0)
    else:
        y_u = y_u - np.dot(y_u, dir_pca) * dir_pca
        y_u = y_u / (np.linalg.norm(y_u) or 1.0)

    y_coords = residuals.dot(y_u)

    coords = np.vstack([x_coords, y_coords]).T
    coords_centered = coords - coords.mean(axis=0)
    max_abs = np.max(np.abs(coords_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in all_vectors]
    scaled = coords_centered / max_abs
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def song_alchemy(add_ids: List[str], subtract_ids: List[str], n_results: int = None, subtract_distance: float = None) -> dict:
    """Perform Song Alchemy:
    - add_ids: items to include in positive centroid
    - subtract_ids: items to include in negative centroid
    - n_results: number of similar songs to fetch (default from config)

    Returns list of song detail dicts (using get_score_data_by_ids mapping)
    """
    if n_results is None:
        n_results = config.ALCHEMY_DEFAULT_N_RESULTS
    n_results = min(n_results, config.ALCHEMY_MAX_N_RESULTS)

    # Allow one or more songs in the ADD set — a single-song centroid is valid
    if not add_ids or len(add_ids) < 1:
        raise ValueError("At least one song must be in the ADD set")
    # Remove rigid 10-song per group limit; allow any reasonable number (server-side configs still cap results)

    add_centroid = _compute_centroid_from_ids(add_ids)
    if add_centroid is None:
        return {"results": [], "filtered_out": [], "centroid_2d": None}

    subtract_centroid = None
    if subtract_ids:
        subtract_centroid = _compute_centroid_from_ids(subtract_ids)

    # Find nearest neighbors to add_centroid using Voyager
    neighbors = find_nearest_neighbors_by_vector(add_centroid, n=n_results * 3)
    if not neighbors:
        return {"results": [], "filtered_out": [], "centroid_2d": None}

    # neighbors is a list of dicts with item_id and score; keep candidate ids
    candidate_ids = [n['item_id'] for n in neighbors]

    # If subtract centroid present, filter candidates by distance
    filtered_out = []
    filtered = candidate_ids # Start with all candidates
    if subtract_centroid is not None:
        # Compute distances and keep those farther than threshold
        filtered = []
        # Use provided override or default from config depending on metric
        if subtract_distance is None:
            if config.PATH_DISTANCE_METRIC == 'angular':
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
            else:
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
        else:
            threshold = subtract_distance
            
        for cid in candidate_ids:
            vec = get_vector_by_id(cid)
            if vec is None: continue

            v_sub = np.array(vec, dtype=float)
            # Use same metric as PATH: angular => cosine-derived; else euclidean
            if config.PATH_DISTANCE_METRIC == 'angular':
                # angular distance as in path_manager: arccos(cosine)/pi
                v1 = subtract_centroid / (np.linalg.norm(subtract_centroid) or 1.0)
                v2 = v_sub / (np.linalg.norm(v_sub) or 1.0)
                cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angular_distance = np.arccos(cosine) / np.pi
                keep = angular_distance >= threshold
            else:
                # Euclidean
                dist = np.linalg.norm(subtract_centroid - v_sub)
                keep = dist >= threshold
            
            if keep:
                filtered.append(cid)
            else:
                filtered_out.append(cid)

    candidate_ids = filtered

    # Trim to desired n_results
    candidate_ids = candidate_ids[:n_results]

    # Compute distance from add_centroid for each candidate (to display in results)
    # Gather vectors for projection
    proj_vectors = []
    proj_ids = []
    # include add_centroid and subtract_centroid in the matrix so projection aligns
    # include individual add/subtract song vectors first so we can show them explicitly
    add_meta = []
    if add_ids:
        add_details = get_score_data_by_ids(add_ids)
        add_map = {d['item_id']: d for d in add_details}
        for aid in add_ids:
            vec = get_vector_by_id(aid)
            if vec is not None:
                proj_vectors.append(np.array(vec, dtype=float))
                proj_ids.append(f'__add_id__{aid}')
                # store metadata placeholder; we'll attach embedding_2d later
                add_meta.append({'item_id': aid, 'title': add_map.get(aid, {}).get('title'), 'author': add_map.get(aid, {}).get('author')})

    sub_meta = []
    if subtract_ids:
        sub_details = get_score_data_by_ids(subtract_ids)
        sub_map = {d['item_id']: d for d in sub_details}
        for sid in subtract_ids:
            vec = get_vector_by_id(sid)
            if vec is not None:
                proj_vectors.append(np.array(vec, dtype=float))
                proj_ids.append(f'__sub_id__{sid}')
                sub_meta.append({'item_id': sid, 'title': sub_map.get(sid, {}).get('title'), 'author': sub_map.get(sid, {}).get('author')})

    # include centroids as well so they are in the same projection space
    if add_centroid is not None:
        proj_vectors.append(add_centroid)
        proj_ids.append('__add_centroid__')
    if subtract_centroid is not None:
        proj_vectors.append(subtract_centroid)
        proj_ids.append('__subtract_centroid__')

    # keep track of mapping for candidate and filtered_out
    for cid in candidate_ids:
        vec = get_vector_by_id(cid)
        if vec is None:
            continue
        proj_vectors.append(np.array(vec, dtype=float))
        proj_ids.append(cid)
    for fid in filtered_out:
        vec = get_vector_by_id(fid)
        if vec is None:
            continue
        proj_vectors.append(np.array(vec, dtype=float))
        proj_ids.append(fid)

    # Project to 2D - Choose the best method based on input
    projection_used = 'none'
    projections = None
    
    # Build lists of add and sub vectors (raw vectors) for projection methods
    add_vecs = [np.array(get_vector_by_id(aid), dtype=float) for aid in add_ids if get_vector_by_id(aid) is not None]
    sub_vecs = [np.array(get_vector_by_id(sid), dtype=float) for sid in subtract_ids if get_vector_by_id(sid) is not None]

    if add_vecs and sub_vecs:
        # Case 1: Both ADD and SUBTRACT songs are provided. Prioritize separation.
        logger.info("Trying discriminant projection...")
        try:
            projections = _project_with_discriminant(add_vecs, sub_vecs, proj_vectors)
            projection_used = 'discriminant'
            logger.info("Using discriminant projection.")
        except Exception as e:
            logger.warning(f"Discriminant projection failed: {e}. Falling back.")
            try:
                logger.info("Trying aligned_add_sub projection...")
                projections = _project_aligned_add_sub(proj_vectors, add_centroid, subtract_centroid)
                projection_used = 'aligned_add_sub'
                logger.info("Using aligned_add_sub projection.")
            except Exception as e2:
                logger.warning(f"Aligned_add_sub projection failed: {e2}. Falling back to UMAP.")
                projections = None # Ensure fallback to next block
    
    if projections is None:
        # Case 2: Only ADD songs, or previous methods failed. Prioritize structure discovery.
        try:
            logger.info("Trying UMAP projection...")
            projections = _project_with_umap(proj_vectors)
            projection_used = 'umap'
            logger.info("Using UMAP projection.")
        except (ImportError, Exception) as e:
            logger.warning(f"UMAP projection failed: {e}. Falling back to PCA.")
            try:
                logger.info("Trying PCA projection...")
                projections = _project_to_2d(proj_vectors)
                projection_used = 'pca'
                logger.info("Using PCA projection.")
            except Exception as e2:
                logger.error(f"PCA projection also failed: {e2}. Returning no projections.")
                projections = [(0.0, 0.0) for _ in proj_vectors]
                projection_used = 'failed'

    proj_map = {pid: coord for pid, coord in zip(proj_ids, projections)}

    # Compute distances from add_centroid for display
    distances = {}
    for cid in candidate_ids:
        vec = get_vector_by_id(cid)
        if vec is None:
            continue
        v = np.array(vec, dtype=float)
        if config.PATH_DISTANCE_METRIC == 'angular':
            a = add_centroid / (np.linalg.norm(add_centroid) or 1.0)
            b = v / (np.linalg.norm(v) or 1.0)
            cosine = np.clip(np.dot(a, b), -1.0, 1.0)
            dist = float(np.arccos(cosine) / np.pi)
        else:
            dist = float(np.linalg.norm(add_centroid - v))
        distances[cid] = dist

    # Fetch details
    details = get_score_data_by_ids(candidate_ids)
    # Preserve order in candidate_ids
    details_map = {d['item_id']: d for d in details}
    ordered = []
    for i in candidate_ids:
        if i in details_map:
            item = details_map[i]
            # attach distance if available
            item['distance'] = distances.get(i)
            # attach 2d projection if available
            item['embedding_2d'] = proj_map.get(i)
            ordered.append(item)

    # Prepare filtered_out details
    filtered_details = []
    if filtered_out:
        details_f = get_score_data_by_ids(filtered_out)
        details_f_map = {d['item_id']: d for d in details_f}
        for fid in filtered_out:
            if fid in details_f_map:
                fd = details_f_map[fid]
                fd['embedding_2d'] = proj_map.get(fid)
                filtered_details.append(fd)

    # Centroid projections
    centroid_2d = proj_map.get('__add_centroid__')
    subtract_centroid_2d = proj_map.get('__subtract_centroid__')

    # Attach 2D coords to add/sub selected items
    add_points = []
    for m in add_meta:
        pid = f"__add_id__{m['item_id']}"
        coord = proj_map.get(pid)
        add_points.append({**m, 'embedding_2d': coord})

    sub_points = []
    for m in sub_meta:
        pid = f"__sub_id__{m['item_id']}"
        coord = proj_map.get(pid)
        sub_points.append({**m, 'embedding_2d': coord})

    return {
        'results': ordered,
        'filtered_out': filtered_details,
        'centroid_2d': centroid_2d,
        'add_centroid_2d': centroid_2d,
        'subtract_centroid_2d': subtract_centroid_2d,
        'add_points': add_points,
        'sub_points': sub_points,
        'projection': projection_used,
    }

