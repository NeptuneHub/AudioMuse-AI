import logging
from typing import List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .voyager_manager import find_nearest_neighbors_by_vector, find_nearest_neighbors_by_id, get_vector_by_id
from app_helper import get_score_data_by_ids, load_map_projection
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
    reducer = umap.UMAP(n_components=n_components, random_state=None, n_jobs=-1)
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


def song_alchemy(add_ids: List[str], subtract_ids: List[str], n_results: int = None, subtract_distance: float = None, temperature: float = None) -> dict:
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

    # Normalize temperature early so downstream logic (including the special-case
    # branch below) can safely compare/convert it. If the frontend omitted the
    # parameter or provided a non-numeric/null value, fall back to the configured
    # default. This ensures temperature is optional from the API perspective.
    try:
        if temperature is None:
            # Use configured default
            temperature = float(config.ALCHEMY_TEMPERATURE)
        else:
            # Coerce numeric-like strings as well
            temperature = float(temperature)
    except Exception:
        logger.warning(f"Invalid temperature value passed to song_alchemy: {temperature!r}; falling back to config default")
        try:
            temperature = float(config.ALCHEMY_TEMPERATURE)
        except Exception:
            temperature = 1.0

    # Find nearest neighbors to add_centroid using Voyager
    # Special-case: if user provided exactly one ADD track and temperature==0 (deterministic)
    # then use the id-based neighbor query so results match the "similar song" path.
    if temperature is not None and float(temperature) == 0.0 and add_ids and len(add_ids) == 1:
        try:
            neighbors = find_nearest_neighbors_by_id(add_ids[0], n=n_results)
        except Exception:
            neighbors = find_nearest_neighbors_by_vector(add_centroid, n=n_results * 3)
    else:
        neighbors = find_nearest_neighbors_by_vector(add_centroid, n=n_results * 3)
    if not neighbors:
        return {"results": [], "filtered_out": [], "centroid_2d": None}

    # neighbors is a list of dicts with item_id and score; keep candidate ids
    candidate_ids = [n['item_id'] for n in neighbors]

    # Remove any user-provided ADD or SUBTRACT items from candidate list so we don't
    # return the input tracks themselves as results. This mirrors find_nearest_neighbors_by_id
    # behavior which excludes the original item.
    if add_ids:
        add_set = set(add_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in add_set]
    if subtract_ids:
        sub_set = set(subtract_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in sub_set]

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

    # Trim to desired n_results (we'll sample probabilistically from these candidates)
    candidate_ids = candidate_ids[: max(n_results * 3, n_results)]

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

    # Try to use a precomputed 2D projection saved in the DB (same approach as app_map.py).
    # If a precomputed map exists, use its coords for any matching item_ids. Only compute
    # projections locally for the subset of proj_ids that are missing from the precomputed map.
    projection_used = 'none'
    proj_map = {}

    try:
        id_map, precomp_proj = load_map_projection('main_map')
    except Exception:
        id_map, precomp_proj = None, None

    id_to_coord = {}
    if id_map is not None and precomp_proj is not None:
        try:
            # id_map is expected to be a list of item_ids in the same order as rows in precomp_proj
            # Use string keys to be robust (DB item_ids are text)
            for iid, coord in zip(id_map, precomp_proj.tolist()):
                id_to_coord[str(iid)] = (float(coord[0]), float(coord[1]))
        except Exception:
            id_to_coord = {}

    # Fill proj_map from precomputed projection where possible
    missing_ids = []
    missing_vectors = []
    for pid in proj_ids:
        # markers like '__add_id__{id}' or '__sub_id__{id}' map to underlying item ids
        if isinstance(pid, str) and pid.startswith('__add_id__'):
            item_id = pid.replace('__add_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
            else:
                missing_ids.append(pid)
        elif isinstance(pid, str) and pid.startswith('__sub_id__'):
            item_id = pid.replace('__sub_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
            else:
                missing_ids.append(pid)
        elif pid in ('__add_centroid__', '__subtract_centroid__'):
            # compute centroid coords later (from member point coords) if possible
            continue
        else:
            # regular item id
            coord = id_to_coord.get(str(pid))
            if coord is not None:
                proj_map[pid] = coord
            else:
                missing_ids.append(pid)

    # For centroids, attempt to compute centroid coordinates from precomputed member points
    def _centroid_from_member_coords(member_ids, prefix=''):
        coords = []
        for mid in member_ids:
            c = id_to_coord.get(str(mid))
            if c is not None:
                coords.append(np.array(c, dtype=float))
        if not coords:
            return None
        mean = np.mean(np.vstack(coords), axis=0)
        return (float(mean[0]), float(mean[1]))

    add_centroid_2d_db = None
    subtract_centroid_2d_db = None
    try:
        if add_ids:
            add_centroid_2d_db = _centroid_from_member_coords(add_ids)
        if subtract_ids:
            subtract_centroid_2d_db = _centroid_from_member_coords(subtract_ids)
        if add_centroid_2d_db is not None:
            proj_map['__add_centroid__'] = add_centroid_2d_db
        if subtract_centroid_2d_db is not None:
            proj_map['__subtract_centroid__'] = subtract_centroid_2d_db
    except Exception:
        # non-fatal; we'll compute missing projections below
        pass

    # Collect vectors for any proj_ids that are still missing (we will compute only these)
    for pid in proj_ids:
        if pid in proj_map:
            continue
        if pid in ('__add_centroid__', '__subtract_centroid__'):
            # if not set from member coords, skip for now
            continue
        # resolve underlying item id for add/sub markers
        if isinstance(pid, str) and pid.startswith('__add_id__'):
            item_id = pid.replace('__add_id__', '')
        elif isinstance(pid, str) and pid.startswith('__sub_id__'):
            item_id = pid.replace('__sub_id__', '')
        else:
            item_id = pid

        vec = get_vector_by_id(item_id)
        if vec is None:
            # can't project without vector; leave missing
            continue
        missing_ids.append(pid)
        missing_vectors.append(np.array(vec, dtype=float))

    # If we have missing vectors, compute projections for them only
    if missing_vectors:
        try:
            # Try discriminant/aligned/umap/pca similar to previous logic but only for missing subset
            local_projections = None
            # For discriminant we need add/sub vectors; attempt only if both exist within missing set
            try:
                # Build add/sub vectors intersecting missing vectors
                local_add_vecs = [np.array(get_vector_by_id(a), dtype=float) for a in add_ids if get_vector_by_id(a) is not None and (f'__add_id__{a}' in missing_ids or a in missing_ids)]
                local_sub_vecs = [np.array(get_vector_by_id(s), dtype=float) for s in subtract_ids if get_vector_by_id(s) is not None and (f'__sub_id__{s}' in missing_ids or s in missing_ids)]
                if local_add_vecs and local_sub_vecs and _project_with_discriminant is not None:
                    local_projections = _project_with_discriminant(local_add_vecs, local_sub_vecs, missing_vectors)
                    projection_used = 'discriminant'
            except Exception:
                local_projections = None

            if local_projections is None:
                try:
                    local_projections = _project_with_umap(missing_vectors)
                    projection_used = 'umap'
                except Exception:
                    try:
                        local_projections = _project_to_2d(missing_vectors)
                        projection_used = 'pca'
                    except Exception:
                        # fallback zeros
                        local_projections = [(0.0, 0.0) for _ in missing_vectors]

            # Assign local projections back into proj_map in the same order
            for pid, coord in zip(missing_ids, local_projections):
                proj_map[pid] = (float(coord[0]), float(coord[1]))
        except Exception as e:
            logger.warning(f"Failed to compute local projections for missing ids: {e}")

    # Ensure every proj_id has something (fill remaining with zeros)
    for pid in proj_ids:
        if pid not in proj_map:
            proj_map[pid] = (0.0, 0.0)

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
    details_map = {d['item_id']: d for d in details}

    # Build a list of scored candidates for probabilistic sampling
    scored_candidates = []
    for cid in candidate_ids:
        if cid in details_map and cid in distances:
            scored_candidates.append((cid, distances[cid]))

    # Determine temperature: use provided value or config default
    if temperature is None:
        try:
            from config import ALCHEMY_TEMPERATURE as _cfg_temp
            temperature = float(_cfg_temp)
        except Exception:
            temperature = 1.0

    # Convert distances into similarity-like scores (smaller distance => higher similarity)
    # We'll negate distances so higher is better
    import math, random

    ids = [c[0] for c in scored_candidates]
    raw_scores = [ -float(c[1]) for c in scored_candidates ]

    ordered = []
    if ids:
        # If temperature is exactly zero, use deterministic selection (best matches first)
        try:
            if temperature is not None and float(temperature) == 0.0:
                ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
                for i in ids_sorted[:n_results]:
                    item = details_map.get(i, {})
                    item['distance'] = distances.get(i)
                    item['embedding_2d'] = proj_map.get(i)
                    ordered.append(item)
            else:
                # Softmax with temperature (temperature may be None or >0)
                temps = [s / (temperature or 1.0) for s in raw_scores]
                max_t = max(temps)
                exps = [math.exp(t - max_t) for t in temps]
                total = sum(exps)
                if total <= 0:
                    probs = [1.0 / len(exps)] * len(exps)
                else:
                    probs = [e / total for e in exps]

                # Weighted sampling without replacement to get n_results items (preserve projection/metadata)
                chosen = []
                avail_ids = ids.copy()
                avail_probs = probs.copy()
                k = min(n_results, len(avail_ids))
                for _ in range(k):
                    # Normalize
                    s = sum(avail_probs)
                    if s <= 0:
                        idx = random.randrange(len(avail_ids))
                    else:
                        r = random.random() * s
                        acc = 0.0
                        idx = 0
                        for j, p in enumerate(avail_probs):
                            acc += p
                            if r <= acc:
                                idx = j
                                break
                    chosen_id = avail_ids.pop(idx)
                    avail_probs.pop(idx)
                    chosen.append(chosen_id)

                # Build ordered results from chosen ids in the order selected
                for cid in chosen:
                    item = details_map.get(cid, {})
                    item['distance'] = distances.get(cid)
                    item['embedding_2d'] = proj_map.get(cid)
                    ordered.append(item)
        except Exception as e:
            # Fallback deterministic ordering by best match
            logger.warning(f"Sampling failed, falling back to deterministic selection: {e}")
            ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
            for i in ids_sorted[:n_results]:
                item = details_map.get(i, {})
                item['distance'] = distances.get(i)
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

