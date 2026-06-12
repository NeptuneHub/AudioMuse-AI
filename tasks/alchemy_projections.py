"""2D projection helpers shared by song_alchemy, app_map, and app_helper.

These functions are pure numpy/sklearn math with no dependency on the
voyager index, the media server layer, or the database, so consumers that
only need a projection (e.g. the music map) can import this module without
pulling the full song_alchemy import chain.
"""

import logging
from typing import List, Tuple

import numpy as np

try:
    # sklearn is already a dependency; import lazily for environments where it's present
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
except Exception:
    PCA = None
    LogisticRegression = None

logger = logging.getLogger(__name__)


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
        clf = LogisticRegression(l1_ratio=0, C=1.0, solver='saga', max_iter=1000)
        clf.fit(Xp, y_train)
    except Exception:
        # Fallback with less regularization if solver fails
        clf = LogisticRegression(l1_ratio=0, C=0.1, solver='saga', max_iter=1000)
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
