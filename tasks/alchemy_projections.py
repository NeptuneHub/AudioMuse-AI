import logging
from typing import List, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
except Exception:
    PCA = None
    LogisticRegression = None

logger = logging.getLogger(__name__)


def _project_to_2d(vectors: List[np.ndarray]) -> List[Tuple[float, float]]:
    if not vectors:
        return []
    mat = np.vstack(vectors)
    mean = np.mean(mat, axis=0)
    mat_c = mat - mean
    try:
        _, _, vh = np.linalg.svd(mat_c, full_matrices=False)
    except Exception:
        return [(0.0, 0.0) for _ in vectors]
    pcs = vh[:2]
    proj = mat_c.dot(pcs.T)
    if proj.size == 0:
        return [(0.0, 0.0) for _ in vectors]
    proj_centered = proj - proj.mean(axis=0)
    max_abs = np.max(np.abs(proj_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in vectors]
    scaled = proj_centered / max_abs
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def _project_aligned_add_sub(
    vectors: List[np.ndarray], add_centroid: np.ndarray, subtract_centroid: np.ndarray
) -> List[Tuple[float, float]]:
    if not vectors:
        return []
    mat = np.vstack(vectors)
    rel = mat - add_centroid
    axis = subtract_centroid - add_centroid
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return _project_to_2d(vectors)
    axis_u = axis / axis_norm

    x_coords = rel.dot(axis_u)

    proj_on_axis = np.outer(x_coords, axis_u)
    residuals = rel - proj_on_axis

    try:
        _, _, vh = np.linalg.svd(residuals, full_matrices=False)
        y_u = vh[0]
    except Exception:
        y_u = None

    if y_u is None or np.linalg.norm(y_u) == 0:
        idx = int(np.argmin(np.abs(axis_u)))
        e = np.zeros_like(axis_u)
        e[idx] = 1.0
        y_u = e - np.dot(e, axis_u) * axis_u
        norm_y = np.linalg.norm(y_u)
        if norm_y == 0:
            return _project_to_2d(vectors)
        y_u = y_u / norm_y
    else:
        y_u = y_u - np.dot(y_u, axis_u) * axis_u
        y_u_norm = np.linalg.norm(y_u)
        if y_u_norm == 0:
            return _project_to_2d(vectors)
        y_u = y_u / y_u_norm

    y_coords = residuals.dot(y_u)

    coords = np.vstack([x_coords, y_coords]).T
    coords_centered = coords - coords.mean(axis=0)
    max_abs = np.max(np.abs(coords_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in vectors]
    scaled = coords_centered / max_abs
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def _project_with_umap(
    vectors: List[np.ndarray], n_components: int = 2
) -> List[Tuple[float, float]]:
    import umap

    if not vectors:
        return []
    mat = np.vstack(vectors)
    reducer = umap.UMAP(n_components=n_components, random_state=None, n_jobs=-1)
    embedding = reducer.fit_transform(mat)
    emb_centered = embedding - embedding.mean(axis=0)
    max_abs = np.max(np.abs(emb_centered))
    if max_abs == 0:
        return [(0.0, 0.0) for _ in vectors]
    scaled = emb_centered / max_abs
    scaled = np.clip(scaled, -1.0, 1.0)
    return [(float(x), float(y)) for x, y in scaled]


def _project_with_discriminant(
    add_vectors: List[np.ndarray], sub_vectors: List[np.ndarray], all_vectors: List[np.ndarray]
) -> List[Tuple[float, float]]:
    if LogisticRegression is None or PCA is None:
        raise RuntimeError('sklearn not available')
    if not add_vectors or not sub_vectors:
        raise RuntimeError('Insufficient classes for discriminant')

    X_train = np.vstack([np.vstack(add_vectors), np.vstack(sub_vectors)])
    y_train = np.array([1] * len(add_vectors) + [0] * len(sub_vectors))

    n_samples, n_features = X_train.shape
    max_components = min(32, n_samples - 1, n_features)
    if max_components < 1:
        raise RuntimeError('Not enough samples for discriminant PCA')

    pca = PCA(n_components=max_components, random_state=42)
    x_pca = pca.fit_transform(X_train)

    try:
        clf = LogisticRegression(l1_ratio=0, C=1.0, solver='saga', max_iter=1000)
        clf.fit(x_pca, y_train)
    except Exception:
        clf = LogisticRegression(l1_ratio=0, C=0.1, solver='saga', max_iter=1000)
        clf.fit(x_pca, y_train)

    coef = clf.coef_.ravel()
    norm = np.linalg.norm(coef)
    if norm == 0:
        raise RuntimeError('Discriminant produced zero vector')
    dir_pca = coef / norm

    all_mat = np.vstack(all_vectors)
    all_pca = pca.transform(all_mat)
    x_coords = all_pca.dot(dir_pca)

    proj_on_dir = np.outer(x_coords, dir_pca)
    residuals = all_pca - proj_on_dir
    try:
        _, _, vh = np.linalg.svd(residuals, full_matrices=False)
        y_u = vh[0]
    except Exception:
        y_u = None

    if y_u is None or np.linalg.norm(y_u) == 0:
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
