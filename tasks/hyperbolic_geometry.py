# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Pure NumPy Poincare-ball geometry over the MusiCNN embeddings.

Implements the hyperbolic projection, radius, exact Poincare distance and the
quantile radial bands used by the Hyperbolic Explorer feature, with no
dependency beyond NumPy (this project is ONNX-only by design, so no
geoopt/PyTorch is allowed).

Main Features:
* project_to_poincare maps raw vectors into the open Poincare ball with
  proj(x) = tanh(||x|| / s) * (x / ||x||)
* poincare_radius returns R = ||proj(x)|| for 1-D or 2-D input
* hyperbolic_distance / hyperbolic_distances_to implement the exact Poincare
  metric d_H(u, v) = arccosh(1 + 2*||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
* calibrate_scale picks the scale s from a percentile of the norm distribution
  so tanh stays in its active region instead of saturating
* split_radial_bands / assign_radial_bands derive quantile band boundaries
  from the actual radius distribution
"""

import numpy as np


def project_to_poincare(vectors, scale):
    scale = float(scale) if scale else 1.0
    vecs = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = np.where(norms <= 1e-12, 1.0, norms)
    unit = vecs / safe
    # tanh mathematically stays in (-1, 1), but the separate unit-vector
    # division can reintroduce enough float error that the product's norm
    # lands fractionally over 1.0 - clip so callers always get a point
    # strictly inside the open ball, never exactly on or past the boundary.
    radii = np.minimum(np.tanh(norms / scale), 1.0 - 1e-7)
    out = unit * radii
    if np.ndim(vectors) == 1:
        return out.reshape(-1).astype(np.float32)
    return out.astype(np.float32)


def poincare_radius(vectors, scale):
    scale = float(scale) if scale else 1.0
    vecs = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=-1)
    return np.minimum(np.tanh(norms / scale), 1.0 - 1e-7)


def hyperbolic_distance(u, v):
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    u_norm2 = float(np.sum(uu * uu))
    v_norm2 = float(np.sum(vv * vv))
    diff2 = float(np.sum((uu - vv) ** 2))
    denom = max((1.0 - u_norm2) * (1.0 - v_norm2), 1e-12)
    arg = 1.0 + 2.0 * diff2 / denom
    return float(np.arccosh(max(arg, 1.0)))


def hyperbolic_distances_to(target, candidates):
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    c = np.asarray(candidates, dtype=np.float64)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    u_norm2 = float(np.sum(t * t))
    c_norm2 = np.sum(c * c, axis=1)
    diff2 = np.sum((c - t) ** 2, axis=1)
    denom = np.maximum((1.0 - u_norm2) * (1.0 - c_norm2), 1e-12)
    arg = 1.0 + 2.0 * diff2 / denom
    return np.arccosh(np.clip(arg, 1.0, None))


def hyperbolic_distance_matrix(targets, candidates):
    """Pairwise Poincare distances between two sets of points.

    Returns an (n, k) matrix where row i / column j is d_H(targets[i],
    candidates[j]). Vectorized so partitioning a large catalogue against a
    handful of centroids (mood, genre, subgenre) is one NumPy pass instead of
    a Python loop of scalar distance calls, which dominated tree build time.
    """
    t = np.asarray(targets, dtype=np.float64)
    c = np.asarray(candidates, dtype=np.float64)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    t_norm2 = np.sum(t * t, axis=1)
    c_norm2 = np.sum(c * c, axis=1)
    diff2 = t_norm2[:, None] + c_norm2[None, :] - 2.0 * (t @ c.T)
    denom = np.maximum((1.0 - t_norm2[:, None]) * (1.0 - c_norm2[None, :]), 1e-12)
    arg = 1.0 + 2.0 * diff2 / denom
    return np.arccosh(np.clip(arg, 1.0, None))


def calibrate_scale(norms, percentile=95.0):
    norms = np.asarray(norms, dtype=np.float64)
    if norms.size == 0:
        return 1.0
    pct = float(np.percentile(norms, float(percentile)))
    if not np.isfinite(pct) or pct <= 0.0:
        return 1.0
    return pct


def split_radial_bands(radii, n_bands):
    radii = np.asarray(radii, dtype=np.float64)
    n_bands = max(1, int(n_bands))
    if radii.size == 0:
        return []
    if radii.size == 1 or n_bands == 1:
        return [(float(radii.min()), float(radii.max()))]
    quantiles = np.quantile(radii, np.linspace(0.0, 1.0, n_bands + 1))
    qs = np.unique(quantiles)
    if qs.size < 2:
        return [(float(qs[0]), float(qs[-1]))]
    return [(float(qs[i]), float(qs[i + 1])) for i in range(qs.size - 1)]


def assign_radial_bands(radii, boundaries):
    radii = np.asarray(radii, dtype=np.float64)
    if not boundaries or radii.size == 0:
        return np.zeros(radii.shape[0], dtype=np.int64)
    edges = np.array([b[0] for b in boundaries] + [boundaries[-1][1]], dtype=np.float64)
    idx = np.searchsorted(edges, radii, side="right") - 1
    return np.clip(idx, 0, len(boundaries) - 1)
