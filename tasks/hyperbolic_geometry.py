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
* mobius_add / mobius_scalar_mul are the gyrovector operations of the Poincare
  ball, from which poincare_geodesic builds the exact constant-speed geodesic
  gamma(t) = x (+) (t (x) (-x (+) y)) with gamma(0) = x and gamma(1) = y
* einstein_midpoint is the closed-form Lorentz-weighted average taken in the
  Klein model, and karcher_mean refines it into the true Frechet mean with
  Riemannian gradient steps. The step is divided by the mean of d*coth(d) over
  the members because the Hessian of d^2 grows with d in negative curvature: an
  undamped unit step overshoots as soon as the members span more than a couple
  of units of hyperbolic distance, and the iteration then walks OUTWARD to the
  boundary with the Frechet cost rising on every pass instead of converging
* poincare_kmeans is k-means done entirely in the Poincare metric: k-means++
  seeding on hyperbolic distance, assignment through nearest_centroid, and the
  Frechet mean above as the centroid update. nearest_centroid skips the arccosh
  because argmin over centroids of arccosh(1 + 2*d2/((1-|t|^2)(1-|c|^2))) is the
  argmin of d2/(1-|c|^2) once the target-only factor is dropped, which turns a
  full-catalogue assignment pass into one BLAS matmul per chunk. Both the tree
  builder and the disk-paged Poincare IVF index partition through this
* geodesic_apex returns the point of the geodesic closest to the origin: the
  continuous analogue of the lowest common ancestor of the two endpoints,
  because a geodesic between two points bows inward toward the more general
  region that contains both
* apply_radial_dive deepens that inward bow by a bump that is zero at both
  endpoints, so a caller can ask the walk to travel further back toward the
  shared root without moving where it starts or ends
* unproject_from_poincare inverts the projection exactly (the map is radial, so
  direction is preserved), which is what lets a synthetic ball point be looked
  up in the raw-space IVF index
* geodesic_plane_basis / plane_angles give the 2-plane the whole geodesic lives
  in, so a Poincare disk drawing of it is an exact picture and not a sketch
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


def hyperbolic_distance_matrix(targets, candidates, target_norms2=None, candidate_norms2=None):
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
    t_norm2 = (
        np.sum(t * t, axis=1) if target_norms2 is None
        else np.asarray(target_norms2, dtype=np.float64).reshape(-1)
    )
    c_norm2 = (
        np.sum(c * c, axis=1) if candidate_norms2 is None
        else np.asarray(candidate_norms2, dtype=np.float64).reshape(-1)
    )
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


_BALL_LIMIT = 1.0 - 1e-7


def clip_into_ball(vectors):
    v = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    factor = np.where(norms > _BALL_LIMIT, _BALL_LIMIT / np.maximum(norms, 1e-12), 1.0)
    return v * factor


def mobius_add(x, y):
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    x2 = np.sum(xx * xx, axis=-1, keepdims=True)
    y2 = np.sum(yy * yy, axis=-1, keepdims=True)
    xy = np.sum(xx * yy, axis=-1, keepdims=True)
    num = (1.0 + 2.0 * xy + y2) * xx + (1.0 - x2) * yy
    den = 1.0 + 2.0 * xy + x2 * y2
    return clip_into_ball(num / np.where(np.abs(den) < 1e-15, 1e-15, den))


def mobius_scalar_mul(t, x):
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    tt = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    norms = np.linalg.norm(xx, axis=-1, keepdims=True)
    safe = np.where(norms <= 1e-12, 1.0, norms)
    radii = np.tanh(tt * np.arctanh(np.minimum(norms, _BALL_LIMIT)))
    return clip_into_ball(radii * (xx / safe))


def einstein_midpoint(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[0] == 0:
        return None
    norms2 = np.sum(pts * pts, axis=-1, keepdims=True)
    klein = 2.0 * pts / (1.0 + norms2)
    klein2 = np.minimum(np.sum(klein * klein, axis=-1, keepdims=True), _BALL_LIMIT ** 2)
    gamma = 1.0 / np.sqrt(1.0 - klein2)
    total = float(np.sum(gamma))
    if not np.isfinite(total) or total <= 1e-12:
        return clip_into_ball(pts.mean(axis=0))
    centre = clip_into_ball(np.sum(gamma * klein, axis=0) / total)
    centre2 = float(np.sum(centre * centre))
    return clip_into_ball(centre / (1.0 + np.sqrt(max(1.0 - centre2, 0.0))))


def karcher_mean(points, iterations=10):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[0] == 0:
        return None
    mean = einstein_midpoint(pts)
    for _ in range(max(0, int(iterations))):
        diff = mobius_add(-mean, pts)
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        safe = np.where(norms <= 1e-12, 1.0, norms)
        theta = np.arctanh(np.minimum(norms, _BALL_LIMIT))
        lam = 1.0 - float(np.sum(mean * mean))
        step = ((2.0 / max(lam, 1e-12)) * theta * (diff / safe)).mean(axis=0)
        step_norm = float(np.linalg.norm(step))
        if step_norm <= 1e-12:
            break
        spread = 2.0 * theta
        stiffness = float(np.mean(spread / np.tanh(np.maximum(spread, 1e-9))))
        travel = lam * step_norm / max(stiffness, 1.0)
        if travel <= 1e-12:
            break
        mean = clip_into_ball(mobius_add(mean, np.tanh(0.5 * travel) * (step / step_norm)))
    return mean


def nearest_centroid(points, centroids, chunk=None):
    pts = clip_into_ball(np.asarray(points, dtype=np.float64))
    cent = clip_into_ball(np.asarray(centroids, dtype=np.float64))
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if cent.ndim == 1:
        cent = cent.reshape(1, -1)
    n, k = pts.shape[0], cent.shape[0]
    p_norm2 = np.sum(pts * pts, axis=1)
    c_norm2 = np.sum(cent * cent, axis=1)
    inv = 1.0 / np.maximum(1.0 - c_norm2, 1e-12)
    block = int(chunk) if chunk else max(64, min(4096, 2_000_000 // max(k, 1)))
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, block):
        stop = start + block
        diff2 = p_norm2[start:stop, None] + c_norm2[None, :] - 2.0 * (pts[start:stop] @ cent.T)
        labels[start:stop] = np.argmin(np.maximum(diff2, 0.0) * inv[None, :], axis=1)
    return labels


def poincare_kmeans(points, k, iterations=10, seed=0, chunk=None):
    pts = clip_into_ball(np.asarray(points, dtype=np.float64))
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, pts.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.int64)
    k = max(1, min(int(k), n))
    norms2 = np.sum(pts * pts, axis=1)
    rng = np.random.RandomState(int(seed))
    centroids = clip_into_ball(pts[_kmeans_plus_plus(pts, norms2, k, rng)])
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        new_labels = nearest_centroid(pts, centroids, chunk=chunk)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        order = np.argsort(labels, kind="stable")
        bounds = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=k))))
        moved = np.empty_like(centroids)
        for j in range(k):
            lo, hi = int(bounds[j]), int(bounds[j + 1])
            if hi <= lo:
                moved[j] = centroids[j]
                continue
            centre = karcher_mean(pts[order[lo:hi]])
            moved[j] = centroids[j] if centre is None else centre
        centroids = clip_into_ball(moved)
    return centroids, labels


def _kmeans_plus_plus(pts, norms2, k, rng):
    n = pts.shape[0]

    def spread_from(idx):
        return hyperbolic_distance_matrix(
            pts, pts[idx:idx + 1],
            target_norms2=norms2, candidate_norms2=norms2[idx:idx + 1],
        )[:, 0]

    first = int(rng.randint(n))
    chosen = [first]
    taken = {first}
    best = spread_from(first)
    while len(chosen) < k:
        total = float(best.sum())
        if not np.isfinite(total) or total <= 1e-12:
            for idx in range(n):
                if len(chosen) >= k:
                    break
                if idx not in taken:
                    chosen.append(idx)
                    taken.add(idx)
            break
        pick = int(rng.choice(n, p=best / total))
        if pick in taken:
            pick = int(np.argmax(best))
        chosen.append(pick)
        taken.add(pick)
        best = np.minimum(best, spread_from(pick))
    return chosen


def poincare_geodesic(start, end, ts):
    u = np.asarray(start, dtype=np.float64).reshape(1, -1)
    v = np.asarray(end, dtype=np.float64).reshape(1, -1)
    direction = mobius_add(-u, v)
    steps = mobius_scalar_mul(np.asarray(ts, dtype=np.float64).reshape(-1), direction)
    return mobius_add(u, steps)


def geodesic_apex(start, end, samples=129, refinements=30):
    ts = np.linspace(0.0, 1.0, max(3, int(samples)))
    points = poincare_geodesic(start, end, ts)
    norms = np.linalg.norm(points, axis=1)
    best = int(np.argmin(norms))
    lo = float(ts[max(best - 1, 0)])
    hi = float(ts[min(best + 1, ts.size - 1)])
    for _ in range(max(0, int(refinements))):
        third = (hi - lo) / 3.0
        probe = poincare_geodesic(start, end, [lo + third, hi - third])
        if np.linalg.norm(probe[0]) <= np.linalg.norm(probe[1]):
            hi -= third
        else:
            lo += third
    t = 0.5 * (lo + hi)
    return float(t), poincare_geodesic(start, end, [t])[0]


def apply_radial_dive(points, ts, dive):
    depth = min(max(float(dive or 0.0), 0.0), 0.95)
    pts = np.asarray(points, dtype=np.float64)
    if depth <= 0.0:
        return pts
    t = np.asarray(ts, dtype=np.float64).reshape(-1, 1)
    bump = 4.0 * t * (1.0 - t)
    return clip_into_ball(pts * (1.0 - depth * bump))


def unproject_from_poincare(points, scale):
    scale = float(scale) if scale else 1.0
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, -1)
    norms = np.linalg.norm(pts, axis=-1, keepdims=True)
    safe = np.where(norms <= 1e-12, 1.0, norms)
    raw_norms = scale * np.arctanh(np.minimum(norms, _BALL_LIMIT))
    out = (pts / safe) * raw_norms
    if single:
        return out.reshape(-1).astype(np.float32)
    return out.astype(np.float32)


def geodesic_plane_basis(start, end):
    u = np.asarray(start, dtype=np.float64).reshape(-1)
    v = np.asarray(end, dtype=np.float64).reshape(-1)
    first_norm = float(np.linalg.norm(u))
    if first_norm > 1e-12:
        e1 = u / first_norm
    else:
        second_norm = float(np.linalg.norm(v))
        if second_norm <= 1e-12:
            return None, None
        return v / second_norm, None
    residual = v - float(np.dot(v, e1)) * e1
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= 1e-9:
        return e1, None
    return e1, residual / residual_norm


def plane_angles(points, e1, e2):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if e1 is None:
        return np.zeros(pts.shape[0], dtype=np.float64)
    along = pts @ np.asarray(e1, dtype=np.float64)
    if e2 is None:
        return np.where(along >= 0.0, 0.0, np.pi)
    across = pts @ np.asarray(e2, dtype=np.float64)
    return np.arctan2(across, along)
