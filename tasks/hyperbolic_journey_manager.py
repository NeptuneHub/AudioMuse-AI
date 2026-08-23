# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Geodesic Journey engine: walk the Poincare geodesic between two songs.

Given a start and an end song this builds the exact hyperbolic geodesic
between their Poincare projections, samples it at constant hyperbolic speed,
and snaps every waypoint to a real song. Because a geodesic in negatively
curved space bows toward the origin, the resulting playlist does not blend the
two songs the way a straight line in raw space does (that is what the Sonic
Path page already offers): it descends through the region general enough to
contain both, then climbs back out to the destination. The deepest point of
that bow is the continuous analogue of the lowest common ancestor of the two
songs, which is what the page reports as the shared root.

Scale: the snapping reads the whole projected catalogue once (every row with
a finite Poincare projection) and ranks it by exact Poincare distance, so
there is no IVF probe and no cosine/angular shortcut anywhere in the walk.
The cost scales with the library size by design - the walk must be able to
follow its own geodesic inward, where songs are rare, without a directional
index quietly dropping them.

Main Features:
* build_hyperbolic_journey resolves both endpoints, samples the geodesic,
  optionally deepens its inward bow by ancestry_dive (default 0.20), snaps
  each interior waypoint to the nearest unused real song by exact Poincare
  distance, and returns the ordered walk with the endpoints pinned at both
  ends
* Candidate generation reads tasks.hyperbolic_manager.fetch_all_poincare_rows
  once for the whole walk and picks against the full projected catalogue by
  exact Poincare distance
* Content de-duplication and the MAX_SONGS_PER_ARTIST cap are enforced while
  picking rather than afterwards, so enforcing them shortens the walk instead
  of tearing holes in the middle of it
* The apex (lowest common ancestor) and every picked track are labelled with
  the nearest genre/subgenre centroid, so the journey narrates the regions it
  crosses instead of returning bare ids
* geodesic_plane_coordinates exposes the 2-plane the whole geodesic lives in,
  letting the frontend draw a Poincare disk that is an exact picture of the
  path rather than a decorative curve
"""

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)

_REGION_CACHE = {"source": None, "labels": None, "matrix": None}


def _clamp_length(length):
    if length is None:
        length = config.HYPERBOLIC_JOURNEY_DEFAULT_LENGTH
    try:
        length = int(length)
    except (TypeError, ValueError):
        raise ValueError('Invalid "length" value.') from None
    return max(3, length)


def _clamp_dive(dive):
    if dive is None:
        dive = config.HYPERBOLIC_JOURNEY_ANCESTRY_DIVE
    try:
        dive = float(dive)
    except (TypeError, ValueError):
        raise ValueError('Invalid "ancestry_dive" value.') from None
    if not (0.0 <= dive <= 0.95):
        raise ValueError('"ancestry_dive" must be between 0 and 0.95.')
    return dive


def _region_centroids():
    from tasks.hyperbolic_manager import get_projected_genre_subgenres

    tree = get_projected_genre_subgenres()
    if not tree:
        return None, None
    if _REGION_CACHE["source"] is tree:
        return _REGION_CACHE["labels"], _REGION_CACHE["matrix"]
    labels = []
    vectors = []
    for genre, info in tree.items():
        for sub in info.get("subgenres") or []:
            labels.append((genre, sub["name"]))
            vectors.append(sub["vec"])
    if not vectors:
        return None, None
    matrix = np.stack(vectors).astype(np.float64)
    _REGION_CACHE["source"] = tree
    _REGION_CACHE["labels"] = labels
    _REGION_CACHE["matrix"] = matrix
    return labels, matrix


def _region_for_points(points):
    from tasks.hyperbolic_geometry import hyperbolic_distance_matrix

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    labels, matrix = _region_centroids()
    if labels is None or matrix.shape[1] != pts.shape[1]:
        return [None] * pts.shape[0]
    best = np.argmin(hyperbolic_distance_matrix(pts, matrix), axis=1)
    out = []
    for idx in best:
        genre, subgenre = labels[int(idx)]
        out.append({
            "genre": genre,
            "subgenre": subgenre,
            "label": f"{genre.title()} / {subgenre.title()}",
        })
    return out


def _endpoint_rows(start_item_id, end_item_id):
    from tasks.hyperbolic_manager import fetch_poincare_rows

    if not start_item_id or not end_item_id:
        raise ValueError("Both a start and an end song are required.")
    if start_item_id == end_item_id:
        raise ValueError("The start and end song must be different.")
    rows = fetch_poincare_rows([start_item_id, end_item_id])
    if start_item_id not in rows or end_item_id not in rows:
        raise ValueError(
            "One of the chosen songs has no hyperbolic projection; run the "
            "hyperbolic backfill job first"
        )
    return rows[start_item_id], rows[end_item_id]


def _gather_journey_candidates(interior_points, excluded):
    from tasks.hyperbolic_manager import fetch_all_poincare_rows

    rows = fetch_all_poincare_rows()
    candidate_ids = [i for i in rows if i not in excluded]
    if not candidate_ids:
        return [], None, None
    vectors = np.stack([rows[i][0] for i in candidate_ids]).astype(np.float64)
    radii = np.array([rows[i][1] for i in candidate_ids], dtype=np.float64)
    return candidate_ids, vectors, radii


def _pick_steps(interior_points, candidate_ids, candidate_vecs, details, seed_details):
    from tasks.hyperbolic_geometry import hyperbolic_distance_matrix
    from tasks.search_shaping import is_same_song

    distances = hyperbolic_distance_matrix(interior_points, candidate_vecs)
    ranked = np.argsort(distances, axis=1)
    cap = config.MAX_SONGS_PER_ARTIST
    used = set()
    artist_counts = {}
    taken = [info for info in seed_details if info]
    for info in taken:
        author = info.get("author")
        if author:
            artist_counts[author] = artist_counts.get(author, 0) + 1
    picks = []
    for step, ranking in enumerate(ranked):
        chosen = _choose_candidate(
            ranking, candidate_ids, details, used, taken, artist_counts, cap, is_same_song
        )
        if chosen is None:
            continue
        item_id, column, info = chosen
        used.add(item_id)
        taken.append(info)
        author = info.get("author")
        if author:
            artist_counts[author] = artist_counts.get(author, 0) + 1
        picks.append({
            "item_id": item_id,
            "step": step + 1,
            "distance": float(distances[step, column]),
            "column": column,
        })
    return picks


def _choose_candidate(ranking, candidate_ids, details, used, taken, artist_counts, cap, is_same_song):
    for column in ranking:
        item_id = candidate_ids[int(column)]
        if item_id in used:
            continue
        info = details.get(item_id)
        if not info:
            continue
        author = info.get("author")
        if cap and cap > 0 and (not author or artist_counts.get(author, 0) >= cap):
            continue
        if any(
            is_same_song(info.get("title"), author, prev.get("title"), prev.get("author"))
            for prev in taken
        ):
            continue
        return item_id, int(column), info
    return None


def _journey_row(item_id, step, t, distance, radius, waypoint_radius, is_endpoint):
    return {
        "item_id": item_id,
        "step": int(step),
        "t": float(t),
        "distance": float(distance),
        "hyperbolic_radius": float(radius),
        "waypoint_radius": float(waypoint_radius),
        "is_endpoint": bool(is_endpoint),
    }


def _path_samples(start_vec, end_vec, dive, e1, e2, samples):
    from tasks.hyperbolic_geometry import apply_radial_dive, plane_angles, poincare_geodesic

    ts = np.linspace(0.0, 1.0, max(3, int(samples)))
    points = apply_radial_dive(poincare_geodesic(start_vec, end_vec, ts), ts, dive)
    radii = np.linalg.norm(points, axis=1)
    angles = plane_angles(points, e1, e2)
    return [
        {"t": float(t), "radius": float(r), "angle": float(a)}
        for t, r, a in zip(ts, radii, angles)
    ]


def geodesic_plane_coordinates(vectors, e1, e2):
    from tasks.hyperbolic_geometry import plane_angles

    pts = np.asarray(vectors, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return np.linalg.norm(pts, axis=1), plane_angles(pts, e1, e2)


def _apex_payload(start_vec, end_vec, dive, e1, e2):
    from tasks.hyperbolic_geometry import apply_radial_dive, geodesic_apex, plane_angles

    t, point = geodesic_apex(start_vec, end_vec)
    if dive > 0.0:
        point = apply_radial_dive(point.reshape(1, -1), [t], dive)[0]
    return {
        "t": float(t),
        "radius": float(np.linalg.norm(point)),
        "angle": float(plane_angles(point, e1, e2)[0]),
        "region": _region_for_points(point)[0],
    }


def _snap_interior(interior, start_item_id, end_item_id, seed_details):
    from database import get_score_data_by_ids

    if not interior.shape[0]:
        return [], None, None
    candidate_ids, candidate_vecs, candidate_radii = _gather_journey_candidates(
        interior, {start_item_id, end_item_id}
    )
    if not candidate_ids:
        return [], None, None
    details = {d["item_id"]: d for d in get_score_data_by_ids(candidate_ids)}
    picks = _pick_steps(interior, candidate_ids, candidate_vecs, details, seed_details)
    return picks, candidate_vecs, candidate_radii


def build_hyperbolic_journey(start_item_id, end_item_id, length=None, ancestry_dive=None):
    from database import get_score_data_by_ids
    from tasks.hyperbolic_geometry import (
        apply_radial_dive,
        geodesic_plane_basis,
        hyperbolic_distance,
        poincare_geodesic,
    )

    length = _clamp_length(length)
    dive = _clamp_dive(ancestry_dive)
    (start_vec, start_radius), (end_vec, end_radius) = _endpoint_rows(
        start_item_id, end_item_id
    )

    start_vec = np.asarray(start_vec, dtype=np.float64)
    end_vec = np.asarray(end_vec, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, length)
    waypoints = apply_radial_dive(poincare_geodesic(start_vec, end_vec, ts), ts, dive)
    interior = waypoints[1:-1]
    e1, e2 = geodesic_plane_basis(start_vec, end_vec)

    seed_rows = get_score_data_by_ids([start_item_id, end_item_id])
    seed_details = {d["item_id"]: d for d in seed_rows}
    picks, candidate_vecs, candidate_radii = _snap_interior(
        interior, start_item_id, end_item_id,
        [seed_details.get(start_item_id) or {}, seed_details.get(end_item_id) or {}],
    )

    rows = [_journey_row(
        start_item_id, 0, 0.0, 0.0, start_radius,
        float(np.linalg.norm(waypoints[0])), True,
    )]
    picked_vectors = [start_vec]
    for pick in picks:
        step = pick["step"]
        rows.append(_journey_row(
            pick["item_id"], step, ts[step], pick["distance"],
            candidate_radii[pick["column"]],
            float(np.linalg.norm(interior[step - 1])), False,
        ))
        picked_vectors.append(candidate_vecs[pick["column"]])
    rows.append(_journey_row(
        end_item_id, length - 1, 1.0, 0.0, end_radius,
        float(np.linalg.norm(waypoints[-1])), True,
    ))
    picked_vectors.append(end_vec)

    stacked = np.stack(picked_vectors)
    _, angles = geodesic_plane_coordinates(stacked, e1, e2)
    for row, angle, region in zip(rows, angles, _region_for_points(stacked)):
        row["plane_angle"] = float(angle)
        row["region"] = region

    if len(rows) < length:
        logger.info(
            "Geodesic journey returned %d of the %d requested steps: the candidate "
            "pool ran out after content de-duplication and the per-artist cap.",
            len(rows), length,
        )

    return {
        "results": rows,
        "count": len(rows),
        "requested_length": length,
        "ancestry_dive": dive,
        "geodesic_length": hyperbolic_distance(start_vec, end_vec),
        "start_radius": float(start_radius),
        "end_radius": float(end_radius),
        "apex": _apex_payload(start_vec, end_vec, dive, e1, e2),
        "path": _path_samples(
            start_vec, end_vec, dive, e1, e2,
            int(config.HYPERBOLIC_JOURNEY_PATH_SAMPLES),
        ),
    }
