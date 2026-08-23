# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the Poincare geodesic math and the Geodesic Journey walk.

Covers the gyrovector operations the journey is built on, the properties that
make the walk meaningful (constant hyperbolic speed, an inward bow toward the
shared root, an exact 2-plane the disk drawing can use), and the snapping
engine's ordering, de-duplication and per-artist behaviour.

Main Features:
* The geodesic starts at one endpoint, ends at the other, and every step of an
  evenly spaced t covers the same hyperbolic distance
* The geodesic bows inward: its apex radius never exceeds either endpoint's
* The whole geodesic lies in the 2-plane spanned by its endpoints, so the disk
  drawing is exact rather than a projection that loses the curve
* apply_radial_dive deepens the bow while pinning both endpoints exactly
* unproject_from_poincare inverts the projection exactly
* The journey pins the two chosen songs at the ends and orders interior picks
  by their waypoint
* A step whose only candidates are content duplicates or over the artist cap is
  dropped instead of being filled with a repeat
* Bad length / ancestry_dive values are rejected, and identical endpoints or a
  song with no projection raise instead of returning a degenerate walk
* The snapping engine ranks the whole projected catalogue directly, with no
  IVF probe and no cosine shortcut
* POST /api/hyperbolic/journey validates its endpoints, answers 400 on a
  rejected walk, and never returns an internal canonical id
"""

from unittest.mock import patch

import numpy as np
import pytest

import config
from tasks.hyperbolic_geometry import (
    apply_radial_dive,
    geodesic_apex,
    geodesic_plane_basis,
    hyperbolic_distance,
    mobius_add,
    plane_angles,
    poincare_geodesic,
    project_to_poincare,
    unproject_from_poincare,
)
import tasks.hyperbolic_journey_manager as hjm


def _ball_point(*values):
    vec = np.array(values, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm >= 1.0:
        vec = vec / (norm * 1.05)
    return vec


def test_geodesic_hits_both_endpoints_exactly():
    u = _ball_point(0.6, 0.1, -0.2)
    v = _ball_point(-0.3, 0.7, 0.25)
    points = poincare_geodesic(u, v, np.linspace(0.0, 1.0, 9))
    np.testing.assert_allclose(points[0], u, atol=1e-9)
    np.testing.assert_allclose(points[-1], v, atol=1e-9)


def test_evenly_spaced_t_gives_equal_hyperbolic_steps():
    u = _ball_point(0.55, -0.2, 0.1)
    v = _ball_point(-0.4, 0.5, -0.35)
    points = poincare_geodesic(u, v, np.linspace(0.0, 1.0, 13))
    segments = [hyperbolic_distance(points[i], points[i + 1]) for i in range(len(points) - 1)]
    assert max(segments) - min(segments) < 1e-6
    assert abs(sum(segments) - hyperbolic_distance(u, v)) < 1e-6


def test_geodesic_bows_inward_toward_the_shared_root():
    u = _ball_point(0.8, 0.0)
    v = _ball_point(-0.8, 0.05)
    _, apex = geodesic_apex(u, v)
    apex_radius = float(np.linalg.norm(apex))
    assert apex_radius < min(np.linalg.norm(u), np.linalg.norm(v))


def test_apex_is_the_minimum_radius_of_the_whole_geodesic():
    u = _ball_point(0.7, 0.3, -0.1)
    v = _ball_point(-0.6, 0.4, 0.2)
    t, apex = geodesic_apex(u, v)
    sampled = poincare_geodesic(u, v, np.linspace(0.0, 1.0, 1001))
    assert 0.0 <= t <= 1.0
    assert float(np.linalg.norm(apex)) <= float(np.linalg.norm(sampled, axis=1).min()) + 1e-6


def test_geodesic_stays_in_the_plane_spanned_by_its_endpoints():
    u = _ball_point(0.5, -0.3, 0.2, 0.1)
    v = _ball_point(-0.2, 0.6, -0.1, 0.3)
    points = poincare_geodesic(u, v, np.linspace(0.0, 1.0, 25))
    e1, e2 = geodesic_plane_basis(u, v)
    residual = points - np.outer(points @ e1, e1) - np.outer(points @ e2, e2)
    assert float(np.abs(residual).max()) < 1e-9


def test_plane_angle_puts_the_start_on_the_positive_axis():
    u = _ball_point(0.4, 0.4, 0.2)
    v = _ball_point(-0.5, 0.2, -0.3)
    e1, e2 = geodesic_plane_basis(u, v)
    assert abs(float(plane_angles(u, e1, e2)[0])) < 1e-9


def test_radial_dive_deepens_the_bow_but_pins_both_endpoints():
    u = _ball_point(0.75, 0.1)
    v = _ball_point(-0.2, 0.7)
    ts = np.linspace(0.0, 1.0, 11)
    plain = poincare_geodesic(u, v, ts)
    dived = apply_radial_dive(plain, ts, 0.5)
    np.testing.assert_allclose(dived[0], plain[0], atol=1e-12)
    np.testing.assert_allclose(dived[-1], plain[-1], atol=1e-12)
    assert np.linalg.norm(dived[1:-1], axis=1).max() < np.linalg.norm(plain[1:-1], axis=1).max()


def test_zero_dive_leaves_the_geodesic_untouched():
    u = _ball_point(0.3, 0.5)
    v = _ball_point(-0.6, 0.1)
    ts = np.linspace(0.0, 1.0, 7)
    plain = poincare_geodesic(u, v, ts)
    np.testing.assert_allclose(apply_radial_dive(plain, ts, 0.0), plain, atol=1e-12)


def test_unprojection_inverts_the_projection():
    raw = np.array([[1.5, -2.0, 0.5], [0.2, 0.1, -0.3]], dtype=np.float32)
    projected = project_to_poincare(raw, 2.0)
    np.testing.assert_allclose(
        unproject_from_poincare(projected, 2.0), raw, rtol=1e-4, atol=1e-5
    )


def test_mobius_addition_with_the_origin_is_the_identity():
    x = _ball_point(0.4, -0.5, 0.2)
    np.testing.assert_allclose(mobius_add(np.zeros(3), x).reshape(-1), x, atol=1e-12)
    np.testing.assert_allclose(mobius_add(x, np.zeros(3)).reshape(-1), x, atol=1e-12)


def _details(rows):
    return [dict(row) for row in rows]


def _install_journey_stubs(monkeypatch, rows, details):
    monkeypatch.setattr(hjm, "_region_centroids", lambda: (None, None))
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.fetch_poincare_rows",
        lambda ids: {i: rows[i] for i in ids if i in rows},
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_index.hyperbolic_nearest_multi",
        lambda vectors, k, server_id=None, exclude=frozenset(): [
            i for i in rows if i not in exclude
        ],
    )
    monkeypatch.setattr("database.get_score_data_by_ids", lambda ids: _details(
        [details[i] for i in ids if i in details]
    ))


def _row(*values):
    vec = _ball_point(*values)
    return vec.astype(np.float32), float(np.linalg.norm(vec))


@pytest.fixture
def journey_world(monkeypatch):
    rows = {
        "start": _row(0.80, 0.00),
        "end": _row(-0.75, 0.10),
        "near_start": _row(0.60, 0.05),
        "middle": _row(0.05, 0.02),
        "near_end": _row(-0.55, 0.08),
    }
    details = {
        "start": {"item_id": "start", "title": "Alpha", "author": "A"},
        "end": {"item_id": "end", "title": "Omega", "author": "Z"},
        "near_start": {"item_id": "near_start", "title": "Beta", "author": "B"},
        "middle": {"item_id": "middle", "title": "Gamma", "author": "C"},
        "near_end": {"item_id": "near_end", "title": "Delta", "author": "D"},
    }
    _install_journey_stubs(monkeypatch, rows, details)
    return rows, details


def test_journey_pins_the_chosen_tracks_at_both_ends(journey_world):
    result = hjm.build_hyperbolic_journey("start", "end", length=5)
    ids = [row["item_id"] for row in result["results"]]
    assert ids[0] == "start"
    assert ids[-1] == "end"
    assert result["results"][0]["is_endpoint"] is True
    assert result["results"][-1]["is_endpoint"] is True


def test_journey_orders_interior_picks_by_their_waypoint(journey_world):
    result = hjm.build_hyperbolic_journey("start", "end", length=5)
    steps = [row["step"] for row in result["results"]]
    assert steps == sorted(steps)
    assert [row["item_id"] for row in result["results"]][1:-1] == [
        "near_start", "middle", "near_end"
    ]


def test_journey_never_repeats_a_track(journey_world):
    result = hjm.build_hyperbolic_journey("start", "end", length=12)
    ids = [row["item_id"] for row in result["results"]]
    assert len(ids) == len(set(ids))


def test_journey_reports_the_shared_root_and_a_drawable_path(journey_world):
    result = hjm.build_hyperbolic_journey("start", "end", length=6)
    assert 0.0 <= result["apex"]["t"] <= 1.0
    assert result["apex"]["radius"] < min(result["start_radius"], result["end_radius"])
    assert len(result["path"]) == config.HYPERBOLIC_JOURNEY_PATH_SAMPLES
    assert all("radius" in sample and "angle" in sample for sample in result["path"])
    assert result["geodesic_length"] > 0


def test_journey_drops_a_step_rather_than_repeating_a_content_duplicate(monkeypatch):
    rows = {
        "start": _row(0.80, 0.00),
        "end": _row(-0.75, 0.10),
        "twin": _row(0.10, 0.02),
    }
    details = {
        "start": {"item_id": "start", "title": "Alpha", "author": "A"},
        "end": {"item_id": "end", "title": "Omega", "author": "Z"},
        "twin": {"item_id": "twin", "title": "Alpha", "author": "A"},
    }
    _install_journey_stubs(monkeypatch, rows, details)
    result = hjm.build_hyperbolic_journey("start", "end", length=6)
    assert [row["item_id"] for row in result["results"]] == ["start", "end"]


def test_journey_stops_adding_a_artist_once_the_cap_is_reached(monkeypatch):
    rows = {
        "start": _row(0.80, 0.00),
        "end": _row(-0.75, 0.10),
        "same_a": _row(0.30, 0.02),
        "same_b": _row(0.00, 0.02),
        "same_c": _row(-0.30, 0.02),
    }
    details = {
        "start": {"item_id": "start", "title": "Alpha", "author": "A"},
        "end": {"item_id": "end", "title": "Omega", "author": "Z"},
        "same_a": {"item_id": "same_a", "title": "One", "author": "Repeat"},
        "same_b": {"item_id": "same_b", "title": "Two", "author": "Repeat"},
        "same_c": {"item_id": "same_c", "title": "Three", "author": "Repeat"},
    }
    _install_journey_stubs(monkeypatch, rows, details)
    with patch.object(config, "MAX_SONGS_PER_ARTIST", 2):
        result = hjm.build_hyperbolic_journey("start", "end", length=6)
    picked = [row["item_id"] for row in result["results"]][1:-1]
    assert len(picked) == 2


def test_journey_returns_only_the_endpoints_when_no_candidate_survives(monkeypatch):
    rows = {"start": _row(0.80, 0.00), "end": _row(-0.75, 0.10)}
    details = {
        "start": {"item_id": "start", "title": "Alpha", "author": "A"},
        "end": {"item_id": "end", "title": "Omega", "author": "Z"},
    }
    _install_journey_stubs(monkeypatch, rows, details)
    result = hjm.build_hyperbolic_journey("start", "end", length=8)
    assert result["count"] == 2
    assert result["requested_length"] == 8


def test_journey_rejects_identical_endpoints(journey_world):
    with pytest.raises(ValueError):
        hjm.build_hyperbolic_journey("start", "start")


def test_journey_rejects_a_track_without_a_projection(journey_world):
    with pytest.raises(ValueError):
        hjm.build_hyperbolic_journey("start", "not_projected")


def test_journey_raises_when_the_poincare_index_is_missing(monkeypatch):
    rows = {"start": _row(0.80, 0.00), "end": _row(-0.75, 0.10)}
    details = {
        "start": {"item_id": "start", "title": "Alpha", "author": "A"},
        "end": {"item_id": "end", "title": "Omega", "author": "Z"},
    }
    _install_journey_stubs(monkeypatch, rows, details)
    monkeypatch.setattr(
        "tasks.hyperbolic_index.hyperbolic_nearest_multi",
        lambda vectors, k, server_id=None, exclude=frozenset(): None,
    )
    with pytest.raises(ValueError):
        hjm.build_hyperbolic_journey("start", "end", length=4)


def test_journey_rejects_an_out_of_range_ancestry_dive(journey_world):
    with pytest.raises(ValueError):
        hjm.build_hyperbolic_journey("start", "end", ancestry_dive=1.5)


def test_journey_length_has_no_upper_cap_and_keeps_a_minimum_of_three(journey_world):
    assert hjm._clamp_length(1) == 3
    assert hjm._clamp_length(10 ** 6) == 10 ** 6
    assert hjm._clamp_length(None) == config.HYPERBOLIC_JOURNEY_DEFAULT_LENGTH


def _journey_client():
    from flask import Flask

    import app_hyperbolic

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(app_hyperbolic.hyperbolic_bp)
    return app.test_client()


def test_journey_api_requires_both_endpoints():
    response = _journey_client().post(
        "/api/hyperbolic/journey", json={"start_item_id": "a"}
    )
    assert response.status_code == 400
    assert "end_item_id" in response.get_json()["error"]


def test_journey_api_reports_a_rejected_walk_as_a_bad_request():
    with patch(
        "tasks.hyperbolic_journey_manager.build_hyperbolic_journey",
        side_effect=ValueError("The start and end track must be different."),
    ):
        response = _journey_client().post(
            "/api/hyperbolic/journey", json={"start_item_id": "a", "end_item_id": "a"}
        )
    assert response.status_code == 400
    assert response.get_json()["error"] == "The start and end track must be different."


def test_journey_api_returns_provider_ids_not_canonical_ones():
    walk = {
        "results": [
            {"item_id": "fp_start", "step": 0, "is_endpoint": True},
            {"item_id": "fp_mid", "step": 1, "is_endpoint": False},
            {"item_id": "fp_end", "step": 2, "is_endpoint": True},
        ],
        "count": 3,
        "requested_length": 3,
        "apex": {"t": 0.5, "radius": 0.2, "angle": 0.0, "region": None},
        "path": [],
    }
    mapping = {"fp_start": "srv1", "fp_mid": "srv2", "fp_end": "srv3"}
    with patch(
        "tasks.hyperbolic_journey_manager.build_hyperbolic_journey", return_value=walk
    ), patch("app_hyperbolic._attach_title_author"), patch(
        "app_helper.attach_song_features", side_effect=lambda rows, **kw: rows
    ), patch(
        "app_server_context.resolve_input_item_id", side_effect=lambda raw, data=None: raw
    ), patch(
        "app_server_context.scope_results",
        side_effect=lambda rows, **kw: [{**r, "item_id": mapping[r["item_id"]]} for r in rows],
    ), patch(
        "app_server_context.translate_ids_for_request", return_value=mapping
    ):
        response = _journey_client().post(
            "/api/hyperbolic/journey",
            json={"start_item_id": "fp_start", "end_item_id": "fp_end"},
        )

    body = response.get_json()
    assert response.status_code == 200
    assert [row["item_id"] for row in body["results"]] == ["srv1", "srv2", "srv3"]
    assert body["start_item_id"] == "srv1"
    assert body["end_item_id"] == "srv3"
    assert not any(str(row["item_id"]).startswith("fp_") for row in body["results"])
