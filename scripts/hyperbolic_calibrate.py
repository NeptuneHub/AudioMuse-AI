#!/usr/bin/env python3
# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Validation gate for the Hyperbolic Explorer projection scale.

Streams the real norm distribution from the ``embedding`` table, reports the
percentiles, and prints the recommended ``HYPERBOLIC_RADIUS_SCALE`` (the
configured percentile of the norms) together with the projected radius spread
it produces. Pass ``--persist`` to write that scale into app_config so the
analysis-time incremental path and the backfill job agree on it.

Main Features:
* Streams norms with the same paged reader the backfill job uses.
* Reports norm percentiles (50/75/90/95/99/max) and projected radius spread.
* Prints the exact HYPERBOLIC_RADIUS_SCALE env override to use.
* --persist stores the calibrated scale in app_config for cross-process reuse.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import config


def _norms_from_catalogue():
    from tasks.index_build_helpers import iter_embedding_batches

    batches = []
    for batch, _ids in iter_embedding_batches(
        "embedding",
        "embedding",
        int(config.EMBEDDING_DIMENSION),
        where_clause="embedding IS NOT NULL",
    ):
        batches.append(np.linalg.norm(batch.astype(np.float64), axis=1))
    if not batches:
        return np.array([], dtype=np.float64)
    if len(batches) == 1:
        return batches[0]
    return np.concatenate(batches)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate HYPERBOLIC_RADIUS_SCALE from the real norm distribution."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=float(config.HYPERBOLIC_RADIUS_PERCENTILE),
        help="Norm percentile used as the projection scale (default: config).",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write the calibrated scale into app_config for cross-process reuse.",
    )
    args = parser.parse_args()

    norms = _norms_from_catalogue()
    if norms.size == 0:
        print("No embeddings found; nothing to calibrate.")
        return 1

    print(f"Catalog: {norms.size} embeddings, {int(config.EMBEDDING_DIMENSION)}-dim.")
    for pct in (50, 75, 90, 95, 99):
        print(f"  norm p{pct}: {float(np.percentile(norms, pct)):.4f}")
    print(f"  norm max : {float(norms.max()):.4f}")

    scale = float(np.percentile(norms, args.percentile))
    radii = np.tanh(norms / scale)
    print(f"\nRecommended HYPERBOLIC_RADIUS_SCALE = {scale:.4f} (p{args.percentile:g})")
    print(
        "Projected radius spread at that scale: "
        f"min={float(radii.min()):.4f} median={float(np.median(radii)):.4f} "
        f"max={float(radii.max()):.4f}"
    )
    print(f"\nSet env HYPERBOLIC_RADIUS_SCALE={scale:.4f} (or leave unset and run --persist once).")

    if args.persist:
        from database import set_app_config_value

        set_app_config_value("hyperbolic_radius_scale", repr(scale))
        print("Persisted scale to app_config (key hyperbolic_radius_scale).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
