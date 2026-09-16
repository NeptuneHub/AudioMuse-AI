# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the chromaprint lookup of the fingerprint canonicalizer.

A song mapped on several servers has one chromaprint per mapped file. The
confirmation step must read the fingerprint of the strongest mapping, in the
same match-tier order every other DISTINCT ON over track_server_map uses, so
the same data always yields the same verdict.

Main Features:
* The chromaprint query orders by item id, match-tier rank, provider id
* The tier ranking comes from registry.match_tier_rank_sql, not a local copy
"""

from unittest.mock import MagicMock

import numpy as np

from tasks import fingerprint_canonicalize, simhash
from tasks.mediaserver import registry


def test_the_chromaprint_lookup_picks_the_strongest_mapping_deterministically(monkeypatch):
    monkeypatch.setattr(fingerprint_canonicalize.config, "CHROMAPRINT_GATE_ENABLED", True)
    monkeypatch.setattr(simhash, "confirm_pairs", lambda *a, **k: np.zeros(0, dtype=bool))
    fetch = MagicMock()
    fetch.fetchall.return_value = []

    fingerprint_canonicalize._confirm_slice(
        fetch, ["fp_1", "fp_2"], np.array([0]), np.array([1]), {},
    )

    chromaprint_queries = [
        c.args[0] for c in fetch.execute.call_args_list if "JOIN chromaprint" in c.args[0]
    ]
    assert chromaprint_queries, "the gate is on, so the chromaprint lookup must run"
    sql = chromaprint_queries[0]
    assert "ORDER BY m.item_id, " + registry.match_tier_rank_sql("m.match_tier") + ", m.provider_track_id" in sql
    assert sql.index("DISTINCT ON (m.item_id)") < sql.index("ORDER BY m.item_id")
