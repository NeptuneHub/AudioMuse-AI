# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CodSpeed performance benchmarks for the sanitization helpers.

Measures the string and JSON sanitization paths that run on every database
write and API response, using realistic payload sizes so the benchmarks track
the cost of the regex stripping and numpy conversion hot paths.

Main Features:
* Benchmarks NUL/control-character stripping on plain strings.
* Benchmarks nested-JSON sanitization and numpy-to-native conversion.
"""

import numpy as np
import pytest

import sanitization


DIRTY_STRING = (
    "Song\x00Title \x01with \x02control\x1f chars and a long tail "
    "of unicode text \u00e9\u00e8\u00ea " * 40
)

NESTED_JSON = {
    "tracks": [
        {
            "item_id": f"track-{i}",
            "title": f"Title\x00 {i}",
            "author": f"Artist\x1f {i}",
            "moods": {"happy": 0.8, "energetic": 0.6, "calm": 0.2},
            "tags": [f"tag\x00{j}" for j in range(5)],
        }
        for i in range(50)
    ],
    "meta": {"note": "clean\x00note", "count": 50},
}

NUMPY_PAYLOAD = {
    "embedding": np.random.rand(200),
    "scores": [np.float64(v) for v in np.random.rand(50)],
    "counts": {"a": np.int64(3), "b": np.int32(7)},
    "flag": np.bool_(True),
    "matrix": np.random.rand(20, 20),
}


@pytest.mark.benchmark
def test_bench_sanitize_string_for_db(benchmark):
    result = benchmark(sanitization.sanitize_string_for_db, DIRTY_STRING)
    assert "\x00" not in result


@pytest.mark.benchmark
def test_bench_sanitize_db_field(benchmark):
    result = benchmark(sanitization.sanitize_db_field, DIRTY_STRING, 1000, "title")
    assert "\x00" not in result


@pytest.mark.benchmark
def test_bench_sanitize_json_for_db(benchmark):
    result = benchmark(sanitization.sanitize_json_for_db, NESTED_JSON)
    assert result["meta"]["count"] == 50


@pytest.mark.benchmark
def test_bench_sanitize_for_json(benchmark):
    result = benchmark(sanitization.sanitize_for_json, NUMPY_PAYLOAD)
    assert isinstance(result["embedding"], list)
