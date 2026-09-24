# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Library-calibrated energy and instrument scales for the playlist AI.

The chat tools speak energy as 0.0 (calm) .. 1.0 (intense). This module maps
that scale onto the library's real energy distribution, so ``tools`` (SQL
cuts), ``tool_impl`` (brainstorm gates) and ``rerank`` (soft gradient) agree
on what "calm" or "intense" means for the songs actually analyzed. It does the
same for instruments: how strongly a DCLAP SAE concept fires on a song, as a
percentile of the library.

Main Features:
* energy_to_raw / energy_to_norm: 0..1 is a library PERCENTILE (0.35 = the calmest 35% of songs), interpolated over 21 quantiles read with one percentile_cont query.
* Independent of ENERGY_MIN/ENERGY_MAX: an install whose stored range no longer matches the analyzed values still gets working energy filters; the configured linear scale is only the fallback when the library query fails or the library is empty.
* concept_percentiles scores candidate songs on the SAE concepts a request names (clap_steering.concept_scores over their stored CLAP embeddings) against a sampled library distribution, so "viola" means "among the songs where viola fires most in THIS library".
* Both scales are cached in process for an hour (a failed read is retried after a minute); the concept one keeps only sorted per-concept samples, so there is nothing to unload.
"""

import bisect
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

import config
from tasks.mcp_helper import get_db_connection

logger = logging.getLogger(__name__)

QUANTILE_STEPS = 20
CACHE_SECONDS = 3600.0
FAILURE_RETRY_SECONDS = 60.0

_FRACTIONS = [i / QUANTILE_STEPS for i in range(QUANTILE_STEPS + 1)]
_lock = threading.Lock()
_cache = {'at': None, 'values': None, 'ttl': 0.0}


def _read_quantiles() -> Optional[List[float]]:
    db_conn = get_db_connection()
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT percentile_cont(%s::float8[]) WITHIN GROUP (ORDER BY energy) "
                "FROM public.score WHERE energy IS NOT NULL",
                (_FRACTIONS,),
            )
            row = cur.fetchone()
    finally:
        db_conn.close()
    values = row[0] if row else None
    if not values or any(v is None for v in values):
        return None
    values = [float(v) for v in values]
    if values[-1] <= values[0]:
        return None
    return values


def energy_quantiles() -> Optional[List[float]]:
    now = time.monotonic()
    with _lock:
        if _cache['at'] is not None and now - _cache['at'] < _cache['ttl']:
            return _cache['values']
    try:
        values = _read_quantiles()
        ttl = CACHE_SECONDS
    except Exception:
        logger.exception(
            "Energy calibration query failed; chat energy filters use the configured "
            "ENERGY_MIN/ENERGY_MAX scale until it succeeds"
        )
        values = None
        ttl = FAILURE_RETRY_SECONDS
    with _lock:
        _cache.update(at=now, values=values, ttl=ttl)
    return values


def reset_cache() -> None:
    with _lock:
        _cache.update(at=None, values=None, ttl=0.0)
        _concept_cache.update(at=None, values=None, ttl=0.0)


CONCEPT_SAMPLE_SIZE = 4000
EMBEDDING_FETCH_CHUNK = 5000
_concept_cache = {'at': None, 'values': None, 'ttl': 0.0}


def _vectors(rows) -> Dict[str, np.ndarray]:
    out = {}
    for item_id, raw in rows:
        vec = np.frombuffer(bytes(raw), dtype=np.float32)
        if vec.shape[0] == config.CLAP_EMBEDDING_DIMENSION:
            out[str(item_id)] = vec
    return out


def _read_clap_sample(size: int) -> Optional[np.ndarray]:
    db_conn = get_db_connection()
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.clap_embedding")
            total = int(cur.fetchone()[0] or 0)
            if not total:
                return None
            percent = min(100.0, max(0.01, size * 100.0 / total))
            cur.execute(
                "SELECT item_id, embedding FROM public.clap_embedding TABLESAMPLE SYSTEM (%s) LIMIT %s",
                (percent, size * 2),
            )
            rows = cur.fetchall()
    finally:
        db_conn.close()
    vectors = list(_vectors(rows).values())
    return np.vstack(vectors) if vectors else None


def _read_concept_distributions() -> Optional[Dict[str, np.ndarray]]:
    from tasks.clap_steering import concept_scores, concept_terms

    terms = concept_terms()
    if not terms:
        return None
    sample = _read_clap_sample(CONCEPT_SAMPLE_SIZE)
    if sample is None:
        return None
    scores = concept_scores(sample, terms)
    return {term: np.sort(np.asarray(values, dtype=np.float32)) for term, values in scores.items()}


def concept_distributions() -> Optional[Dict[str, np.ndarray]]:
    now = time.monotonic()
    with _lock:
        if _concept_cache['at'] is not None and now - _concept_cache['at'] < _concept_cache['ttl']:
            return _concept_cache['values']
    try:
        values = _read_concept_distributions()
        ttl = CACHE_SECONDS if values else FAILURE_RETRY_SECONDS
    except Exception:
        logger.exception("Sampling the library for the instrument concepts failed")
        values = None
        ttl = FAILURE_RETRY_SECONDS
    with _lock:
        _concept_cache.update(at=now, values=values, ttl=ttl)
    return values


def _read_clap_embeddings(item_ids: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    db_conn = get_db_connection()
    try:
        with db_conn.cursor() as cur:
            for start in range(0, len(item_ids), EMBEDDING_FETCH_CHUNK):
                cur.execute(
                    "SELECT item_id, embedding FROM public.clap_embedding WHERE item_id = ANY(%s)",
                    (list(item_ids[start:start + EMBEDDING_FETCH_CHUNK]),),
                )
                out.update(_vectors(cur.fetchall()))
    finally:
        db_conn.close()
    return out


def concept_percentiles(item_ids: List[str], terms: List[str]) -> Dict[str, Dict[str, float]]:
    from tasks.clap_steering import concept_scores

    if not item_ids or not terms:
        return {}
    distributions = concept_distributions()
    if not distributions:
        return {}
    wanted = [term for term in terms if term in distributions and len(distributions[term])]
    if not wanted:
        return {}
    vectors = _read_clap_embeddings([str(i) for i in item_ids])
    if not vectors:
        return {}
    ids = list(vectors)
    scores = concept_scores(np.vstack([vectors[i] for i in ids]), wanted)
    out: Dict[str, Dict[str, float]] = {i: {} for i in ids}
    for term, values in scores.items():
        library = distributions[term]
        ranks = np.searchsorted(library, np.asarray(values, dtype=np.float32), side='right') / len(library)
        for item_id, rank in zip(ids, ranks.tolist()):
            out[item_id][term] = float(rank)
    return out


def _clamp01(value) -> float:
    return max(0.0, min(1.0, float(value)))


def energy_to_raw(norm) -> float:
    norm = _clamp01(norm)
    values = energy_quantiles()
    if not values:
        return config.ENERGY_MIN + norm * (config.ENERGY_MAX - config.ENERGY_MIN)
    pos = norm * QUANTILE_STEPS
    lo = min(int(pos), QUANTILE_STEPS - 1)
    frac = pos - lo
    return values[lo] + frac * (values[lo + 1] - values[lo])


def energy_to_norm(raw) -> float:
    raw = float(raw)
    values = energy_quantiles()
    if not values:
        span = (config.ENERGY_MAX - config.ENERGY_MIN) or 1.0
        return _clamp01((raw - config.ENERGY_MIN) / span)
    if raw <= values[0]:
        return 0.0
    if raw >= values[-1]:
        return 1.0
    hi = bisect.bisect_right(values, raw)
    lo = hi - 1
    width = values[hi] - values[lo]
    frac = (raw - values[lo]) / width if width > 0 else 0.0
    return _clamp01((lo + frac) / QUANTILE_STEPS)
