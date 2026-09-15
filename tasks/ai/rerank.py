# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Filter-driven soft re-rank of a candidate pool for the AI planner.

The planner's composition re-rank, extracted behind one public entry point:
score each candidate against the filter dimensions (tag confidence, continuous
gradients, identity matches), blend the primary tool's similarity rank in as an
extra dimension, then order the pool - categorical matches first, continuous
dimensions next, non-song titles (intro/skit/interlude) demoted to the end.

Main Features:
* rerank: the single public seam (formerly planner._rerank_pool); returns the
  ordered pool plus the matched/moved counters the planner reports.
* All scoring, dimension-stats and ordering helpers are private to this module.
"""

import logging
import re
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

CATEGORICAL_DIMS = ('genres', 'voices', 'scale', 'artist', 'album', 'instrumental')

_NON_SONG_TITLE_RE = re.compile(
    r'\b(?:intro|outro|skit|interlude|interludio|prelude|epilogue)\b',
    re.IGNORECASE,
)

YEAR_DECAY_SPAN = 30.0

_KEY_PC = {
    'C': 0,
    'B#': 0,
    'C#': 1,
    'DB': 1,
    'D': 2,
    'D#': 3,
    'EB': 3,
    'E': 4,
    'FB': 4,
    'F': 5,
    'E#': 5,
    'F#': 6,
    'GB': 6,
    'G': 7,
    'G#': 8,
    'AB': 8,
    'A': 9,
    'A#': 10,
    'BB': 10,
    'B': 11,
    'CB': 11,
}


def _key_pitch_class(k) -> Optional[int]:
    if not k:
        return None
    s = str(k).strip().upper().replace('\u266f', '#').replace('\u266d', 'B')
    token = s.split()[0] if s.split() else s
    for cand in (token[:2], token[:1]):
        if cand in _KEY_PC:
            return _KEY_PC[cand]
    return None


def _range_pref_score(v_norm: float, req_lo: float, req_hi: float) -> float:
    v = max(0.0, min(1.0, v_norm))
    prefer_high = req_hi >= 0.99 and req_lo > 0.01
    prefer_low = req_lo <= 0.01 and req_hi < 0.99
    if prefer_high:
        return v
    if prefer_low:
        return 1.0 - v
    center = (req_lo + req_hi) / 2.0
    half = max((req_hi - req_lo) / 2.0, 1e-6)
    return max(0.0, 1.0 - abs(v - center) / half)


def _parse_tag_scores(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not raw or not isinstance(raw, str):
        return out
    for part in raw.split(','):
        if ':' not in part:
            continue
        label, _, score = part.rpartition(':')
        label = label.strip().lower()
        if not label:
            continue
        try:
            out[label] = float(score.strip())
        except ValueError:
            continue
    return out


def _filter_dim_scores(filt: Dict, feats: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not filt or not feats:
        return out

    mv = _parse_tag_scores(feats.get('mood_vector') or '')
    of = _parse_tag_scores(feats.get('other_features') or '')

    def _max_conf(labels, table):
        best = 0.0
        for lab in labels or []:
            c = table.get((lab or '').strip().lower(), 0.0)
            if c > best:
                best = c
        return best

    if filt.get('genres'):
        out['genres'] = _max_conf(filt['genres'], mv)
    if filt.get('voices'):
        out['voices'] = _max_conf(filt['voices'], mv)
    if filt.get('moods'):
        out['moods'] = _max_conf(filt['moods'], of)
    if filt.get('other_features'):
        out['other_features'] = _max_conf(filt['other_features'], of)

    if filt.get('instrumental') is not None:
        want = filt['instrumental']
        if isinstance(want, str):
            want = want.strip().lower() in ('true', '1', 'yes')
        conf = mv.get('instrumental', 0.0)
        out['instrumental'] = conf if want else max(0.0, 1.0 - conf)

    if filt.get('year_min') is not None or filt.get('year_max') is not None:
        year = feats.get('year')
        if year is None:
            out['year'] = 0.0
        else:
            ymin = int(filt['year_min']) if filt.get('year_min') is not None else None
            ymax = int(filt['year_max']) if filt.get('year_max') is not None else None
            if (ymin is None or year >= ymin) and (ymax is None or year <= ymax):
                out['year'] = 1.0
            else:
                dist = (ymin - year) if (ymin is not None and year < ymin) else (year - ymax)
                out['year'] = max(0.0, 1.0 - dist / YEAR_DECAY_SPAN)

    if filt.get('tempo_min') is not None or filt.get('tempo_max') is not None:
        tempo = feats.get('tempo')
        if tempo is None:
            out['tempo'] = 0.0
        else:
            t_lo, t_hi = config.TEMPO_MIN_BPM, config.TEMPO_MAX_BPM
            span = (t_hi - t_lo) or 1.0
            v_norm = (float(tempo) - t_lo) / span
            req_lo = (
                ((float(filt['tempo_min']) - t_lo) / span)
                if filt.get('tempo_min') is not None
                else 0.0
            )
            req_hi = (
                ((float(filt['tempo_max']) - t_lo) / span)
                if filt.get('tempo_max') is not None
                else 1.0
            )
            out['tempo'] = _range_pref_score(
                v_norm, max(0.0, min(1.0, req_lo)), max(0.0, min(1.0, req_hi))
            )

    if filt.get('energy_min') is not None or filt.get('energy_max') is not None:
        energy = feats.get('energy')
        if energy is None:
            out['energy'] = 0.0
        else:
            span = (config.ENERGY_MAX - config.ENERGY_MIN) or 1.0
            v_norm = (float(energy) - config.ENERGY_MIN) / span
            req_lo = float(filt['energy_min']) if filt.get('energy_min') is not None else 0.0
            req_hi = float(filt['energy_max']) if filt.get('energy_max') is not None else 1.0
            out['energy'] = _range_pref_score(
                v_norm, max(0.0, min(1.0, req_lo)), max(0.0, min(1.0, req_hi))
            )

    if filt.get('scale'):
        s = (feats.get('scale') or '').strip().lower()
        out['scale'] = 1.0 if s == str(filt['scale']).strip().lower() else 0.0

    if filt.get('key'):
        sp = _key_pitch_class(feats.get('key'))
        rp = _key_pitch_class(filt.get('key'))
        if sp is None or rp is None:
            k = (feats.get('key') or '').strip().upper()
            out['key'] = 1.0 if k == str(filt['key']).strip().upper() else 0.0
        else:
            d = abs(sp - rp) % 12
            d = min(d, 12 - d)
            out['key'] = 1.0 - d / 6.0

    if filt.get('min_rating') is not None:
        r = feats.get('rating')
        out['min_rating'] = max(0.0, min(1.0, float(r) / 5.0)) if r is not None else 0.0

    if filt.get('artist'):
        a = (feats.get('author') or '').strip().lower()
        out['artist'] = 1.0 if a == str(filt['artist']).strip().lower() else 0.0

    if filt.get('album'):
        alb = (feats.get('album') or '').strip().lower()
        out['album'] = 1.0 if str(filt['album']).strip().lower() in alb else 0.0

    return out


def _filter_dimension_report(filt: Dict, feats_map: Dict, pool_songs: List[Dict]):
    items = [feats_map.get(s.get('item_id'), {}) for s in pool_songs]
    n = len(items) or 1
    lines: List[str] = []
    machine: Dict = {}

    def _tag_stats(labels, column):
        vals = []
        for f in items:
            tags = _parse_tag_scores(f.get(column) or '')
            best = 0.0
            for lab in labels or []:
                c = tags.get((lab or '').strip().lower(), 0.0)
                if c > best:
                    best = c
            vals.append(best)
        nz = sum(1 for v in vals if v > 0)
        return nz, (min(vals) if vals else 0.0), (max(vals) if vals else 0.0)

    if filt.get('genres'):
        nz, lo, hi = _tag_stats(filt['genres'], 'mood_vector')
        lines.append(
            f"   genres {filt['genres']} -> mood_vector (top-5, sparse): {nz}/{n} carry it, rest scored 0 (range {lo:.2f}..{hi:.2f})"
        )
        machine['genres'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('voices'):
        nz, lo, hi = _tag_stats(filt['voices'], 'mood_vector')
        lines.append(
            f"   voices {filt['voices']} -> mood_vector (top-5, sparse): {nz}/{n} carry it, rest scored 0 (range {lo:.2f}..{hi:.2f})"
        )
        machine['voices'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('moods'):
        nz, lo, hi = _tag_stats(filt['moods'], 'other_features')
        lines.append(
            f"   moods {filt['moods']} -> other_features (dense, every song 0..1): range {lo:.2f}..{hi:.2f}"
        )
        machine['moods'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('other_features'):
        nz, lo, hi = _tag_stats(filt['other_features'], 'other_features')
        lines.append(
            f"   other_features {filt['other_features']} -> other_features (dense): range {lo:.2f}..{hi:.2f}"
        )
        machine['other_features'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('instrumental') is not None:
        nz, lo, hi = _tag_stats(['instrumental'], 'mood_vector')
        lines.append(
            f"   instrumental={filt['instrumental']} -> mood_vector (top-5, sparse): {nz}/{n} carry the tag (range {lo:.2f}..{hi:.2f})"
        )
        machine['instrumental'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('energy_min') is not None or filt.get('energy_max') is not None:
        lines.append(
            f"   energy {filt.get('energy_min', '?')}..{filt.get('energy_max', '?')} -> continuous gradient"
        )
        machine['energy'] = (filt.get('energy_min'), filt.get('energy_max'))
    if filt.get('tempo_min') is not None or filt.get('tempo_max') is not None:
        lines.append(
            f"   tempo {filt.get('tempo_min', '?')}..{filt.get('tempo_max', '?')} -> continuous gradient"
        )
        machine['tempo'] = (filt.get('tempo_min'), filt.get('tempo_max'))
    if filt.get('year_min') is not None or filt.get('year_max') is not None:
        lines.append(
            f"   year {filt.get('year_min', '?')}..{filt.get('year_max', '?')} -> proximity gradient"
        )
        machine['year'] = (filt.get('year_min'), filt.get('year_max'))
    if filt.get('min_rating') is not None:
        lines.append(f"   min_rating {filt['min_rating']} -> rating/5 gradient")
        machine['min_rating'] = filt['min_rating']
    if filt.get('scale'):
        lines.append(f"   scale {filt['scale']} -> identity match")
        machine['scale'] = filt['scale']
    if filt.get('key'):
        lines.append(f"   key {filt['key']} -> chromatic-distance gradient")
        machine['key'] = filt['key']
    if filt.get('artist'):
        lines.append(f"   artist {filt['artist']} -> identity match")
        machine['artist'] = filt['artist']
    if filt.get('album'):
        lines.append(f"   album {filt['album']} -> identity substring")
        machine['album'] = filt['album']
    return lines, machine


def _norm_dim(v, lo, hi):
    return (v - lo) / (hi - lo) if hi > lo else 0.0


def _cont_dim_score(d, cont_keys, dim_min, dim_max, sim_score):
    total = sum(_norm_dim(d.get(k, 0.0), dim_min[k], dim_max[k]) for k in cont_keys)
    n_dims = len(cont_keys)
    if sim_score is not None:
        total += sim_score
        n_dims += 1
    return total / n_dims if n_dims else 0.0


def _cat_dim_count(d, cat_keys):
    return sum(1 for k in cat_keys if d.get(k, 0.0) > 0)


def _cat_dim_conf(d, cat_keys, dim_min, dim_max):
    return sum(_norm_dim(d.get(k, 0.0), dim_min[k], dim_max[k]) for k in cat_keys)


def _blend_sim_scores(sim_by_id, pool_songs):
    if not sim_by_id:
        return None
    vals = [float(sim_by_id.get(s.get('item_id'), 0.0)) for s in pool_songs]
    lo, hi = min(vals), max(vals)
    if hi > lo:
        return [(v - lo) / (hi - lo) for v in vals]
    return None


def _dimension_stats(raw_dims):
    dim_keys = sorted({k for d in raw_dims for k in d})
    dim_min = {k: min((d.get(k, 0.0) for d in raw_dims), default=0.0) for k in dim_keys}
    dim_max = {k: max((d.get(k, 0.0) for d in raw_dims), default=0.0) for k in dim_keys}
    cat_keys = [k for k in dim_keys if k in CATEGORICAL_DIMS]
    cont_keys = [k for k in dim_keys if k not in CATEGORICAL_DIMS]
    return dim_keys, dim_min, dim_max, cat_keys, cont_keys


def _log_pool_ranges(log_messages, dim_keys, dim_min, dim_max, sim_scores, n_demoted):
    if dim_keys:
        norm_summary = ", ".join(f"{k}[{dim_min[k]:.2f}..{dim_max[k]:.2f}]" for k in dim_keys)
        if sim_scores is not None:
            norm_summary += ", similarity[primary-tool rank, blended as an extra dimension]"
        log_messages.append(
            f"   per-dim pool range (each normalized 0..1 for the blend): {norm_summary}"
        )
    if n_demoted:
        log_messages.append(
            f"   non-song tracks (intro/skit/interlude titles): {n_demoted} down-ranked to the end"
        )


def _order_by_category(pool_songs, keep_rank, sort_keys, cat_label, cont_label, log_messages):
    N = len(pool_songs)
    matched = sum(1 for t in sort_keys if t[0] > 0)
    order = sorted(
        range(N),
        key=lambda i: (keep_rank[i], sort_keys[i][0], sort_keys[i][1], sort_keys[i][2]),
        reverse=True,
    )
    final = [pool_songs[i] for i in order]
    moved = sum(1 for new_i, old_i in enumerate(order) if new_i != old_i)
    if matched == 0:
        log_messages.append(
            f"   re-rank: 0/{N} match the requested {cat_label}; all ordered by {cont_label}"
        )
    else:
        log_messages.append(
            f"   re-rank: {matched}/{N} match the requested {cat_label} and rank first; "
            f"remaining ordered by {cont_label} (categorical priority, then gradient)"
        )
    return final, matched, moved


def _order_by_similarity(pool_songs, keep_rank, cont_scores, n_demoted, matched, log_messages):
    N = len(pool_songs)
    if matched == 0:
        order = (
            sorted(range(N), key=lambda i: keep_rank[i], reverse=True)
            if n_demoted
            else list(range(N))
        )
    else:
        order = sorted(range(N), key=lambda i: (keep_rank[i], cont_scores[i]), reverse=True)
    final = [pool_songs[i] for i in order]
    moved = sum(1 for new_i, old_i in enumerate(order) if new_i != old_i)
    if matched == 0:
        log_messages.append(
            f"   re-rank: 0/{N} songs matched the filter -> order UNCHANGED (pure similarity)"
        )
    elif moved == 0:
        log_messages.append(
            f"   re-rank: {matched}/{N} matched but scores tied -> no song changed position"
        )
    else:
        log_messages.append(
            f"   re-rank: {matched}/{N} matched the filter and rose to the top; "
            f"{moved} songs shifted position vs pure similarity order "
            f"(per-dim normalized then averaged with primary-tool rank, "
            f"higher score = higher rank)"
        )
    return final, matched, moved


def rerank(
    pool_songs: List[Dict],
    filt: Dict,
    feats: Dict,
    log_messages: List[str],
    sim_by_id: Optional[Dict[str, float]] = None,
):
    N = len(pool_songs)
    clean_filter = {k: v for k, v in filt.items() if k not in ('candidate_item_ids', 'get_songs')}

    log_messages.append(f"\nFILTER (priority re-rank): {N} songs from pool")
    log_messages.append(f"   filter applied: {clean_filter}")
    dim_lines, _dim_machine = _filter_dimension_report(filt, feats, pool_songs)
    for ln in dim_lines:
        log_messages.append(ln)

    raw_dims = [_filter_dim_scores(filt, feats.get(s['item_id'], {})) for s in pool_songs]
    dim_keys, dim_min, dim_max, cat_keys, cont_keys = _dimension_stats(raw_dims)

    sim_scores = _blend_sim_scores(sim_by_id, pool_songs)
    keep_rank = [0 if _NON_SONG_TITLE_RE.search(s.get('title') or '') else 1 for s in pool_songs]
    n_demoted = keep_rank.count(0)

    cont_scores = [
        _cont_dim_score(
            raw_dims[i], cont_keys, dim_min, dim_max,
            sim_scores[i] if sim_scores is not None else None,
        )
        for i in range(N)
    ]

    _log_pool_ranges(log_messages, dim_keys, dim_min, dim_max, sim_scores, n_demoted)

    if cat_keys:
        sort_keys = [
            (
                _cat_dim_count(raw_dims[i], cat_keys),
                cont_scores[i],
                _cat_dim_conf(raw_dims[i], cat_keys, dim_min, dim_max),
            )
            for i in range(N)
        ]
        cat_label = ", ".join(cat_keys)
        cont_label = ", ".join(cont_keys) if cont_keys else "similarity"
        final, matched, moved = _order_by_category(
            pool_songs, keep_rank, sort_keys, cat_label, cont_label, log_messages
        )
    else:
        matched = sum(1 for d in raw_dims if any(v > 0 for v in d.values()))
        final, matched, moved = _order_by_similarity(
            pool_songs, keep_rank, cont_scores, n_demoted, matched, log_messages
        )

    logger.info(
        "soft re-rank: pool=%d matched=%d moved=%d filter=%s dim_range=%s",
        N,
        matched,
        moved,
        clean_filter,
        {k: (round(dim_min[k], 2), round(dim_max[k], 2)) for k in dim_keys},
    )
    return final, matched, moved
