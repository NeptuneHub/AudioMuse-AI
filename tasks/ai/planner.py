# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Single-call plan builder and executor for AI playlist requests.

Orchestrates the AI pipeline between ``api``/``prompts`` (LLM calls),
``vocab`` (label normalization), and ``tools`` (execution): one tool-calling
LLM request over the full tool surface emits the plan, which is then
validated, deduplicated, merged into a normalized plan, run, and composed.

Main Features:
* Regex pre-extraction of years/decades/BPM/tempo/energy/genre (and negated-genre) hints; hints the model omitted are deterministically merged back into the filter (hint backstop), while hallucinated year/instrumental args absent from the request are stripped; unsupported constraints (duration) surface as plan notes.
* Soft categorical-priority re-rank (matching songs first, continuous dims order within tiers) that blends the primary tool's similarity rank as an extra dimension and down-ranks intro/skit/interlude titles; exclude_artists/exclude_genres are the one HARD cut; multiple finder tools get an intersection boost (songs returned by several tools rank first).
* knowledge_lookup (AI brainstorm) results are returned as-is: any parsed filter is dropped (and reported as ignored) so brainstorm output is not re-ranked; score_threshold relax loop backfills when a filter pool is short, and a filter-only query that still underfills the target re-runs without its soft dims (tempo/energy/moods/key/scale/rating) and applies them as the soft re-rank over the broader pool. Duplicate tool calls are dropped, plans cap at 4 calls, and a zero-result run triggers ONE replan with failure feedback.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config

from tasks.ai.vocab import (
    ALIAS_ENERGY,
    ALIAS_GENRE,
    ALIAS_TEMPO,
    GENRE_VOCAB,
    normalize_genre_list,
    normalize_mood_list,
    normalize_scale,
    normalize_voices_list,
)

logger = logging.getLogger(__name__)


PRIMARY_NAMES = {
    'seed_search',
    'text_match',
    'knowledge_lookup',
}
FILTER_NAME = 'search_database'

RELAX_THRESHOLD_STEPS = (0.5, 0.4, 0.3, 0.2)
SCORED_FILTER_KEYS = ('genres', 'voices', 'moods', 'other_features')

CATEGORICAL_DIMS = ('genres', 'voices', 'scale', 'artist', 'album', 'instrumental')

_NON_SONG_TITLE_RE = re.compile(
    r'\b(?:intro|outro|skit|interlude|interludio|prelude|epilogue)\b',
    re.IGNORECASE,
)

_ENERGY_BUCKET_RANGE = {'low': (0.0, 0.33), 'medium': (0.33, 0.66), 'high': (0.66, 1.0)}
_TEMPO_BUCKET_RANGE = {'slow': (None, 90), 'medium': (90, 140), 'fast': (140, None)}

COMPOSITION_POOL_TARGET = 10000

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
    s = str(k).strip().upper().replace('♯', '#').replace('♭', 'B')
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


def _rerank_pool(
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


FILTER_LIST_KEYS = (
    'genres',
    'voices',
    'moods',
    'other_features',
    'exclude_artists',
    'exclude_genres',
)
FILTER_MIN_KEYS = ('tempo_min', 'energy_min', 'year_min', 'min_rating')
FILTER_MAX_KEYS = ('tempo_max', 'energy_max', 'year_max')
FILTER_SCALAR_KEYS = ('key', 'scale', 'album', 'artist', 'instrumental')
FILTER_ALL_KEYS = FILTER_LIST_KEYS + FILTER_MIN_KEYS + FILTER_MAX_KEYS + FILTER_SCALAR_KEYS


def _song_is_excluded(s: Dict, feats: Dict, ex_artist_norms: set, ex_genre_lows: List[str]) -> bool:
    from tasks.ai.tool_impl import _EXCLUDE_GENRE_SCORE, _normalize_for_match

    f = feats.get(s.get('item_id'), {})
    author = f.get('author') or s.get('artist') or ''
    if _normalize_for_match(author) in ex_artist_norms:
        return True
    if ex_genre_lows:
        mv = _parse_tag_scores(f.get('mood_vector') or '')
        return any(mv.get(g, 0.0) >= _EXCLUDE_GENRE_SCORE for g in ex_genre_lows)
    return False


def _apply_exclusions(pool_songs: List[Dict], filt: Dict, feats: Dict, log_messages: List[str]):
    ex_artists = [a for a in (filt.get('exclude_artists') or []) if isinstance(a, str) and a.strip()]
    ex_genres = [g for g in (filt.get('exclude_genres') or []) if isinstance(g, str) and g.strip()]
    if not ex_artists and not ex_genres:
        return pool_songs

    from tasks.ai.tool_impl import _normalize_for_match

    ex_artist_norms = {_normalize_for_match(a) for a in ex_artists}
    ex_genre_lows = [g.strip().lower() for g in ex_genres]

    kept = [s for s in pool_songs if not _song_is_excluded(s, feats, ex_artist_norms, ex_genre_lows)]
    removed = len(pool_songs) - len(kept)
    if removed:
        log_messages.append(
            f"   exclusions (hard cut): removed {removed}/{len(pool_songs)} songs "
            f"(exclude_artists={ex_artists or '-'}, exclude_genres={ex_genres or '-'})"
        )
    return kept


@dataclass
class ToolPlan:
    primaries: List[Dict] = field(default_factory=list)
    filter: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_DECADE_RE = re.compile(r"\b((?:19|20)?(?:60|70|80|90|00|10|20))s\b", re.IGNORECASE)
_BPM_RE = re.compile(r"\b(\d{2,3})\s*bpm\b", re.IGNORECASE)
_ENERGY_NUM_RE = re.compile(
    r"\benergy\s*(?:above|>=?|over|min(?:imum)?)\s*([0-9]*\.[0-9]+|[0-9]+)\b", re.IGNORECASE
)
_INSTRUMENTAL_RE = re.compile(
    r'\b(?:instrumentals?|no\s+(?:vocals?|lyrics|singing|voice)|'
    r'without\s+(?:vocals?|lyrics|singing|voice))\b',
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r'\b(?:under|below|over|above|shorter\s+than|longer\s+than|less\s+than|more\s+than|'
    r'at\s+least|at\s+most|max(?:imum)?|min(?:imum)?)\s+\d{1,3}\s*'
    r'(?:minutes?|mins?|seconds?|secs?)\b',
    re.IGNORECASE,
)
_NEGATION_TAIL_RE = re.compile(
    r"\b(?:no|not|without|except|excluding|avoid|nothing|never|zero)\b[^,.;!?]*$",
    re.IGNORECASE,
)
_YEARISH_RE = re.compile(
    r'\b(?:recent|latest|new(?:est)?|modern|current|today|old(?:er)?|oldies|classic|'
    r'vintage|early|late|decades?|years?|era)\b',
    re.IGNORECASE,
)
_VOCALNESS_RE = re.compile(
    r'\b(?:instrumentals?|vocals?|vocalists?|voices?|singing|singers?|sung|'
    r'karaoke|acapella|a\s+cappella)\b',
    re.IGNORECASE,
)
_GENRE_HINT_SKIP = {'house'}


def _genre_hint_tokens() -> List[str]:
    tokens = {g.lower() for g in GENRE_VOCAB} | set(ALIAS_GENRE.keys())
    tokens -= _GENRE_HINT_SKIP
    return sorted(tokens, key=len, reverse=True)


def _extract_genre_hints(text: str) -> Dict[str, List[str]]:
    masked = text.lower()
    positive_raw: List[str] = []
    negative_raw: List[str] = []
    for token in _genre_hint_tokens():
        pattern = re.compile(rf"\b{re.escape(token)}\b")
        pos = 0
        while True:
            m = pattern.search(masked, pos)
            if not m:
                break
            window = masked[max(0, m.start() - 40):m.start()]
            if _NEGATION_TAIL_RE.search(window):
                negative_raw.append(token)
            else:
                positive_raw.append(token)
            masked = masked[:m.start()] + '\x00' * (m.end() - m.start()) + masked[m.end():]
            pos = m.end()
    positive = normalize_genre_list(positive_raw)['genres']
    negative = normalize_genre_list(negative_raw)['genres']
    positive = [g for g in positive if g not in negative]
    return {'genres': positive, 'exclude_genres': negative}


def _normalize_decade(prefix: str) -> int:
    p = int(prefix)
    if p >= 1000:
        return p
    if p >= 30:
        return 1900 + p
    return 2000 + p


def extract_hints(text: str) -> Dict:
    if not text or not isinstance(text, str):
        return {}

    hints: Dict = {}
    notes: List[str] = []

    years = [int(y) for y in _YEAR_RE.findall(text)]
    if years:
        hints['years'] = years
        hints['year_min'] = min(years)
        hints['year_max'] = max(years)
        notes.append(f"years detected: {years}")

    decade_matches = _DECADE_RE.findall(text)
    if decade_matches:
        decade_starts = [_normalize_decade(d) for d in decade_matches]
        hints.setdefault('year_min', min(decade_starts))
        hints['year_max'] = max(hints.get('year_max', 0), max(d + 9 for d in decade_starts))
        notes.append(f"decade(s) detected: {[f'{d}s' for d in decade_starts]}")

    bpm_match = _BPM_RE.search(text)
    if bpm_match:
        bpm = int(bpm_match.group(1))
        hints['bpm'] = bpm
        notes.append(f"BPM detected: {bpm}")

    low = text.lower()
    for phrase, (tmin, tmax) in ALIAS_TEMPO.items():
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            hints['tempo_min'] = (
                tmin if hints.get('tempo_min') is None else min(hints['tempo_min'], tmin)
            )
            hints['tempo_max'] = (
                tmax if hints.get('tempo_max') is None else max(hints['tempo_max'], tmax)
            )
            notes.append(f"tempo phrase '{phrase}' -> {tmin}-{tmax} BPM")
            break

    for phrase, (emin, emax) in ALIAS_ENERGY.items():
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            hints['energy_min'] = (
                emin if hints.get('energy_min') is None else min(hints['energy_min'], emin)
            )
            hints['energy_max'] = (
                emax if hints.get('energy_max') is None else max(hints['energy_max'], emax)
            )
            notes.append(f"energy phrase '{phrase}' -> {emin}-{emax}")
            break

    energy_num = _ENERGY_NUM_RE.search(text)
    if energy_num:
        try:
            v = float(energy_num.group(1))
            if 0.0 <= v <= 1.0:
                hints['energy_min'] = (
                    v if hints.get('energy_min') is None else min(hints['energy_min'], v)
                )
                notes.append(f"explicit energy floor: {v}")
        except ValueError:
            pass

    if _INSTRUMENTAL_RE.search(text):
        hints['instrumental'] = True
        notes.append("instrumental requested")

    genre_hints = _extract_genre_hints(text)
    if genre_hints['genres']:
        hints['genres'] = genre_hints['genres']
        notes.append(f"genre word(s) detected: {genre_hints['genres']}")
    if genre_hints['exclude_genres']:
        hints['exclude_genres'] = genre_hints['exclude_genres']
        notes.append(f"NEGATED genre word(s) detected: {genre_hints['exclude_genres']}")

    duration_match = _DURATION_RE.search(text)
    if duration_match:
        unsupported = (
            f"duration constraint '{duration_match.group(0)}' is not supported "
            "and was IGNORED (the library has no track-length filter)"
        )
        hints['unsupported'] = [unsupported]
        notes.append(unsupported)

    if notes:
        hints['notes'] = notes
    return hints


def format_hints_block(hints: Optional[Dict]) -> str:
    if not hints:
        return ""
    lines: List[str] = []
    if hints.get('year_min') is not None or hints.get('year_max') is not None:
        lines.append(f"  year: {hints.get('year_min', '?')}..{hints.get('year_max', '?')}")
    if hints.get('bpm') is not None:
        lines.append(f"  bpm: {hints['bpm']}")
    if hints.get('tempo_min') is not None or hints.get('tempo_max') is not None:
        lines.append(f"  tempo: {hints.get('tempo_min', '?')}..{hints.get('tempo_max', '?')}")
    if hints.get('energy_min') is not None or hints.get('energy_max') is not None:
        lines.append(f"  energy: {hints.get('energy_min', '?')}..{hints.get('energy_max', '?')}")
    if hints.get('instrumental') is True:
        lines.append("  instrumental: true (use instrumental=true in search_database)")
    if hints.get('genres'):
        lines.append(f"  genres: {hints['genres']}")
    if hints.get('exclude_genres'):
        lines.append(
            f"  exclude_genres: {hints['exclude_genres']} "
            "(the user does NOT want these; use exclude_genres, never genres)"
        )
    for u in hints.get('unsupported', []):
        lines.append(f"  unsupported: {u}; do not fake it with other filters")
    if not lines:
        return ""
    return (
        "EXTRACTED_HINTS (use these values directly in search_database if relevant):\n"
        + "\n".join(lines)
    )


def _strip_unrequested_filter_args(
    plan: 'ToolPlan',
    hints: Dict,
    original_message: str,
    log_messages: List[str],
) -> None:
    if plan.filter is None:
        return
    filt = plan.filter
    has_year_args = filt.get('year_min') is not None or filt.get('year_max') is not None
    if (
        has_year_args
        and hints.get('year_min') is None
        and hints.get('year_max') is None
        and not _YEARISH_RE.search(original_message)
    ):
        log_messages.append(
            f"   strip hallucinated year range {filt.get('year_min')}..{filt.get('year_max')} "
            "(no year in the request)"
        )
        filt.pop('year_min', None)
        filt.pop('year_max', None)
    if (
        filt.get('instrumental') is not None
        and hints.get('instrumental') is not True
        and not _VOCALNESS_RE.search(original_message)
    ):
        log_messages.append(
            f"   strip hallucinated instrumental={filt['instrumental']} "
            "(no vocal/instrumental wording in the request)"
        )
        filt.pop('instrumental', None)
    if not _has_filter_content(filt):
        plan.notes.append('filter emptied after stripping hallucinated args')
        plan.filter = None


_BPM_HINT_HALF_WINDOW = 10.0


def _backstop_min_max(backstop: Dict, filt: Dict, hints: Dict, min_key: str, max_key: str) -> None:
    if filt.get(min_key) is not None or filt.get(max_key) is not None:
        return
    if hints.get(min_key) is not None:
        backstop[min_key] = hints[min_key]
    if hints.get(max_key) is not None:
        backstop[max_key] = hints[max_key]


def _backstop_tempo(backstop: Dict, filt: Dict, hints: Dict) -> None:
    if filt.get('tempo_min') is not None or filt.get('tempo_max') is not None:
        return
    if hints.get('bpm') is not None:
        backstop['tempo_min'] = float(hints['bpm']) - _BPM_HINT_HALF_WINDOW
        backstop['tempo_max'] = float(hints['bpm']) + _BPM_HINT_HALF_WINDOW
    else:
        _backstop_min_max(backstop, filt, hints, 'tempo_min', 'tempo_max')


def _backstop_missing_list(backstop: Dict, filt: Dict, hints: Dict, key: str) -> None:
    if not hints.get(key):
        return
    existing = {g.lower() for g in (filt.get(key) or [])}
    missing = [g for g in hints[key] if g.lower() not in existing]
    if missing:
        backstop[key] = missing


def _apply_hint_backstop(plan: 'ToolPlan', hints: Dict, log_messages: List[str]) -> None:
    filt = plan.filter or {}
    backstop: Dict = {}

    _backstop_tempo(backstop, filt, hints)
    _backstop_min_max(backstop, filt, hints, 'energy_min', 'energy_max')
    _backstop_min_max(backstop, filt, hints, 'year_min', 'year_max')

    if filt.get('instrumental') is None and hints.get('instrumental') is True:
        backstop['instrumental'] = True

    _backstop_missing_list(backstop, filt, hints, 'genres')
    _backstop_missing_list(backstop, filt, hints, 'exclude_genres')

    if backstop:
        plan.filter = _merge_filter(plan.filter, backstop)
        log_messages.append(
            f"   hint backstop: merged {backstop} into the filter "
            "(detected in the request but missing from the plan)"
        )


def _has_filter_content(args: Dict) -> bool:
    if not isinstance(args, dict):
        return False
    for k in FILTER_ALL_KEYS:
        v = args.get(k)
        if k in FILTER_LIST_KEYS:
            if v:
                return True
        elif v is not None and v != '':
            return True
    return False


def _merge_filter(base: Optional[Dict], incoming: Dict) -> Dict:
    if base is None:
        base = {}
    for k in FILTER_LIST_KEYS:
        if incoming.get(k):
            existing = list(base.get(k) or [])
            for v in incoming[k]:
                if v not in existing:
                    existing.append(v)
            base[k] = existing
    for k in FILTER_MIN_KEYS:
        v = incoming.get(k)
        if v is not None and v != '':
            base[k] = v if base.get(k) is None else min(base[k], v)
    for k in FILTER_MAX_KEYS:
        v = incoming.get(k)
        if v is not None and v != '':
            base[k] = v if base.get(k) is None else max(base[k], v)
    for k in FILTER_SCALAR_KEYS:
        v = incoming.get(k)
        if v is not None and v != '' and k not in base:
            base[k] = v
    return base


def _normalize_filter_inplace(filt: Dict, notes: List[str]) -> Dict:
    if 'genres' in filt and filt['genres']:
        g = normalize_genre_list(filt['genres'])
        filt['genres'] = g['genres']
        for n in g.get('notes') or []:
            notes.append(n)
        if not filt['genres']:
            filt.pop('genres', None)

    if 'exclude_genres' in filt and filt['exclude_genres']:
        eg = normalize_genre_list(filt['exclude_genres'])
        filt['exclude_genres'] = eg['genres']
        for n in eg.get('notes') or []:
            notes.append(n)
        if not filt['exclude_genres']:
            filt.pop('exclude_genres', None)

    if 'exclude_artists' in filt:
        cleaned_ex = []
        for a in filt.get('exclude_artists') or []:
            if isinstance(a, str) and a.strip() and a.strip() not in cleaned_ex:
                cleaned_ex.append(a.strip())
        if cleaned_ex:
            filt['exclude_artists'] = cleaned_ex
        else:
            filt.pop('exclude_artists', None)

    if filt.get('exclude_genres') and filt.get('genres'):
        overlap = [g for g in filt['genres'] if g in filt['exclude_genres']]
        if overlap:
            notes.append(f"genres {overlap} were both requested and excluded; exclusion wins")
            filt['genres'] = [g for g in filt['genres'] if g not in filt['exclude_genres']]
            if not filt['genres']:
                filt.pop('genres', None)

    if 'voices' in filt and filt['voices']:
        v = normalize_voices_list(filt['voices'])
        if v['voices']:
            filt['voices'] = v['voices']
        else:
            filt.pop('voices', None)
        for n in v.get('notes') or []:
            notes.append(n)

    if 'moods' in filt and filt['moods']:
        m = normalize_mood_list(filt['moods'])
        if m.get('voices'):
            existing_v = list(filt.get('voices') or [])
            for vv in m['voices']:
                if vv not in existing_v:
                    existing_v.append(vv)
            filt['voices'] = existing_v
        if m['mood_vector']:
            ignored = [t for t in m['mood_vector'] if t not in m['other_features']]
            if ignored:
                notes.append(
                    f"vocab_normalizer ignored unsupported mood tag(s) {ignored} "
                    f"(moods must be one of: {', '.join(config.OTHER_FEATURE_LABELS)}; "
                    "use 'genres', 'voices' or 'year' for the rest)"
                )
        if m['other_features']:
            existing_of = list(filt.get('moods') or [])
            for o in m['other_features']:
                if o not in existing_of:
                    existing_of.append(o)
            filt['moods'] = existing_of
        else:
            filt.pop('moods', None)
        if m['energy_min'] is not None:
            filt['energy_min'] = (
                m['energy_min']
                if filt.get('energy_min') is None
                else min(filt['energy_min'], m['energy_min'])
            )
        if m['energy_max'] is not None:
            filt['energy_max'] = (
                m['energy_max']
                if filt.get('energy_max') is None
                else max(filt['energy_max'], m['energy_max'])
            )
        if m['tempo_min'] is not None:
            filt['tempo_min'] = (
                m['tempo_min']
                if filt.get('tempo_min') is None
                else min(filt['tempo_min'], m['tempo_min'])
            )
        if m['tempo_max'] is not None:
            filt['tempo_max'] = (
                m['tempo_max']
                if filt.get('tempo_max') is None
                else max(filt['tempo_max'], m['tempo_max'])
            )
        for n in m.get('notes') or []:
            notes.append(n)

    if 'scale' in filt and filt['scale']:
        s = normalize_scale(filt['scale'])
        if s:
            filt['scale'] = s
        else:
            notes.append(f"vocab_normalizer dropped unknown scale: {filt['scale']}")
            filt.pop('scale', None)

    return filt


def _seed_identity(seed: Dict) -> tuple:
    if seed.get('type') == 'artist':
        return ('artist', (seed.get('name') or '').strip().lower(), '')
    return (
        'song',
        (seed.get('title') or '').strip().lower(),
        (seed.get('artist') or '').strip().lower(),
    )


def validate_plan_args(
    tool_calls: List[Dict],
    *,
    user_wants_rating: bool,
    log_messages: Optional[List[str]] = None,
) -> List[Dict]:
    if log_messages is None:
        log_messages = []

    out: List[Dict] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get('name', '')
        args = tc.get('arguments', {}) or {}

        if name == 'seed_search':
            seeds_raw = args.get('seeds') or []
            cleaned_seeds: List[Dict] = []
            for s in seeds_raw:
                if not isinstance(s, dict):
                    continue
                stype = (s.get('type') or '').lower()
                if stype == 'song':
                    title = (s.get('title') or s.get('song_title') or '').strip()
                    artist = (s.get('artist') or s.get('song_artist') or '').strip()
                    if not title or not artist:
                        log_messages.append(f"   skip malformed song seed {s}")
                        continue
                    cleaned_seeds.append({'type': 'song', 'title': title, 'artist': artist})
                elif stype == 'artist':
                    nm = (s.get('name') or s.get('artist') or s.get('id') or '').strip()
                    if not nm:
                        log_messages.append(f"   skip malformed artist seed {s}")
                        continue
                    cleaned_seeds.append({'type': 'artist', 'name': nm})
                else:
                    log_messages.append(f"   skip unknown seed type '{stype}'")
            if not cleaned_seeds:
                log_messages.append(f"   skip {name}: no usable seeds")
                continue
            args['seeds'] = cleaned_seeds

            blend = (args.get('blend_mode') or 'union').lower()
            if blend == 'alchemy' and len(cleaned_seeds) < 2:
                log_messages.append("   coerce blend_mode 'alchemy' -> 'union' (need 2+ seeds)")
                blend = 'union'
            if blend == 'subtract':
                sub_raw = args.get('subtract') or []
                cleaned_sub: List[Dict] = []
                for s in sub_raw:
                    if not isinstance(s, dict):
                        continue
                    stype = (s.get('type') or '').lower()
                    if stype == 'artist':
                        nm = (s.get('name') or s.get('artist') or s.get('id') or '').strip()
                        if nm:
                            cleaned_sub.append({'type': 'artist', 'name': nm})
                    elif stype == 'song':
                        title = (s.get('title') or '').strip()
                        artist = (s.get('artist') or '').strip()
                        if title and artist:
                            cleaned_sub.append({'type': 'song', 'title': title, 'artist': artist})
                seed_keys = {_seed_identity(s) for s in cleaned_seeds}
                self_subtracted = [
                    s for s in cleaned_sub if _seed_identity(s) in seed_keys
                ]
                if self_subtracted:
                    cleaned_sub = [
                        s for s in cleaned_sub if _seed_identity(s) not in seed_keys
                    ]
                    log_messages.append(
                        f"   drop self-subtraction: {len(self_subtracted)} subtract "
                        "item(s) duplicate the seeds"
                    )
                if not cleaned_sub:
                    log_messages.append(
                        "   coerce blend_mode 'subtract' -> 'union' (empty 'subtract' list)"
                    )
                    args.pop('subtract', None)
                    blend = 'union'
                else:
                    args['subtract'] = cleaned_sub
            args['blend_mode'] = blend

        if name == 'text_match':
            query = (args.get('query') or '').strip()
            if not query:
                log_messages.append(f"   skip {name}: empty query")
                continue
            args['query'] = query
            mode = (args.get('mode') or '').lower()
            if mode in ('audio', 'lyrics'):
                args['mode'] = mode
            else:
                if mode:
                    log_messages.append(
                        f"   drop invalid text_match mode '{args.get('mode')}' (dispatch default applies)"
                    )
                args.pop('mode', None)

        if name == 'knowledge_lookup':
            req = (args.get('user_request') or args.get('query') or '').strip()
            if not req:
                log_messages.append(f"   skip {name}: empty user_request")
                continue
            args['user_request'] = req

        if name == FILTER_NAME:
            y_min = args.get('year_min')
            y_max = args.get('year_max')
            try:
                if y_min is not None and int(y_min) < 1900:
                    log_messages.append(f"   strip nonsensical year_min={y_min}")
                    args.pop('year_min', None)
            except (TypeError, ValueError):
                args.pop('year_min', None)
            try:
                if y_max is not None and int(y_max) < 1900:
                    log_messages.append(f"   strip nonsensical year_max={y_max}")
                    args.pop('year_max', None)
            except (TypeError, ValueError):
                args.pop('year_max', None)

            if not user_wants_rating and args.get('min_rating'):
                log_messages.append(
                    f"   strip hallucinated min_rating={args['min_rating']} (user didn't ask for ratings)"
                )
                args.pop('min_rating', None)

            if (
                not _has_filter_content(args)
                and not args.get('min_rating')
                and args.get('year_min') is None
                and args.get('year_max') is None
            ):
                log_messages.append(f"   skip {name}: no filters specified")
                continue

        out.append(tc)
    return out


def validate_and_normalize_plan(tool_calls: List[Dict]) -> ToolPlan:
    plan = ToolPlan()
    if not tool_calls:
        return plan

    merged_filter: Optional[Dict] = None
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get('name')
        args = tc.get('arguments') or {}
        if name == FILTER_NAME:
            if _has_filter_content(args):
                clean = {k: v for k, v in args.items() if k in FILTER_ALL_KEYS}
                merged_filter = _merge_filter(merged_filter, clean)
            else:
                plan.notes.append('search_database call with no filter content was dropped')
        elif name in PRIMARY_NAMES:
            plan.primaries.append(tc)
        elif name:
            plan.notes.append(f"unknown tool '{name}' was dropped")

    if merged_filter is not None:
        plan.filter = _normalize_filter_inplace(merged_filter, plan.notes)
        if not _has_filter_content(plan.filter):
            plan.notes.append('filter was emptied after vocab normalization')
            plan.filter = None

    return plan


def plan_from_tool_calls(tool_calls: List[Dict]) -> ToolPlan:
    return validate_and_normalize_plan(tool_calls or [])


MAX_TOOL_CALLS = 4


def dedupe_and_cap_calls(
    tool_calls: List[Dict],
    log_messages: Optional[List[str]] = None,
) -> List[Dict]:
    if log_messages is None:
        log_messages = []
    out: List[Dict] = []
    seen: set = set()
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        try:
            key = (tc.get('name'), json.dumps(tc.get('arguments') or {}, sort_keys=True, default=str))
        except (TypeError, ValueError):
            key = (tc.get('name'), str(tc.get('arguments')))
        if key in seen:
            log_messages.append(f"   dropped duplicate {tc.get('name')} call")
            continue
        seen.add(key)
        out.append(tc)
    if len(out) > MAX_TOOL_CALLS:
        log_messages.append(f"   capping {len(out)} tool calls to {MAX_TOOL_CALLS}")
        out = out[:MAX_TOOL_CALLS]
    return out


SOFT_SQL_KEYS = (
    'tempo_min',
    'tempo_max',
    'energy_min',
    'energy_max',
    'moods',
    'key',
    'scale',
    'min_rating',
)


def _broaden_underfilled(
    filter_args: Dict,
    strict_result: Dict,
    ai_config: Dict,
    target_count: int,
    pool_target: int,
    log_messages: List[str],
) -> Dict:
    from tasks.ai.tools import execute_mcp_tool
    from tasks.ai.tool_impl import _fetch_pool_features

    strict_songs = strict_result.get('songs', []) or []
    if len(strict_songs) >= target_count:
        return strict_result
    dropped = [k for k in SOFT_SQL_KEYS if filter_args.get(k) not in (None, '', [])]
    if not dropped:
        return strict_result

    broad_args = {k: v for k, v in filter_args.items() if k not in SOFT_SQL_KEYS}
    broad_args['get_songs'] = pool_target
    if any(broad_args.get(k) for k in SCORED_FILTER_KEYS):
        broad_args['score_threshold'] = RELAX_THRESHOLD_STEPS[-1]
    broad = execute_mcp_tool('search_database', broad_args, ai_config)
    if 'error' in broad:
        return strict_result
    pool_songs = broad.get('songs', []) or []
    if len(pool_songs) <= len(strict_songs):
        return strict_result

    log_messages.append(
        f"   underfilled: hard cut left {len(strict_songs)} songs (target {target_count}) -> "
        f"re-queried without {dropped} ({len(pool_songs)}-song pool), "
        "applying them as a soft re-rank instead"
    )
    feats = _fetch_pool_features([s['item_id'] for s in pool_songs])
    final, _matched, _moved = _rerank_pool(pool_songs, filter_args, feats, log_messages)
    return {"songs": final, "message": broad.get('message', '')}


def _run_search_database_with_relax(
    filter_args: Dict,
    ai_config: Dict,
    target_count: int,
    log_messages: List[str],
    pool_target: Optional[int] = None,
) -> Dict:
    from tasks.ai.tools import execute_mcp_tool

    has_scored = any(filter_args.get(k) for k in SCORED_FILTER_KEYS)
    if not has_scored:
        result = execute_mcp_tool('search_database', filter_args, ai_config)
    else:
        result = None
        for step_threshold in RELAX_THRESHOLD_STEPS:
            args = dict(filter_args)
            args['score_threshold'] = step_threshold
            step_result = execute_mcp_tool('search_database', args, ai_config)
            if 'error' in step_result:
                return step_result
            songs = step_result.get('songs', [])
            log_messages.append(
                f"   relax: score_threshold={step_threshold} -> {len(songs)} songs"
            )
            result = step_result
            if len(songs) >= target_count:
                break
        if result is None:
            result = {"songs": [], "message": "relax loop produced no result"}
    if 'error' in result or not pool_target:
        return result
    return _broaden_underfilled(
        filter_args, result, ai_config, target_count, pool_target, log_messages
    )


def call_ai_for_plan(
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    from tasks.ai.api import call_with_tools as _call_with_tools

    return _call_with_tools(
        user_message=user_message,
        tools=tools,
        ai_config=ai_config,
        log_messages=log_messages,
        library_context=library_context,
    )


def plan_and_execute_once(
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str],
    *,
    library_context: Optional[Dict] = None,
    user_wants_rating: bool = False,
    collection_cap: int = 1000,
    target_song_count: int = 100,
    replan_feedback: Optional[str] = None,
):
    from tasks.ai.tools import execute_mcp_tool
    from tasks.ai.tool_impl import _fetch_pool_features

    log_messages.append("\n--- AI Decision ---")

    original_user_message = user_message
    log_messages.append(
        f"   tools offered: {', '.join(t.get('name', '') for t in tools)}"
    )

    hints = extract_hints(original_user_message)
    hints_block = format_hints_block(hints)
    if hints_block:
        for n in hints.get('notes', []):
            log_messages.append(f"   pre-extract: {n}")
        user_message = f"{user_message}\n\n{hints_block}"
    if replan_feedback:
        user_message = f"{user_message}\n\n{replan_feedback}"

    yield

    raw = call_ai_for_plan(user_message, tools, ai_config, log_messages, library_context)
    if 'error' in raw:
        return {"error": raw['error']}

    reasoning = raw.get('reasoning')
    if isinstance(reasoning, str) and reasoning.strip():
        log_messages.append(f"AI reasoning: {reasoning.strip()}")

    raw_calls = raw.get('tool_calls', []) or []
    log_messages.append(f"AI emitted {len(raw_calls)} tool call(s)")

    yield

    raw_calls = dedupe_and_cap_calls(raw_calls, log_messages=log_messages)
    raw_calls = validate_plan_args(
        raw_calls, user_wants_rating=user_wants_rating, log_messages=log_messages
    )

    if not raw_calls:
        return {"error": "No valid tool calls after validation"}

    plan = validate_and_normalize_plan(raw_calls)
    for note in plan.notes:
        log_messages.append(f"   plan: {note}")
    if not plan.primaries and plan.filter is None:
        return {"error": "Plan was empty after normalization"}

    for p in plan.primaries:
        if not isinstance(p, dict) or p.get('name') != 'text_match':
            continue
        pargs = p.get('arguments') or {}
        derived: Dict = {}
        ef = pargs.pop('energy_filter', None)
        if isinstance(ef, str) and ef.lower() in _ENERGY_BUCKET_RANGE:
            lo, hi = _ENERGY_BUCKET_RANGE[ef.lower()]
            if lo is not None:
                derived['energy_min'] = lo
            if hi is not None:
                derived['energy_max'] = hi
        tf = pargs.pop('tempo_filter', None)
        if isinstance(tf, str) and tf.lower() in _TEMPO_BUCKET_RANGE:
            lo, hi = _TEMPO_BUCKET_RANGE[tf.lower()]
            if lo is not None:
                derived['tempo_min'] = lo
            if hi is not None:
                derived['tempo_max'] = hi
        if derived:
            plan.filter = _merge_filter(plan.filter, derived)
            log_messages.append(
                f"   text_match audio buckets -> filter {derived} (handled by the shared soft re-rank)"
            )

    has_knowledge = any(
        isinstance(p, dict) and p.get('name') == 'knowledge_lookup' for p in plan.primaries
    )

    _strip_unrequested_filter_args(plan, hints, original_user_message, log_messages)
    if not has_knowledge:
        _apply_hint_backstop(plan, hints, log_messages)

    for u in hints.get('unsupported', []):
        plan.notes.append(u)

    if plan.filter is not None and has_knowledge:
        dropped_filter = {
            k: v for k, v in plan.filter.items() if k not in ('candidate_item_ids', 'get_songs')
        }
        log_messages.append(
            f"   AI brainstorming: filter {dropped_filter} NOT applied - "
            "knowledge results returned as-is"
        )
        plan.notes.append(
            f"constraints {dropped_filter} were IGNORED: brainstorm (knowledge_lookup) "
            "results are always returned as-is"
        )
        plan.filter = None

    detected_min_rating: Optional[int] = None
    if plan.filter and plan.filter.get('min_rating'):
        try:
            detected_min_rating = int(plan.filter['min_rating'])
        except (TypeError, ValueError):
            pass

    all_songs: List[Dict] = []
    ids_seen: set = set()
    keys_seen: set = set()
    song_sources: Dict[str, int] = {}
    tools_used_history: List[Dict] = []
    tool_execution_summary: List[str] = []
    tool_call_counter = 0

    def _add_songs(songs, call_index):
        added = 0
        for s in songs or []:
            iid = s.get('item_id')
            if not iid or iid in ids_seen:
                continue
            key = (s.get('title', '').strip().lower(), s.get('artist', '').strip().lower())
            if key in keys_seen:
                continue
            all_songs.append(s)
            ids_seen.add(iid)
            keys_seen.add(key)
            song_sources[iid] = call_index
            added += 1
            if len(all_songs) >= collection_cap:
                break
        return added

    def _summary(name: str, args: Dict, n_added: int) -> str:
        parts = []
        if name == 'search_database':
            for k in ('artist', 'album'):
                v = args.get(k)
                if v:
                    parts.append(f"{k}='{v}'")
            for k in ('genres', 'voices', 'moods', 'exclude_artists', 'exclude_genres'):
                v = args.get(k)
                if v:
                    parts.append(f"{k}={v}")
            for k in ('scale', 'key', 'min_rating'):
                v = args.get(k)
                if v:
                    parts.append(f"{k}={v}")
            if args.get('tempo_min') is not None or args.get('tempo_max') is not None:
                parts.append(f"tempo={args.get('tempo_min', '')}..{args.get('tempo_max', '')}")
            if args.get('energy_min') is not None or args.get('energy_max') is not None:
                parts.append(f"energy={args.get('energy_min', '')}..{args.get('energy_max', '')}")
            if args.get('year_min') is not None or args.get('year_max') is not None:
                parts.append(f"year={args.get('year_min', '')}..{args.get('year_max', '')}")
        elif name == 'seed_search':
            seeds = args.get('seeds') or []
            blend = args.get('blend_mode', 'union')
            seed_summary = []
            for s in seeds[:4]:
                if isinstance(s, dict):
                    if s.get('type') == 'song':
                        seed_summary.append(f"song:'{s.get('title', '')}'")
                    elif s.get('type') == 'artist':
                        seed_summary.append(f"artist:'{s.get('name', '')}'")
            if seed_summary:
                parts.append(f"seeds=[{', '.join(seed_summary)}]")
            if blend and blend != 'union':
                parts.append(f"blend={blend}")
        elif name == 'text_match':
            q = (args.get('query') or '')[:40]
            mode = args.get('mode', 'audio')
            if q:
                parts.append(f"query='{q}'")
            parts.append(f"mode={mode}")
        elif name == 'knowledge_lookup':
            req = (args.get('user_request') or '')[:35]
            if req:
                parts.append(f"req='{req}...'")
        body = ", ".join(parts) if parts else ""
        return f"{name}({body}, +{n_added})" if body else f"{name}(+{n_added})"

    if plan.primaries and plan.filter:
        pool_target = COMPOSITION_POOL_TARGET
        db_total = library_context.get('total_songs', 0) if library_context else 0
        if db_total and db_total < pool_target:
            pool_target = db_total
        log_messages.append(
            f"\n--- Composition: {len(plan.primaries)} primary call(s) + 1 filter "
            f"(re-rank pool target {pool_target}) ---"
        )
        pool_songs: List[Dict] = []
        pool_ids: set = set()
        sim_by_id: Dict[str, float] = {}
        primary_logs: List[tuple] = []
        for tc in plan.primaries:
            tn = tc.get('name')
            ta = dict(tc.get('arguments', {}) or {})
            ta['get_songs'] = pool_target
            pretty = {k: v for k, v in ta.items() if k != 'get_songs'}
            log_messages.append(f"\nPRIMARY: {tn}")
            try:
                log_messages.append(f"   Arguments: {json.dumps(pretty, indent=2, default=str)}")
            except TypeError:
                log_messages.append(f"   Arguments: {pretty}")
            res = execute_mcp_tool(tn, ta, ai_config)
            if 'error' in res:
                log_messages.append(f"   error {tn}: {res['error']}")
                primary_logs.append((tn, ta, 0, True, res.get('error', '')))
                continue
            songs = res.get('songs', [])
            if res.get('message'):
                for line in res['message'].split('\n'):
                    if line.strip():
                        log_messages.append(f"   {line}")
            added_to_pool = 0
            n_songs = len(songs)
            for idx, s in enumerate(songs):
                iid = s.get('item_id')
                if not iid:
                    continue
                sim_by_id[iid] = sim_by_id.get(iid, 0.0) + (1.0 - idx / n_songs)
                if iid not in pool_ids:
                    pool_songs.append(s)
                    pool_ids.add(iid)
                    added_to_pool += 1
            log_messages.append(
                f"   pooled {added_to_pool}/{len(songs)} unique (pool={len(pool_songs)})"
            )
            primary_logs.append((tn, ta, added_to_pool, False, res.get('message', '')))
            yield

        feats = _fetch_pool_features([s['item_id'] for s in pool_songs])
        pool_songs = _apply_exclusions(pool_songs, plan.filter, feats, log_messages)
        yield

        for tn, ta, pooled, errored, msg in primary_logs:
            tools_used_history.append(
                {
                    'name': tn,
                    'args': ta,
                    'songs': pooled if pool_songs else 0,
                    'error': errored,
                    'call_index': tool_call_counter,
                    'result_message': msg,
                }
            )
            tool_execution_summary.append(_summary(tn, ta, pooled if pool_songs else 0))
            tool_call_counter += 1

        if pool_songs:
            N = len(pool_songs)
            final, matched, _moved = _rerank_pool(
                pool_songs, plan.filter, feats, log_messages, sim_by_id=sim_by_id
            )
            yield

            filter_call_index = tool_call_counter
            added = _add_songs(final, filter_call_index)
            tools_used_history.append(
                {
                    'name': 'search_database',
                    'args': dict(plan.filter),
                    'songs': added,
                    'call_index': filter_call_index,
                    'result_message': f"priority re-rank: {matched}/{N} matched filter",
                }
            )
            tool_execution_summary.append(_summary('search_database', plan.filter, added))
            tool_call_counter += 1
        else:
            log_messages.append("   composition pool empty (no songs, or all excluded)")

    else:
        all_calls: List[Dict] = list(plan.primaries)
        if plan.filter is not None:
            all_calls.append({'name': 'search_database', 'arguments': dict(plan.filter)})
        primary_hits: Dict[str, int] = {}
        primary_rank: Dict[str, float] = {}
        n_primaries_with_songs = 0
        for tc in all_calls:
            tn = tc.get('name')
            ta = dict(tc.get('arguments', {}) or {})
            if 'get_songs' not in ta:
                ta['get_songs'] = 200
            pretty = {k: v for k, v in ta.items() if k != 'get_songs'}
            log_messages.append(f"\nTOOL: {tn}")
            try:
                log_messages.append(f"   Arguments: {json.dumps(pretty, indent=2, default=str)}")
            except TypeError:
                log_messages.append(f"   Arguments: {pretty}")
            if tn == 'search_database':
                relax_pool_target = COMPOSITION_POOL_TARGET
                db_total = library_context.get('total_songs', 0) if library_context else 0
                if db_total and db_total < relax_pool_target:
                    relax_pool_target = db_total
                res = _run_search_database_with_relax(
                    ta, ai_config, target_song_count, log_messages,
                    pool_target=relax_pool_target,
                )
            else:
                res = execute_mcp_tool(tn, ta, ai_config)
            if 'error' in res:
                log_messages.append(f"   error: {res['error']}")
                tools_used_history.append(
                    {
                        'name': tn,
                        'args': ta,
                        'songs': 0,
                        'error': True,
                        'call_index': tool_call_counter,
                        'result_message': res.get('error', ''),
                    }
                )
                tool_execution_summary.append(_summary(tn, ta, 0))
                tool_call_counter += 1
                continue
            songs = res.get('songs', [])
            if res.get('message'):
                for line in res['message'].split('\n'):
                    if line.strip():
                        log_messages.append(f"   {line}")
            if tn != 'search_database' and songs:
                n_primaries_with_songs += 1
                n_songs = len(songs)
                for idx, s in enumerate(songs):
                    iid = s.get('item_id')
                    if not iid:
                        continue
                    primary_hits[iid] = primary_hits.get(iid, 0) + 1
                    primary_rank[iid] = primary_rank.get(iid, 0.0) + (1.0 - idx / n_songs)
            added = _add_songs(songs, tool_call_counter)
            log_messages.append(f"   retrieved {len(songs)} songs, added {added} new")
            tools_used_history.append(
                {
                    'name': tn,
                    'args': ta,
                    'songs': added,
                    'call_index': tool_call_counter,
                    'result_message': res.get('message', ''),
                }
            )
            tool_execution_summary.append(_summary(tn, ta, added))
            tool_call_counter += 1
            yield
            if len(all_songs) >= collection_cap:
                log_messages.append(f"collection cap {collection_cap} reached, stopping")
                break

        if n_primaries_with_songs >= 2 and not has_knowledge:
            boosted = sum(1 for c in primary_hits.values() if c > 1)
            if boosted:
                all_songs.sort(
                    key=lambda s: (
                        -primary_hits.get(s['item_id'], 0),
                        -primary_rank.get(s['item_id'], 0.0),
                    )
                )
                log_messages.append(
                    f"   intersection boost: {boosted} songs returned by MULTIPLE finder "
                    "tools moved to the front (likely what the user meant by combining them)"
                )

    if not all_songs and replan_feedback is None:
        detail_lines: List[str] = []
        for h in tools_used_history[-4:]:
            msg_lines = [ln for ln in (h.get('result_message') or '').splitlines() if ln.strip()]
            if msg_lines:
                detail_lines.append(f"- {h.get('name')}: {msg_lines[-1][:160]}")
        feedback = (
            "PREVIOUS ATTEMPT FAILED: every tool call returned 0 songs.\n"
            f"Calls tried: {' -> '.join(tool_execution_summary) or 'none'}\n"
            + ("\n".join(detail_lines) + "\n" if detail_lines else "")
            + "Make a DIFFERENT plan for the same request: relax or drop the least "
            "essential constraint, fix likely misspellings, or switch tool "
            "(seed_search for similar-artist requests, text_match for sound "
            "descriptions). Do not repeat the same calls."
        )
        log_messages.append("\nZERO RESULTS -> replanning once with failure feedback")
        replan = yield from plan_and_execute_once(
            original_user_message,
            tools,
            ai_config,
            log_messages,
            library_context=library_context,
            user_wants_rating=user_wants_rating,
            collection_cap=collection_cap,
            target_song_count=target_song_count,
            replan_feedback=feedback,
        )
        if isinstance(replan, dict) and 'error' not in replan:
            return replan
        log_messages.append("   replan attempt failed; returning the original empty result")

    return {
        "songs": all_songs,
        "song_sources": song_sources,
        "tools_used_history": tools_used_history,
        "tool_execution_summary": tool_execution_summary,
        "detected_min_rating": detected_min_rating,
        "plan_notes": plan.notes,
        "executed_query_str": f"MCP single-pass ({len(tools_used_history)} tools): {' -> '.join(tool_execution_summary)}",
        "filter_applied": plan.filter is not None,
    }
