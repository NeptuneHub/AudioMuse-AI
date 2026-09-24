# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Single-call plan builder and executor for AI playlist requests.

Orchestrates the pipeline between ``api``/``prompts`` (LLM calls), ``vocab``
(label normalization) and ``tools`` (execution): one tool-calling request over
the full tool surface emits the plan, which is then validated, deduplicated,
merged, run and composed.

Main Features:
* Regex hints (multilingual negation and decades): years, relative eras, BPM bounds, tempo/energy/activity words, key/scale, track length, time budget, song count, per-artist cap, excluded versions, recently added, genres, sound words. Missing hints are merged back (backstop); relative eras and instrumental wording override the model; hallucinated year/instrumental/exclusion args are stripped (exclusions need a negation cue).
* Deterministic repair of what small models get wrong: repeated array values collapsed (Ollama ignores uniqueItems); contradicting exclusions dropped (seed/filter artist, a requested genre, a name absent from the request); point ranges widened; a named album looked up; min_rating dropped in an unrated library; a journey cue fixes blend_mode; "like X but calmer/faster" is relative to the seeds; sound words add an audio text_match to a filter-only plan; a genre-filtered pool short of full matches is backfilled; filter-only genre results lead with songs whose main style is that genre.
* Soft categorical-priority re-rank (rerank.py); exclude_artists/exclude_genres are the one HARD cut (reverted only if they empty the pool); excluded versions go by title; multi-finder songs get an intersection boost; a journey runs alone, in order.
* Named instruments (DCLAP SAE concepts) always get a sound search steered x3 toward them; each candidate is SAE-checked, and a pool short of songs that carry them is topped up by an x10 instrument-led search. A text_match query copying a prompt example becomes the request; a negation-only one is dropped.
* knowledge_lookup output is never post-filtered: the parsed filter is injected INTO the tool call (grounded recipe, gated channels) and the planner filter is cleared.
* A request always yields a plan that finds songs: an empty plan or a zero-result run replans ONCE with feedback, then falls back to a direct match of the user's words (also when the provider is unreachable); a short filter pool relaxes score_threshold, then re-runs without its soft dims and re-ranks by them.
"""

import datetime
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config

from tasks.ai.vocab import (
    ACTIVITY_TEMPO,
    ALIAS_ENERGY,
    ALIAS_GENRE,
    ALIAS_TEMPO,
    GENRE_VOCAB,
    INSTRUMENT_WORDS,
    MOOD_WORDS,
    OUT_OF_VOCAB_GENRES,
    SOUND_DESCRIPTORS,
    normalize_genre_list,
    normalize_mood_list,
    normalize_scale,
    normalize_voices_list,
)

from .prompts import EXAMPLE_TEXT_QUERIES
from .rerank import (
    EXACT_TEMPO_KEY,
    INSTRUMENT_HIT,
    _NON_SONG_TITLE_RE,
    _parse_tag_scores,
    _short_track_floor,
    count_full_matches,
    genre_style_rank,
    rerank,
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

_ENERGY_BUCKET_RANGE = {'low': (0.0, 0.33), 'medium': (0.33, 0.66), 'high': (0.66, 1.0)}
_TEMPO_BUCKET_RANGE = {'slow': (None, 90), 'medium': (90, 140), 'fast': (140, None)}

COMPOSITION_POOL_TARGET = 10000


FILTER_LIST_KEYS = (
    'genres',
    'voices',
    'moods',
    'other_features',
    'exclude_artists',
    'exclude_genres',
    'instruments',
)
FILTER_MIN_KEYS = ('tempo_min', 'energy_min', 'year_min', 'min_rating', 'duration_min')
FILTER_MAX_KEYS = ('tempo_max', 'energy_max', 'year_max', 'duration_max', 'added_within_days')
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


def _apply_exclusions(
    pool_songs: List[Dict],
    filt: Dict,
    feats: Dict,
    log_messages: List[str],
    notes: Optional[List[str]] = None,
):
    ex_artists = [a for a in (filt.get('exclude_artists') or []) if isinstance(a, str) and a.strip()]
    ex_genres = [g for g in (filt.get('exclude_genres') or []) if isinstance(g, str) and g.strip()]
    if not ex_artists and not ex_genres:
        return pool_songs

    from tasks.ai.tool_impl import _normalize_for_match

    ex_artist_norms = {_normalize_for_match(a) for a in ex_artists}
    ex_genre_lows = [g.strip().lower() for g in ex_genres]

    kept = [s for s in pool_songs if not _song_is_excluded(s, feats, ex_artist_norms, ex_genre_lows)]
    removed = len(pool_songs) - len(kept)

    if pool_songs and not kept:
        log_messages.append(
            f"   exclusions removed every one of the {len(pool_songs)} candidates; "
            "kept the pool and recorded the conflict instead"
        )
        if notes is not None:
            notes.append(
                f"exclusions (exclude_artists={ex_artists or '-'}, "
                f"exclude_genres={ex_genres or '-'}) matched the whole pool and were "
                "not applied; they most likely did not mean what the request said"
            )
        return pool_songs

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


_DECADE_NUM = r"((?:19|20)?(?:[3-9]0|00|10|20))"
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_DECADE_RE = re.compile(rf"\b{_DECADE_NUM}'?s\b", re.IGNORECASE)
_DECADE_MOD_RE = re.compile(rf"\b(early|mid|late)[\s-]*{_DECADE_NUM}'?s\b", re.IGNORECASE)
_DECADE_MOD_SPAN = {'early': (0, 4), 'mid': (3, 6), 'late': (5, 9)}
_DECADE_WORDS = {
    'thirties': 1930, 'forties': 1940, 'fifties': 1950, 'sixties': 1960,
    'seventies': 1970, 'eighties': 1980, 'nineties': 1990, 'noughties': 2000,
}
_DECADE_WORD_RE = re.compile(r"\b(" + "|".join(_DECADE_WORDS) + r")\b", re.IGNORECASE)
_DECADE_FOREIGN_RE = re.compile(
    r"\b(?:anni|a[\u00f1n]os|ann[\u00e9e]es|jaren|d[\u00e9e]cada)\s+(?:de\s+|dos\s+|'|\u2019)?"
    + _DECADE_NUM + r"\b|\b" + _DECADE_NUM + r"er(?:\s+jahre)?\b",
    re.IGNORECASE,
)
_LAST_N_YEARS_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten|"
    r"fifteen|twenty|few|couple\s+of)\s+years\b",
    re.IGNORECASE,
)
_THIS_YEAR_RE = re.compile(r"\bthis\s+year(?:'s)?\b", re.IGNORECASE)
_LAST_YEAR_RE = re.compile(r"\blast\s+year(?:'s)?\b", re.IGNORECASE)
_RECENT_RE = re.compile(
    r"\b(?:recent|latest|newest|brand[\s-]new)\s+(?:songs?|tracks?|music|releases?|hits?|"
    r"stuff|albums?)\b|\bnew\s+releases?\b|\breleased\s+recently\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    'one': 1, 'a': 1, 'an': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'fifteen': 15, 'twenty': 20,
    'few': 3, 'couple of': 2, 'half an': 0.5, 'half a': 0.5,
}
_BPM_RANGE_RE = re.compile(
    r"\b(\d{2,3})\s*(?:-|to|and|\u2013)\s*(\d{2,3})\s*bpm\b", re.IGNORECASE
)
_BPM_MIN_RE = re.compile(
    r"(?:\b(?:over|above|faster\s+than|at\s+least|more\s+than|min(?:imum)?)\s*|>=?\s*)"
    r"(\d{2,3})\s*\+?\s*bpm\b|\b(\d{2,3})\s*\+\s*bpm\b|"
    r"\b(\d{2,3})\s*bpm\s+(?:or\s+(?:more|faster|higher|above)|and\s+(?:up|above|faster))\b",
    re.IGNORECASE,
)
_BPM_MAX_RE = re.compile(
    r"(?:\b(?:under|below|slower\s+than|at\s+most|less\s+than|max(?:imum)?)\s*|<=?\s*)"
    r"(\d{2,3})\s*bpm\b|\b(\d{2,3})\s*bpm\s+or\s+(?:less|slower|lower|below)\b",
    re.IGNORECASE,
)
_BPM_RE = re.compile(r"\b(\d{2,3})\s*bpm\b", re.IGNORECASE)
_ENERGY_NUM_RE = re.compile(
    r"\benergy\s*(?:above|>=?|over|min(?:imum)?)\s*([0-9]*\.[0-9]+|[0-9]+)\b", re.IGNORECASE
)
_ACCIDENTAL = r"(#|b|\u266f|\u266d|\s+sharp|\s+flat)?"
_KEY_SCALE_RE = re.compile(r"\b([A-G])" + _ACCIDENTAL + r"\s*(?i:(major|minor|maj|min))\b")
_KEY_OF_RE = re.compile(
    r"\bkey\s+of\s+([A-Ga-g])" + _ACCIDENTAL + r"(?:\s+(major|minor))?(?!\w)", re.IGNORECASE
)
_SCALE_WORD_RE = re.compile(
    r"\b(major|minor)[\s-]+(?:keys?|scales?|mode|tonality)\b|\bin\s+(?:a\s+)?(major|minor)\b",
    re.IGNORECASE,
)
_NO_VOCALS_RE = re.compile(
    r"\b(?:no|without|zero|sans|senza|sin|ohne|sem)\s+(?:vocals?|lyrics|singing|voices?|words|"
    r"voce|voci|voz|voces|voix|stimme|gesang|testo|paroles?|letras?)\b|"
    r"\bvocal[\s-]?free\b|\blyric[\s-]?less\b|\bwordless\b",
    re.IGNORECASE,
)
_INSTRUMENTAL_WORD_RE = re.compile(
    r"\b(?:instrumentals?|instrumentale[ns]?|instrumentales|strumental[ei]|instrumentalmusik)\b"
    r"(?!\s+(?:passages?|sections?|parts?|breaks?|bits?|solos?|interludes?|intros?|outros?|"
    r"moments?|jams?|stretches?|versions?)\b)",
    re.IGNORECASE,
)
_DURATION_UNIT = r"(\d{1,3}(?:[.,]\d+)?)\s*(minutes?|mins?|seconds?|secs?)\b"
_DURATION_MAX_RE = re.compile(
    r"\b(?:under|below|shorter\s+than|less\s+than|at\s+most|max(?:imum)?|up\s+to|"
    r"no\s+longer\s+than)\s+" + _DURATION_UNIT,
    re.IGNORECASE,
)
_DURATION_MIN_RE = re.compile(
    r"\b(?:over|above|longer\s+than|more\s+than|at\s+least|min(?:imum)?)\s+" + _DURATION_UNIT,
    re.IGNORECASE,
)
_SHORT_TRACKS_RE = re.compile(
    r"\b(?:short|quick|brief)\s+(?:songs?|tracks?|tunes?|ones)\b|\bshort\s+and\s+sweet\b",
    re.IGNORECASE,
)
_LONG_TRACKS_RE = re.compile(
    r"\b(?:long|lengthy|extended|longer)\s+(?:[a-z]+\s+)?(?:songs?|tracks?|tunes?|jams?|pieces?|ones)\b|"
    r"\blong[\s-]form\b",
    re.IGNORECASE,
)
_TOTAL_LENGTH_RE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d+)?|an?|one|two|three|four|five|half\s+an?)[\s-]*"
    r"(hours?|hrs?|minutes?|mins?)(?:\s+(?:of|long|worth)\b|"
    r"(?:[\s-]+[a-z]+)?\s+(?:playlist|mix|set|session)\b)",
    re.IGNORECASE,
)
_MAX_PER_ARTIST_RE = re.compile(
    r"\b(?:(?:max(?:imum)?|at\s+most|up\s+to|no\s+more\s+than|only|just)\s+)?"
    r"(\d|one|a\s+single|two|three|four|five)\s+(?:(?:songs?|tracks?)\s+)?"
    r"(?:per|from\s+each|by\s+each|for\s+each|of\s+each|each)\s+(?:artists?|bands?|singers?)\b",
    re.IGNORECASE,
)
_DISTINCT_ARTISTS_RE = re.compile(
    r"\b(?:all\s+|only\s+)?different\s+artists\b|\bno\s+(?:repeated|repeat|duplicate)\s+artists?\b|"
    r"\bevery\s+(?:song|track)\s+(?:by|from)\s+a\s+different\s+artist\b|"
    r"\bone[\s-]hit[\s-]wonders?\b",
    re.IGNORECASE,
)
_SONG_COUNT_RE = re.compile(
    r"\b(\d{1,3})\s+(?:(?!(?:hours?|hrs?|minutes?|mins?|seconds?|years?|per|from|by|of|bpm)\b)"
    r"[a-z-]+\s+){0,3}(?:songs?|tracks?|tunes)\b"
    r"(?!\s+(?:per|from\s+each|by\s+each|for\s+each|each)\b)|"
    r"\b(\d{1,3})[\s-](?:song|track)\s+(?:playlist|mix|list)\b",
    re.IGNORECASE,
)
_VERSION_PATTERNS = {
    'live': r"[\(\[\-\u2013]\s*live\b|\blive\s+(?:at|in|from|version|recording|session)\b",
    'remix': r"\bre-?mix(?:ed|es)?\b|\b(?:club|extended|dub|radio)\s+mix\b",
    'cover': r"\bcover(?:ed)?\b",
    'demo': r"\bdemo\b",
    'acoustic': r"[\(\[\-\u2013]\s*acoustic\b|\bacoustic\s+version\b",
    'karaoke': r"\bkaraoke\b",
    'remaster': r"\bremaster(?:ed)?\b",
    'edit': r"\b(?:radio|single)\s+edit\b",
    'instrumental version': r"[\(\[\-\u2013]\s*instrumental\b|\binstrumental\s+version\b",
}
_VERSION_WORDS = {
    'live': 'live', 'remix': 'remix', 'remixes': 'remix', 'cover': 'cover', 'covers': 'cover',
    'demo': 'demo', 'demos': 'demo', 'acoustic version': 'acoustic',
    'acoustic versions': 'acoustic', 'karaoke': 'karaoke', 'remaster': 'remaster',
    'remasters': 'remaster', 'remastered': 'remaster', 'edits': 'edit', 'radio edits': 'edit',
    'instrumental versions': 'instrumental version',
}
_VERSION_EXCLUDE_RE = re.compile(
    r"\b(?:no|without|exclude|excluding|skip|avoid|not)\s+(?:any\s+|the\s+)?"
    r"(" + "|".join(sorted((re.escape(w) for w in _VERSION_WORDS), key=len, reverse=True)) + r")"
    r"(?:\s+(?:versions?|tracks?|songs?|recordings?))?\b|"
    r"\b(studio|original)\s+(?:versions?|recordings?)\s+only\b|\bonly\s+(studio|original)\s+"
    r"(?:versions?|recordings?)\b",
    re.IGNORECASE,
)
_ADDED_RE = re.compile(
    r"\b(?:recently|newly|just|latest|last)\s+added\b|\badded\s+(?:recently|lately)\b|"
    r"\bnew(?:est)?\s+(?:additions?|arrivals?)\b|\badded\s+(?:to\s+(?:my|the)\s+library\s+)?"
    r"(?:this|last|in\s+the\s+(?:last|past))\s+(\d{1,3}\s+days?|week|month|year)\b",
    re.IGNORECASE,
)
_ADDED_PERIOD_DAYS = {'week': 7, 'month': 31, 'year': 365}
_MORE_ENERGY_RE = re.compile(
    r"\b(?:more|mroe|moer|much\s+more|a\s+bit\s+more|even\s+more)\s+(?:upbeat|energetic|intense|aggressive|"
    r"lively|danceable|powerful|hype|punchy)\b|\b(?:harder|heavier|louder|livelier)\b|"
    r"\bless\s+(?:chill|calm|mellow|relaxed|soft|quiet)\b",
    re.IGNORECASE,
)
_LESS_ENERGY_RE = re.compile(
    r"\b(?:more|mroe|moer|much\s+more|a\s+bit\s+more|even\s+more)\s+(?:chill|calm|mellow|relaxed|relaxing|"
    r"laid[\s-]back|quiet|soft|gentle|peaceful)\b|\b(?:calmer|softer|quieter|mellower|chiller|gentler)\b|"
    r"\bless\s+(?:intense|energetic|aggressive|upbeat|loud|heavy)\b",
    re.IGNORECASE,
)
_FASTER_RE = re.compile(r"\b(?:faster|quicker|more\s+uptempo|higher\s+tempo)\b", re.IGNORECASE)
_SLOWER_RE = re.compile(r"\b(?:slower|lower\s+tempo|less\s+fast)\b", re.IGNORECASE)
RELATIVE_ENERGY_STEP = 0.15
RELATIVE_TEMPO_STEP = 10.0
RECENTLY_ADDED_DAYS = 30
_JOURNEY_WORD_RE = re.compile(r"\b(?:journey|transition(?:s|ing)?)\b", re.IGNORECASE)
_FROM_TO_RE = re.compile(r"\bfrom\b(.+?)\bto\b(.+)", re.IGNORECASE)
_NEGATION_WORDS = (
    r"no|not|without|except|excluding|exclude|avoid|nothing|never|zero|skip|hates?|dislikes?|"
    r"niente|nessun[oa]?|senza|non(?!-)|tranne|evita(?:re)?|sin|nada|ning[u\u00fa]n[oa]?|"
    r"excepto|sans|aucune?|ohne|keine?[nrs]?|nicht|au(?:ss|\u00df)er|sem|nenhuma?|geen|zonder"
)
_NEGATION_TAIL_RE = re.compile(
    rf"(?:\b(?:{_NEGATION_WORDS})\b|\bpas\s+de\b)[^,.;!?]*$",
    re.IGNORECASE,
)
_NEGATION_CUE_RE = re.compile(
    rf"\b(?:{_NEGATION_WORDS})\b|anything but|\bpas\s+de\b",
    re.IGNORECASE,
)
_NEGATION_NEAR_RE = re.compile(rf"(?:\b(?:{_NEGATION_WORDS})\b|\bpas\s+de\b)\W*(?:\w+\W+)?$", re.IGNORECASE)
_YEARISH_RE = re.compile(
    r'\b(?:recent|latest|new(?:est)?|modern|current|today|old(?:er)?|oldies|classic|'
    r'vintage|early|late|decades?|years?|era)\b',
    re.IGNORECASE,
)
_VOCALNESS_RE = re.compile(
    r'\b(?:instrumentals?|vocals?|vocalists?|voices?|singing|singers?|sung|lyrics|words|'
    r'karaoke|acapella|a\s+cappella|instrumentale[ns]?|instrumentales|strumental[ei]|'
    r'voce|voci|voz|voces|voix|paroles?|letras?|testo|testi|gesang|stimme|cantad[oa]s?|'
    r'cantat[oa]|chant[\u00e9e]e?s?|gesungen)\b',
    re.IGNORECASE,
)
_LENGTHISH_RE = re.compile(
    r'\b(?:minutes?|mins?|seconds?|secs?|hours?|long|longer|short|shorter|length|lengthy|'
    r'duration|quick|brief|extended|epic)\b',
    re.IGNORECASE,
)
_ADDED_WORDING_RE = re.compile(r'\b(?:added|additions?|arrivals?|imported)\b', re.IGNORECASE)
_GENRE_HINT_SKIP = {'house', 'dance'}
_MUSIC_NOUN_SUFFIX = r"(?:musik|\s+(?:music|songs?|tracks?|hits?|pop|party|anthems?|floor|beats?|mix))?"


def _genre_hint_tokens() -> List[str]:
    tokens = {g.lower() for g in GENRE_VOCAB} | set(ALIAS_GENRE.keys())
    tokens -= _GENRE_HINT_SKIP
    return sorted(tokens, key=len, reverse=True)


def _is_compound_part(text: str, start: int, end: int, token: str) -> bool:
    if '-' in token:
        return False
    before = text[start - 1] if start > 0 else ''
    after = text[end] if end < len(text) else ''
    return before == '-' or after == '-'


def _extract_genre_hints(text: str) -> Dict[str, List[str]]:
    masked = text.lower()
    positive_raw: List[str] = []
    negative_raw: List[str] = []
    for token in _genre_hint_tokens():
        suffix = _MUSIC_NOUN_SUFFIX if ' ' not in token else ''
        pattern = re.compile(rf"\b{re.escape(token)}{suffix}\b")
        pos = 0
        while True:
            m = pattern.search(masked, pos)
            if not m:
                break
            window = masked[max(0, m.start() - 40):m.start()]
            if _is_compound_part(masked, m.start(), m.end(), token):
                pass
            elif _NEGATION_TAIL_RE.search(window):
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


def _word_number(raw: str) -> Optional[float]:
    raw = re.sub(r"\s+", " ", (raw or '').strip().lower().replace(',', '.'))
    if raw in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[raw])
    try:
        return float(raw)
    except ValueError:
        return None


def _decade_ranges(text: str) -> List[tuple]:
    ranges: List[tuple] = []
    masked = text
    for m in _DECADE_MOD_RE.finditer(text):
        start = _normalize_decade(m.group(2))
        lo, hi = _DECADE_MOD_SPAN[m.group(1).lower()]
        ranges.append((start + lo, start + hi))
        masked = masked[:m.start()] + ' ' * (m.end() - m.start()) + masked[m.end():]
    for d in _DECADE_RE.findall(masked):
        start = _normalize_decade(d)
        ranges.append((start, start + 9))
    for w in _DECADE_WORD_RE.findall(masked):
        start = _DECADE_WORDS[w.lower()]
        ranges.append((start, start + 9))
    for a, b in _DECADE_FOREIGN_RE.findall(masked):
        start = _normalize_decade(a or b)
        ranges.append((start, start + 9))
    return ranges


def _relative_year_range(text: str, today_year: int, library_year_max: Optional[int]) -> Optional[tuple]:
    m = _LAST_N_YEARS_RE.search(text)
    if m:
        n = _word_number(m.group(1))
        if n and n >= 1:
            return today_year - int(n) + 1, today_year
    if _THIS_YEAR_RE.search(text):
        return today_year, today_year
    if _LAST_YEAR_RE.search(text):
        return today_year - 1, today_year - 1
    if _RECENT_RE.search(text):
        ref = min(today_year, library_year_max) if library_year_max else today_year
        return ref - 2, ref
    return None


def _year_hints(text: str, hints: Dict, notes: List[str], library_year_max: Optional[int]) -> None:
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if years:
        hints['years'] = years
        hints['year_min'] = min(years)
        hints['year_max'] = max(years)
        notes.append(f"years detected: {years}")

    decades = _decade_ranges(text)
    if decades:
        hints['year_min'] = min([lo for lo, _hi in decades] + ([hints['year_min']] if years else []))
        hints['year_max'] = max([hi for _lo, hi in decades] + ([hints['year_max']] if years else []))
        notes.append(f"decade(s) detected: {[f'{lo}-{hi}' for lo, hi in decades]}")

    if years or decades:
        return
    relative = _relative_year_range(text, datetime.date.today().year, library_year_max)
    if relative:
        hints['year_min'], hints['year_max'] = relative
        hints['year_relative'] = True
        notes.append(f"relative era detected: {relative[0]}-{relative[1]}")


_SOFT_TEMPO_PHRASES = {'upbeat', 'uptempo'}


def _tempo_hints(text: str, hints: Dict, notes: List[str]) -> None:
    m = _BPM_RANGE_RE.search(text)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        hints['tempo_min'], hints['tempo_max'] = float(lo), float(hi)
        hints['tempo_explicit'] = True
        notes.append(f"BPM range detected: {lo}-{hi}")
        return
    m_min = _BPM_MIN_RE.search(text)
    m_max = _BPM_MAX_RE.search(text)
    if m_min or m_max:
        if m_min:
            hints['tempo_min'] = float(next(g for g in m_min.groups() if g))
        if m_max:
            hints['tempo_max'] = float(next(g for g in m_max.groups() if g))
        hints['tempo_explicit'] = True
        notes.append(f"BPM bound detected: {hints.get('tempo_min', '?')}..{hints.get('tempo_max', '?')}")
        return
    bpm_match = _BPM_RE.search(text)
    if bpm_match:
        bpm = int(bpm_match.group(1))
        hints['bpm'] = bpm
        hints['tempo_explicit'] = True
        notes.append(f"BPM detected: {bpm}")
        return

    low = text.lower()
    for table in (ACTIVITY_TEMPO, ALIAS_TEMPO):
        for phrase, (tmin, tmax) in table.items():
            if re.search(rf"\b{re.escape(phrase)}\b", low):
                hints['tempo_min'], hints['tempo_max'] = tmin, tmax
                if table is ALIAS_TEMPO and phrase not in _SOFT_TEMPO_PHRASES:
                    hints['tempo_explicit'] = True
                notes.append(f"tempo phrase '{phrase}' -> {tmin}-{tmax} BPM")
                return


def _key_hints(text: str, hints: Dict, notes: List[str]) -> None:
    for m in _KEY_SCALE_RE.finditer(text):
        note, accidental, mode = m.group(1), (m.group(2) or '').strip().lower(), m.group(3).lower()
        preceding = text[:m.start()].rstrip().lower()
        if note == 'A' and not accidental and not re.search(r"\b(?:in|of)$", preceding):
            continue
        accidental = {'sharp': '#', '\u266f': '#', 'flat': 'b', '\u266d': 'b'}.get(accidental, accidental)
        hints['key'] = note + accidental
        hints['scale'] = 'major' if mode.startswith('maj') else 'minor'
        notes.append(f"key detected: {hints['key']} {hints['scale']}")
        return
    m = _KEY_OF_RE.search(text)
    if m:
        accidental = (m.group(2) or '').strip().lower()
        accidental = {'sharp': '#', '\u266f': '#', 'flat': 'b', '\u266d': 'b'}.get(accidental, accidental)
        hints['key'] = m.group(1).upper() + accidental
        if m.group(3):
            hints['scale'] = m.group(3).lower()
        notes.append(f"key detected: {hints['key']} {hints.get('scale', '')}".rstrip())
        return
    m = _SCALE_WORD_RE.search(text)
    if m:
        hints['scale'] = (m.group(1) or m.group(2)).lower()
        notes.append(f"scale detected: {hints['scale']}")


def _instrumental_hint(text: str, hints: Dict, notes: List[str]) -> None:
    if _NO_VOCALS_RE.search(text):
        hints['instrumental'] = True
        notes.append("instrumental requested")
        return
    m = _INSTRUMENTAL_WORD_RE.search(text)
    if not m:
        return
    if _NEGATION_NEAR_RE.search(text[max(0, m.start() - 25):m.start()]):
        hints['instrumental'] = False
        notes.append("instrumental EXCLUDED (vocal tracks only)")
    else:
        hints['instrumental'] = True
        notes.append("instrumental requested")


def _minutes_to_seconds(value: str, unit: str) -> Optional[float]:
    number = _word_number(value)
    if number is None:
        return None
    return number if unit.lower().startswith('s') else number * 60.0


_TOTAL_LENGTH_GUARD_RE = re.compile(
    r"\b(?:under|below|over|above|less\s+than|more\s+than|at\s+least|at\s+most|shorter\s+than|"
    r"longer\s+than|max(?:imum)?|min(?:imum)?|up\s+to)\s*$",
    re.IGNORECASE,
)
TRACK_DURATION_LIMIT_SECONDS = 15 * 60
SHORT_TRACK_SECONDS = 180.0
LONG_TRACK_SECONDS = 360.0


def _duration_hints(text: str, hints: Dict, notes: List[str]) -> None:
    for regex, key in ((_DURATION_MAX_RE, 'duration_max'), (_DURATION_MIN_RE, 'duration_min')):
        m = regex.search(text)
        if m:
            seconds = _minutes_to_seconds(m.group(1), m.group(2))
            if seconds and seconds < TRACK_DURATION_LIMIT_SECONDS:
                hints[key] = seconds
    if hints.get('duration_max') is None and hints.get('duration_min') is None:
        if _SHORT_TRACKS_RE.search(text):
            hints['duration_max'] = SHORT_TRACK_SECONDS
        elif _LONG_TRACKS_RE.search(text):
            hints['duration_min'] = LONG_TRACK_SECONDS
    if hints.get('duration_min') is not None or hints.get('duration_max') is not None:
        notes.append(
            f"track length detected: {hints.get('duration_min', '?')}..{hints.get('duration_max', '?')} s"
        )

    for m in _TOTAL_LENGTH_RE.finditer(text):
        if _TOTAL_LENGTH_GUARD_RE.search(text[:m.start()]):
            continue
        number = _word_number(m.group(1))
        if number is None:
            continue
        seconds = number * (3600.0 if m.group(2).lower().startswith('h') else 60.0)
        if seconds >= 10 * 60:
            hints['total_seconds'] = seconds
            notes.append(f"playlist length detected: {int(seconds // 60)} minutes")
            return


def _playlist_shape_hints(text: str, hints: Dict, notes: List[str]) -> None:
    m = _MAX_PER_ARTIST_RE.search(text)
    if m:
        n = _word_number(m.group(1).replace('a single', 'one'))
        if n and n >= 1:
            hints['max_per_artist'] = int(n)
    elif _DISTINCT_ARTISTS_RE.search(text):
        hints['max_per_artist'] = 1
    if hints.get('max_per_artist'):
        notes.append(f"at most {hints['max_per_artist']} song(s) per artist")

    for m in _SONG_COUNT_RE.finditer(text):
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= config.INSTANT_PLAYLIST_MAX_N_RESULTS:
            hints['song_count'] = n
            notes.append(f"song count detected: {n}")
            break

    added = _ADDED_RE.search(text)
    if added:
        period = (added.group(1) or '').lower()
        digits = re.match(r"(\d+)", period)
        hints['added_within_days'] = (
            int(digits.group(1)) if digits else _ADDED_PERIOD_DAYS.get(period, RECENTLY_ADDED_DAYS)
        )
        notes.append(f"recently added: last {hints['added_within_days']} days")

    versions: List[str] = []
    for m in _VERSION_EXCLUDE_RE.finditer(text):
        if m.group(1):
            versions.append(_VERSION_WORDS[m.group(1).lower()])
        elif (m.group(2) or m.group(3) or '').lower() == 'studio':
            versions.extend(['live', 'demo'])
        else:
            versions.extend(['cover', 'remix', 'live', 'karaoke'])
    if versions:
        hints['exclude_versions'] = list(dict.fromkeys(versions))
        notes.append(f"excluded versions: {hints['exclude_versions']}")


def _fold_accents(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def _instrument_hints(text: str, hints: Dict, notes: List[str]) -> None:
    masked = ' ' + re.sub(r"[^a-z0-9 ]+", ' ', _fold_accents(text).lower()) + ' '
    canonical: List[str] = []
    surface: List[str] = []
    for word in sorted(INSTRUMENT_WORDS, key=len, reverse=True):
        m = re.search(rf"(?<=\s){re.escape(word)}(?=\s)", masked)
        if not m:
            continue
        if not _NEGATION_NEAR_RE.search(masked[max(0, m.start() - 25):m.start()]):
            if INSTRUMENT_WORDS[word]:
                canonical.append(INSTRUMENT_WORDS[word])
            surface.append(word)
        masked = masked[:m.start()] + ' ' * (m.end() - m.start()) + masked[m.end():]
    if surface:
        order = {w: text.lower().find(w) for w in surface}
        surface.sort(key=lambda w: order[w])
        hints['instrument_words'] = surface
        notes.append(f"instrument(s) detected: {surface}")
    if canonical:
        hints['instruments'] = list(dict.fromkeys(canonical))


def _mood_hints(text: str) -> List[str]:
    folded = _fold_accents(text).lower()
    moods: List[str] = []
    for mood, words in MOOD_WORDS.items():
        for word in words:
            m = re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", folded)
            if m and not _NEGATION_NEAR_RE.search(folded[max(0, m.start() - 25):m.start()]):
                moods.append(mood)
                break
    return moods


def _sound_words(text: str) -> List[str]:
    text = _fold_accents(text)
    masked = text.lower()
    found: List[str] = []
    for phrase in sorted(OUT_OF_VOCAB_GENRES + SOUND_DESCRIPTORS, key=len, reverse=True):
        m = re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", masked)
        if m:
            if not _NEGATION_TAIL_RE.search(masked[max(0, m.start() - 40):m.start()]):
                found.append(phrase)
            masked = masked[:m.start()] + ' ' * (m.end() - m.start()) + masked[m.end():]
    order = {w: text.lower().find(w) for w in found}
    return sorted(found, key=lambda w: order[w])


def extract_hints(text: str, library_year_max: Optional[int] = None) -> Dict:
    if not text or not isinstance(text, str):
        return {}

    hints: Dict = {}
    notes: List[str] = []

    _year_hints(text, hints, notes, library_year_max)
    _tempo_hints(text, hints, notes)

    low = text.lower()
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

    _instrumental_hint(text, hints, notes)
    _key_hints(text, hints, notes)
    _duration_hints(text, hints, notes)
    _playlist_shape_hints(text, hints, notes)

    genre_text = _NO_VOCALS_RE.sub(',', _VERSION_EXCLUDE_RE.sub(',', text))
    genre_hints = _extract_genre_hints(genre_text)
    if genre_hints['genres']:
        hints['genres'] = genre_hints['genres']
        notes.append(f"genre word(s) detected: {genre_hints['genres']}")
    if genre_hints['exclude_genres']:
        hints['exclude_genres'] = genre_hints['exclude_genres']
        notes.append(f"NEGATED genre word(s) detected: {genre_hints['exclude_genres']}")

    moods = _mood_hints(genre_text)
    if moods:
        hints['moods'] = moods
        notes.append(f"mood word(s) detected: {moods}")

    _instrument_hints(genre_text, hints, notes)
    if _is_holiday(text):
        hints['holiday'] = True

    sound = _sound_words(genre_text)
    if sound:
        hints['sound_words'] = sound
        notes.append(f"sound word(s) detected: {sound}")

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
    if hints.get('duration_min') is not None or hints.get('duration_max') is not None:
        lines.append(
            f"  track length seconds: {hints.get('duration_min', '?')}..{hints.get('duration_max', '?')}"
        )
    if hints.get('added_within_days'):
        lines.append(f"  added_within_days: {hints['added_within_days']}")
    if hints.get('key'):
        lines.append(f"  key: {hints['key']}")
    if hints.get('scale'):
        lines.append(f"  scale: {hints['scale']}")
    if hints.get('instrumental') is True:
        lines.append("  instrumental: true (use instrumental=true in search_database)")
    elif hints.get('instrumental') is False:
        lines.append("  instrumental: false (only songs with vocals)")
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
    for keys, hint_keys, wording_re, label in (
        (('duration_min', 'duration_max'), ('duration_min', 'duration_max'), _LENGTHISH_RE, 'track length'),
        (('added_within_days',), ('added_within_days',), _ADDED_WORDING_RE, 'recently added'),
    ):
        present = {k: filt[k] for k in keys if filt.get(k) is not None}
        if present and not any(hints.get(k) for k in hint_keys) and not wording_re.search(original_message):
            log_messages.append(
                f"   strip hallucinated {label} {present} (the request says nothing about it)"
            )
            for k in keys:
                filt.pop(k, None)
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
    if (
        (filt.get('exclude_genres') or filt.get('exclude_artists'))
        and not hints.get('exclude_genres')
        and not _NEGATION_CUE_RE.search(original_message)
    ):
        dropped_ex = {
            k: filt[k] for k in ('exclude_genres', 'exclude_artists') if filt.get(k)
        }
        log_messages.append(
            f"   strip hallucinated exclusions {dropped_ex} "
            "(nothing is excluded in the request)"
        )
        filt.pop('exclude_genres', None)
        filt.pop('exclude_artists', None)
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


def _override_from_hints(plan: 'ToolPlan', hints: Dict, log_messages: List[str]) -> None:
    filt = plan.filter
    if filt is None:
        return
    if hints.get('year_relative') and (
        filt.get('year_min') != hints['year_min'] or filt.get('year_max') != hints['year_max']
    ):
        log_messages.append(
            f"   relative era: year {filt.get('year_min')}..{filt.get('year_max')} -> "
            f"{hints['year_min']}..{hints['year_max']} (resolved against today's date)"
        )
        filt['year_min'], filt['year_max'] = hints['year_min'], hints['year_max']
    want = hints.get('instrumental')
    if want is not None and filt.get('instrumental') is not None and bool(filt['instrumental']) != want:
        log_messages.append(
            f"   instrumental {filt['instrumental']} -> {want} (the request wording says so)"
        )
        filt['instrumental'] = want


def _apply_hint_backstop(plan: 'ToolPlan', hints: Dict, log_messages: List[str]) -> None:
    _override_from_hints(plan, hints, log_messages)
    filt = plan.filter or {}
    backstop: Dict = {}

    _backstop_tempo(backstop, filt, hints)
    _backstop_min_max(backstop, filt, hints, 'energy_min', 'energy_max')
    _backstop_min_max(backstop, filt, hints, 'year_min', 'year_max')
    _backstop_min_max(backstop, filt, hints, 'duration_min', 'duration_max')

    if filt.get('instrumental') is None and hints.get('instrumental') is not None:
        backstop['instrumental'] = hints['instrumental']
    for key in ('key', 'scale', 'added_within_days'):
        if not filt.get(key) and hints.get(key):
            backstop[key] = hints[key]

    _backstop_missing_list(backstop, filt, hints, 'genres')
    _backstop_missing_list(backstop, filt, hints, 'exclude_genres')
    if not plan.primaries and not filt.get('moods') and hints.get('moods'):
        backstop['moods'] = list(hints['moods'])
    excluded = {
        g.lower() for g in (filt.get('exclude_genres') or []) + (backstop.get('exclude_genres') or [])
    }
    if backstop.get('genres') and excluded:
        backstop['genres'] = [g for g in backstop['genres'] if g.lower() not in excluded]
        if not backstop['genres']:
            backstop.pop('genres')

    if backstop:
        plan.filter = _merge_filter(plan.filter, backstop)
        log_messages.append(
            f"   hint backstop: merged {backstop} into the filter "
            "(detected in the request but missing from the plan)"
        )
        if backstop.get('genres'):
            dropped: List[str] = []
            _drop_broader_genres(plan.filter, dropped)
            log_messages.extend(f"   {d}" for d in dropped)
    if hints.get('tempo_explicit') and plan.filter is not None and (
        plan.filter.get('tempo_min') is not None or plan.filter.get('tempo_max') is not None
    ):
        plan.filter[EXACT_TEMPO_KEY] = True


def _add_sound_primary(
    plan: 'ToolPlan', hints: Dict, tool_names: set, log_messages: List[str]
) -> None:
    if plan.primaries or plan.filter is None or 'text_match' not in tool_names:
        return
    if not config.CLAP_ENABLED:
        return
    filt = plan.filter
    if filt.get('artist') or filt.get('album'):
        return
    words = list(hints.get('instrument_words') or []) + list(hints.get('sound_words') or [])
    if not words:
        return
    genres = [g for g in (filt.get('genres') or []) if g.lower() not in words]
    voices = {str(v).lower() for v in filt.get('voices') or []}
    voice = ['female vocals'] if any(v.startswith('female') for v in voices) else (
        ['male vocals'] if voices else []
    )
    query = ' '.join(words[:6] + genres[:2]) + ' music' + (' with ' + voice[0] if voice else '')
    plan.primaries.append({'name': 'text_match', 'arguments': {'query': query, 'mode': 'audio'}})
    log_messages.append(
        f"   sound match: {words} has no exact metadata field -> added "
        f"text_match(audio, '{query}'); the filter re-ranks that sound-matched pool"
    )


INSTRUMENT_STEER_WEIGHT = 3.0
INSTRUMENT_BACKFILL_WEIGHT = 10.0
INSTRUMENT_BACKFILL_SONGS = 2000


def _apply_instruments(plan: 'ToolPlan', hints: Dict, tool_names: set, log_messages: List[str]) -> None:
    wanted = list(hints.get('instruments') or [])
    if not wanted or not config.CLAP_ENABLED or 'text_match' not in tool_names:
        return
    if plan.filter is not None and (plan.filter.get('artist') or plan.filter.get('album')):
        return
    from tasks.clap_steering import concept_terms

    known = set(concept_terms())
    terms = [t for t in wanted if t in known]
    if not terms:
        return
    plan.filter = _merge_filter(plan.filter, {'instruments': terms})
    log_messages.append(
        f"   instrument(s) {terms}: each candidate is checked with the DCLAP concept model and "
        "songs where it clearly plays rank first"
    )


def _steering_for(filt: Optional[Dict], weight: float) -> List[Dict]:
    return [
        {'term': t, 'weight': weight, 'direction': 'more'}
        for t in (filt or {}).get('instruments') or []
    ]


def _relative_change(request: str, hints: Dict, profile: Dict) -> Dict:
    from tasks.ai.calibration import energy_to_norm

    change: Dict = {}
    up, down = _MORE_ENERGY_RE.search(request), _LESS_ENERGY_RE.search(request)
    if bool(up) != bool(down):
        pct = energy_to_norm(profile['energy'])
        if up:
            change['energy_min'] = round(min(0.95, pct + RELATIVE_ENERGY_STEP), 2)
        else:
            change['energy_max'] = round(max(0.05, pct - RELATIVE_ENERGY_STEP), 2)
    tempo = profile.get('tempo')
    if tempo and not hints.get('tempo_explicit'):
        faster, slower = _FASTER_RE.search(request), _SLOWER_RE.search(request)
        if faster and not slower:
            change['tempo_min'] = round(tempo + RELATIVE_TEMPO_STEP)
        elif slower and not faster:
            change['tempo_max'] = round(tempo - RELATIVE_TEMPO_STEP)
    return change


def _apply_seed_relative(plan: 'ToolPlan', request: str, hints: Dict, log_messages: List[str]) -> None:
    if not request:
        return
    seeds = [
        s
        for p in plan.primaries
        if isinstance(p, dict) and p.get('name') == 'seed_search'
        for s in (p.get('arguments') or {}).get('seeds') or []
        if isinstance(s, dict)
    ]
    if not seeds or not any(
        r.search(request) for r in (_MORE_ENERGY_RE, _LESS_ENERGY_RE, _FASTER_RE, _SLOWER_RE)
    ):
        return
    from tasks.ai.tool_impl import _seed_profile

    try:
        profile = _seed_profile(seeds)
    except Exception:
        logger.exception("Reading the seed energy for a relative request failed")
        return
    if not profile:
        return
    change = _relative_change(request, hints, profile)
    if not change:
        return
    filt = plan.filter or {}
    for prefix in {k.split('_')[0] for k in change}:
        filt.pop(f'{prefix}_min', None)
        filt.pop(f'{prefix}_max', None)
    filt.update(change)
    filt.pop(EXACT_TEMPO_KEY, None)
    plan.filter = filt
    log_messages.append(
        f"   relative to the seeds (energy {profile['energy']:.3f}, tempo "
        f"{(profile.get('tempo') or 0):.0f}): {change}"
    )


_ALBUM_WORD_RE = re.compile(r"\b(?:albums?|[\u00e1a]lbum|lp)\b", re.IGNORECASE)


def _backstop_album(plan: 'ToolPlan', request: str, log_messages: List[str]) -> None:
    if plan.primaries or not request or not _ALBUM_WORD_RE.search(request):
        return
    filt = plan.filter or {}
    if filt.get('album'):
        return
    from tasks.ai.tool_impl import _album_named_in_request

    try:
        hit = _album_named_in_request(request, filt.get('artist'))
    except Exception:
        logger.exception("Album lookup for the chat request failed")
        return
    if not hit:
        return
    backstop = {'album': hit['album']}
    if not filt.get('artist') and hit.get('artist_named'):
        backstop['artist'] = hit['author']
    plan.filter = _merge_filter(plan.filter, backstop)
    log_messages.append(
        f"   album backstop: the request names the album '{hit['album']}' -> merged {backstop} "
        "into the filter"
    )


def _strip_unrated_filter(
    plan: 'ToolPlan', library_context: Optional[Dict], log_messages: List[str]
) -> None:
    if plan.filter is None or not plan.filter.get('min_rating'):
        return
    if not library_context or library_context.get('has_ratings', True):
        return
    log_messages.append(
        f"   strip min_rating={plan.filter['min_rating']} (no song in the library has a rating)"
    )
    plan.filter.pop('min_rating', None)
    plan.notes.append("your library has no song ratings yet, so the rating filter was skipped")
    if not _has_filter_content(plan.filter):
        plan.filter = None


def _drop_versions(songs: List[Dict], versions: List[str], log_messages: List[str]) -> List[Dict]:
    patterns = [_VERSION_PATTERNS[v] for v in versions if v in _VERSION_PATTERNS]
    if not patterns or not songs:
        return songs
    version_re = re.compile('|'.join(patterns), re.IGNORECASE)

    def _is_version(s):
        text = f"{s.get('title') or ''} | {s.get('album') or ''}" if 'live' in versions else s.get('title') or ''
        return bool(version_re.search(text))

    kept = [s for s in songs if not _is_version(s)]
    if kept and len(kept) < len(songs):
        log_messages.append(
            f"   versions excluded {versions}: removed {len(songs) - len(kept)} of {len(songs)} songs"
        )
        return kept
    return songs


def _is_holiday(*texts) -> bool:
    from tasks.album_creation_manager import is_holiday_text

    return bool(is_holiday_text(*texts))


def _demote_holidays(songs: List[Dict], log_messages: List[str]) -> List[Dict]:
    seasonal = [s for s in songs if _is_holiday(s.get('title'), s.get('album'))]
    if not seasonal or len(seasonal) == len(songs):
        return songs
    moved = {id(s) for s in seasonal}
    log_messages.append(
        f"   holiday songs: {len(seasonal)} moved to the end (the request does not ask for them)"
    )
    return [s for s in songs if id(s) not in moved] + seasonal


def _shape_result(result: Dict, hints: Dict, log_messages: List[str], plan: 'ToolPlan') -> Dict:
    if hints.get('exclude_versions') and result.get('songs'):
        result['songs'] = _drop_versions(result['songs'], hints['exclude_versions'], log_messages)
    if not hints.get('holiday') and result.get('songs') and _journey_call(plan) is None:
        result['songs'] = _demote_holidays(result['songs'], log_messages)
    result['max_per_artist'] = hints.get('max_per_artist')
    result['keep_order'] = _journey_call(plan) is not None
    return result


def requested_playlist_shape(text: str) -> Dict:
    hints: Dict = {}
    notes: List[str] = []
    if text and isinstance(text, str):
        _duration_hints(text, hints, notes)
        _playlist_shape_hints(text, hints, notes)
    return {k: hints[k] for k in ('song_count', 'total_seconds', 'max_per_artist') if hints.get(k)}


def _synthesize_rescue_plan(
    raw_request: str,
    hints: Dict,
    log_messages: List[str],
) -> 'ToolPlan':
    from tasks.ai.tools import _YEAR_ONLY_RE

    plan = ToolPlan()
    _apply_hint_backstop(plan, hints, log_messages)

    query = (raw_request or '').strip()
    year_only = bool(query) and bool(_YEAR_ONLY_RE.match(query))

    if query and not year_only:
        if config.CLAP_ENABLED:
            plan.primaries.append(
                {'name': 'text_match', 'arguments': {'query': query, 'mode': 'audio'}}
            )
        elif config.LYRICS_ENABLED:
            plan.primaries.append(
                {'name': 'text_match', 'arguments': {'query': query, 'mode': 'lyrics'}}
            )
        else:
            plan.primaries.append(
                {'name': 'knowledge_lookup', 'arguments': {'user_request': query}}
            )

    if plan.primaries or plan.filter is not None:
        kinds = ', '.join(p.get('name', '') for p in plan.primaries) or 'filter only'
        log_messages.append(f"   rescue: matching your words directly ({kinds})")
    return plan


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


def _drop_broader_genres(filt: Dict, notes: List[str]) -> None:
    genres = filt.get('genres') or []
    if len(genres) < 2:
        return
    broader = [
        g for g in genres
        if any(h != g and re.search(rf"\b{re.escape(g.lower())}\b", h.lower()) for h in genres)
    ]
    if broader:
        filt['genres'] = [g for g in genres if g not in broader]
        notes.append(f"dropped broader genre(s) {broader}: a more specific one was requested")


def _normalize_filter_inplace(filt: Dict, notes: List[str]) -> Dict:
    if 'genres' in filt and filt['genres']:
        g = normalize_genre_list(filt['genres'])
        filt['genres'] = g['genres']
        for n in g.get('notes') or []:
            notes.append(n)
        if not filt['genres']:
            filt.pop('genres', None)

    _drop_broader_genres(filt, notes)

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

    _widen_narrow_range(filt, 'tempo_min', 'tempo_max', _BPM_HINT_HALF_WINDOW, 'BPM', notes)
    _widen_narrow_range(filt, 'energy_min', 'energy_max', _ENERGY_HALF_WINDOW, 'energy', notes)
    _sanitize_duration(filt, notes)
    return filt


_ENERGY_HALF_WINDOW = 0.1
_POINT_RANGE_FRACTION = 0.2


def _widen_narrow_range(filt: Dict, lo_key: str, hi_key: str, half: float, unit: str, notes: List[str]) -> None:
    lo, hi = filt.get(lo_key), filt.get(hi_key)
    if lo is None or hi is None:
        return
    try:
        lo, hi = sorted((float(lo), float(hi)))
    except (TypeError, ValueError):
        return
    filt[lo_key], filt[hi_key] = lo, hi
    if hi - lo >= 2 * half * _POINT_RANGE_FRACTION:
        return
    center = (lo + hi) / 2.0
    filt[lo_key], filt[hi_key] = round(center - half, 3), round(center + half, 3)
    notes.append(
        f"{unit} {lo:g}-{hi:g} widened to {filt[lo_key]:g}-{filt[hi_key]:g} "
        "(an exact point matches almost nothing)"
    )


def _sanitize_duration(filt: Dict, notes: List[str]) -> None:
    for key in ('duration_min', 'duration_max'):
        v = filt.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        if v <= 0:
            notes.append(f"dropped nonsensical {key}={filt[key]}")
            filt.pop(key, None)
        else:
            filt[key] = v
    lo, hi = filt.get('duration_min'), filt.get('duration_max')
    if lo is not None and hi is not None and lo > hi:
        filt['duration_min'], filt['duration_max'] = hi, lo


def _seed_identity(seed: Dict) -> tuple:
    if seed.get('type') == 'artist':
        return ('artist', (seed.get('name') or '').strip().lower(), '')
    return (
        'song',
        (seed.get('title') or '').strip().lower(),
        (seed.get('artist') or '').strip().lower(),
    )


_DEDUPE_LIST_KEYS = ('genres', 'voices', 'moods', 'exclude_artists', 'exclude_genres')


def _dedupe_by_key(values, wanted_type, identity) -> tuple:
    out: List = []
    seen: set = set()
    for value in values or []:
        if not isinstance(value, wanted_type):
            continue
        key = identity(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out, len(out) != len(values or [])


def _dedupe_strings(values) -> tuple:
    return _dedupe_by_key(values, str, lambda v: v.strip().lower())


def _dedupe_seed_list(values) -> tuple:
    return _dedupe_by_key(values, dict, _seed_identity)


def _dedupe_call_lists(
    tool_calls: List[Dict],
    log_messages: Optional[List[str]] = None,
) -> List[Dict]:
    if log_messages is None:
        log_messages = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get('name')
        args = tc.get('arguments')
        if not isinstance(args, dict):
            continue
        if name == FILTER_NAME:
            for key in _DEDUPE_LIST_KEYS:
                if not isinstance(args.get(key), list):
                    continue
                before = len(args[key])
                cleaned, changed = _dedupe_strings(args[key])
                if changed:
                    args[key] = cleaned
                    log_messages.append(
                        f"   dedupe: {name}.{key} {before} -> {len(cleaned)} value(s) "
                        "(the model repeated a value to fill the array)"
                    )
        elif name == 'seed_search':
            for key in ('seeds', 'subtract'):
                if not isinstance(args.get(key), list):
                    continue
                before = len(args[key])
                cleaned, changed = _dedupe_seed_list(args[key])
                if changed:
                    args[key] = cleaned
                    log_messages.append(
                        f"   dedupe: {name}.{key} {before} -> {len(cleaned)} item(s) "
                        "(the model repeated a seed)"
                    )
    return tool_calls or []


def _positive_subjects(plan: 'ToolPlan') -> set:
    from tasks.ai.tool_impl import _normalize_for_match

    subjects: set = set()
    filt = plan.filter or {}
    for key in ('artist', 'album'):
        v = filt.get(key)
        if isinstance(v, str) and v.strip():
            subjects.add(_normalize_for_match(v))
    for p in plan.primaries:
        if not isinstance(p, dict) or p.get('name') != 'seed_search':
            continue
        for s in (p.get('arguments') or {}).get('seeds') or []:
            if not isinstance(s, dict):
                continue
            for key in ('name', 'artist'):
                v = s.get(key)
                if isinstance(v, str) and v.strip():
                    subjects.add(_normalize_for_match(v))
    return {s for s in subjects if s}


def _named_in_request(value: str, message: str) -> bool:
    from rapidfuzz import fuzz

    v = (value or '').strip().lower()
    if not v:
        return False
    if v in (message or '').lower():
        return True
    return fuzz.partial_ratio(v, (message or '').lower()) >= 85


def _genre_named(genre: str, message: str) -> bool:
    text = _fold_accents(message or '').lower()
    names = {genre.lower()} | {alias for alias, target in ALIAS_GENRE.items() if target == genre}
    names |= {n.replace('-', ' ') for n in names}
    return any(re.search(rf"(?<![\w-]){re.escape(n)}(?:s|es|p?ers?)?(?![\w-])", text) for n in names)


def _strip_contradictory_exclusions(
    plan: 'ToolPlan',
    hints: Dict,
    original_message: str,
    log_messages: List[str],
) -> None:
    from tasks.ai.tool_impl import _normalize_for_match

    if plan.filter is None:
        return
    filt = plan.filter

    subjects = _positive_subjects(plan)
    hint_genres = {g.lower() for g in (hints.get('genres') or [])}
    hint_excluded = {g.lower() for g in (hints.get('exclude_genres') or [])}

    if filt.get('exclude_artists'):
        kept = []
        for a in filt['exclude_artists']:
            if not isinstance(a, str):
                continue
            if _normalize_for_match(a) in subjects:
                log_messages.append(
                    f"   contradiction: dropped exclude_artists '{a}' "
                    "(the request asks for that same artist)"
                )
                plan.notes.append(
                    f"'{a}' was both requested and excluded; kept it as the subject"
                )
                continue
            if not _named_in_request(a, original_message):
                log_messages.append(
                    f"   contradiction: dropped exclude_artists '{a}' "
                    "(not named anywhere in the request)"
                )
                plan.notes.append(f"invented exclusion of '{a}' was dropped")
                continue
            kept.append(a)
        if kept:
            filt['exclude_artists'] = kept
        else:
            filt.pop('exclude_artists', None)

    if filt.get('exclude_genres'):
        kept = []
        for g in filt['exclude_genres']:
            if not isinstance(g, str):
                continue
            low = g.lower()
            if low in hint_genres and low not in hint_excluded:
                log_messages.append(
                    f"   contradiction: dropped exclude_genres '{g}' "
                    "(the request asks for that same genre)"
                )
                plan.notes.append(
                    f"'{g}' was both requested and excluded; kept it as a positive filter"
                )
                continue
            if low not in hint_excluded and not _genre_named(g, original_message):
                log_messages.append(
                    f"   contradiction: dropped exclude_genres '{g}' "
                    "(the request does not name that genre)"
                )
                plan.notes.append(f"invented exclusion of '{g}' was dropped")
                continue
            kept.append(g)
        if kept:
            filt['exclude_genres'] = kept
        else:
            filt.pop('exclude_genres', None)

    if not _has_filter_content(filt):
        plan.notes.append('filter emptied after dropping contradictory exclusions')
        plan.filter = None


COPIED_EXAMPLE_MIN_WORDS = 3
_NEGATED_QUERY_RE = re.compile(r"^\W*(?:no|not|nothing|without|never|zero|avoid|excluding)\b", re.IGNORECASE)


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (text or '').lower()) if len(w) > 2}


def _copied_example_words(query: str, request: str) -> List[str]:
    if not request or not request.strip():
        return []
    asked = _content_words(request)
    said = _content_words(query)
    for example in EXAMPLE_TEXT_QUERIES:
        borrowed = sorted((_content_words(example) & said) - asked)
        if len(borrowed) >= COPIED_EXAMPLE_MIN_WORDS:
            return borrowed
    return []


def _seed_label(seed: Dict) -> str:
    return (seed.get('title') if seed.get('type') == 'song' else seed.get('name')) or ''


def _is_journey_request(seeds: List[Dict], text: str) -> bool:
    if len(seeds) != 2 or not text:
        return False
    if _JOURNEY_WORD_RE.search(text):
        return True
    m = _FROM_TO_RE.search(text)
    return bool(m) and _named_in_request(_seed_label(seeds[0]), m.group(1)) and _named_in_request(
        _seed_label(seeds[1]), m.group(2)
    )


def _journey_call(plan: 'ToolPlan') -> Optional[Dict]:
    for p in plan.primaries:
        if (
            isinstance(p, dict)
            and p.get('name') == 'seed_search'
            and (p.get('arguments') or {}).get('blend_mode') == 'journey'
        ):
            return p
    return None


def _prepare_journey(plan: 'ToolPlan', target_song_count: Optional[int], log_messages: List[str]) -> None:
    journey = _journey_call(plan)
    if journey is None:
        return
    journey['arguments']['journey_length'] = (
        target_song_count or config.INSTANT_PLAYLIST_DEFAULT_N_RESULTS
    )
    dropped = [p.get('name') for p in plan.primaries if p is not journey]
    if dropped or plan.filter:
        log_messages.append(
            f"   journey: kept the song path alone (dropped {dropped or '-'} and the filter "
            f"{plan.filter or '-'}) so its order stays intact"
        )
        plan.notes.append("a journey keeps its path order, so the other constraints were not applied")
    plan.primaries = [journey]
    plan.filter = None


def validate_plan_args(
    tool_calls: List[Dict],
    *,
    user_wants_rating: bool,
    log_messages: Optional[List[str]] = None,
    request_text: str = '',
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
            journey_ok = blend != 'subtract' or not args.get('subtract')
            if blend != 'journey' and journey_ok and _is_journey_request(cleaned_seeds, request_text):
                log_messages.append(
                    f"   coerce blend_mode '{blend}' -> 'journey' (the request goes from one seed to the other)"
                )
                blend = 'journey'
                args.pop('subtract', None)
            if blend == 'journey' and len(cleaned_seeds) != 2:
                log_messages.append("   coerce blend_mode 'journey' -> 'union' (a journey needs exactly 2 seeds)")
                blend = 'union'
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
            if _NEGATED_QUERY_RE.match(query):
                log_messages.append(
                    f"   skip {name}: '{query}' is a negation, which a sound or lyric match reads as its opposite"
                )
                continue
            copied = _copied_example_words(query, request_text)
            if copied:
                log_messages.append(
                    f"   text_match query '{query}' repeats the prompt example ({', '.join(copied)}); "
                    f"using the request's own words '{request_text.strip()}'"
                )
                query = request_text.strip()
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


MAX_TOOL_CALLS = config.AI_MAX_TOOL_CALLS


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
    'duration_min',
    'duration_max',
    'added_within_days',
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
    final, _matched, _moved = rerank(pool_songs, filter_args, feats, log_messages)
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


_GROUNDING_FILTER_KEYS = (
    'genres', 'moods', 'voices',
    'year_min', 'year_max',
    'energy_min', 'energy_max',
    'tempo_min', 'tempo_max',
)
_GATE_FILTER_KEYS = (
    'key', 'scale', 'min_rating', 'instrumental', 'album',
    'exclude_artists', 'exclude_genres', 'duration_min', 'duration_max', 'added_within_days',
)


def _ground_knowledge_lookup(plan: 'ToolPlan', log_messages: List[str]) -> None:
    filt = plan.filter or {}
    grounding = {
        k: v for k in _GROUNDING_FILTER_KEYS
        if (v := filt.get(k)) not in (None, '', [], {})
    }
    gate = {
        k: v for k in _GATE_FILTER_KEYS
        if (v := filt.get(k)) not in (None, '', [], {})
    }

    for p in plan.primaries:
        if not isinstance(p, dict) or p.get('name') != 'knowledge_lookup':
            continue
        args = p.setdefault('arguments', {})
        if grounding:
            args['grounding_filter'] = dict(grounding)
        if gate:
            args['gate_filter'] = dict(gate)

    if grounding or gate:
        log_messages.append(
            f"   AI brainstorming: filter grounding={grounding or '-'} gate={gate or '-'} "
            "merged INTO the recipe (applied inside the tool, not as a post-filter)"
        )
        plan.notes.append(
            f"constraints {dict(grounding, **gate)} were applied inside the brainstorm "
            "search itself, so the suggestions are never re-filtered afterwards"
        )

    if filt.get('artist'):
        plan.notes.append(
            f"artist '{filt['artist']}' was not applied to the brainstorm: a popularity "
            "request narrowed to one artist would return only that artist"
        )

    plan.filter = None


_EMPTY_PLAN_FEEDBACK = (
    "PREVIOUS ATTEMPT FAILED: the plan was empty. search_database was emitted with no "
    "arguments, or every call was dropped in validation.\n"
    "Emit at least one finder tool (seed_search for a named artist or song, text_match "
    "for a described sound or lyric topic, knowledge_lookup for a popularity or cultural "
    "ask), or fill search_database with the constraints the request actually states. "
    "Never emit search_database with no fields."
)


def _finish_plan(
    plan: 'ToolPlan',
    hints: Dict,
    ai_config: Dict,
    log_messages: List[str],
    *,
    library_context: Optional[Dict] = None,
    collection_cap: int = 1000,
    target_song_count: Optional[int] = None,
    raw_request: str = '',
    allow_rescue: bool = True,
):
    for u in hints.get('unsupported', []):
        if u not in plan.notes:
            plan.notes.append(u)

    has_knowledge = any(
        isinstance(p, dict) and p.get('name') == 'knowledge_lookup' for p in plan.primaries
    )

    exec_result = yield from _execute_plan(
        plan,
        ai_config,
        log_messages,
        library_context=library_context,
        collection_cap=collection_cap,
        target_song_count=target_song_count,
        has_knowledge=has_knowledge,
    )

    if not exec_result['songs'] and allow_rescue:
        rescue = _synthesize_rescue_plan(raw_request, hints, log_messages)
        if rescue.primaries or rescue.filter is not None:
            rescue.notes = list(plan.notes) + rescue.notes
            return (
                yield from _finish_plan(
                    rescue, hints, ai_config, log_messages,
                    library_context=library_context,
                    collection_cap=collection_cap,
                    target_song_count=target_song_count,
                    raw_request=raw_request,
                    allow_rescue=False,
                )
            )

    history = exec_result['tools_used_history']
    summary = exec_result['tool_execution_summary']
    return _shape_result({
        "songs": exec_result['songs'],
        "song_sources": exec_result['song_sources'],
        "tools_used_history": history,
        "tool_execution_summary": summary,
        "detected_min_rating": exec_result['detected_min_rating'],
        "plan_notes": plan.notes,
        "executed_query_str": f"MCP single-pass ({len(history)} tools): {' -> '.join(summary)}",
        "filter_applied": plan.filter is not None,
    }, hints, log_messages, plan)


def plan_and_execute_once(
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str],
    *,
    library_context: Optional[Dict] = None,
    user_wants_rating: bool = False,
    collection_cap: int = 1000,
    target_song_count: Optional[int] = None,
    replan_feedback: Optional[str] = None,
    raw_user_request: Optional[str] = None,
):
    log_messages.append("\n--- AI Decision ---")

    original_user_message = user_message
    raw_request = raw_user_request or original_user_message
    log_messages.append(
        f"   tools offered: {', '.join(t.get('name', '') for t in tools)}"
    )

    hints = extract_hints(
        raw_request, (library_context or {}).get('year_max') if library_context else None
    )
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
        log_messages.append(
            "   the AI planner did not answer; matching your words directly instead"
        )
        plan = _synthesize_rescue_plan(raw_request, hints, log_messages)
        plan.notes.append(
            "the AI planner was unreachable, so this playlist comes from a direct "
            "match of your request instead of a tool plan"
        )
        return (
            yield from _finish_plan(
                plan, hints, ai_config, log_messages,
                library_context=library_context,
                collection_cap=collection_cap,
                target_song_count=target_song_count,
                raw_request=raw_request,
                allow_rescue=False,
            )
        )

    reasoning = raw.get('reasoning')
    if isinstance(reasoning, str) and reasoning.strip():
        log_messages.append(f"AI reasoning: {reasoning.strip()}")

    raw_calls = raw.get('tool_calls', []) or []
    log_messages.append(f"AI emitted {len(raw_calls)} tool call(s)")

    yield

    raw_calls = _dedupe_call_lists(raw_calls, log_messages=log_messages)
    raw_calls = dedupe_and_cap_calls(raw_calls, log_messages=log_messages)
    raw_calls = validate_plan_args(
        raw_calls,
        user_wants_rating=user_wants_rating,
        log_messages=log_messages,
        request_text=raw_request,
    )

    plan = validate_and_normalize_plan(raw_calls)
    for note in plan.notes:
        log_messages.append(f"   plan: {note}")

    if not plan.primaries and plan.filter is None:
        if replan_feedback is None:
            log_messages.append("\nEMPTY PLAN -> replanning once with feedback")
            replan = yield from plan_and_execute_once(
                original_user_message,
                tools,
                ai_config,
                log_messages,
                library_context=library_context,
                user_wants_rating=user_wants_rating,
                collection_cap=collection_cap,
                target_song_count=target_song_count,
                replan_feedback=_EMPTY_PLAN_FEEDBACK,
                raw_user_request=raw_request,
            )
            if isinstance(replan, dict) and replan.get('songs'):
                return replan
            log_messages.append(
                "   replan produced no usable plan either; matching your words directly"
            )
        plan = _synthesize_rescue_plan(raw_request, hints, log_messages)
        return (
            yield from _finish_plan(
                plan, hints, ai_config, log_messages,
                library_context=library_context,
                collection_cap=collection_cap,
                target_song_count=target_song_count,
                raw_request=raw_request,
                allow_rescue=False,
            )
        )

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

    _strip_unrequested_filter_args(plan, hints, raw_request, log_messages)
    _strip_contradictory_exclusions(plan, hints, raw_request, log_messages)
    _strip_unrated_filter(plan, library_context, log_messages)
    _apply_hint_backstop(plan, hints, log_messages)
    _apply_seed_relative(plan, raw_request, hints, log_messages)
    _backstop_album(plan, raw_request, log_messages)
    _add_sound_primary(plan, hints, {t.get('name') for t in tools}, log_messages)
    _apply_instruments(plan, hints, {t.get('name') for t in tools}, log_messages)
    _prepare_journey(plan, target_song_count, log_messages)

    for u in hints.get('unsupported', []):
        plan.notes.append(u)

    if plan.filter is not None and has_knowledge:
        _ground_knowledge_lookup(plan, log_messages)

    exec_result = yield from _execute_plan(
        plan,
        ai_config,
        log_messages,
        library_context=library_context,
        collection_cap=collection_cap,
        target_song_count=target_song_count,
        has_knowledge=has_knowledge,
    )
    all_songs = exec_result['songs']
    tools_used_history = exec_result['tools_used_history']
    tool_execution_summary = exec_result['tool_execution_summary']

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
            raw_user_request=raw_request,
        )
        if isinstance(replan, dict) and replan.get('songs'):
            return replan
        log_messages.append("   replan attempt failed; matching your words directly")

    if not all_songs:
        rescue = _synthesize_rescue_plan(raw_request, hints, log_messages)
        if rescue.primaries or rescue.filter is not None:
            rescue.notes = list(plan.notes) + rescue.notes
            return (
                yield from _finish_plan(
                    rescue, hints, ai_config, log_messages,
                    library_context=library_context,
                    collection_cap=collection_cap,
                    target_song_count=target_song_count,
                    raw_request=raw_request,
                    allow_rescue=False,
                )
            )

    return _shape_result({
        "songs": all_songs,
        "song_sources": exec_result['song_sources'],
        "tools_used_history": tools_used_history,
        "tool_execution_summary": tool_execution_summary,
        "detected_min_rating": exec_result['detected_min_rating'],
        "plan_notes": plan.notes,
        "executed_query_str": f"MCP single-pass ({len(tools_used_history)} tools): {' -> '.join(tool_execution_summary)}",
        "filter_applied": plan.filter is not None,
    }, hints, log_messages, plan)


BACKFILL_POOL_FACTOR = 4


def _backfill_genre_matches(
    filt: Dict,
    pool_songs: List[Dict],
    feats: Dict,
    ai_config: Dict,
    target_song_count: int,
    log_messages: List[str],
) -> List[Dict]:
    from tasks.ai.tools import execute_mcp_tool
    from tasks.ai.tool_impl import _fetch_pool_features

    if not filt.get('genres'):
        return pool_songs
    full = count_full_matches(pool_songs, filt, feats)
    if full >= target_song_count:
        return pool_songs
    args = {k: v for k, v in filt.items() if k in FILTER_ALL_KEYS}
    args['get_songs'] = max(200, target_song_count * BACKFILL_POOL_FACTOR)
    res = execute_mcp_tool('search_database', args, ai_config)
    pooled = {s.get('item_id') for s in pool_songs}
    extra = [s for s in res.get('songs') or [] if s.get('item_id') and s['item_id'] not in pooled]
    if not extra:
        return pool_songs
    feats.update(_fetch_pool_features([s['item_id'] for s in extra]))
    log_messages.append(
        f"   pool backfill: only {full} of {len(pool_songs)} pooled songs match every requested "
        f"value (target {target_song_count}); added {len(extra)} library songs that do, "
        "ranked after the similar ones"
    )
    return pool_songs + extra


def _attach_instruments(filt: Dict, songs: List[Dict], feats: Dict, log_messages: List[str]) -> None:
    from tasks.ai.calibration import concept_percentiles

    terms = list(filt.get('instruments') or [])
    todo = [s['item_id'] for s in songs if 'instrument_pct' not in (feats.get(s['item_id']) or {})]
    if not terms or not todo:
        return
    try:
        scored = concept_percentiles(todo, terms)
    except Exception:
        logger.exception("Scoring the pool for the requested instruments failed")
        scored = {}
    for item_id in todo:
        feats.setdefault(item_id, {})['instrument_pct'] = scored.get(item_id, {})
    clear = sum(
        1 for s in songs
        if all((feats.get(s['item_id']) or {}).get('instrument_pct', {}).get(t, 0.0) >= INSTRUMENT_HIT for t in terms)
    )
    log_messages.append(
        f"   instrument check {terms}: {clear} of {len(songs)} pooled songs are in the library's "
        f"top {int(round((1 - INSTRUMENT_HIT) * 100))}% for it"
    )


def _backfill_instrument_matches(
    plan: 'ToolPlan',
    pool_songs: List[Dict],
    feats: Dict,
    ai_config: Dict,
    target_song_count: int,
    log_messages: List[str],
) -> List[Dict]:
    from tasks.ai.tools import execute_mcp_tool

    filt = plan.filter
    if count_full_matches(pool_songs, filt, feats) >= target_song_count:
        return pool_songs
    queries = [
        (p.get('arguments') or {}).get('query') for p in plan.primaries
        if p.get('name') == 'text_match' and (p.get('arguments') or {}).get('mode', 'audio') == 'audio'
    ]
    query = next((q for q in queries if q), None) or ' '.join(filt['instruments'])
    res = execute_mcp_tool('text_match', {
        'query': query,
        'mode': 'audio',
        'get_songs': INSTRUMENT_BACKFILL_SONGS,
        'steering': _steering_for(filt, INSTRUMENT_BACKFILL_WEIGHT),
    }, ai_config)
    pooled = {s.get('item_id') for s in pool_songs}
    extra = [s for s in res.get('songs') or [] if s.get('item_id') and s['item_id'] not in pooled]
    if not extra:
        return pool_songs
    log_messages.append(
        f"   instrument backfill: too few pooled songs carry every requested value; added "
        f"{len(extra)} songs from a search led by {filt['instruments']}"
    )
    from tasks.ai.tool_impl import _fetch_pool_features

    feats.update(_fetch_pool_features([s['item_id'] for s in extra]))
    merged = pool_songs + extra
    _attach_instruments(filt, extra, feats, log_messages)
    return merged


def _genre_purity_sort(songs: List[Dict], genres: List[str], log_messages: List[str]) -> List[Dict]:
    from tasks.ai.tool_impl import _fetch_pool_features

    wanted = {g.lower() for g in genres}
    try:
        feats = _fetch_pool_features([s['item_id'] for s in songs])
    except Exception:
        logger.exception("Reading genre tags for the purity order failed")
        return songs
    ranks = [
        genre_style_rank((feats.get(s['item_id']) or {}).get('mood_vector') or '', wanted)
        for s in songs
    ]
    order = sorted(range(len(songs)), key=lambda i: ranks[i])
    if order != list(range(len(songs))):
        log_messages.append(
            f"   genre purity: {ranks.count(0)} of {len(songs)} songs have {sorted(wanted)} as their "
            "main style and lead the list"
        )
    return [songs[i] for i in order]


def _demote_non_songs(songs: List[Dict], log_messages: List[str], short_floor: Optional[float]) -> None:
    from tasks.ai.tool_impl import _fetch_pool_features

    lengths: Dict = {}
    if short_floor:
        try:
            feats = _fetch_pool_features([s['item_id'] for s in songs if s.get('item_id')])
            lengths = {k: (v or {}).get('duration') for k, v in feats.items()}
        except Exception:
            logger.exception("Reading track lengths for the non-song demotion failed")

    def _is_non_song(s):
        if _NON_SONG_TITLE_RE.search(s.get('title') or ''):
            return True
        length = lengths.get(s.get('item_id'))
        return bool(short_floor) and length is not None and 0 < float(length) < short_floor

    demoted = [s for s in songs if _is_non_song(s)]
    if not demoted or len(demoted) == len(songs):
        return
    demoted_ids = {id(s) for s in demoted}
    songs[:] = [s for s in songs if id(s) not in demoted_ids] + demoted
    log_messages.append(
        f"   non-song tracks (intro/skit/interlude titles or very short): {len(demoted)} moved to the end"
    )


def _execute_plan(
    plan: 'ToolPlan',
    ai_config: Dict,
    log_messages: List[str],
    *,
    library_context: Optional[Dict] = None,
    collection_cap: int = 1000,
    target_song_count: Optional[int] = None,
    has_knowledge: bool = False,
):
    from tasks.ai.tools import execute_mcp_tool
    from tasks.ai.tool_impl import _fetch_pool_features

    if target_song_count is None:
        target_song_count = config.INSTANT_PLAYLIST_DEFAULT_N_RESULTS

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
            if args.get('duration_min') is not None or args.get('duration_max') is not None:
                parts.append(f"duration={args.get('duration_min', '')}..{args.get('duration_max', '')}s")
            if args.get('added_within_days'):
                parts.append(f"added_within_days={args['added_within_days']}")
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
            if tn == 'text_match' and ta.get('mode', 'audio') == 'audio' and plan.filter.get('instruments'):
                ta['steering'] = _steering_for(plan.filter, INSTRUMENT_STEER_WEIGHT)
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
        pool_songs = _apply_exclusions(
            pool_songs, plan.filter, feats, log_messages, notes=plan.notes
        )
        instruments = bool(plan.filter.get('instruments'))
        if pool_songs and instruments:
            _attach_instruments(plan.filter, pool_songs, feats, log_messages)
        if pool_songs:
            pool_songs = _backfill_genre_matches(
                plan.filter, pool_songs, feats, ai_config, target_song_count, log_messages
            )
        if pool_songs and instruments:
            _attach_instruments(plan.filter, pool_songs, feats, log_messages)
            pool_songs = _backfill_instrument_matches(
                plan, pool_songs, feats, ai_config, target_song_count, log_messages
            )
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
                    'role': 'pool',
                }
            )
            tool_execution_summary.append(_summary(tn, ta, pooled if pool_songs else 0))
            tool_call_counter += 1

        if pool_songs:
            N = len(pool_songs)
            final, matched, _moved = rerank(
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
                    'role': 'rerank',
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
                ta['get_songs'] = max(200, target_song_count * 2)
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
                if ta.get('genres') and not ta.get('album') and not ta.get('artist') and res.get('songs'):
                    res['songs'] = _genre_purity_sort(res['songs'], ta['genres'], log_messages)
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
        if not (plan.filter and plan.filter.get('album')) and _journey_call(plan) is None:
            _demote_non_songs(all_songs, log_messages, _short_track_floor(plan.filter or {}))

    return {
        "songs": all_songs,
        "song_sources": song_sources,
        "tools_used_history": tools_used_history,
        "tool_execution_summary": tool_execution_summary,
        "detected_min_rating": detected_min_rating,
    }
