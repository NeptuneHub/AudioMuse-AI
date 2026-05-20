"""Chat-pipeline orchestration: pre-extract, classifier, validation, plan execution.

Single home for everything that turns a user's natural-language request into
executed tool calls. Replaces the old split across ``tool_plan.py`` +
``tool_planner.py`` + ``intent_classifier.py`` + ``intent_preextract.py``.

Public surface (call sites elsewhere import from here):

    Pre-extract:
        extract_hints(text)            -> dict
        format_hints_block(hints)      -> str

    Plan model + validation:
        ToolPlan                       (dataclass)
        validate_plan_args(...)        pre-execution sanity checks
        validate_and_normalize_plan(...) raw tool_calls -> ToolPlan
        plan_from_tool_calls(...)      thin alias

    Intent classifier (stage 1):
        classify(user_message, ai_config, log_messages=None) -> Optional[dict]
        tools_for_intent(intent, needs_filter, all_tools)    -> filtered tool list

    Orchestrators:
        call_ai_for_plan(...)          one transport call, returns raw tool_calls
        plan_and_execute(...)          legacy single-call planner with retry
        plan_and_execute_once(...)     two-stage classifier + execute pipeline
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config

from tasks.ai.prompts import INTENT_CLASSES, build_intent_classifier_prompt
from tasks.ai.vocab import (
    ALIAS_ENERGY,
    ALIAS_TEMPO,
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

# Presence/identity dims the user names explicitly. In the composition re-rank
# these act as a priority tier (songs that HAVE them rank above songs that
# don't); the continuous dims only order songs within each tier. Everything else
# requested (moods, energy, tempo, year, min_rating, key) is a gradient.
CATEGORICAL_DIMS = ('genres', 'voices', 'scale', 'artist', 'album')

COMPOSITION_POOL_TARGET = 10000

YEAR_DECAY_SPAN = 30.0

_KEY_PC = {
    'C': 0, 'B#': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3,
    'E': 4, 'FB': 4, 'F': 5, 'E#': 5, 'F#': 6, 'GB': 6, 'G': 7,
    'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11, 'CB': 11,
}


def _key_pitch_class(k) -> Optional[int]:
    """Map a key label ('C', 'F# minor', 'Bb') to a 0..11 pitch class, or None."""
    if not k:
        return None
    s = str(k).strip().upper().replace('♯', '#').replace('♭', 'B')
    token = s.split()[0] if s.split() else s
    for cand in (token[:2], token[:1]):
        if cand in _KEY_PC:
            return _KEY_PC[cand]
    return None


def _range_pref_score(v_norm: float, req_lo: float, req_hi: float) -> float:
    """Continuous 0..1 fit of a normalized value against a requested [lo,hi] band.

    All args are in [0,1]. Directional so that 'high X' (band open at the top)
    rewards higher values, 'low X' (band open at the bottom) rewards lower
    values, and a bounded band rewards proximity to its centre. Always
    differentiates, so songs never tie on a continuous feature.
    """
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
    """Parse a 'label:score,label:score' column into {label_lower: float}."""
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
    """Raw 0..1 per-dimension match scores for one song (only requested dims).

    Routing follows the data model: genres/voices -> mood_vector;
    moods/other_features -> other_features; energy/tempo -> directional
    gradient; year -> distance-decay; rating -> /5; key -> chromatic distance;
    scale/artist/album -> identity. Keyed by dimension name so the caller can
    min-max normalize each dimension across the pool before blending (otherwise
    a wide-range dim like energy drowns a narrow one like a mood confidence).
    """
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
            req_lo = ((float(filt['tempo_min']) - t_lo) / span) if filt.get('tempo_min') is not None else 0.0
            req_hi = ((float(filt['tempo_max']) - t_lo) / span) if filt.get('tempo_max') is not None else 1.0
            out['tempo'] = _range_pref_score(v_norm, max(0.0, min(1.0, req_lo)), max(0.0, min(1.0, req_hi)))

    if filt.get('energy_min') is not None or filt.get('energy_max') is not None:
        energy = feats.get('energy')
        if energy is None:
            out['energy'] = 0.0
        else:
            span = (config.ENERGY_MAX - config.ENERGY_MIN) or 1.0
            v_norm = (float(energy) - config.ENERGY_MIN) / span
            req_lo = float(filt['energy_min']) if filt.get('energy_min') is not None else 0.0
            req_hi = float(filt['energy_max']) if filt.get('energy_max') is not None else 1.0
            out['energy'] = _range_pref_score(v_norm, max(0.0, min(1.0, req_lo)), max(0.0, min(1.0, req_hi)))

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


def _filter_score(filt: Dict, feats: Dict) -> float:
    """Unnormalized mean of the per-dimension scores for a single song.

    Kept for callers without a pool to normalize against. The composition
    re-rank uses ``_filter_dim_scores`` + per-pool min-max normalization
    instead, so wide-range dims don't dominate narrow ones.
    """
    dim = _filter_dim_scores(filt, feats)
    return sum(dim.values()) / len(dim) if dim else 0.0


def _filter_dimension_report(filt: Dict, feats_map: Dict, pool_songs: List[Dict]):
    """Per-dimension truthful stats for the composition re-rank log.

    Returns (human_lines, machine_dict). Tag dims report how many pool songs
    carry the requested label(s) and the score range, noting dense
    (other_features, every song 0..1) vs sparse (mood_vector top-5, absence=0).
    """
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
        lines.append(f"   genres {filt['genres']} -> mood_vector (top-5, sparse): {nz}/{n} carry it, rest scored 0 (range {lo:.2f}..{hi:.2f})")
        machine['genres'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('voices'):
        nz, lo, hi = _tag_stats(filt['voices'], 'mood_vector')
        lines.append(f"   voices {filt['voices']} -> mood_vector (top-5, sparse): {nz}/{n} carry it, rest scored 0 (range {lo:.2f}..{hi:.2f})")
        machine['voices'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('moods'):
        nz, lo, hi = _tag_stats(filt['moods'], 'other_features')
        lines.append(f"   moods {filt['moods']} -> other_features (dense, every song 0..1): range {lo:.2f}..{hi:.2f}")
        machine['moods'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('other_features'):
        nz, lo, hi = _tag_stats(filt['other_features'], 'other_features')
        lines.append(f"   other_features {filt['other_features']} -> other_features (dense): range {lo:.2f}..{hi:.2f}")
        machine['other_features'] = (nz, round(lo, 2), round(hi, 2))
    if filt.get('energy_min') is not None or filt.get('energy_max') is not None:
        lines.append(f"   energy {filt.get('energy_min', '?')}..{filt.get('energy_max', '?')} -> continuous gradient")
        machine['energy'] = (filt.get('energy_min'), filt.get('energy_max'))
    if filt.get('tempo_min') is not None or filt.get('tempo_max') is not None:
        lines.append(f"   tempo {filt.get('tempo_min', '?')}..{filt.get('tempo_max', '?')} -> continuous gradient")
        machine['tempo'] = (filt.get('tempo_min'), filt.get('tempo_max'))
    if filt.get('year_min') is not None or filt.get('year_max') is not None:
        lines.append(f"   year {filt.get('year_min', '?')}..{filt.get('year_max', '?')} -> proximity gradient")
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


FILTER_LIST_KEYS = ('genres', 'voices', 'moods', 'other_features')
FILTER_MIN_KEYS = ('tempo_min', 'energy_min', 'year_min', 'min_rating')
FILTER_MAX_KEYS = ('tempo_max', 'energy_max', 'year_max')
FILTER_SCALAR_KEYS = ('key', 'scale', 'album', 'artist')
FILTER_ALL_KEYS = (
    FILTER_LIST_KEYS + FILTER_MIN_KEYS + FILTER_MAX_KEYS + FILTER_SCALAR_KEYS
)


@dataclass
class ToolPlan:
    primaries: List[Dict] = field(default_factory=list)
    filter: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_DECADE_RE = re.compile(r"\b(60|70|80|90|00|10|20)s\b", re.IGNORECASE)
_BPM_RE = re.compile(r"\b(\d{2,3})\s*bpm\b", re.IGNORECASE)
_ENERGY_NUM_RE = re.compile(
    r"\benergy\s*(?:above|>=?|over|min(?:imum)?)\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE
)


def _normalize_decade(prefix: str) -> int:
    p = int(prefix)
    if p >= 30:
        return 1900 + p
    return 2000 + p


def extract_hints(text: str) -> Dict:
    """Return a dict of deterministically-extracted hints from raw user input.

    Keys (only present when the corresponding pattern matched):
        years, year_min, year_max, bpm, tempo_min/tempo_max, energy_min/energy_max, notes.
    """
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
            hints['tempo_min'] = tmin if hints.get('tempo_min') is None else min(hints['tempo_min'], tmin)
            hints['tempo_max'] = tmax if hints.get('tempo_max') is None else max(hints['tempo_max'], tmax)
            notes.append(f"tempo phrase '{phrase}' -> {tmin}-{tmax} BPM")
            break

    for phrase, (emin, emax) in ALIAS_ENERGY.items():
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            hints['energy_min'] = emin if hints.get('energy_min') is None else min(hints['energy_min'], emin)
            hints['energy_max'] = emax if hints.get('energy_max') is None else max(hints['energy_max'], emax)
            notes.append(f"energy phrase '{phrase}' -> {emin}-{emax}")
            break

    energy_num = _ENERGY_NUM_RE.search(text)
    if energy_num:
        try:
            v = float(energy_num.group(1))
            if 0.0 <= v <= 1.0:
                hints['energy_min'] = v if hints.get('energy_min') is None else min(hints['energy_min'], v)
                notes.append(f"explicit energy floor: {v}")
        except ValueError:
            pass

    if notes:
        hints['notes'] = notes
    return hints


def format_hints_block(hints: Optional[Dict]) -> str:
    """Render hints as a compact prompt block. Empty string if no hints."""
    if not hints:
        return ""
    lines: List[str] = []
    if hints.get('year_min') is not None or hints.get('year_max') is not None:
        lines.append(
            f"  year: {hints.get('year_min', '?')}..{hints.get('year_max', '?')}"
        )
    if hints.get('bpm') is not None:
        lines.append(f"  bpm: {hints['bpm']}")
    if hints.get('tempo_min') is not None or hints.get('tempo_max') is not None:
        lines.append(
            f"  tempo: {hints.get('tempo_min', '?')}..{hints.get('tempo_max', '?')}"
        )
    if hints.get('energy_min') is not None or hints.get('energy_max') is not None:
        lines.append(
            f"  energy: {hints.get('energy_min', '?')}..{hints.get('energy_max', '?')}"
        )
    if not lines:
        return ""
    return "EXTRACTED_HINTS (use these values directly in search_database if relevant):\n" + "\n".join(lines)


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
            filt['energy_min'] = m['energy_min'] if filt.get('energy_min') is None else min(filt['energy_min'], m['energy_min'])
        if m['energy_max'] is not None:
            filt['energy_max'] = m['energy_max'] if filt.get('energy_max') is None else max(filt['energy_max'], m['energy_max'])
        if m['tempo_min'] is not None:
            filt['tempo_min'] = m['tempo_min'] if filt.get('tempo_min') is None else min(filt['tempo_min'], m['tempo_min'])
        if m['tempo_max'] is not None:
            filt['tempo_max'] = m['tempo_max'] if filt.get('tempo_max') is None else max(filt['tempo_max'], m['tempo_max'])
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


def validate_plan_args(
    tool_calls: List[Dict],
    *,
    user_wants_rating: bool,
    log_messages: Optional[List[str]] = None,
) -> List[Dict]:
    """Pre-execution validation + coercion on a raw tool_calls list.

    Drops or coerces obviously-wrong calls before they reach the executor:
    - seed_search with no usable seeds -> dropped
    - seed_search blend_mode='alchemy' with <2 seeds -> coerced to 'union'
    - seed_search blend_mode='subtract' with empty 'subtract' list -> dropped
    - text_match with empty query -> dropped; unknown mode -> coerced to 'audio'
    - knowledge_lookup with empty user_request -> dropped
    - search_database with year <1900 -> stripped
    - search_database with min_rating but user didn't mention ratings -> stripped
    - search_database with no filters at all -> dropped
    """
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
                if not cleaned_sub:
                    log_messages.append("   skip seed_search(subtract): empty 'subtract' list")
                    continue
                args['subtract'] = cleaned_sub
            args['blend_mode'] = blend

        if name == 'text_match':
            query = (args.get('query') or '').strip()
            if not query:
                log_messages.append(f"   skip {name}: empty query")
                continue
            args['query'] = query
            mode = (args.get('mode') or 'audio').lower()
            if mode not in ('audio', 'lyrics'):
                log_messages.append(f"   coerce text_match mode '{mode}' -> 'audio'")
                mode = 'audio'
            args['mode'] = mode

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

            if not _has_filter_content(args) and not args.get('min_rating') \
                    and args.get('year_min') is None and args.get('year_max') is None:
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


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_first_json_object(text: str) -> Optional[Dict]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    if "```" in candidate:
        candidate = candidate.split("```", 1)[0].strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_classifier_result(parsed: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(parsed, dict):
        return None
    intent = parsed.get("intent")
    if not isinstance(intent, str):
        return None
    intent = intent.strip().lower()
    if intent not in INTENT_CLASSES:
        return None
    needs_filter = parsed.get("needs_filter", False)
    if isinstance(needs_filter, str):
        needs_filter = needs_filter.strip().lower() in ("true", "1", "yes")
    return {"intent": intent, "needs_filter": bool(needs_filter)}


def classify(
    user_message: str,
    ai_config: Dict,
    log_messages: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Stage-1 classifier: returns {"intent": <class>, "needs_filter": bool} or None on any failure."""
    if log_messages is None:
        log_messages = []

    provider = (ai_config.get("provider") or "NONE").upper()
    if provider == "NONE":
        return None

    prompt = build_intent_classifier_prompt(user_message)

    try:
        from tasks.ai.api import generate_text
        # NOTE: do NOT pass a low max_tokens here. A reasoning model (e.g. qwen3.5
        # on OpenRouter) spends tokens thinking first; a tight cap truncates it
        # before the JSON answer -> empty -> retry loop. Keep the generous default.
        response = generate_text(prompt, ai_config, skip_delay=True, temperature=0.0)
    except Exception as e:
        logger.warning("intent_classifier transport error: %s", e)
        log_messages.append(f"intent_classifier: transport error ({e}), falling back to full tools")
        return None

    if not isinstance(response, str) or response.startswith("Error"):
        log_messages.append(f"intent_classifier: provider returned error ({response[:120] if response else 'empty'}), falling back to full tools")
        return None

    parsed = _extract_first_json_object(response)
    result = _normalize_classifier_result(parsed)
    if result is None:
        log_messages.append(f"intent_classifier: could not parse a valid JSON intent from response: {response[:160]!r}")
        return None

    log_messages.append(
        f"intent_classifier: intent={result['intent']}, needs_filter={result['needs_filter']}"
    )
    return result


def tools_for_intent(intent: str, needs_filter: bool, all_tools: List[Dict]) -> List[Dict]:
    """Filter the full tool list down to the subset the Stage-2 call should see."""
    by_name = {t.get("name"): t for t in all_tools if isinstance(t, dict)}

    primary_map = {
        "seed": "seed_search",
        "text": "text_match",
        "knowledge": "knowledge_lookup",
        "metadata": "search_database",
    }
    primary = primary_map.get(intent)

    chosen: List[Dict] = []
    if primary and primary in by_name:
        chosen.append(by_name[primary])
    if intent != "metadata" and needs_filter and "search_database" in by_name:
        chosen.append(by_name["search_database"])

    return chosen if chosen else list(all_tools)


def _run_search_database_with_relax(
    filter_args: Dict,
    ai_config: Dict,
    target_count: int,
    log_messages: List[str],
) -> Dict:
    """Run search_database, progressively relaxing the score threshold until target_count is met.

    Loops ``RELAX_THRESHOLD_STEPS`` (0.5 -> 0.4 -> 0.3 -> 0.2). Returns the LAST
    result that still meets target OR the last step's result (whichever is closer
    to target). Only relaxes when the filter actually uses a scored column
    (genres / voices / moods / other_features); otherwise runs once at the
    default threshold.
    """
    from tasks.ai.tools import execute_mcp_tool

    has_scored = any(filter_args.get(k) for k in SCORED_FILTER_KEYS)
    if not has_scored:
        return execute_mcp_tool('search_database', filter_args, ai_config)

    last_result: Optional[Dict] = None
    for step_threshold in RELAX_THRESHOLD_STEPS:
        args = dict(filter_args)
        args['score_threshold'] = step_threshold
        result = execute_mcp_tool('search_database', args, ai_config)
        if 'error' in result:
            return result
        songs = result.get('songs', [])
        log_messages.append(
            f"   relax: score_threshold={step_threshold} -> {len(songs)} songs"
        )
        last_result = result
        if len(songs) >= target_count:
            return result
    return last_result if last_result is not None else {"songs": [], "message": "relax loop produced no result"}


def call_ai_for_plan(
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    """Call the AI transport once and return the raw tool-calling result."""
    from tasks.ai.api import call_with_tools as _call_with_tools
    return _call_with_tools(
        user_message=user_message,
        tools=tools,
        ai_config=ai_config,
        log_messages=log_messages,
        library_context=library_context,
    )


def plan_and_execute(
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    """Call the AI, classify the response into a ToolPlan, and return both.

    Returns ``{"plan": ToolPlan, "tool_calls": [...], "raw": <ai_result>}`` on
    success, ``{"error": "..."}`` on AI failure.
    """
    raw = call_ai_for_plan(user_message, tools, ai_config, log_messages, library_context)
    if 'error' in raw:
        return raw

    tool_calls = raw.get('tool_calls', []) or []
    plan = plan_from_tool_calls(tool_calls)

    if not plan.primaries and plan.filter is None and tool_calls:
        log_messages.append('Plan: AI returned tool_calls but none were valid primaries or filters; retrying once with explicit guidance')
        retry_msg = (
            f"{user_message}\n\n"
            "IMPORTANT: Your previous response had no valid tool calls. "
            "Emit at least ONE tool call. Pick a PRIMARY (seed_search / text_match / "
            "knowledge_lookup) or a FILTER (search_database with at least one of "
            "genres / voices / moods / year / artist / album / scale / tempo / energy / rating)."
        )
        raw = call_ai_for_plan(retry_msg, tools, ai_config, log_messages, library_context)
        if 'error' in raw:
            return raw
        tool_calls = raw.get('tool_calls', []) or []
        plan = plan_from_tool_calls(tool_calls)

    return {"plan": plan, "tool_calls": tool_calls, "raw": raw}


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
) -> Dict:
    """Two-stage orchestrator: classifier narrows tools -> single AI plan -> execute.

    Returns ``{songs, song_sources, tools_used_history, tool_execution_summary,
    detected_min_rating, plan_notes, executed_query_str, filter_applied}`` or
    ``{"error": ...}``. Progress streams to the caller via ``log_messages.append``
    (the streaming endpoint passes a queue-backed list that pushes each line to the
    client as it is added).
    """
    from tasks.ai.tools import execute_mcp_tool
    from tasks.ai.tool_impl import _fetch_pool_features

    log_messages.append("\n--- AI Decision (two-stage) ---")

    original_user_message = user_message

    classification = classify(user_message, ai_config, log_messages=log_messages)
    if classification is not None:
        narrowed = tools_for_intent(classification['intent'], classification['needs_filter'], tools)
        if narrowed and len(narrowed) < len(tools):
            kept = ", ".join(t.get('name', '') for t in narrowed)
            log_messages.append(f"   stage-2 tools narrowed to: {kept}")
            tools = narrowed
        else:
            log_messages.append("   stage-2 tools: keeping full surface (no narrowing)")
    else:
        log_messages.append("   classifier returned no intent; using full 4-tool surface for stage-2")

    if classification is not None and classification['intent'] == 'knowledge':
        user_message = (
            "INTENT=knowledge. You MUST emit a knowledge_lookup tool call with "
            "user_request set to the user's request below. Add a search_database "
            "call ONLY for explicit metadata constraints the user mentioned.\n\n"
            f"{user_message}"
        )

    hints = extract_hints(original_user_message)
    hints_block = format_hints_block(hints)
    if hints_block:
        for n in hints.get('notes', []):
            log_messages.append(f"   pre-extract: {n}")
        user_message = f"{user_message}\n\n{hints_block}"
    raw = call_ai_for_plan(user_message, tools, ai_config, log_messages, library_context)
    if 'error' in raw:
        return {"error": raw['error']}

    raw_calls = raw.get('tool_calls', []) or []
    log_messages.append(f"AI emitted {len(raw_calls)} tool call(s)")

    raw_calls = validate_plan_args(
        raw_calls, user_wants_rating=user_wants_rating, log_messages=log_messages
    )

    if classification is not None and classification['intent'] == 'knowledge':
        has_knowledge = any(
            isinstance(tc, dict) and tc.get('name') == 'knowledge_lookup'
            for tc in raw_calls
        )
        if not has_knowledge:
            log_messages.append(
                "   intent=knowledge but no knowledge_lookup in plan -- injecting one"
            )
            raw_calls.insert(0, {
                'name': 'knowledge_lookup',
                'arguments': {'user_request': original_user_message, 'get_songs': 200},
            })

    if not raw_calls:
        return {"error": "No valid tool calls after validation"}

    plan = validate_and_normalize_plan(raw_calls)
    for note in plan.notes:
        log_messages.append(f"   plan: {note}")
    if not plan.primaries and plan.filter is None:
        return {"error": "Plan was empty after normalization"}

    # HARD RULE: never apply a filter/re-rank to AI brainstorming. Brainstormed
    # (knowledge_lookup) songs are returned exactly as suggested.
    if plan.filter is not None and any(
            isinstance(p, dict) and p.get('name') == 'knowledge_lookup' for p in plan.primaries):
        log_messages.append("   AI brainstorming: filter NOT applied — knowledge results returned as-is")
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
            for k in ('genres', 'voices', 'moods'):
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
                        seed_summary.append(f"song:'{s.get('title','')}'")
                    elif s.get('type') == 'artist':
                        seed_summary.append(f"artist:'{s.get('name','')}'")
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
            for s in songs:
                iid = s.get('item_id')
                if iid and iid not in pool_ids:
                    pool_songs.append(s)
                    pool_ids.add(iid)
                    added_to_pool += 1
            log_messages.append(f"   pooled {added_to_pool}/{len(songs)} unique (pool={len(pool_songs)})")
            primary_logs.append((tn, ta, added_to_pool, False, res.get('message', '')))

        if not pool_songs:
            for (tn, ta, _added, errored, msg) in primary_logs:
                tools_used_history.append({
                    'name': tn, 'args': ta, 'songs': 0, 'error': errored,
                    'call_index': tool_call_counter, 'result_message': msg,
                })
                tool_execution_summary.append(_summary(tn, ta, 0))
                tool_call_counter += 1
            return {
                "songs": [],
                "song_sources": {},
                "tools_used_history": tools_used_history,
                "tool_execution_summary": tool_execution_summary,
                "detected_min_rating": detected_min_rating,
                "plan_notes": plan.notes,
                "executed_query_str": f"MCP single-pass ({len(tools_used_history)} tools): {' -> '.join(tool_execution_summary)}",
                "filter_applied": plan.filter is not None,
            }

        N = len(pool_songs)
        feats = _fetch_pool_features([s['item_id'] for s in pool_songs])
        clean_filter = {k: v for k, v in plan.filter.items() if k not in ('candidate_item_ids', 'get_songs')}

        log_messages.append(f"\nFILTER (priority re-rank): {N} songs from seed pool")
        log_messages.append(f"   filter applied: {clean_filter}")
        dim_lines, _dim_machine = _filter_dimension_report(plan.filter, feats, pool_songs)
        for ln in dim_lines:
            log_messages.append(ln)

        raw_dims = [_filter_dim_scores(plan.filter, feats.get(s['item_id'], {})) for s in pool_songs]

        dim_keys = sorted({k for d in raw_dims for k in d})
        dim_min = {k: min((d.get(k, 0.0) for d in raw_dims), default=0.0) for k in dim_keys}
        dim_max = {k: max((d.get(k, 0.0) for d in raw_dims), default=0.0) for k in dim_keys}

        def _norm(k, v):
            lo, hi = dim_min[k], dim_max[k]
            return (v - lo) / (hi - lo) if hi > lo else 0.0

        cat_keys = [k for k in dim_keys if k in CATEGORICAL_DIMS]
        cont_keys = [k for k in dim_keys if k not in CATEGORICAL_DIMS]

        def _cont_score(d):
            if not cont_keys:
                return 0.0
            return sum(_norm(k, d.get(k, 0.0)) for k in cont_keys) / len(cont_keys)

        def _cat_count(d):
            return sum(1 for k in cat_keys if d.get(k, 0.0) > 0)

        def _cat_conf(d):
            return sum(_norm(k, d.get(k, 0.0)) for k in cat_keys)

        if dim_keys:
            norm_summary = ", ".join(f"{k}[{dim_min[k]:.2f}..{dim_max[k]:.2f}]" for k in dim_keys)
            log_messages.append(f"   per-dim pool range (each normalized 0..1 for the blend): {norm_summary}")

        if cat_keys:
            # Tiered: songs matching the requested categorical(s) rank above those
            # that don't; the continuous dims (and categorical confidence) only
            # order songs WITHIN each tier. Soft -- non-matching songs backfill.
            matched = sum(1 for d in raw_dims if _cat_count(d) > 0)
            order = sorted(
                range(N),
                key=lambda i: (_cat_count(raw_dims[i]), _cont_score(raw_dims[i]), _cat_conf(raw_dims[i])),
                reverse=True,
            )
            final = [pool_songs[i] for i in order]
            moved = sum(1 for new_i, old_i in enumerate(order) if new_i != old_i)
            cat_label = ", ".join(cat_keys)
            cont_label = ", ".join(cont_keys) if cont_keys else "similarity"
            if matched == 0:
                log_messages.append(
                    f"   re-rank: 0/{N} match the requested {cat_label}; "
                    f"all ordered by {cont_label}"
                )
            else:
                log_messages.append(
                    f"   re-rank: {matched}/{N} match the requested {cat_label} and rank first; "
                    f"remaining ordered by {cont_label} (categorical priority, then gradient)"
                )
        else:
            # Continuous-only: blend the normalized gradient dims (unchanged).
            matched = sum(1 for d in raw_dims if any(v > 0 for v in d.values()))
            fscores = [_cont_score(d) for d in raw_dims]
            if matched == 0:
                final = list(pool_songs)
                order = list(range(N))
                moved = 0
                log_messages.append(f"   re-rank: 0/{N} songs matched the filter -> order UNCHANGED (pure similarity)")
            else:
                order = sorted(range(N), key=lambda i: fscores[i], reverse=True)
                final = [pool_songs[i] for i in order]
                moved = sum(1 for new_i, old_i in enumerate(order) if new_i != old_i)
                if moved == 0:
                    log_messages.append(f"   re-rank: {matched}/{N} matched but scores tied -> no song changed position")
                else:
                    log_messages.append(
                        f"   re-rank: {matched}/{N} matched the filter and rose to the top; "
                        f"{moved} songs shifted position vs pure similarity order "
                        f"(per-dim normalized then averaged, higher score = higher rank)"
                    )

        logger.info(
            "composition re-rank: pool=%d matched=%d moved=%d filter=%s dim_range=%s",
            N, matched, moved, clean_filter,
            {k: (round(dim_min[k], 2), round(dim_max[k], 2)) for k in dim_keys},
        )

        for (tn, ta, pooled, errored, msg) in primary_logs:
            tools_used_history.append({
                'name': tn, 'args': ta, 'songs': pooled, 'error': errored,
                'call_index': tool_call_counter, 'result_message': msg,
            })
            tool_execution_summary.append(_summary(tn, ta, pooled))
            tool_call_counter += 1

        filter_call_index = tool_call_counter
        added = _add_songs(final, filter_call_index)
        tools_used_history.append({
            'name': 'search_database', 'args': dict(plan.filter), 'songs': added,
            'call_index': filter_call_index,
            'result_message': f"priority re-rank: {matched}/{N} matched filter",
        })
        tool_execution_summary.append(_summary('search_database', plan.filter, added))
        tool_call_counter += 1

    else:
        all_calls: List[Dict] = list(plan.primaries)
        if plan.filter is not None:
            all_calls.append({'name': 'search_database', 'arguments': dict(plan.filter)})
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
                res = _run_search_database_with_relax(ta, ai_config, target_song_count, log_messages)
            else:
                res = execute_mcp_tool(tn, ta, ai_config)
            if 'error' in res:
                log_messages.append(f"   error: {res['error']}")
                tools_used_history.append({
                    'name': tn, 'args': ta, 'songs': 0, 'error': True,
                    'call_index': tool_call_counter,
                    'result_message': res.get('error', ''),
                })
                tool_execution_summary.append(_summary(tn, ta, 0))
                tool_call_counter += 1
                continue
            songs = res.get('songs', [])
            if res.get('message'):
                for line in res['message'].split('\n'):
                    if line.strip():
                        log_messages.append(f"   {line}")
            added = _add_songs(songs, tool_call_counter)
            log_messages.append(f"   retrieved {len(songs)} songs, added {added} new")
            tools_used_history.append({
                'name': tn, 'args': ta, 'songs': added,
                'call_index': tool_call_counter,
                'result_message': res.get('message', ''),
            })
            tool_execution_summary.append(_summary(tn, ta, added))
            tool_call_counter += 1
            if len(all_songs) >= collection_cap:
                log_messages.append(f"collection cap {collection_cap} reached, stopping")
                break

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
