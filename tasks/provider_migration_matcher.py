# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Match existing tracks to a target provider's library during migration.

Pure matching helpers used by the provider-migration orchestration; the
per-provider track fetching lives elsewhere and is not touched here.

Main Features:
* Path-normalisation that strips a wide set of common mount prefixes and
  file:// URLs so paths from different servers compare on their library tails.
* Tiered matching (normalised path, path tail, exact metadata, normalised
  metadata, and an optional title+artist fallback) with disc/track
  disambiguation when several candidates share a metadata key.
"""

import re
from urllib.parse import unquote


_MOUNT_PREFIXES_TO_STRIP = (
    '/media/music/',
    '/media/media/',
    '/media/',
    '/mnt/media/music/',
    '/mnt/media/',
    '/mnt/music/',
    '/mnt/data/music/',
    '/mnt/data/',
    '/mnt/',
    '/data/music/',
    '/data/',
    '/music/',
    '/share/music/',
    '/share/',
    '/volume1/music/',
    '/volume1/',
    '/srv/music/',
    '/srv/',
    '/home/music/',
    '/storage/music/',
    '/opt/music/',
    '/nas/music/',
    '/library/music/',
)


def normalize_path(raw):
    if not raw:
        return None
    p = str(raw)
    if p.startswith('file://'):
        p = unquote(p[len('file://') :])
    p = p.replace('\\', '/').lower()
    for prefix in _MOUNT_PREFIXES_TO_STRIP:
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p.lstrip('/')


def path_tail_key(path, n=3):
    if not path:
        return None
    p = str(path).replace('\\', '/').strip('/').lower()
    if not p:
        return None
    parts = p.split('/')
    if len(parts) < 2:
        return None
    tail = parts[-n:] if len(parts) >= n else parts
    return '/'.join(tail)


_DISC_TRACK_RE = re.compile(r'^(\d+)[\s._-]+(\d+)(?=\D|$)')


def extract_disc_track(path):
    if not path:
        return None
    p = str(path).replace('\\', '/')
    basename = p.rsplit('/', 1)[-1]
    m = _DISC_TRACK_RE.match(basename)
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


_META_NOISE_WORDS = (
    'remaster',
    'remastered',
    'feat',
    'ft',
    'featuring',
    'explicit',
    'clean',
    'radio edit',
    'radio version',
    'single version',
    'album version',
    'extended',
    'club mix',
    'acoustic',
    'live',
    'demo',
    'version',
    'mix',
)
_META_NOISE_ALT = '|'.join(re.escape(w) for w in _META_NOISE_WORDS)
_META_NOISE_PAREN_RE = re.compile(r'\s*\([^)]*(?:' + _META_NOISE_ALT + r')[^)]*\)', re.IGNORECASE)
_META_NOISE_BRACKET_RE = re.compile(
    r'\s*\[[^\]]*(?:' + _META_NOISE_ALT + r')[^\]]*\]', re.IGNORECASE
)
_LEADING_THE_RE = re.compile(r'^the\s+', re.IGNORECASE)
_COLLAPSE_WS_RE = re.compile(r'\s+')


def normalize_meta(s):
    if not s:
        return ''
    out = str(s).lower()
    out = _META_NOISE_PAREN_RE.sub('', out)
    out = _META_NOISE_BRACKET_RE.sub('', out)
    out = _LEADING_THE_RE.sub('', out)
    out = _COLLAPSE_WS_RE.sub(' ', out).strip()
    return out


_TIERS = ('fingerprint', 'mbid', 'path', 'tail', 'exact_meta', 'norm_meta')
_OPT_TIER_TITLE_ARTIST = 'title_artist'


def normalize_fingerprint(raw):
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def normalize_mbid(raw):
    if not raw:
        return None
    m = str(raw).strip().lower()
    return m or None


def _best_artist_old(row):
    return row.get('author') or row.get('artist') or row.get('album_artist')


def _best_artist_new(row):
    return row.get('artist') or row.get('album_artist')


def _old_exact_meta_key(old):
    t = (old.get('title') or '').lower()
    a = (_best_artist_old(old) or '').lower()
    alb = (old.get('album') or '').lower()
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _new_exact_meta_key(new):
    t = (new.get('title') or '').lower()
    a = (_best_artist_new(new) or '').lower()
    alb = (new.get('album') or '').lower()
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _old_norm_meta_key(old):
    t = normalize_meta(old.get('title'))
    a = normalize_meta(_best_artist_old(old))
    alb = normalize_meta(old.get('album'))
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _new_norm_meta_key(new):
    t = normalize_meta(new.get('title'))
    a = normalize_meta(_best_artist_new(new))
    alb = normalize_meta(new.get('album'))
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _old_title_artist_key(old):
    t = normalize_meta(old.get('title'))
    a = normalize_meta(_best_artist_old(old))
    if not (t and a):
        return None
    return (t, a)


def _new_title_artist_key(new):
    t = normalize_meta(new.get('title'))
    a = normalize_meta(_best_artist_new(new))
    if not (t and a):
        return None
    return (t, a)


def match_tracks(old_rows, new_tracks, allow_title_artist_only=False):
    tiers = list(_TIERS)
    if allow_title_artist_only:
        tiers.append(_OPT_TIER_TITLE_ARTIST)

    by_fingerprint = {}
    by_mbid = {}
    by_norm_path = {}
    by_tail = {}
    by_exact_meta = {}
    by_norm_meta = {}
    by_title_artist = {}
    for n in new_tracks:
        fp = normalize_fingerprint(n.get('fingerprint'))
        if fp is not None and fp not in by_fingerprint:
            by_fingerprint[fp] = n['id']
        mb = normalize_mbid(n.get('mbid'))
        if mb and mb not in by_mbid:
            by_mbid[mb] = n['id']
        np = normalize_path(n.get('path'))
        if np and np not in by_norm_path:
            by_norm_path[np] = n['id']
        tk = path_tail_key(np)
        if tk and tk not in by_tail:
            by_tail[tk] = n['id']
        ek = _new_exact_meta_key(n)
        if ek:
            by_exact_meta.setdefault(ek, []).append(n)
        nk = _new_norm_meta_key(n)
        if nk:
            by_norm_meta.setdefault(nk, []).append(n)
        if allow_title_artist_only:
            tak = _new_title_artist_key(n)
            if tak:
                by_title_artist.setdefault(tak, []).append(n)

    tier_counts = {t: 0 for t in tiers}
    tier_rank = {t: i for i, t in enumerate(tiers)}

    def _pick_meta_candidate(old, candidates):
        if len(candidates) == 1:
            return candidates[0]
        old_dt = extract_disc_track(old.get('file_path'))
        if old_dt is not None:
            for c in candidates:
                if extract_disc_track(c.get('path')) == old_dt:
                    return c
        return candidates[0]

    proposals = []
    for old in old_rows:
        matched = False
        fp = normalize_fingerprint(old.get('fingerprint'))
        if fp is not None and fp in by_fingerprint:
            proposals.append(('fingerprint', old, by_fingerprint[fp]))
            matched = True
            continue
        mb = normalize_mbid(old.get('mbid'))
        if mb and mb in by_mbid:
            proposals.append(('mbid', old, by_mbid[mb]))
            matched = True
            continue
        np = normalize_path(old.get('file_path'))
        if np and np in by_norm_path:
            proposals.append(('path', old, by_norm_path[np]))
            matched = True
            continue
        tk = path_tail_key(np) if np else None
        if tk and tk in by_tail:
            proposals.append(('tail', old, by_tail[tk]))
            matched = True
            continue
        ek = _old_exact_meta_key(old)
        if ek and ek in by_exact_meta:
            chosen = _pick_meta_candidate(old, by_exact_meta[ek])
            proposals.append(('exact_meta', old, chosen['id']))
            matched = True
            continue
        nk = _old_norm_meta_key(old)
        if nk and nk in by_norm_meta:
            chosen = _pick_meta_candidate(old, by_norm_meta[nk])
            proposals.append(('norm_meta', old, chosen['id']))
            matched = True
            continue
        if allow_title_artist_only:
            tak = _old_title_artist_key(old)
            if tak and tak in by_title_artist:
                chosen = _pick_meta_candidate(old, by_title_artist[tak])
                proposals.append((_OPT_TIER_TITLE_ARTIST, old, chosen['id']))
                matched = True
                continue
        if not matched:
            proposals.append((None, old, None))

    best_for_new = {}
    for tier, old, new_id in proposals:
        if tier is None:
            continue
        cur = best_for_new.get(new_id)
        if cur is None or tier_rank[tier] < tier_rank[cur[0]]:
            best_for_new[new_id] = (tier, old)

    winners = {id(old): new_id for new_id, (_tier, old) in best_for_new.items()}

    matches = {}
    match_tiers = {}
    unmatched = []
    for tier, old, new_id in proposals:
        if tier is not None and winners.get(id(old)) == new_id:
            matches[old['item_id']] = new_id
            match_tiers[old['item_id']] = tier
            tier_counts[tier] += 1
        else:
            unmatched.append(old)

    unmatched_by_album = {}
    for old in unmatched:
        key = (old.get('album_artist') or old.get('author'), old.get('album'))
        unmatched_by_album.setdefault(key, []).append(old)

    return {
        'matches': matches,
        'match_tiers': match_tiers,
        'tier_counts': tier_counts,
        'unmatched': unmatched,
        'unmatched_by_album': unmatched_by_album,
    }
