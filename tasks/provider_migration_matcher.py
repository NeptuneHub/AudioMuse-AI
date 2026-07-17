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
* ``CandidateIndex`` builds the target-side lookups once from slim copies of
  the candidate rows so callers can stream their own rows through
  ``match_chunk`` in bounded-memory chunks, with a shared claimed-id set
  keeping one provider track mapped to at most one canonical row.
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


_TIERS = ('path', 'tail', 'exact_meta', 'norm_meta')
_OPT_TIER_TITLE_ARTIST = 'title_artist'


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


def old_paths(old):
    """Every path known for this catalogue row, across ALL servers that hold it.

    The path is a property of a FILE ON A SERVER, so one song can legitimately
    have a different path on each server. Matching a new server against only ONE
    of them (historically the default server's) left the path and tail tiers with
    no evidence at all for any track the default did not happen to have.
    """
    paths = old.get('file_paths')
    if paths:
        return [p for p in paths if p]
    single = old.get('file_path')
    return [single] if single else []


def _pick_meta_candidate(old, candidates):
    if len(candidates) == 1:
        return candidates[0]
    for path in old_paths(old):
        old_dt = extract_disc_track(path)
        if old_dt is None:
            continue
        for c in candidates:
            if extract_disc_track(c.get('path')) == old_dt:
                return c
    return candidates[0]


class CandidateIndex:
    """Build-once lookup structures over a target catalogue.

    Stores slim copies of the candidate rows (id, path, and the metadata keys
    the tiers compare) so the caller can release the full fetched catalogue,
    then matches any number of row chunks via ``match_chunk`` without holding
    them all in memory at once.
    """

    def __init__(self, new_tracks, allow_title_artist_only=False):
        self.tiers = list(_TIERS)
        self._allow_title_artist_only = allow_title_artist_only
        if allow_title_artist_only:
            self.tiers.append(_OPT_TIER_TITLE_ARTIST)
        self._tier_rank = {t: i for i, t in enumerate(self.tiers)}
        self.by_norm_path = {}
        self.by_tail = {}
        self.by_exact_meta = {}
        self.by_norm_meta = {}
        self.by_title_artist = {}
        self.path_by_id = {}
        self.size = 0
        for n in new_tracks:
            self.add(n)

    def add(self, n):
        slim = {
            'id': n['id'],
            'path': n.get('path'),
            'title': n.get('title'),
            'artist': n.get('artist'),
            'album_artist': n.get('album_artist'),
            'album': n.get('album'),
        }
        self.size += 1
        if slim['path']:
            self.path_by_id[str(slim['id'])] = slim['path']
        np = normalize_path(slim['path'])
        if np and np not in self.by_norm_path:
            self.by_norm_path[np] = slim['id']
        tk = path_tail_key(np)
        if tk and tk not in self.by_tail:
            self.by_tail[tk] = slim['id']
        ek = _new_exact_meta_key(slim)
        if ek:
            self.by_exact_meta.setdefault(ek, []).append(slim)
        nk = _new_norm_meta_key(slim)
        if nk:
            self.by_norm_meta.setdefault(nk, []).append(slim)
        if self._allow_title_artist_only:
            tak = _new_title_artist_key(slim)
            if tak:
                self.by_title_artist.setdefault(tak, []).append(slim)

    def _propose(self, old):
        norm_paths = [p for p in (normalize_path(p) for p in old_paths(old)) if p]
        for np in norm_paths:
            if np in self.by_norm_path:
                return ('path', self.by_norm_path[np])
        for np in norm_paths:
            tk = path_tail_key(np)
            if tk and tk in self.by_tail:
                return ('tail', self.by_tail[tk])
        ek = _old_exact_meta_key(old)
        if ek and ek in self.by_exact_meta:
            return ('exact_meta', _pick_meta_candidate(old, self.by_exact_meta[ek])['id'])
        nk = _old_norm_meta_key(old)
        if nk and nk in self.by_norm_meta:
            return ('norm_meta', _pick_meta_candidate(old, self.by_norm_meta[nk])['id'])
        if self._allow_title_artist_only:
            tak = _old_title_artist_key(old)
            if tak and tak in self.by_title_artist:
                return (
                    _OPT_TIER_TITLE_ARTIST,
                    _pick_meta_candidate(old, self.by_title_artist[tak])['id'],
                )
        return (None, None)

    def match_chunk(self, old_rows, claimed_new_ids=None):
        """Match ``old_rows`` against the index and return the usual result dict.

        ``claimed_new_ids`` (mutated in place when given) carries the provider
        track ids already assigned by EARLIER chunks, as ``{new_id: tier_rank}``,
        so one provider track never maps to two canonical rows (the unique DB
        constraint). A later chunk holding a STRICTLY BETTER tier still takes it:
        the upsert moves the mapping and the weaker previous owner is simply left
        unmapped for a later sweep. Without the rank, the first chunk to see a
        provider track would own it forever - a normalized-metadata guess made in
        chunk 1 would permanently outrank an exact path match in chunk 2, and the
        best-match tie-break (which only looks inside one chunk) would never see
        the pair.
        """
        claimed = claimed_new_ids if claimed_new_ids is not None else {}
        proposals = []
        for old in old_rows:
            tier, new_id = self._propose(old)
            if tier is not None and new_id in claimed:
                if self._tier_rank[tier] >= claimed[new_id]:
                    tier, new_id = None, None
            proposals.append((tier, old, new_id))

        best_for_new = {}
        for tier, old, new_id in proposals:
            if tier is None:
                continue
            cur = best_for_new.get(new_id)
            if cur is None or self._tier_rank[tier] < self._tier_rank[cur[0]]:
                best_for_new[new_id] = (tier, old)

        winners = {id(old): new_id for new_id, (_tier, old) in best_for_new.items()}

        matches = {}
        match_tiers = {}
        tier_counts = {t: 0 for t in self.tiers}
        unmatched = []
        for tier, old, new_id in proposals:
            if tier is not None and winners.get(id(old)) == new_id:
                matches[old['item_id']] = new_id
                match_tiers[old['item_id']] = tier
                tier_counts[tier] += 1
                claimed[new_id] = self._tier_rank[tier]
            else:
                unmatched.append(old)

        return {
            'matches': matches,
            'match_tiers': match_tiers,
            'tier_counts': tier_counts,
            'unmatched': unmatched,
        }


def match_tracks(old_rows, new_tracks, allow_title_artist_only=False):
    index = CandidateIndex(new_tracks, allow_title_artist_only=allow_title_artist_only)
    result = index.match_chunk(old_rows)

    unmatched_by_album = {}
    for old in result['unmatched']:
        key = (old.get('album_artist') or old.get('author'), old.get('album'))
        unmatched_by_album.setdefault(key, []).append(old)

    result['unmatched_by_album'] = unmatched_by_album
    return result
