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
* One song held as N files keeps all N: every other known path of a matched
  row that lands exactly on a target file (path, then path tail) is returned
  in ``extra_matches``, from the lookups the index already built, so a sweep
  or a provider migration never collapses duplicate files to one mapping.
  An extra only takes a file no row has claimed and claims it below every
  tier, so it never unbinds another song and any song's own match takes it back.
* Several target files under one path key are told apart by an exact full path,
  then by the longest shared path suffix; a key still tied is used only when
  every tied file is the same song (title and artist), otherwise it falls
  through to the metadata tiers.
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


def _path_parts(raw):
    if not raw:
        return []
    p = str(raw)
    if p.startswith('file://'):
        p = unquote(p[len('file://') :])
    return [part for part in p.replace('\\', '/').lower().split('/') if part]


def path_match_strength(old_path, new_path):
    a = _path_parts(old_path)
    b = _path_parts(new_path)
    shared = 0
    while shared < len(a) and shared < len(b) and a[-1 - shared] == b[-1 - shared]:
        shared += 1
    return shared, bool(a) and shared == len(a) == len(b)


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


_LEADING_NUMBER_RE = re.compile(r'^(\d+)(?=\D|$)')


def extract_track_number(path):
    disc_track = extract_disc_track(path)
    if disc_track is not None:
        return disc_track[1]
    if not path:
        return None
    m = _LEADING_NUMBER_RE.match(str(path).replace('\\', '/').rsplit('/', 1)[-1])
    return int(m.group(1)) if m else None


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


def _exact_meta_key(row, best_artist):
    t = (row.get('title') or '').lower()
    a = (best_artist(row) or '').lower()
    alb = (row.get('album') or '').lower()
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _norm_meta_key(row, best_artist):
    t = normalize_meta(row.get('title'))
    a = normalize_meta(best_artist(row))
    alb = normalize_meta(row.get('album'))
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _title_artist_key(row, best_artist):
    t = normalize_meta(row.get('title'))
    a = normalize_meta(best_artist(row))
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
    for path in old_paths(old):
        old_track = extract_track_number(path)
        if old_track is None:
            continue
        same_number = [c for c in candidates if extract_track_number(c.get('path')) == old_track]
        if len(same_number) == 1:
            return same_number[0]
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
        self._slim_by_id = {}
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
            key = str(slim['id'])
            self.path_by_id[key] = slim['path']
            self._slim_by_id[key] = slim
        np = normalize_path(slim['path'])
        self._add_key(self.by_norm_path, np, slim['id'])
        self._add_key(self.by_tail, path_tail_key(np), slim['id'])
        ek = _exact_meta_key(slim, _best_artist_new)
        if ek:
            self.by_exact_meta.setdefault(ek, []).append(slim)
        nk = _norm_meta_key(slim, _best_artist_new)
        if nk:
            self.by_norm_meta.setdefault(nk, []).append(slim)
        if self._allow_title_artist_only:
            tak = _title_artist_key(slim, _best_artist_new)
            if tak:
                self.by_title_artist.setdefault(tak, []).append(slim)

    @staticmethod
    def _add_key(lookup, key, new_id):
        if not key:
            return
        held = lookup.get(key)
        if held is None:
            lookup[key] = new_id
        elif type(held) is list:
            if new_id not in held:
                held.append(new_id)
        elif held != new_id:
            lookup[key] = [held, new_id]

    @staticmethod
    def _candidates(lookup, key):
        held = lookup.get(key) if key else None
        if held is None:
            return []
        return held if type(held) is list else [held]

    def _closest(self, old_path, candidates):
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        scored = [
            (path_match_strength(old_path, self.path_by_id.get(str(c))), c) for c in candidates
        ]
        exact = [c for (_shared, is_exact), c in scored if is_exact]
        if exact:
            return exact[0]
        best = max(shared for (shared, _exact), _c in scored)
        tied = [c for (shared, _exact), c in scored if shared == best]
        if len(tied) == 1:
            return tied[0]
        songs = {
            _title_artist_key(self._slim_by_id.get(str(c)) or {}, _best_artist_new) for c in tied
        }
        return tied[0] if len(songs) == 1 and None not in songs else None

    def _propose(self, old):
        paths = [(raw, normalize_path(raw)) for raw in old_paths(old)]
        for raw, np in paths:
            candidates = self._candidates(self.by_norm_path, np)
            if candidates:
                chosen = self._closest(raw, candidates)
                if chosen is not None:
                    return ('path', chosen)
        for raw, np in paths:
            chosen = self._closest(raw, self._candidates(self.by_tail, path_tail_key(np)))
            if chosen is not None:
                return ('tail', chosen)
        ek = _exact_meta_key(old, _best_artist_old)
        if ek and ek in self.by_exact_meta:
            return ('exact_meta', _pick_meta_candidate(old, self.by_exact_meta[ek])['id'])
        nk = _norm_meta_key(old, _best_artist_old)
        if nk and nk in self.by_norm_meta:
            return ('norm_meta', _pick_meta_candidate(old, self.by_norm_meta[nk])['id'])
        if self._allow_title_artist_only:
            tak = _title_artist_key(old, _best_artist_old)
            if tak and tak in self.by_title_artist:
                return (
                    _OPT_TIER_TITLE_ARTIST,
                    _pick_meta_candidate(old, self.by_title_artist[tak])['id'],
                )
        return (None, None)

    def _path_siblings(self, old):
        seen = set()
        for raw in old_paths(old):
            key = '/'.join(_path_parts(raw))
            if not key or key in seen:
                continue
            seen.add(key)
            np = normalize_path(raw)
            candidates = self._candidates(self.by_norm_path, np)
            if candidates:
                chosen, tier = self._closest(raw, candidates), 'path'
            else:
                chosen, tier = self._closest(raw, self._candidates(self.by_tail, path_tail_key(np))), 'tail'
            if chosen is not None:
                yield tier, chosen

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

        extra_matches = {}
        extra_match_tiers = {}
        for old in old_rows:
            primary = matches.get(old['item_id'])
            if primary is None:
                continue
            for tier, sibling in self._path_siblings(old):
                if sibling == primary or sibling in claimed:
                    continue
                extra_matches[sibling] = old['item_id']
                extra_match_tiers[sibling] = tier
                claimed[sibling] = len(self.tiers)

        return {
            'matches': matches,
            'match_tiers': match_tiers,
            'tier_counts': tier_counts,
            'unmatched': unmatched,
            'extra_matches': extra_matches,
            'extra_match_tiers': extra_match_tiers,
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
