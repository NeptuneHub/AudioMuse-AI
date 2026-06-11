"""Path + metadata matching for the provider migration tool.

Pure-stdlib module. No DB, no HTTP, no config imports. Deterministic.

Used by the dry-run step of the migration wizard to match existing
``score.item_id`` rows against tracks fetched from a target provider via
``tasks.provider_probe``. Logic is ported from lessons learned in the
multi-provider-setup-gui branch.

Tiered matching strategy (first match wins):
  1. ``path``         - normalized absolute path (mount prefixes stripped)
  2. ``tail``         - last 3 path components only
  3. ``exact_meta``   - exact (title, artist, album) lowercased
  4. ``norm_meta``    - metadata with noise stripped ("(Remastered)", "feat.",
                       leading "the ", etc.)
  5. ``title_artist`` - normalized (title, artist) only, ignoring album.
                       Disabled by default — must be enabled by the caller
                       because it can pair tracks from different album versions
                       (studio vs. compilation vs. live).
"""
import re
from urllib.parse import unquote


# Mount prefixes to strip when normalizing absolute paths. Ordered longest-first
# so that more specific prefixes match before their parents (e.g. '/media/music/'
# before '/media/'). All matched case-insensitively.
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
    """Normalize a file path for cross-provider matching.

    Steps:
      1. ``None``/empty -> ``None``
      2. Strip ``file://`` URL scheme and URL-decode
      3. Convert backslashes to forward slashes (Windows)
      4. Lowercase
      5. Strip the longest matching mount prefix from ``_MOUNT_PREFIXES_TO_STRIP``
      6. ``lstrip('/')``

    Returns a lowercased, prefix-stripped relative path, or ``None``.
    """
    if not raw:
        return None
    p = str(raw)
    if p.startswith('file://'):
        # file:///path -> /path ; handle %20 etc.
        p = unquote(p[len('file://'):])
    p = p.replace('\\', '/').lower()
    for prefix in _MOUNT_PREFIXES_TO_STRIP:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return p.lstrip('/')


def path_tail_key(path, n=3):
    """Return the last ``n`` path components (lowercased), joined with ``/``.

    Used as a coarser fallback when full-path normalization fails because
    providers use completely different mount points (e.g. Jellyfin
    ``/media/music/...`` vs Navidrome relative ``Artist/Album/...``).

    Returns ``None`` if the path has fewer than 2 components (too ambiguous).
    """
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


# Regex for extracting disc/track number from a filename. Matches common
# multi-disc naming conventions like "01-05 Title.flac", "1-5 Title.flac",
# "02 04 Title.flac", "1.05 - Title.flac". Applied to the filename basename
# only (directory parts are stripped first). Leading zero-padding is
# removed via int() so "01-05" and "1-5" compare equal.
_DISC_TRACK_RE = re.compile(r'^(\d+)[\s._-]+(\d+)(?=\D|$)')


def extract_disc_track(path):
    """Return ``(disc, track)`` as ``(int, int)`` parsed from the filename,
    or ``None`` if the pattern doesn't match.

    Used to disambiguate multi-disc albums where two tracks on different
    discs share the same (title, artist, album). The matcher's metadata
    tiers would otherwise collapse them into one target.
    """
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


# Regex patterns for metadata noise stripping. Applied to lowercased input.
# Each pattern matches a parenthesized or bracketed group containing the noise.
_META_NOISE_WORDS = (
    'remaster', 'remastered',
    'feat', 'ft', 'featuring',
    'explicit', 'clean',
    'radio edit', 'radio version',
    'single version', 'album version',
    'extended', 'club mix',
    'acoustic', 'live', 'demo',
    'version', 'mix',
)
_META_NOISE_ALT = '|'.join(re.escape(w) for w in _META_NOISE_WORDS)
# Match "(...noise...)" or "[...noise...]" at any position
_META_NOISE_PAREN_RE = re.compile(r'\s*\([^)]*(?:' + _META_NOISE_ALT + r')[^)]*\)', re.IGNORECASE)
_META_NOISE_BRACKET_RE = re.compile(r'\s*\[[^\]]*(?:' + _META_NOISE_ALT + r')[^\]]*\]', re.IGNORECASE)
_LEADING_THE_RE = re.compile(r'^the\s+', re.IGNORECASE)
_COLLAPSE_WS_RE = re.compile(r'\s+')


def normalize_meta(s):
    """Strip cosmetic noise from a metadata string (title/artist/album).

    Lowercases, removes ``(Remastered)`` / ``[Explicit]`` / ``(feat. X)`` style
    noise, strips leading ``"the "``, and collapses whitespace.

    Returns ``''`` for ``None``/empty input.
    """
    if not s:
        return ''
    out = str(s).lower()
    out = _META_NOISE_PAREN_RE.sub('', out)
    out = _META_NOISE_BRACKET_RE.sub('', out)
    out = _LEADING_THE_RE.sub('', out)
    out = _COLLAPSE_WS_RE.sub(' ', out).strip()
    return out


# Tier names in priority order (highest first). ``title_artist`` is opt-in and
# is always the lowest-priority tier when enabled — callers must explicitly
# set ``allow_title_artist_only=True`` because it can pair different album
# versions of the same song.
_TIERS = ('path', 'tail', 'exact_meta', 'norm_meta')
_OPT_TIER_TITLE_ARTIST = 'title_artist'


def _best_artist_old(row):
    """Track-level artist for a source (``score``) row.

    Precedence: ``author`` → ``artist`` → ``album_artist``.
    ``score.author`` holds the track performer that mediaserver_*.py picked via
    ``_select_best_artist``, while ``score.album_artist`` preserves the
    album-level artist (often "Various Artists" on compilations). Preferring
    ``author`` keeps compilation tracks matchable to their real performer on
    the target provider. The ``artist`` fallback is defensive — the score
    schema has no ``artist`` column today, but future importers may write one.
    """
    return row.get('author') or row.get('artist') or row.get('album_artist')


def _best_artist_new(row):
    """Track-level artist for a target (probe) track.

    Precedence: ``artist`` → ``album_artist``. ``provider_probe.py`` already
    collapses the provider-specific hierarchy (e.g. Jellyfin's
    ``ArtistItems[0].Name`` → ``Artists[0]`` → ``AlbumArtist``) into the
    unified ``artist`` field, so ``album_artist`` is only consulted when the
    probe couldn't resolve a track artist at all.
    """
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
    """Match existing score rows against a probed target-provider track list.

    Args:
        old_rows: iterable of dicts with keys ``item_id``, ``file_path``,
            ``title``, ``author``, ``album``, ``album_artist``.
        new_tracks: iterable of dicts with keys ``id``, ``path``, ``title``,
            ``artist``, ``album``, ``album_artist``.
        allow_title_artist_only: opt-in looser fallback that matches by
            normalized (title, artist) only. Only fires when all strict tiers
            miss. Can pair tracks from different album versions.

    Returns a dict:
        {
          'matches':            dict[old_item_id -> new_id],
          'tier_counts':        dict[tier_name -> int],
          'unmatched':          list[old_row],
          'unmatched_by_album': dict[(album_artist, album) -> list[old_row]],
        }

    Collision handling: if two old rows would map to the same new_id, the one
    matched by the higher-priority tier wins. Losers become unmatched.
    """
    tiers = list(_TIERS)
    if allow_title_artist_only:
        tiers.append(_OPT_TIER_TITLE_ARTIST)

    # Build new-track indexes. Path/tail are first-write-wins (unique enough);
    # metadata indexes store candidate LISTS because multi-disc albums can
    # have several tracks sharing (title, artist, album), and we need to
    # pick the right one by disc/track at match time.
    by_norm_path = {}
    by_tail = {}
    by_exact_meta = {}     # key -> list of new tracks
    by_norm_meta = {}      # key -> list of new tracks
    by_title_artist = {}   # key -> list of new tracks (opt-in tier)
    for n in new_tracks:
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
    # tier rank: lower number = higher priority
    tier_rank = {t: i for i, t in enumerate(tiers)}

    def _pick_meta_candidate(old, candidates):
        """Pick the best new track from a list of candidates that all share
        the same (title, artist, album). Uses disc/track extracted from the
        file path as a tiebreaker; falls back to the first candidate.
        """
        if len(candidates) == 1:
            return candidates[0]
        old_dt = extract_disc_track(old.get('file_path'))
        if old_dt is not None:
            for c in candidates:
                if extract_disc_track(c.get('path')) == old_dt:
                    return c
        # No disc/track info or no candidate matched — fall back to first.
        # This preserves legacy behavior for single-disc albums.
        return candidates[0]

    # First pass: determine best tier match per old row.
    proposals = []  # list of (tier, old_row, new_id)
    for old in old_rows:
        matched = False
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

    # Second pass: resolve collisions (multiple old rows → same new_id).
    # The proposal with the best (lowest-rank) tier keeps the match.
    best_for_new = {}  # new_id -> (tier, old_row)
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
