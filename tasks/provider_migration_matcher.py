"""Path + metadata matching for the provider migration tool.

Pure-stdlib module. No DB, no HTTP, no config imports. Deterministic.

Used by the dry-run step of the migration wizard to match existing
``score.item_id`` rows against tracks fetched from a target provider via
``tasks.provider_probe``. Logic is ported from lessons learned in the
multi-provider-setup-gui branch.

Tiered matching strategy (first match wins):
  1. ``path``      - normalized absolute path (mount prefixes stripped)
  2. ``tail``      - last 3 path components only
  3. ``exact_meta``- exact (title, artist, album) lowercased
  4. ``norm_meta`` - metadata with noise stripped ("(Remastered)", "feat.",
                    leading "the ", etc.)
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


# Tier names in priority order (highest first).
_TIERS = ('path', 'tail', 'exact_meta', 'norm_meta')


def _best_artist(row):
    """Prefer album_artist, then artist/author."""
    return row.get('album_artist') or row.get('artist') or row.get('author')


def _old_exact_meta_key(old):
    t = (old.get('title') or '').lower()
    a = (_best_artist(old) or '').lower()
    alb = (old.get('album') or '').lower()
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _new_exact_meta_key(new):
    t = (new.get('title') or '').lower()
    a = ((new.get('album_artist') or new.get('artist') or '')).lower()
    alb = (new.get('album') or '').lower()
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _old_norm_meta_key(old):
    t = normalize_meta(old.get('title'))
    a = normalize_meta(_best_artist(old))
    alb = normalize_meta(old.get('album'))
    if not (t and a and alb):
        return None
    return (t, a, alb)


def _new_norm_meta_key(new):
    t = normalize_meta(new.get('title'))
    a = normalize_meta(new.get('album_artist') or new.get('artist'))
    alb = normalize_meta(new.get('album'))
    if not (t and a and alb):
        return None
    return (t, a, alb)


def match_tracks(old_rows, new_tracks):
    """Match existing score rows against a probed target-provider track list.

    Args:
        old_rows: iterable of dicts with keys ``item_id``, ``file_path``,
            ``title``, ``author``, ``album``, ``album_artist``.
        new_tracks: iterable of dicts with keys ``id``, ``path``, ``title``,
            ``artist``, ``album``, ``album_artist``.

    Returns a dict:
        {
          'matches':            dict[old_item_id -> new_id],
          'tier_counts':        dict[tier_name -> int] (always has all 4 tiers),
          'unmatched':          list[old_row],
          'unmatched_by_album': dict[(album_artist, album) -> list[old_row]],
        }

    Collision handling: if two old rows would map to the same new_id, the one
    matched by the higher-priority tier wins. Losers become unmatched.
    """
    # Build new-track indexes. First-write-wins per key.
    by_norm_path = {}
    by_tail = {}
    by_exact_meta = {}
    by_norm_meta = {}
    for n in new_tracks:
        np = normalize_path(n.get('path'))
        if np and np not in by_norm_path:
            by_norm_path[np] = n['id']
        tk = path_tail_key(np)
        if tk and tk not in by_tail:
            by_tail[tk] = n['id']
        ek = _new_exact_meta_key(n)
        if ek and ek not in by_exact_meta:
            by_exact_meta[ek] = n['id']
        nk = _new_norm_meta_key(n)
        if nk and nk not in by_norm_meta:
            by_norm_meta[nk] = n['id']

    tier_counts = {t: 0 for t in _TIERS}
    # tier rank: lower number = higher priority
    tier_rank = {t: i for i, t in enumerate(_TIERS)}

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
            proposals.append(('exact_meta', old, by_exact_meta[ek]))
            matched = True
            continue
        nk = _old_norm_meta_key(old)
        if nk and nk in by_norm_meta:
            proposals.append(('norm_meta', old, by_norm_meta[nk]))
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
    unmatched = []
    for tier, old, new_id in proposals:
        if tier is not None and winners.get(id(old)) == new_id:
            matches[old['item_id']] = new_id
            tier_counts[tier] += 1
        else:
            unmatched.append(old)

    unmatched_by_album = {}
    for old in unmatched:
        key = (old.get('album_artist') or old.get('author'), old.get('album'))
        unmatched_by_album.setdefault(key, []).append(old)

    return {
        'matches': matches,
        'tier_counts': tier_counts,
        'unmatched': unmatched,
        'unmatched_by_album': unmatched_by_album,
    }
