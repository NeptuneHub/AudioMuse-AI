"""Real provider-migration integration test.

Exercises the actual migration decision pipeline — ``match_tracks`` (the
tiered path/metadata matcher) followed by ``build_mapping`` (the dry-run ->
execute mapping prep with collision dedup) — for every combination of
supported media-server providers, including migrating from a provider to
itself.

The execute step proper (``execute_provider_migration``) rewrites item_ids
inside a Postgres transaction and is covered by the unit suite with a mocked
cursor; it needs a live database and is therefore out of scope for this
infra-free integration test (the GitHub workflow provisions no DB and no live
media servers, exactly like the analysis integration test).

The four providers are exactly those the migration probe supports
(``tasks.provider_probe._SUPPORTED_PROVIDERS``); MPD is intentionally excluded
because migration probing was never implemented for it. They are modelled with
realistic id + path conventions taken from how each ``mediaserver_*.py`` and
``tasks.provider_probe`` actually expose tracks:

  * jellyfin  - 32-char hex item id, absolute path under /media/music/...
  * emby      - integer item id, absolute path under a different mount
  * lyrion    - numeric track id, file:// URL-encoded path
  * navidrome - hex item id, relative path (Report Real Path off)

Because the same physical library is rendered under each provider's
conventions, every track must find its counterpart on the target:

  * providers that share a path family (absolute vs relative) match on the
    normalized ``path`` tier,
  * providers in different families fall back to the ``tail`` tier (last 3
    path components),
  * migrating a provider to itself models a library re-scan that re-issues
    item ids (a genuine remap).

Run locally:
    pytest test/test_provider_migration_integration.py -s -v --tb=short
"""
import importlib.util
import os
import sys
from urllib.parse import quote

import pytest


def _load_module(mod_name, *rel_parts):
    """Load a ``tasks.*`` module straight from its file.

    Mirrors the loader used by the provider-migration unit tests: it keeps the
    import self-contained and independent of how the rest of the package is
    wired together in any given environment.
    """
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mod_path = os.path.join(repo_root, *rel_parts)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


matcher = _load_module(
    'tasks.provider_migration_matcher', 'tasks', 'provider_migration_matcher.py'
)
mig = _load_module(
    'tasks.provider_migration_tasks', 'tasks', 'provider_migration_tasks.py'
)


PROVIDERS = ('jellyfin', 'emby', 'navidrome', 'lyrion')

_PATH_FAMILY = {
    'jellyfin': 'absolute',
    'emby': 'absolute',
    'lyrion': 'absolute',
    'navidrome': 'relative',
}

_ID_BASE = {
    'jellyfin': 0x1000,
    'emby': 5000,
    'navidrome': 0xABCD00,
    'lyrion': 90000,
}

_TARGET_ROLE_SHIFT = 500000


CANONICAL_LIBRARY = [
    {'artist': 'Daft Punk', 'album': 'Discovery', 'album_artist': 'Daft Punk',
     'title': 'One More Time', 'disc': 1, 'track': 1, 'ext': 'flac'},
    {'artist': 'Daft Punk', 'album': 'Discovery', 'album_artist': 'Daft Punk',
     'title': 'Aerodynamic', 'disc': 1, 'track': 2, 'ext': 'flac'},
    {'artist': 'Green Day', 'album': 'American Idiot (Japanese Edition)',
     'album_artist': 'Green Day', 'title': 'Are We The Waiting',
     'disc': 1, 'track': 5, 'ext': 'flac'},
    {'artist': 'Green Day', 'album': 'American Idiot (Japanese Edition)',
     'album_artist': 'Green Day', 'title': 'Are We The Waiting',
     'disc': 2, 'track': 4, 'ext': 'flac'},
    {'artist': 'Eagles', 'album': 'Ultimate Rock Hits',
     'album_artist': 'Various Artists', 'title': 'Hotel California',
     'disc': 1, 'track': 3, 'ext': 'mp3'},
]


def _relative_path(track):
    """Library-relative path shared by every provider (folder/album/file)."""
    folder_artist = track['album_artist'] or track['artist']
    filename = f"{track['disc']}-{track['track']:02d} - {track['title']}.{track['ext']}"
    return f"{folder_artist}/{track['album']}/{filename}"


def _provider_id(provider, role, index):
    n = _ID_BASE[provider] + (_TARGET_ROLE_SHIFT if role == 'target' else 0) + index
    if provider == 'jellyfin':
        return format(n, '032x')
    if provider == 'navidrome':
        return format(n, '016x')
    return str(n)


def _provider_path(provider, rel):
    if provider == 'jellyfin':
        return '/media/music/MyTunes/' + rel
    if provider == 'emby':
        return '/mnt/media/MyTunes/' + rel
    if provider == 'lyrion':
        return 'file:///media/music/MyTunes/' + quote(rel)
    return rel


def _render(provider, role):
    """Render the canonical library as ``provider`` would expose it."""
    rendered = []
    for index, track in enumerate(CANONICAL_LIBRARY):
        rel = _relative_path(track)
        rendered.append({
            'index': index,
            'id': _provider_id(provider, role, index),
            'path': _provider_path(provider, rel),
            'title': track['title'],
            'artist': track['artist'],
            'album': track['album'],
            'album_artist': track['album_artist'],
        })
    return rendered


def _score_rows(rendered):
    return [
        {
            'item_id': r['id'],
            'file_path': r['path'],
            'title': r['title'],
            'author': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in rendered
    ]


def _probe_tracks(rendered):
    return [
        {
            'id': r['id'],
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in rendered
    ]


@pytest.mark.integration
@pytest.mark.parametrize('target', PROVIDERS)
@pytest.mark.parametrize('source', PROVIDERS)
def test_provider_migration_all_combinations(source, target):
    """Every (source -> target) combination remaps the whole library."""
    source_rendered = _render(source, 'source')
    target_rendered = _render(target, 'target')

    old_rows = _score_rows(source_rendered)
    new_tracks = _probe_tracks(target_rendered)
    expected = {
        s['id']: t['id'] for s, t in zip(source_rendered, target_rendered)
    }

    print(f"\n=== Migrating {source} -> {target} ===")
    print(f"  source item_id sample: {old_rows[0]['item_id']}")
    print(f"  source file_path sample: {old_rows[0]['file_path']}")
    print(f"  target id sample: {new_tracks[0]['id']}")
    print(f"  target path sample: {new_tracks[0]['path']}")

    result = matcher.match_tracks(old_rows, new_tracks)

    n = len(CANONICAL_LIBRARY)
    assert result['unmatched'] == [], (
        f"{source}->{target}: {len(result['unmatched'])} track(s) failed to "
        f"migrate: {[r['item_id'] for r in result['unmatched']]}"
    )
    assert result['matches'] == expected, (
        f"{source}->{target}: mapping mismatch\n"
        f"  expected: {expected}\n  got:      {result['matches']}"
    )
    assert len(result['matches']) == n

    tier_counts = result['tier_counts']
    same_family = _PATH_FAMILY[source] == _PATH_FAMILY[target]
    if same_family:
        assert tier_counts['path'] == n, (
            f"{source}->{target}: same path family should match on the path "
            f"tier, got {tier_counts}"
        )
        assert tier_counts['tail'] == 0
    else:
        assert tier_counts['tail'] == n, (
            f"{source}->{target}: cross path family should match on the tail "
            f"tier, got {tier_counts}"
        )
        assert tier_counts['path'] == 0
    assert tier_counts['exact_meta'] == 0
    assert tier_counts['norm_meta'] == 0

    if source == target:
        assert all(k != v for k, v in result['matches'].items()), (
            f"{source}->{source}: a re-scan re-issues item ids, so every "
            f"row must be remapped to a new id"
        )

    state = {
        'dry_run': {'matches': dict(result['matches'])},
        'manual_matches': {},
        'manual_unmatches': [],
    }
    deduped, dropped = mig.build_mapping(state)
    assert deduped == result['matches'], (
        f"{source}->{target}: build_mapping changed the mapping unexpectedly"
    )
    assert dropped == [], (
        f"{source}->{target}: build_mapping reported collisions: {dropped}"
    )
    print(f"  ok: {n} tracks remapped, no orphans, no collisions")


def test_all_provider_combinations_are_covered():
    """Guard: the parametrization spans the full provider x provider matrix,
    including each provider migrated to itself."""
    combos = {(s, t) for s in PROVIDERS for t in PROVIDERS}
    assert len(combos) == len(PROVIDERS) ** 2
    assert all((p, p) in combos for p in PROVIDERS)
