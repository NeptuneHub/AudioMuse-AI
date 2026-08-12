# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Standalone research script: build genre_subgenre.json from live data + web tags.

Offline experiment (not part of the app runtime) that connects read-only to a
Postgres AudioMuse-AI database, assigns every track to its top-mood genre
(one of the genre labels the system itself maps), clusters each genre's real
200-dim MusiCNN embeddings into subgenre clusters, then names each cluster from
the aggregated REAL genre tags of the songs inside it. Tags come from public
music APIs queried per sampled track (Last.fm when LASTFM_API_KEY is set,
otherwise keyless Deezer and MusicBrainz) plus each cluster's own dominant MSD
tags. No artist names are used or stored, only music-server item ids. The
output mirrors mood_centroids_real_080_clap.json so the same Poincare-projection
machinery can later consume it for a main-genre to subgenre directory tree.

Main Features:
* Read-only access; every track assigned to exactly one top-mood genre label
* MiniBatchKMeans clustering per genre over the real 200-dim embeddings
* Cluster naming is data-driven: web genre tags (Last.fm / Deezer / MusicBrainz)
  sampled per track, fused with the cluster's own dominant MSD tags
* Never stores artist names or per-track sample ids, so the file stays small
  and portable across libraries
* Output generalizes to other libraries: the genre list is the top-mood
  vocabulary and the naming is tag-driven, not library- or artist-specific
"""

import json
import os
import re
import sys
import time
import urllib.parse

import numpy as np
import psycopg2

DB_CONFIG = {
    'host': '192.168.3.208',
    'port': 5432,
    'user': 'audiomuse',
    'password': 'audiomusepassword',
    'dbname': 'audiomusedb',
    'options': '-c default_transaction_read_only=on',
}

OUTPUT_FILE = 'genre_subgenre.json'

LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY', '')
WEB_UA = 'AudioMuseResearch/1.0 (genre-subgenre research)'
MIN_CLUSTER_SONGS = 40
MIN_PER_GENRE = 80
MAX_K = 20
WEB_SAMPLE_PER_CLUSTER = 4
MAX_WEB_LOOKUPS = 900
WEB_SLEEP = 0.35
SEED = 42

GENRE_LABELS = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'jazz', 'metal',
    'classic rock', 'soul', 'indie rock', 'electronica', 'folk', 'punk',
    'blues', 'hard rock', 'ambient', 'acoustic', 'experimental', 'Hip-Hop',
    'country', 'funk', 'electro', 'heavy metal', 'Progressive rock', 'rnb',
    'indie pop', 'House', 'alternative rock', 'dance',
]

NON_GENRE_TAGS = {
    'female vocalists', 'male vocalists', 'female vocalist', 'guitar', 'oldies',
    'beautiful', 'sexy', 'catchy', 'mellow', 'chill', 'instrumental', 'party',
    'happy', 'sad', '90s', '80s', '70s', '60s', '00s', '50s', '40s', '30s',
    'easy listening', 'live', 'seen live', 'favorites', 'love', 'chillout',
    'lo-fi', 'lofi', 'american', 'billboard hot 100', 'love at first listen',
    'peak', 'motomano', 'hot hits', 'top 40', 'hit', 'hits', 'mainstream',
    'beautiful voice', 'nice voice', 'great voice', 'danceable', 'sing along',
    'epic', 'marco', 'amazing', 'awesome', 'greatest hits', 'london',
    'spanish', 'religious', 'british', 'american classic', 'thank u', 'feat',
    'featuring', 'remix', 'trumpet', 'saxophone', 'piano', 'canadian',
    'canada', 'usa', 'england', 'atlanta', 'new york', 'chicago', 'detroit',
    'memphis', 'los angeles', 'young money', 'swag', 'crew', 'label',
    'under 2000 listeners', 'my top songs', 'legend',
}

_WEB_CACHE = {}


def parse_scores(feat_str):
    out = {}
    if not feat_str:
        return out
    for pair in str(feat_str).split(','):
        label, _, value = pair.partition(':')
        label = label.strip()
        if not label:
            continue
        try:
            out[label] = float(value)
        except ValueError:
            continue
    return out


def top_genre(tags):
    best, best_score = None, 0.0
    for g in GENRE_LABELS:
        s = tags.get(g, 0.0)
        if s > best_score:
            best_score = s
            best = g
    return best if best_score >= 0.05 else None


def norm_tag(tag):
    return str(tag).strip().lower().replace('&amp;', '&').replace('  ', ' ')


def name_key(tag):
    return re.sub(r'[^a-z0-9]+', '-', str(tag).lower()).strip('-')


_CHART_NOISE = re.compile(
    r'(billboard|hot 100|top 40|top 100|viral|trending|playlist|stream|'
    r'chart|20[0-9]{2}|hits?$)', re.IGNORECASE)


def is_genre_like(tag):
    t = norm_tag(tag)
    return (bool(t) and len(t) >= 2 and t not in NON_GENRE_TAGS
            and not _CHART_NOISE.search(t))


def _http_json(url):
    import requests

    r = requests.get(url, headers={'User-Agent': WEB_UA}, timeout=12)
    r.raise_for_status()
    return r.json()


def _load_deezer_genres():
    try:
        data = _http_json('https://api.deezer.com/genre')
        return {g['id']: norm_tag(g['name']) for g in data.get('data', [])}
    except Exception:
        return {}


_DEZER_GENRES = {}


def _lastfm_tags(title, artist):
    tags = []
    params = urllib.parse.urlencode({
        'method': 'track.getInfo', 'api_key': LASTFM_API_KEY,
        'artist': artist, 'track': title, 'format': 'json', 'autocorrect': 1,
    })
    try:
        data = _http_json('https://ws.audioscrobbler.com/2.0/?' + params)
        toptags = data.get('track', {}).get('toptags', {}).get('tag', []) or []
        for tg in toptags[:12]:
            name = norm_tag(tg.get('name', ''))
            if name and is_genre_like(name):
                tags.append((name, max(0.1, float(tg.get('count', 1)))))
    except Exception:
        pass
    return tags


def _deezer_tags(title, artist):
    q = urllib.parse.quote(f'artist:"{artist}" track:"{title}"')
    tags = []
    try:
        data = _http_json(f'https://api.deezer.com/search?q={q}&limit=3')
        for item in data.get('data', [])[:3]:
            gid = item.get('genre_id')
            gname = _DEZER_GENRES.get(gid) if gid else None
            if gname and is_genre_like(gname):
                tags.append((gname, 1.0))
    except Exception:
        pass
    return tags


def _musicbrainz_tags(title, artist):
    tags = []
    try:
        q = urllib.parse.quote(f'recording:"{title}" AND artist:"{artist}"')
        data = _http_json(f'https://musicbrainz.org/ws/2/recording/?query={q}&fmt=json&limit=1')
        recs = data.get('recordings', [])
        if recs:
            rec = recs[0]
            for g in rec.get('genres', []) + rec.get('tags', []):
                n = norm_tag(g.get('name', ''))
                if n and is_genre_like(n):
                    tags.append((n, 0.7))
            rel = (rec.get('releases') or [{}])[0]
            rg = rel.get('release-group', {})
            for g in rg.get('genres', []) + rg.get('tags', []):
                n = norm_tag(g.get('name', ''))
                if n and is_genre_like(n):
                    tags.append((n, 0.6))
    except Exception:
        pass
    return tags


def web_genre_tags(title, artist):
    key = (norm_tag(title), norm_tag(artist))
    if key in _WEB_CACHE:
        return _WEB_CACHE[key]
    tags = []
    if LASTFM_API_KEY:
        tags.extend(_lastfm_tags(title, artist))
    if not tags:
        tags.extend(_deezer_tags(title, artist))
        tags.extend(_musicbrainz_tags(title, artist))
    _WEB_CACHE[key] = tags
    time.sleep(WEB_SLEEP)
    return tags


def aggregate_web_tags(sample):
    agg = {}
    counts = {}
    for title, artist in sample:
        seen = set()
        for name, weight in web_genre_tags(title, artist):
            agg[name] = agg.get(name, 0.0) + weight
            if name not in seen:
                counts[name] = counts.get(name, 0) + 1
                seen.add(name)
    return {name: w for name, w in agg.items() if counts[name] >= 2}


def lastfm_genre_subgenres(genre, main_keys, limit_artists=20, min_artists=2, top_n=20):
    """Real Last.fm subgenres for a genre, discovered through its top artists.

    tag.getTopTags returns the same global tag cloud for every tag, so it cannot
    describe a genre's subgenres. The hierarchy is instead derived from artists:
    the top artists of a genre each carry their own tags, and the most common
    genre-like tags among them are the real subgenres. Any name that equals
    another main genre is dropped, so no main genre ever appears as a subgenre
    of another genre.
    """
    if not LASTFM_API_KEY:
        return []
    agg = {}
    try:
        url = 'https://ws.audioscrobbler.com/2.0/?' + urllib.parse.urlencode({
            'method': 'tag.getTopArtists', 'api_key': LASTFM_API_KEY,
            'tag': genre, 'limit': limit_artists, 'format': 'json'})
        data = _http_json(url)
        artists = data.get('topartists', {}).get('artist', []) or []
    except Exception:
        return []
    for a in artists:
        aname = a.get('name', '')
        if not aname:
            continue
        try:
            url = 'https://ws.audioscrobbler.com/2.0/?' + urllib.parse.urlencode({
                'method': 'artist.getTopTags', 'api_key': LASTFM_API_KEY,
                'artist': aname, 'autocorrect': 1, 'format': 'json'})
            t = _http_json(url)
        except Exception:
            continue
        for tg in t.get('toptags', {}).get('tag', []) or []:
            n = norm_tag(tg.get('name', ''))
            if not n or not is_genre_like(n):
                continue
            if name_key(n) in main_keys:
                continue
            agg[n] = agg.get(n, 0) + 1
        time.sleep(0.2)
    ranked = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, c in ranked[:top_n] if c >= min_artists]


def load_library():
    print('Loading library (read-only)...')
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute(
        "SELECT item_id, embedding, mood_vector, title, album_artist "
        "FROM embedding JOIN score USING (item_id)"
    )
    ids, vecs, tags, titles, artists = [], [], [], [], []
    skipped = 0
    for item_id, emb, mv, title, artist in cur:
        vec = np.frombuffer(bytes(emb), dtype=np.float32)
        if vec.shape[0] != 200:
            skipped += 1
            continue
        ids.append(item_id)
        vecs.append(vec.astype(np.float32))
        tags.append(parse_scores(mv))
        titles.append((title or '').strip())
        artists.append((artist or '').strip())
    cur.close()
    conn.close()
    print(f'  loaded {len(ids)} tracks (skipped {skipped})')
    return ids, np.stack(vecs), tags, titles, artists


def main():
    global _DEZER_GENRES
    _DEZER_GENRES = _load_deezer_genres()
    print(f'  deezer genres loaded: {len(_DEZER_GENRES)}')

    ids, X, tags, titles, artists = load_library()

    genre_of = [top_genre(t) for t in tags]
    by_genre = {}
    for i, g in enumerate(genre_of):
        if g is None:
            continue
        by_genre.setdefault(g, []).append(i)
    print('top-mood genre assignment:')
    for g in sorted(by_genre, key=lambda g: -len(by_genre[g])):
        print(f'  {g:20s} {len(by_genre[g])}')
    main_keys = {name_key(g) for g in by_genre}

    from sklearn.cluster import MiniBatchKMeans

    results = {}
    web_budget = MAX_WEB_LOOKUPS
    for genre, members in by_genre.items():
        if len(members) < MIN_PER_GENRE:
            print(f'== {genre}: too few ({len(members)}), skipping')
            continue
        n = len(members)
        k = min(MAX_K, max(4, round(n / 400)))
        k = min(k, max(4, round(n / MIN_CLUSTER_SONGS)))
        print(f'\n== {genre}: {n} songs, k={k}')
        sub_x = X[members]
        km = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=4,
                             max_iter=100, random_state=SEED)
        labels = km.fit_predict(sub_x)

        cluster_idx = [[] for _ in range(k)]
        for j, lab in enumerate(labels):
            cluster_idx[int(lab)].append(j)

        centroids, sizes, mean_tags, top_msd, sample_tracks = [], [], [], [], []
        for c in range(k):
            cids = cluster_idx[c]
            sizes.append(len(cids))
            cent = sub_x[cids].mean(axis=0)
            centroids.append(cent)
            prof = {}
            for j in cids:
                for tag, val in tags[members[j]].items():
                    prof[tag] = prof.get(tag, 0.0) + val
            mn = len(cids)
            mean_tags.append({t: v / mn for t, v in prof.items()})
            ranked = sorted(mean_tags[c].items(), key=lambda kv: -kv[1])
            top_msd.append([t for t, _ in ranked if is_genre_like(t)][:10])
            dists = np.linalg.norm(sub_x[cids] - cent, axis=1)
            order = np.argsort(dists)[:WEB_SAMPLE_PER_CLUSTER]
            sample_tracks.append([
                (titles[members[cids[j]]], artists[members[cids[j]]])
                for j in order if titles[members[cids[j]]]
            ])

        order = sorted(range(k), key=lambda c: -sizes[c])
        used = set()
        genre_key = name_key(genre)
        vocab = lastfm_genre_subgenres(genre, main_keys)
        vocab_by_key = {}
        for v in vocab:
            vk = name_key(v)
            if vk != genre_key and vk not in vocab_by_key:
                vocab_by_key[vk] = v
        names = {}
        scored = []
        for c in range(k):
            if sizes[c] < MIN_CLUSTER_SONGS:
                names[c] = None
                continue
            cands = {}
            if web_budget > 0:
                sample = sample_tracks[c][:WEB_SAMPLE_PER_CLUSTER]
                if sample:
                    web_budget -= len(sample)
                    for name, w in aggregate_web_tags(sample).items():
                        ck = name_key(name)
                        if ck in vocab_by_key and ck not in used:
                            cands[ck] = cands.get(ck, 0.0) + w * 2.0
            for name, val in mean_tags[c].items():
                ck = name_key(name)
                if ck in vocab_by_key and ck not in used:
                    cands[ck] = cands.get(ck, 0.0) + val
            scored.append((c, cands))
        for c, cands in sorted(scored, key=lambda x: -max(x[1].values()) if x[1] else -1):
            if names.get(c) is not None:
                continue
            avail = {k: v for k, v in cands.items() if k not in used}
            if not avail:
                continue
            best_k = max(avail.items(), key=lambda kv: kv[1])[0]
            names[c] = vocab_by_key[best_k]
            used.add(best_k)
        for c in order:
            if names.get(c) is not None:
                continue
            nxt = next((v for k, v in vocab_by_key.items() if k not in used), None)
            if nxt is None:
                top = next(
                    (t for t in top_msd[c]
                     if name_key(t) not in main_keys and name_key(t) not in used
                     and is_genre_like(t)),
                    None,
                )
                names[c] = top or 'Mixed'
                if top:
                    used.add(name_key(top))
            else:
                names[c] = nxt
                used.add(name_key(nxt))

        sub_list = []
        for c in range(k):
            if sizes[c] < MIN_CLUSTER_SONGS or names.get(c) is None:
                continue
            top = sorted(mean_tags[c].items(), key=lambda kv: -kv[1])[:8]
            sub_list.append({
                'name': names[c],
                'n_songs': int(sizes[c]),
                'centroid': centroids[c].astype(np.float32).tolist(),
                'top_tags': {t: float(v) for t, v in top},
            })
        sub_list.sort(key=lambda s: -s['n_songs'])
        results[genre] = {'n_songs': int(n), 'subgenres': sub_list}
        for s in sub_list:
            print(f"  {s['n_songs']:6d}  {s['name']:24s} tags={list(s['top_tags'].keys())[:4]}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {OUTPUT_FILE}')
    total = sum(len(m) for m in by_genre.values())
    print(f'SUMMARY: {total:,d} tracks assigned to {len(results)} genres, '
          f'web lookups used: {MAX_WEB_LOOKUPS - web_budget}')


if __name__ == '__main__':
    sys.exit(main())
