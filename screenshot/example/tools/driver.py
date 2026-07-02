# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Playwright screenshot driver for the AudioMuse-AI documentation images.

Development tool that drives a headless browser against a running instance,
logs in, walks the main UI pages and captures the screenshots used in the docs.
It intercepts ``/api/**`` responses to substitute generated placeholder data
(fake playlist/artist names) so published screenshots contain no real library
content. ``driver2.py`` is the flow-driven companion.

Main Features:
* Automates login and per-page navigation, then captures screenshots.
* Masks API JSON with LLM-generated placeholder names to avoid leaking data.
"""

import os
import json
import traceback
from playwright.sync_api import sync_playwright

BASE = os.environ.get("AUDIOMUSE_BASE", "http://YOUR-SERVER:8000")
OUT = os.environ.get("AUDIOMUSE_OUT", "screenshot/example")
USER = os.environ.get("AUDIOMUSE_USER", "admin")
PW = os.environ.get("AUDIOMUSE_PW", "admin")
OPENAI_URL = "https://api.atlascloud.ai/v1/chat/completions"
OPENAI_MODEL = "deepseek-ai/DeepSeek-V3-0324"

ADJ = [
    "Velvet",
    "Crimson",
    "Neon",
    "Hollow",
    "Golden",
    "Silver",
    "Midnight",
    "Electric",
    "Wild",
    "Quiet",
    "Lunar",
    "Solar",
    "Paper",
    "Glass",
    "Marble",
    "Cobalt",
    "Amber",
    "Violet",
    "Saffron",
    "Northern",
    "Coastal",
    "Restless",
    "Tidal",
    "Wandering",
    "Ember",
    "Faded",
    "Brass",
    "Scarlet",
    "Frozen",
    "Gilded",
]
NOUN = [
    "Echo",
    "Harbor",
    "Foxglove",
    "Tigers",
    "Lantern",
    "District",
    "Aviary",
    "Society",
    "Wren",
    "Owls",
    "Cartographers",
    "Birches",
    "Compass",
    "Iris",
    "Underground",
    "Fields",
    "Hours",
    "Pines",
    "Tideway",
    "Avenue",
    "Static",
    "Garden",
    "Cathedral",
    "Meadow",
    "Circuit",
    "Anchor",
    "Mirage",
    "Parade",
    "Willow",
    "Signal",
]
TADJ = [
    "Burning",
    "Falling",
    "Golden",
    "Endless",
    "Silent",
    "Restless",
    "Fading",
    "Rising",
    "Hollow",
    "Electric",
    "Velvet",
    "Midnight",
    "Paper",
    "Glass",
    "Distant",
    "Crimson",
    "Frozen",
    "Wandering",
    "Bright",
    "Lonely",
    "Gentle",
    "Savage",
    "Dizzy",
    "Reckless",
    "Tender",
]
TNOUN = [
    "Skyline",
    "Tides",
    "Hearts",
    "Lights",
    "Avenue",
    "Rain",
    "Ghosts",
    "Embers",
    "Horizon",
    "Daydream",
    "Machine",
    "Static",
    "Highway",
    "Shadows",
    "Mornings",
    "Echoes",
    "Fever",
    "Parade",
    "Wilderness",
    "Lullaby",
    "Gravity",
    "Reverie",
    "Mirage",
    "Compass",
    "Bloom",
]
ALBUMS = [
    "After the Tide",
    "Paper Cities",
    "Neon Wilderness",
    "Slow Motion Skies",
    "Golden Static",
    "Velvet Horizons",
    "Northern Lullabies",
    "Glass Gardens",
    "Midnight Cartography",
    "Echoes & Embers",
    "Coastal Reverie",
    "The Quiet Hours",
    "Wandering Signals",
    "Bright Mirage",
    "Restless Bloom",
    "Amber Frequencies",
    "Distant Parade",
    "Hollow Daydreams",
    "Electric Meadow",
    "Frozen Avenues",
    "Tidal Reverb",
    "Saffron Skyline",
    "Lunar Compass",
    "Crimson Static",
]
AIPL = [
    "Midnight Velvet Grooves",
    "Sunlit Acoustic Mornings",
    "Rainy Day Indie Haze",
    "Neon Nights & City Lights",
    "Coffeehouse Acoustic Calm",
    "Deep Focus Ambient Flow",
    "Golden Hour Soul",
    "Late Night Jazz Lounge",
    "Summer Road Trip Anthems",
    "Mellow Sunday Reset",
    "Electro Pulse After Dark",
    "Campfire Folk Stories",
    "Dreamy Shoegaze Drift",
    "Power Workout Surge",
    "Vintage Vinyl Warmth",
    "Rooftop Sunset Chill",
    "Monday Motivation Mix",
    "Slow Dance Reverie",
    "Indie Discovery Radio",
    "Stormy Night Piano",
]
NAMES = [
    "Evening Wind Down",
    "Focus Session",
    "Throwback Favorites",
    "Fresh Discoveries",
    "Weekend Energy",
    "Chill Vibes",
    "Deep Cuts",
    "Morning Boost",
    "Sunset Sessions",
    "Rainy Mood",
]
UNAMES = [
    "demo_admin",
    "alex_dj",
    "mia_listens",
    "sam_audio",
    "casey_v",
    "jordan_b",
    "noah_m",
    "riley_k",
]

_counters, _maps = {}, {}


def _idx(cat, val):
    k = (cat, val)
    if k in _maps:
        return _maps[k]
    _counters[cat] = _counters.get(cat, 0) + 1
    _maps[k] = _counters[cat] - 1
    return _maps[k]


def fake(cat, val):
    if not isinstance(val, str) or not val.strip():
        return val
    n = _idx(cat, val)
    if cat == 'artist':
        pre = "The " if (n % 3 == 0) else ""
        return pre + ADJ[n % len(ADJ)] + " " + NOUN[(n // len(ADJ)) % len(NOUN)]
    if cat == 'title':
        return TADJ[n % len(TADJ)] + " " + TNOUN[(n // len(TADJ)) % len(TNOUN)]
    if cat == 'album':
        return ALBUMS[n % len(ALBUMS)]
    if cat == 'playlist':
        return AIPL[n % len(AIPL)]
    if cat == 'name':
        return NAMES[n % len(NAMES)]
    if cat == 'user':
        return UNAMES[n % len(UNAMES)]
    if cat == 'email':
        return UNAMES[n % len(UNAMES)] + "@example.com"
    if cat == 'host':
        return "muse-worker-%02d" % (n + 1)
    if cat == 'path':
        return "/music/library/%s/%s.flac" % (
            ADJ[n % len(ADJ)].lower(),
            TNOUN[n % len(TNOUN)].lower(),
        )
    if cat == 'url':
        return "http://media.example.local:8096"
    if cat == 'redact':
        return "********"
    return "sample_%d" % n


def classify(key):
    k = key.lower()
    if any(x in k for x in ('openai', 'ollama', 'gemini', 'mistral')) or 'provider' in k:
        return None
    if any(x in k for x in ('token', 'secret', 'password', 'apikey', 'api_key', 'jwt')):
        return 'redact'
    if 'worker' in k or k in ('host', 'hostname', 'node'):
        return 'host'
    if k.endswith('url') or '_url' in k or k in ('server', 'address', 'base_url', 'server_url'):
        return 'url'
    if 'email' in k:
        return 'email'
    if k in ('username', 'user', 'sub', 'user_id') or k.endswith('_user') or k.endswith('user_id'):
        return 'user'
    if 'path' in k:
        return 'path'
    if k in ('artist', 'author', 'album_artist', 'albumartist', 'artist_name'):
        return 'artist'
    if k in ('title', 'track', 'track_name', 'song_title', 'song'):
        return 'title'
    if k == 'album':
        return 'album'
    if k in ('playlist_name', 'playlist'):
        return 'playlist'
    if k in ('name', 'display_name', 'item_name', 'library_name', 'radio_name', 'anchor_name'):
        return 'name'
    return None


def transform(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            cat = classify(k)
            if cat and isinstance(v, str) and v.strip():
                obj[k] = fake(cat, v)
                continue
            if cat and isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                obj[k] = [fake(cat, x) for x in v]
                continue
            transform(v)
    elif isinstance(obj, list):
        for x in obj:
            transform(x)


def fake_playlists(data):
    if not isinstance(data, dict):
        return data
    out = {}
    for i, (k, songs) in enumerate(data.items()):
        if isinstance(songs, list):
            transform(songs)
            out[AIPL[i % len(AIPL)]] = songs[:14]
    return out


def fake_dashboard(data):
    c = data.get('content') if isinstance(data, dict) else None
    if isinstance(c, dict):
        c['total_songs'] = 2437
        c['distinct_artists'] = 612
        c['distinct_albums'] = 348
        for kk in ('musicnn_indexed', 'clap_indexed', 'gmm_indexed'):
            if kk in c:
                c[kk] = 2437
        tg = c.get('top_genre')
        if isinstance(tg, list):
            for i, e in enumerate(tg):
                if isinstance(e, dict):
                    e['count'] = max(38, 560 - i * 42)
        mc = c.get('moods_coverage')
        if isinstance(mc, list):
            for i, e in enumerate(mc):
                if isinstance(e, dict):
                    e['score'] = round(max(55.0, 1850.0 - i * 250.0), 2)
        tp = c.get('tempo_profile')
        if isinstance(tp, dict):
            tp.update(
                {'slow': 540, 'medium': 980, 'fast': 720, 'very_fast': 197, 'avg_tempo': 118.4}
            )
    return data


def fake_map(data):
    if isinstance(data, dict) and isinstance(data.get('items'), list):
        items = data['items'][:280]
        transform(items)
        data['items'] = items
    return data


def fake_similar_artists():
    out = []
    for j in range(12):
        out.append(
            {
                "artist": ("The " if j % 3 else "")
                + ADJ[(j * 5) % len(ADJ)]
                + " "
                + NOUN[(j * 3 + 1) % len(NOUN)],
                "artist_id": "aid-%03d" % (100 + j),
                "divergence": round(0.05 + j * 0.013, 3),
                "component_matches": [],
            }
        )
    return out


def fake_waveform():
    import math

    peaks = []
    for i in range(250):
        amp = 0.12 + 0.85 * abs(math.sin(i * 0.075)) * (0.5 + 0.5 * math.sin(i * 0.021 + 0.6))
        amp = min(0.98, amp + (((i * 53) % 13) / 120.0))
        peaks.append(round(-amp, 3))
        peaks.append(round(amp, 3))
    return {
        "peaks": peaks,
        "duration": 214.6,
        "title": "Endless Skyline",
        "author": "The Ember Echo",
    }


def fake_chat_response():
    seeds_title = "Midnight Harbor"
    seeds_artist = "The Velvet Echo"
    msg = "\n".join(
        [
            "Request: 'Similar to %s by %s'" % (seeds_title, seeds_artist),
            "AI Provider: OPENAI",
            "AI reasoning: Songs similar to a named track, via seed_search.",
            "AI emitted 1 tool call",
            "--- Composition: 1 primary",
            "PRIMARY: seed_search",
            "pooled 48/60 unique (pool=48)",
            "Pool: 48 collected -> 30 after diversity cap -> 14 in final",
            "Playlist ordered for smooth transitions",
            "SUCCESS! Generated playlist",
        ]
    )
    qr = []
    for j in range(14):
        qr.append(
            {
                "item_id": "id-%04d" % (1000 + j),
                "title": TADJ[(j * 3) % len(TADJ)] + " " + TNOUN[(j * 5 + 2) % len(TNOUN)],
                "artist": ("The " if j % 3 else "")
                + ADJ[(j * 7) % len(ADJ)]
                + " "
                + NOUN[(j * 2 + 1) % len(NOUN)],
            }
        )
    return {
        "response": {
            "message": msg,
            "original_request": "Similar to %s by %s" % (seeds_title, seeds_artist),
            "ai_provider_used": "OPENAI",
            "ai_model_selected": OPENAI_MODEL,
            "executed_query": "executed_query: seed_search -> search_database",
            "query_results": qr,
        }
    }


def handle_route(route):
    url = route.request.url
    low = url.lower()
    if 'chatplayliststream' in low:
        route.fulfill(status=500, content_type='application/json', body=b'{"error":"stream off"}')
        return
    if 'chatplaylist' in low:
        route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps(fake_chat_response()).encode(),
        )
        return
    if '/api/similar_artists' in low:
        route.fulfill(
            status=200,
            content_type='application/json',
            body=json.dumps(fake_similar_artists()).encode(),
        )
        return
    if '/api/waveform' in low:
        route.fulfill(
            status=200, content_type='application/json', body=json.dumps(fake_waveform()).encode()
        )
        return
    if 'lyrics/search/axes' in low:
        try:
            r = route.fetch(
                url='%s/api/lyrics/search/text' % BASE,
                method='POST',
                headers={'content-type': 'application/json'},
                post_data=json.dumps({'query': 'city lights and quiet heartbreak', 'limit': 40}),
            )
            d = r.json()
            transform(d)
            route.fulfill(status=200, content_type='application/json', body=json.dumps(d).encode())
        except Exception:
            route.fulfill(
                status=200,
                content_type='application/json',
                body=json.dumps({'results': [], 'count': 0}).encode(),
            )
        return
    try:
        resp = route.fetch()
    except Exception:
        try:
            route.continue_()
        except Exception:
            pass
        return
    headers = {
        k: v
        for k, v in (resp.headers or {}).items()
        if k.lower() not in ('content-length', 'content-encoding')
    }
    ct = (resp.headers or {}).get('content-type', '')
    if 'json' in ct:
        data = None
        try:
            data = resp.json()
        except Exception:
            data = None
        if data is not None:
            try:
                if '/api/playlists' in low:
                    data = fake_playlists(data)
                elif '/api/dashboard/summary' in low:
                    transform(data)
                    fake_dashboard(data)
                elif '/api/map' in low:
                    data = fake_map(data)
                else:
                    transform(data)
            except Exception:
                pass
            route.fulfill(
                status=resp.status,
                headers=headers,
                content_type='application/json',
                body=json.dumps(data).encode(),
            )
            return
    try:
        route.fulfill(status=resp.status, headers=headers, body=resp.body())
    except Exception:
        try:
            route.fulfill(response=resp)
        except Exception:
            try:
                route.continue_()
            except Exception:
                pass


INIT_JS = r"""
() => {
  try { localStorage.setItem('menuOpen','true'); localStorage.setItem('theme','light'); } catch(e){}
  const SENS = /(token|secret|password|passwd|api[_-]?key|apikey|jwt|_url|url$|server|host|address|identifier|jellyfin|navidrome|emby|lyrion|credential|username|_user|user_id|userid)/i;
  const LLM  = /(openai|ollama|gemini|mistral|model|provider)/i;
  function valFor(s){
    s=(s||'').toLowerCase();
    if(/token|secret|password|passwd|key|jwt/.test(s)) return '********';
    if(/url|server|address|host|jellyfin|navidrome|emby|lyrion/.test(s)) return 'http://media.example.local:8096';
    if(/path/.test(s)) return '/music/library';
    if(/email/.test(s)) return 'demo_user@example.com';
    if(/user|identifier/.test(s)) return 'demo_user';
    return 'sample';
  }
  function mask(){
    var pth = (location.pathname || '');
    if (pth.indexOf('/login') >= 0 || pth.indexOf('/auth') >= 0) return;
    document.querySelectorAll('input, textarea').forEach(el=>{
      try{
        const cred = el.getAttribute('data-cred');
        const sig = (el.id||'')+' '+(el.name||'')+' '+(cred||'');
        if(el.type==='password'){ if(el.value) el.value='********'; return; }
        if(LLM.test(sig)) return;
        if(cred){ el.value = valFor(cred); return; }
        if(SENS.test(sig) && el.value){ el.value = valFor(sig); }
      }catch(e){}
    });
  }
  const run=()=>{ try{mask();}catch(e){} };
  document.addEventListener('DOMContentLoaded', run);
  setInterval(run, 400);
}
"""


def shot(page, idx, key, state=''):
    name = "%02d_%s%s.png" % (idx, key, ('_' + state) if state else '')
    try:
        page.screenshot(path="%s/%s" % (OUT, name), full_page=True)
        print("SHOT", name, flush=True)
    except Exception as e:
        print("SHOT_FAIL", name, repr(e), flush=True)


def settle(page, secs):
    try:
        page.wait_for_load_state('domcontentloaded', timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(int(secs * 1000))


def click(page, sel, t=4000):
    try:
        page.click(sel, timeout=t)
        return True
    except Exception:
        return False


def set_openai(page, sel, url_in, model_in):
    try:
        page.select_option(sel, 'OPENAI', timeout=4000)
        page.wait_for_timeout(400)
        page.fill(url_in, OPENAI_URL, timeout=4000)
        page.fill(model_in, OPENAI_MODEL, timeout=4000)
        page.wait_for_timeout(300)
        return True
    except Exception as e:
        print("LLM_SET_FAIL", sel, repr(e), flush=True)
        return False


def autocomplete(page, inp, wait_sel=None, q='a'):
    try:
        page.fill(inp, q, timeout=4000)
        page.wait_for_timeout(300)
        if len(q) == 1:
            page.type(inp, 'r', delay=60)
        if wait_sel:
            try:
                page.wait_for_selector(wait_sel, timeout=4000)
            except Exception:
                pass
        page.wait_for_timeout(1200)
        return True
    except Exception:
        return False


TEXT2KEY = {
    "Dashboard": "dashboard",
    "Analysis and Clustering": "index",
    "Instant Playlist": "chat",
    "Playlist from Similar Song": "similarity",
    "Artist Similarity": "artist_similarity",
    "Song Path": "path",
    "Song Alchemy": "alchemy",
    "Text Search (DCLAP)": "clap_search",
    "Lyrics Search": "lyrics_search",
    "Music Map": "map",
    "Sonic Fingerprint": "sonic_fingerprint",
    "Waveform": "waveform",
    "Cleaning": "cleaning",
    "Scheduled Tasks": "cron",
    "Backup and Restore": "backup",
    "Provider Migration": "provider_migration",
    "Setup Wizard": "setup",
    "Users": "users",
}


def recipe(page, idx, key):
    if key == 'index':
        settle(page, 3)
        set_openai(
            page,
            '#config-ai_model_provider',
            '#config-openai_server_url',
            '#config-openai_model_name',
        )
        shot(page, idx, key, '1_config_openai')
        if click(page, '#fetch-playlists-btn'):
            try:
                page.wait_for_selector('#playlists-container .playlist-name', timeout=6000)
            except Exception:
                pass
            page.wait_for_timeout(1200)
            try:
                page.eval_on_selector('#playlists-section', "e=>e.scrollIntoView()")
            except Exception:
                pass
            shot(page, idx, key, '2_clustering_playlists')
        if click(page, '#advanced-view-btn'):
            page.wait_for_timeout(600)
            shot(page, idx, key, '3_advanced')
    elif key == 'chat':
        settle(page, 2.5)
        set_openai(page, '#aiProvider', '#openaiServerUrl', '#openaiModel')
        shot(page, idx, key, '1_openai_config')
        try:
            page.fill('#userInput', 'Similar to Midnight Harbor by The Velvet Echo', timeout=4000)
            page.click('#submitChat', timeout=4000)
            try:
                page.wait_for_selector(
                    '#responseArea ol, #createPlaylistSection:not(.hidden)', timeout=9000
                )
            except Exception:
                pass
            page.wait_for_timeout(1500)
            shot(page, idx, key, '2_similar_song_result')
        except Exception as e:
            print("CHAT_FAIL", repr(e), flush=True)
    elif key == 'similarity':
        settle(page, 2.5)
        shot(page, idx, key, '1_song')
        if click(page, '#similarity-mode-toggle'):
            page.wait_for_timeout(800)
            shot(page, idx, key, '2_mood')
        if click(page, '#similarity-mode-toggle'):
            page.wait_for_timeout(800)
            shot(page, idx, key, '3_anchor')
    elif key == 'alchemy':
        settle(page, 2.5)
        shot(page, idx, key, '1_alchemy')
        if click(page, '#tab-anchors'):
            page.wait_for_timeout(700)
            shot(page, idx, key, '2_anchors')
        if click(page, '#tab-radio'):
            page.wait_for_timeout(700)
            shot(page, idx, key, '3_radio')
    elif key == 'lyrics_search':
        settle(page, 2.5)
        shot(page, idx, key, '1_axis')
        if click(page, '.tab-btn[data-tab="text"]'):
            page.wait_for_timeout(500)
            shot(page, idx, key, '2_text')
        if click(page, '.tab-btn[data-tab="song"]'):
            page.wait_for_timeout(500)
            shot(page, idx, key, '3_song')
    elif key == 'artist_similarity':
        settle(page, 2)
        if autocomplete(page, '#artist_search', '#autocomplete-results', q='a'):
            shot(page, idx, key, '1_autocomplete')
        if click(page, '#find-artists-btn'):
            try:
                page.wait_for_selector('#results-table-wrapper .results-table', timeout=6000)
            except Exception:
                pass
            page.wait_for_timeout(1200)
            shot(page, idx, key, '2_results')
    elif key == 'path':
        settle(page, 2.5)
        shot(page, idx, key, '1_form')
        if autocomplete(
            page, '#start_search', '#start-autocomplete-results .autocomplete-item', q='a'
        ):
            shot(page, idx, key, '2_autocomplete')
    elif key == 'waveform':
        settle(page, 2)
        if autocomplete(page, '#search_query', '#autocomplete-results .autocomplete-item', q='a'):
            shot(page, idx, key, '1_autocomplete')
    elif key == 'clap_search':
        settle(page, 2)
        shot(page, idx, key, '1_form')
        try:
            page.fill('#search-query', 'energetic indie rock', timeout=3000)
            page.click('#search-form button[type="submit"]', timeout=3000)
            try:
                page.wait_for_selector('#results-list .result-item', timeout=7000)
            except Exception:
                pass
            page.wait_for_timeout(1500)
            shot(page, idx, key, '2_results')
        except Exception:
            pass
    elif key == 'map':
        settle(page, 6)
        shot(page, idx, key)
    elif key == 'dashboard':
        settle(page, 4)
        shot(page, idx, key)
    elif key in ('setup', 'users', 'provider_migration', 'sonic_fingerprint'):
        settle(page, 3)
        shot(page, idx, key)
    else:
        settle(page, 2.5)
        shot(page, idx, key)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        ctx = browser.new_context(viewport={'width': 1440, 'height': 900}, ignore_https_errors=True)
        ctx.add_init_script(INIT_JS)
        ctx.route("**/api/**", handle_route)
        page = ctx.new_page()

        page.goto(BASE + "/login", wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(800)
        shot(page, 0, 'login')

        page.fill('#login-user', USER)
        page.fill('#login-password', PW)
        page.click('button[type="submit"]')
        try:
            page.wait_for_url(lambda u: '/login' not in u, timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(1500)
        print("AFTER_LOGIN_URL", page.url, flush=True)

        links = page.eval_on_selector_all(
            '.sidebar-nav a[href]',
            "els => els.map(e => ({text: e.textContent.trim(), href: e.getAttribute('href')}))",
        )
        seen, pages = set(), []
        for link in links:
            href = (link.get('href') or '').strip()
            text = (link.get('text') or '').strip()
            if not href or href.startswith('#') or href.startswith('javascript'):
                continue
            key = TEXT2KEY.get(text)
            if not key:
                for t, k in TEXT2KEY.items():
                    if t in text:
                        key = k
                        break
            if not key or href in seen:
                continue
            seen.add(href)
            pages.append((key, href))
        print("PAGES", json.dumps(pages), flush=True)

        idx = 1
        for key, href in pages:
            url = href if href.startswith('http') else BASE + href
            print("VISIT", idx, key, url, flush=True)
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=30000)
                recipe(page, idx, key)
            except Exception as e:
                print("PAGE_ERR", key, repr(e), flush=True)
                traceback.print_exc()
            idx += 1

        browser.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
