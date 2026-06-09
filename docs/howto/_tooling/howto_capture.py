"""Capture every AudioMuse-AI page into docs/howto/<version>/screenshots/.

Drives a real browser (Playwright) against a running, analysed AudioMuse-AI
instance. Two safety properties make the output publishable:

  * Copyright masking — every /api/** JSON response is intercepted and the
    track title / artist / album fields are rewritten to neutral placeholders
    (Song Title N, Artist Name N, Album Name N) BEFORE the page renders them.
  * Secret redaction — server URLs, user IDs, tokens and passwords on the
    Setup, Sonic Fingerprint, Analysis, Instant Playlist and Users pages are
    blanked just before each screenshot.

A few "result" screenshots (analysis/clustering status, AI playlist, artist
results, sonic fingerprint) are produced by overriding the relevant API with
representative placeholder data, or injecting an equivalent result into the
DOM, so the success state is shown without running heavy or destructive jobs.

Only read-only features are exercised; nothing is written to the media server.

Usage (local, against your own instance):
    python howto_capture.py --base-url http://192.168.3.204:8000 \
        --user root --password root
Credentials and URL also read from HOWTO_BASE_URL / HOWTO_USER /
HOWTO_PASSWORD. The output version defaults to APP_VERSION in config.py.
"""
import argparse
import json
import math
import os
import re
import traceback
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from _version import REPO_ROOT, resolve

# --------------------------------------------------------------------------
# Metadata masking
# --------------------------------------------------------------------------
KEY_CATEGORY = {
    "title": "title", "track": "title", "song": "title",
    "old_track": "title", "new_track": "title",
    "loser_title": "title", "winner_title": "title", "target_title": "title",
    "author": "artist", "artist": "artist", "artist_name": "artist",
    "album_artist": "artist",
    "old_artist": "artist", "new_artist": "artist",
    "old_album_artist": "artist", "new_album_artist": "artist",
    "loser_artist": "artist", "winner_artist": "artist", "target_artist": "artist",
    "album": "album",
    "old_album": "album", "new_album": "album",
    "loser_album": "album", "winner_album": "album", "target_album": "album",
}
LIST_TITLE_KEYS = {"sample_titles", "representative_songs",
                   "artist1_representative_songs", "artist2_representative_songs"}
PATH_KEYS = {"loser_path", "winner_path", "target_path", "old_path", "new_path", "path"}
CATEGORY_LABEL = {"title": "Song Title", "artist": "Artist Name", "album": "Album Name"}

_counters = {"title": 0, "artist": 0, "album": 0}
_cache = {}


def _placeholder(category, real):
    key = (category, str(real))
    if key in _cache:
        return _cache[key]
    _counters[category] += 1
    val = "%s %d" % (CATEGORY_LABEL[category], _counters[category])
    _cache[key] = val
    return val


_EMBED_RE = re.compile(r"(title|artist|author|album)\s*=\s*'([^']*)'", re.IGNORECASE)


def _scrub_string(s):
    if not isinstance(s, str) or "=" not in s:
        return s

    def repl(m):
        field = m.group(1).lower()
        cat = "title" if field == "title" else ("album" if field == "album" else "artist")
        return "%s='%s'" % (m.group(1), _placeholder(cat, m.group(2)))

    return _EMBED_RE.sub(repl, s)


def mask(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = k.lower() if isinstance(k, str) else k
            if lk in KEY_CATEGORY and isinstance(v, str) and v.strip():
                out[k] = _placeholder(KEY_CATEGORY[lk], v)
            elif lk in PATH_KEYS and isinstance(v, str) and v.strip():
                out[k] = "/music/Artist Name/Album Name/track.flac"
            elif lk in LIST_TITLE_KEYS and isinstance(v, list):
                out[k] = [_placeholder("title", x) if isinstance(x, str)
                          else (mask(x) if isinstance(x, (dict, list)) else x) for x in v]
            elif isinstance(v, (dict, list)):
                out[k] = mask(v)
            elif isinstance(v, str):
                out[k] = _scrub_string(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [mask(x) for x in obj]
    if isinstance(obj, str):
        return _scrub_string(obj)
    return obj


def make_route_handler(mock_all=False):
    def handle(route):
        url = route.request.url
        if "stream" in url:
            return route.continue_()
        if mock_all:
            data = build_mock(url, route.request.method)
            if data is not None:
                try:
                    return route.fulfill(status=200, content_type="application/json",
                                         body=json.dumps(data, ensure_ascii=False))
                except Exception:
                    pass
        try:
            resp = route.fetch()
            ct = (resp.headers or {}).get("content-type", "")
            if "application/json" not in ct:
                return route.fulfill(response=resp)
            body = json.dumps(mask(resp.json()), ensure_ascii=False)
            return route.fulfill(response=resp, body=body, content_type="application/json")
        except Exception:
            try:
                return route.continue_()
            except Exception:
                return
    return handle


def fulfill_json(data):
    def handler(route):
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps(data, ensure_ascii=False))
    return handler


# --------------------------------------------------------------------------
# Representative placeholder data for emulated "result" screenshots
# --------------------------------------------------------------------------
GENRES = ["rock", "pop", "jazz", "electronic", "hip hop", "folk", "metal", "classical"]


def _mood(i):
    h = round(0.30 + (i * 7 % 60) / 100, 2)
    s = round(0.90 - h, 2)
    a = round(0.25 + (i * 11 % 55) / 100, 2)
    r = round(0.95 - a, 2)
    party = round(0.40 + (i * 13 % 55) / 100, 2)
    dance = round(0.45 + (i * 17 % 50) / 100, 2)
    return "happy:%s,sad:%s,aggressive:%s,relaxed:%s,party:%s,danceable:%s" % (h, s, a, r, party, dance)


SONIC_DATA = [
    {"item_id": "demo-%d" % i, "title": "Song Title %d" % (200 + i),
     "author": "Artist Name %d" % (1 + (i % 18)), "album": "Album Name %d" % (1 + (i % 9)),
     "distance": round(0.05 + i * 0.018, 4), "top_genre": GENRES[i % len(GENRES)],
     "other_features": _mood(i), "mood_vector": _mood(i)}
    for i in range(1, 31)
]

ARTIST_DATA = []
for _i in range(1, 9):
    _cm = [{"component1_index": 0, "component2_index": 1, "distance": round(0.08 + 0.03 * _i, 3),
            "artist1_representative_songs": [{"item_id": "a%ds1" % _i, "title": "Song Title %d" % (_i * 10 + 1)},
                                             {"item_id": "a%ds2" % _i, "title": "Song Title %d" % (_i * 10 + 2)}],
            "artist2_representative_songs": [{"item_id": "b%ds1" % _i, "title": "Song Title %d" % (_i * 10 + 3)},
                                             {"item_id": "b%ds2" % _i, "title": "Song Title %d" % (_i * 10 + 4)}]}]
    if _i == 1:
        _cm.append({"component1_index": 2, "component2_index": 0, "distance": 0.142,
                    "artist1_representative_songs": [{"item_id": "a1s3", "title": "Song Title 15"}],
                    "artist2_representative_songs": [{"item_id": "b1s3", "title": "Song Title 16"}]})
    ARTIST_DATA.append({"artist": "Artist Name %d" % _i, "artist_id": "art%d" % _i,
                        "divergence": round(0.42 + 0.13 * _i, 4), "component_matches": _cm})

# --------------------------------------------------------------------------
# Mock-all data builders (used only with --mock-all, i.e. an empty CI instance)
# --------------------------------------------------------------------------
def _songs(n, album=True, metric="distance"):
    out = []
    for i in range(1, n + 1):
        s = {"item_id": "trk-%d" % i, "title": "Song Title %d" % i,
             "author": "Artist Name %d" % (((i - 1) % 40) + 1),
             "top_genre": GENRES[i % len(GENRES)],
             "other_features": _mood(i), "mood_vector": _mood(i)}
        if album:
            s["album"] = "Album Name %d" % (((i - 1) % 15) + 1)
            s["album_artist"] = s["author"]
        if metric == "distance":
            s["distance"] = round(0.05 + i * 0.01, 4)
        elif metric == "similarity":
            s["similarity"] = round(max(0.40, 0.98 - i * 0.02), 3)
        out.append(s)
    return out


def _map_items(n=600):
    items = []
    for i in range(n):
        g = GENRES[i % len(GENRES)]
        ang = 2 * math.pi * ((i % len(GENRES)) / len(GENRES))
        cx, cy = math.cos(ang) * 3.0, math.sin(ang) * 3.0
        jx = math.cos(i * 1.7) * ((i % 17) / 17.0)
        jy = math.sin(i * 2.3) * ((i % 13) / 13.0)
        items.append({"item_id": str(i), "title": "Song Title %d" % (i + 1),
                      "artist": "Artist Name %d" % ((i % 200) + 1),
                      "embedding_2d": [round(cx + jx, 3), round(cy + jy, 3)],
                      "mood_vector": g})
    return {"items": items, "projection": "PCA → t-SNE (demo projection)"}


def _dashboard():
    genres = [{"label": g, "count": 420 - 30 * i} for i, g in enumerate(GENRES)]
    moods = [{"label": m, "score": round(300 - 20 * i, 2)}
             for i, m in enumerate(["happy", "relaxed", "party", "danceable", "aggressive", "sad"])]
    return {
        "generated_at": "2026-01-01 12:00:00", "stats_updated_at": "2026-01-01 11:00:00",
        "workers": [{"hostname": "audiomuse-worker-1", "queues": ["high", "default"],
                     "state": "idle", "current_job_id": None,
                     "successful_jobs": 128, "failed_jobs": 0}],
        "recent_tasks": [
            {"task_id": "a1b2", "task_type": "main_analysis", "status": "SUCCESS",
             "duration_seconds": 134.0, "note": "Analyzed 25 albums (412 tracks).",
             "timestamp": "2026-01-01 02:02:14"},
            {"task_id": "c3d4", "task_type": "main_clustering", "status": "SUCCESS",
             "duration_seconds": 408.0, "note": "Generated 200 playlists.",
             "timestamp": "2025-12-31 02:06:48"}],
        "content": {"total_songs": 12480, "distinct_artists": 1320, "distinct_albums": 2140,
                    "musicnn_indexed": 12480, "clap_indexed": 12480, "gmm_indexed": 1320,
                    "top_genre": genres, "moods_coverage": moods,
                    "tempo_profile": {"slow": 2200, "medium": 4100, "fast": 4300,
                                      "very_fast": 1880, "avg_tempo": 118.4}},
        "cron": [{"id": 1, "name": "Nightly Analysis", "task_type": "main_analysis",
                  "cron_expr": "0 2 * * 0-5", "enabled": True, "last_run": "2026-01-01 02:00:00"}],
    }


def _playlists():
    names = ["Evening Drive_automatic", "Late Night Focus_automatic",
             "Sunday Morning_automatic", "Workout Energy_automatic"]
    return {nm: [{"title": "Song Title %d" % (j + 1), "author": "Artist Name %d" % ((j % 20) + 1)}
                 for j in range(8)] for nm in names}


def build_mock(url, method):
    """Return placeholder JSON for a data endpoint, or None to pass the request
    through to the real app (e.g. /api/setup, /api/users, /api/config, health,
    warmup — whose real empty-DB/config responses render fine and are masked)."""
    path = urlparse(url).path

    def has(*subs):
        return any(s in path for s in subs)

    if path.endswith("/api/search_tracks"):
        return _songs(20, album=True)
    if path.endswith("/api/search_artists"):
        return [{"artist": "Artist Name %d" % i, "track_count": 400 - 13 * i} for i in range(1, 21)]
    if path.endswith("/api/similar_tracks"):
        return _songs(25, metric="distance")
    if path.endswith("/api/similar_artists"):
        return ARTIST_DATA
    if path.endswith("/api/artist_tracks"):
        return [{"item_id": "at%d" % i, "title": "Song Title %d" % i, "author": "Artist Name 1"}
                for i in range(1, 13)]
    if path.endswith("/api/find_path"):
        return {"path": _songs(15, metric="distance")}
    if path.endswith("/api/alchemy"):
        s = _songs(20, metric="distance")
        for i, x in enumerate(s):
            x["embedding_2d"] = [round(math.cos(i) * 2, 3), round(math.sin(i) * 2, 3)]
        return {"results": s, "filtered_out": [], "add_points": s[:2], "sub_points": []}
    if path.endswith("/api/clap/search"):
        return {"results": _songs(20, metric="similarity"), "query": "energetic upbeat saxophone jazz"}
    if has("/api/lyrics/search/axes", "/api/lyrics/search/text"):
        return {"results": _songs(20, metric="similarity")}
    if has("/api/sem_grove/search"):
        s = _songs(20, metric="similarity")
        s[0]["is_seed"] = True
        return {"results": s}
    if path.endswith("/api/map"):
        return _map_items()
    if path.endswith("/api/waveform"):
        return {"peaks": [round(abs(math.sin(i / 6.0)) * (0.4 + 0.6 * ((i % 50) / 50.0)), 3)
                          for i in range(500)],
                "title": "Song Title 1", "author": "Artist Name 1"}
    if has("/api/sonic_fingerprint/generate"):
        return SONIC_DATA
    if path.endswith("/api/playlists"):
        return _playlists()
    if path.endswith("/api/dashboard/summary"):
        return _dashboard()
    return None


# --------------------------------------------------------------------------
# Injected JS
# --------------------------------------------------------------------------
FORCE_UI = """
() => { try {
  localStorage.setItem('menuOpen','true');
  localStorage.setItem('theme','light');
  document.documentElement.classList.add('sidebar-open');
} catch (e) {} }
"""

REDACT_JS = r"""
(pageKey) => {
  const ipRe = /\b\d{1,3}(?:\.\d{1,3}){3}\b/;
  const setVal=(sel,v)=>document.querySelectorAll(sel).forEach(e=>{e.value=v;});
  if (pageKey === 'aiurls') {
    document.querySelectorAll('input[type=text], input:not([type]), input[type=url]').forEach(e=>{
      const v=e.value||''; if(/^https?:\/\//i.test(v)||ipRe.test(v)) e.value='http://ai-server.example.com:11434/api/generate';
    });
  }
  if (pageKey === 'setup') {
    document.querySelectorAll('input[type=password]').forEach(e=>e.value='');
    document.querySelectorAll('input[type=text], input:not([type]), input[type=url]').forEach(e=>{
      const v=(e.value||'').trim();
      if(/^https?:\/\//i.test(v)||ipRe.test(v)){e.value='https://media.example.com';return;}
      if(/^[0-9a-fA-F]{16,}$/.test(v)||/^[0-9a-fA-F]{8}-[0-9a-fA-F-]{20,}$/.test(v)){e.value='demo-user-id-0000';return;}
    });
    setVal('#AUDIOMUSE_USER','demo_admin');
  }
  if (pageKey === 'sonic') {
    setVal('#jellyfin_user_identifier','your-username');
    setVal('#navidrome_user','your-username');
    document.querySelectorAll('input[type=password]').forEach(e=>e.value='');
  }
  if (pageKey === 'users') {
    document.querySelectorAll('#additional-users-tbody tr').forEach((tr,i)=>{
      const td=tr.querySelector('td'); if(td&&td.textContent.trim()&&td.textContent.trim()!=='root') td.textContent='user'+(i+1);
    });
  }
  if (pageKey === 'dashboard') {
    document.querySelectorAll('#workers-table tbody tr').forEach((tr,i)=>{
      const td=tr.querySelector('td'); if(td&&!td.classList.contains('muted')) td.textContent='audiomuse-worker-'+(i+1);
    });
  }
}
"""

CHAT_RESULT_JS = """
() => {
  const ra=document.getElementById('responseArea'); if(!ra) return;
  ra.innerHTML=''; ra.classList.remove('text-red-600');
  const ok='var(--color-success)';
  const cw=document.createElement('div'); cw.style.margin='0.25rem 0 1rem 0';
  const chain=document.createElement('div'); chain.className='tool-chain';
  ['Understand request','Brainstorm themes','Find similar tracks','Assemble playlist'].forEach((n,i)=>{
    if(i>0){const a=document.createElement('span');a.className='tool-chain-arrow';a.textContent='\\u2192';chain.appendChild(a);}
    const node=document.createElement('span');node.className='tool-chain-node';node.textContent=n;chain.appendChild(node);
  });
  cw.appendChild(chain); ra.appendChild(cw);
  const d=document.createElement('details'); d.style.marginBottom='1rem';
  const s=document.createElement('summary'); s.style.cursor='pointer'; s.style.fontWeight='600'; s.style.padding='0.75rem'; s.style.borderRadius='0.375rem'; s.style.background='var(--bg-code,#f3f4f6)'; s.style.color='var(--text-muted)'; s.textContent='Technical details (raw log)';
  d.appendChild(s); ra.appendChild(d);
  const rd=document.createElement('div'); rd.style.marginTop='1.5rem'; rd.style.padding='1.5rem'; rd.style.borderRadius='0.5rem'; rd.style.border='2px solid '+ok; rd.style.backgroundColor='var(--status-success-bg)';
  const h=document.createElement('h3'); h.style.fontWeight='700'; h.style.marginBottom='1rem'; h.style.color=ok;
  const N=12; h.textContent='Generated Playlist ('+N+' songs)'; rd.appendChild(h);
  const ol=document.createElement('ol'); ol.className='song-list'; ol.style.maxHeight='400px'; ol.style.overflowY='auto';
  for(let i=1;i<=N;i++){const li=document.createElement('li'); li.textContent='Song Title '+(100+i)+' by Artist Name '+i; ol.appendChild(li);}
  rd.appendChild(ol); ra.appendChild(rd);
  const cps=document.getElementById('createPlaylistSection'); if(cps) cps.classList.remove('hidden');
}
"""

ANALYSIS_STATUS_JS = """
(s) => {
  const set=(id,v)=>{const e=document.getElementById(id); if(e) e.textContent=v;};
  set('status-task-id', s.taskId); set('status-running-time', s.runtime);
  set('status-task-type', s.type); set('status-status', s.status);
  set('status-log', s.log); set('status-progress', '100');
  const pb=document.getElementById('progress-bar'); if(pb){ pb.style.width='100%'; pb.style.background='var(--color-success,#16A34A)'; }
  const sd=document.getElementById('status-details'); if(sd){ sd.textContent=s.details; }
  const ss=document.getElementById('status-status'); if(ss){ ss.className='status-success'; }
}
"""

MAP_PATH_JS = """
() => {
  const pts=(window._plotPointsFull||[]).slice();
  if(pts.length<12) return;
  let cur=pts[Math.floor(pts.length*0.4)];
  const used=new Set([cur.id]); const path=[cur];
  for(let k=0;k<11;k++){
    let best=null,bd=Infinity;
    for(const p of pts){ if(used.has(p.id))continue; const d=(p.x-cur.x)*(p.x-cur.x)+(p.y-cur.y)*(p.y-cur.y); if(d<bd){bd=d;best=p;} }
    if(!best)break; used.add(best.id); path.push(best); cur=best;
  }
  const items=path.map(p=>({item_id:p.id,x:p.x,y:p.y,embedding_2d:[p.x,p.y],title:p.title,artist:p.artist,mood_vector:p.genre}));
  try{ appendSongsToSelectionPanel(items); }catch(e){}
  try{ drawPathOnMap(items,'rgba(20,20,20,0.9)'); }catch(e){}
}
"""

AXIS_SELECT_JS = """
() => {
  const sels=[...document.querySelectorAll('#axes-container select')]; let set=0;
  for(const s of sels){ if(set>=2) break; if(s.options.length>1){ s.selectedIndex=1; s.dispatchEvent(new Event('change')); set++; } }
}
"""


# When the backend indexes aren't built (empty CI DB), lyrics/DCLAP render their
# inputs disabled with an "index not built" banner. In --mock-all mode we enable
# them and hide the banner so the search demo can run (data comes from build_mock).
ENABLE_INPUTS_JS = """
(scope) => {
  document.querySelectorAll(scope).forEach(root => {
    root.querySelectorAll('[disabled]').forEach(e => e.removeAttribute('disabled'));
  });
  document.querySelectorAll('.error-message').forEach(e => { e.style.display = 'none'; });
}
"""


# --------------------------------------------------------------------------
# Playwright helpers
# --------------------------------------------------------------------------
def launch(p, channel):
    if channel:
        try:
            return p.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return p.chromium.launch(headless=True)


def capture(base_url, user, password, out_dir, channel="chrome", mock_all=False):
    base_url = base_url.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    def shot(page, name, full=True):
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(out_dir, name), full_page=full)
        print("  saved", name)

    def goto(page, path):
        page.goto(base_url + path, wait_until="networkidle", timeout=40000)
        page.wait_for_timeout(800)

    def redact(page, key):
        try:
            page.evaluate(REDACT_JS, key)
        except Exception as e:
            print("  redact warn:", e)

    def enable_mock(scope):
        if not mock_all:
            return
        try:
            page.evaluate(ENABLE_INPUTS_JS, scope)
        except Exception as e:
            print("  enable warn:", e)

    def autocomplete(page, input_sel, text, results_sel, item_sel=".autocomplete-item"):
        try:
            page.click(input_sel)
            page.fill(input_sel, "")
            page.type(input_sel, text, delay=70)
            page.wait_for_selector("%s %s" % (results_sel, item_sel), timeout=8000, state="visible")
            page.wait_for_timeout(400)
            return True
        except Exception as e:
            print("  autocomplete miss (%s): %s" % (input_sel, e))
            return False

    def pick_first(page, results_sel, item_sel=".autocomplete-item"):
        try:
            page.locator("%s %s" % (results_sel, item_sel)).first.click(timeout=4000)
            page.wait_for_timeout(500)
            return True
        except Exception as e:
            print("  pick miss:", e)
            return False

    with sync_playwright() as p:
        browser = launch(p, channel)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        ctx.add_init_script(FORCE_UI)
        ctx.route("**/api/**", make_route_handler(mock_all))
        page = ctx.new_page()
        page.set_default_timeout(15000)

        def block(label, fn):
            print("==", label)
            try:
                fn()
            except Exception:
                print("  ERROR", label)
                traceback.print_exc()

        # --- 00 login (unauthenticated) ---
        block("login", lambda: (goto(page, "/login"), shot(page, "00-login.png", full=False)))

        page.fill("#login-user", user)
        page.fill("#login-password", password)
        page.click("#login-form button[type=submit]")
        page.wait_for_load_state("networkidle", timeout=40000)
        print("logged in ->", page.url)

        # --- 01 dashboard ---
        def dashboard():
            goto(page, "/")
            page.wait_for_timeout(3500)
            redact(page, "dashboard")
            shot(page, "01-dashboard.png")
        block("dashboard", dashboard)

        # --- 02a analysis result ---
        def analysis_result():
            goto(page, "/analysis")
            page.wait_for_timeout(600)
            redact(page, "aiurls")
            page.evaluate(ANALYSIS_STATUS_JS, {
                "taskId": "a1b2c3d4-e5f6-47a8-9b0c-analysis001", "runtime": "00 : 02 : 14",
                "type": "main_analysis", "status": "SUCCESS",
                "log": "Main analysis complete. Analyzed 25 new albums (412 tracks). Skipped 0 already-analyzed albums.",
                "details": '{\n  "task_type": "main_analysis",\n  "status": "SUCCESS",\n  "albums_analyzed": 25,\n  "tracks_analyzed": 412,\n  "skipped": 0\n}'})
            shot(page, "02a-analysis-result.png")
        block("analysis-result", analysis_result)

        # --- 02b clustering result ---
        def clustering_result():
            goto(page, "/analysis")
            try:
                page.click("#fetch-playlists-btn", timeout=4000)
                page.wait_for_timeout(2500)
            except Exception as e:
                print("  fetch warn:", e)
            redact(page, "aiurls")
            page.evaluate(ANALYSIS_STATUS_JS, {
                "taskId": "f9e8d7c6-b5a4-4321-8765-clustering01", "runtime": "00 : 06 : 48",
                "type": "main_clustering", "status": "SUCCESS",
                "log": "Clustering complete. Generated 200 playlists from 8 clustering runs (best score 0.8123).",
                "details": '{\n  "task_type": "main_clustering",\n  "status": "SUCCESS",\n  "playlists_created": 200,\n  "best_score": 0.8123,\n  "clustering_runs": 8\n}'})
            shot(page, "02b-clustering-result.png")
        block("clustering-result", clustering_result)

        # --- 03 instant playlist form ---
        def chat_form():
            goto(page, "/chat/")
            try:
                page.fill("#userInput", "Calm rainy-day piano songs for focus")
            except Exception as e:
                print("  fill warn:", e)
            redact(page, "aiurls")
            shot(page, "03-instant-playlist.png")
        block("chat-form", chat_form)

        # --- 03b instant playlist AI result ---
        def chat_result():
            goto(page, "/chat/")
            try:
                page.fill("#userInput", "Calm rainy-day piano songs for focus")
            except Exception:
                pass
            redact(page, "aiurls")
            page.evaluate(CHAT_RESULT_JS)
            shot(page, "03b-instant-playlist-result.png")
        block("chat-result", chat_result)

        # --- 04 similarity (autocomplete + results) ---
        def similarity():
            goto(page, "/similarity")
            if autocomplete(page, "#search_query", "love", "#autocomplete-results"):
                shot(page, "04b-similarity-autocomplete.png", full=False)
                pick_first(page, "#autocomplete-results")
            try:
                page.click("#similarity-form button[type=submit]", timeout=4000)
                page.wait_for_selector("#results-table-wrapper .result-item", timeout=25000)
                page.wait_for_timeout(800)
            except Exception as e:
                print("  similarity run warn:", e)
            shot(page, "04-similarity.png")
        block("similarity", similarity)

        # --- 05b artist autocomplete ---
        def artist_autocomplete():
            goto(page, "/artist_similarity")
            autocomplete(page, "#artist_search", "the", "#autocomplete-results")
            page.wait_for_timeout(500)
            shot(page, "05b-artist-autocomplete.png", full=False)
        block("artist-autocomplete", artist_autocomplete)

        # --- 05 artist similarity results (API override) ---
        def artist_result():
            page.route("**/api/similar_artists**", fulfill_json(ARTIST_DATA))
            try:
                goto(page, "/artist_similarity")
                page.evaluate("() => { const e=document.getElementById('artist_search'); if(e) e.value='Artist Name 1'; }")
                page.click("#find-artists-btn", timeout=5000)
                page.wait_for_selector("#results-table-wrapper table", timeout=15000)
                page.wait_for_timeout(500)
                try:
                    page.click('.expand-btn[data-mode="components"]', timeout=4000)
                    page.wait_for_timeout(700)
                except Exception as e:
                    print("  expand warn:", e)
                shot(page, "05-artist-similarity.png")
            finally:
                page.unroute("**/api/similar_artists**")
        block("artist-result", artist_result)

        # --- 06 song path ---
        def song_path():
            goto(page, "/path")
            if autocomplete(page, "#start_search", "love", "#start-autocomplete-results"):
                pick_first(page, "#start-autocomplete-results")
            if autocomplete(page, "#end_search", "night", "#end-autocomplete-results"):
                pick_first(page, "#end-autocomplete-results")
            try:
                page.click("#path-form button[type=submit]", timeout=4000)
                page.wait_for_selector("#results-table-wrapper .result-item", timeout=30000)
                page.wait_for_timeout(1200)
            except Exception as e:
                print("  path run warn:", e)
            shot(page, "06-song-path.png")
        block("song-path", song_path)

        # --- 07 song alchemy ---
        def alchemy():
            goto(page, "/alchemy")
            if autocomplete(page, ".song", "love", ".autocomplete-results"):
                pick_first(page, ".autocomplete-results")
            try:
                page.click("#run-alchemy", timeout=4000)
                page.wait_for_selector("#results-table-wrapper .result-item", timeout=30000)
                page.wait_for_timeout(1200)
            except Exception as e:
                print("  alchemy run warn:", e)
            shot(page, "07-song-alchemy.png")
        block("alchemy", alchemy)

        # --- 08 DCLAP ---
        def clap():
            goto(page, "/clap_search")
            enable_mock('#search-form')
            try:
                page.fill("#search-query", "energetic upbeat saxophone jazz")
                page.click("#search-form button[type=submit]", timeout=4000)
                page.wait_for_selector("#results-list .result-item", timeout=30000)
                page.wait_for_timeout(800)
            except Exception as e:
                print("  clap run warn:", e)
            shot(page, "08-dclap-search.png")
        block("clap", clap)

        # --- 09a lyrics by axis ---
        def lyrics_axis():
            goto(page, "/lyrics_search")
            page.click('.tab-btn[data-tab="axes"]', timeout=5000)
            page.wait_for_timeout(300)
            enable_mock('#axis-form')
            page.evaluate(AXIS_SELECT_JS)
            page.click('#axis-form button[type="submit"]', timeout=5000)
            page.wait_for_selector("#results-list .result-item", timeout=30000)
            page.wait_for_timeout(700)
            shot(page, "09a-lyrics-axis.png")
        block("lyrics-axis", lyrics_axis)

        # --- 09b lyrics by text ---
        def lyrics_text():
            goto(page, "/lyrics_search")
            page.click('.tab-btn[data-tab="text"]', timeout=5000)
            page.wait_for_selector("#search-query", state="visible", timeout=8000)
            enable_mock('#search-form')
            page.fill("#search-query", "love and heartbreak in the city at night")
            page.click('#search-form button[type="submit"]', timeout=5000)
            page.wait_for_selector("#results-list .result-item", timeout=30000)
            page.wait_for_timeout(700)
            shot(page, "09b-lyrics-text.png")
        block("lyrics-text", lyrics_text)

        # --- 09c lyrics by song (SemGrove) ---
        def lyrics_song():
            goto(page, "/lyrics_search")
            page.click('.tab-btn[data-tab="song"]', timeout=5000)
            page.wait_for_selector("#sg-search-query", state="visible", timeout=8000)
            enable_mock('#song-form')
            if autocomplete(page, "#sg-search-query", "love", "#sg-autocomplete-results"):
                pick_first(page, "#sg-autocomplete-results")
            page.click('#song-form button[type="submit"]', timeout=5000)
            page.wait_for_selector("#results-list .result-item", timeout=30000)
            page.wait_for_timeout(700)
            shot(page, "09c-lyrics-song.png")
        block("lyrics-song", lyrics_song)

        # --- 10 music map ---
        def music_map():
            goto(page, "/map")
            try:
                page.click("#btn-pct-25", timeout=4000)
            except Exception as e:
                print("  map pct warn:", e)
            page.wait_for_selector("#plot svg, #plot canvas", timeout=40000)
            page.wait_for_timeout(3500)
            shot(page, "10-music-map.png", full=False)
        block("map", music_map)

        # --- 10b music map with a path drawn ---
        def map_path():
            goto(page, "/map")
            try:
                page.click("#btn-pct-25", timeout=4000)
            except Exception:
                pass
            page.wait_for_selector("#plot svg, #plot canvas", timeout=40000)
            page.wait_for_timeout(3500)
            page.evaluate(MAP_PATH_JS)
            page.wait_for_timeout(1500)
            shot(page, "10b-music-map-path.png", full=False)
        block("map-path", map_path)

        # --- 11 sonic fingerprint form ---
        def sonic_form():
            goto(page, "/sonic_fingerprint")
            page.wait_for_timeout(800)
            redact(page, "sonic")
            shot(page, "11-sonic-fingerprint.png")
        block("sonic-form", sonic_form)

        # --- 11b sonic fingerprint result (API override) ---
        def sonic_result():
            page.route("**/api/sonic_fingerprint/generate**", fulfill_json(SONIC_DATA))
            try:
                goto(page, "/sonic_fingerprint")
                page.wait_for_timeout(700)
                page.evaluate("() => { const j=document.getElementById('jellyfin_user_identifier'); if(j) j.value='demo'; const n=document.getElementById('navidrome_user'); if(n) n.value='demo'; }")
                page.click('#fingerprint-form button[type="submit"]', timeout=5000)
                page.wait_for_selector("#results-table-wrapper .result-item", timeout=20000)
                page.wait_for_timeout(1200)
                shot(page, "11b-sonic-fingerprint-result.png")
            finally:
                page.unroute("**/api/sonic_fingerprint/generate**")
        block("sonic-result", sonic_result)

        # --- 12 waveform ---
        def waveform():
            goto(page, "/waveform")
            if autocomplete(page, "#search_query", "love", "#autocomplete-results"):
                pick_first(page, "#autocomplete-results")
            try:
                page.click("#generate-waveform-btn", timeout=4000)
                page.wait_for_selector("#waveform-canvas", timeout=30000, state="visible")
                page.wait_for_timeout(2500)
            except Exception as e:
                print("  waveform run warn:", e)
            shot(page, "12-waveform.png")
        block("waveform", waveform)

        # --- 13-16 admin (screenshot only) ---
        block("cleaning", lambda: (goto(page, "/cleaning"), shot(page, "13-cleaning.png")))
        block("cron", lambda: (goto(page, "/cron"), page.wait_for_timeout(800), shot(page, "14-scheduled-tasks.png")))
        block("backup", lambda: (goto(page, "/backup"), shot(page, "15-backup-restore.png")))
        block("migration", lambda: (goto(page, "/provider-migration"), page.wait_for_timeout(800), shot(page, "16-provider-migration.png")))

        # --- 17 setup wizard (redact secrets) ---
        def setup():
            goto(page, "/setup")
            page.wait_for_timeout(1500)
            redact(page, "setup")
            shot(page, "17-setup-wizard.png")
        block("setup", setup)

        # --- 18 users (open add panel + redact) ---
        def users():
            goto(page, "/users")
            page.wait_for_timeout(600)
            try:
                page.click("#add-user-toggle", timeout=3000)
                page.wait_for_timeout(500)
            except Exception as e:
                print("  users toggle warn:", e)
            redact(page, "users")
            shot(page, "18-users.png")
        block("users", users)

        print("Capture complete. Placeholders used:", _counters)
        ctx.close()
        browser.close()


def main():
    ap = argparse.ArgumentParser(description="Capture AudioMuse-AI how-to screenshots.")
    ap.add_argument("--base-url", default=os.environ.get("HOWTO_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--user", default=os.environ.get("HOWTO_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("HOWTO_PASSWORD", "root"))
    ap.add_argument("--version", default=None,
                    help="Version/tag (e.g. v2.2.0). Defaults to APP_VERSION in config.py.")
    ap.add_argument("--out", default=None, help="Override the screenshots output directory.")
    ap.add_argument("--browser-channel", default="chrome",
                    help="Browser channel (chrome/msedge). Empty string uses bundled chromium.")
    ap.add_argument("--mock-all", action="store_true",
                    help="Fulfill every data /api/** with placeholder JSON (for an empty CI instance).")
    args = ap.parse_args()

    out = args.out
    if not out:
        _disp, folder = resolve(args.version)
        out = os.path.join(str(REPO_ROOT), "docs", "howto", folder, "screenshots")
    print("Output:", out)
    capture(args.base_url, args.user, args.password, out, channel=args.browser_channel,
            mock_all=args.mock_all)


if __name__ == "__main__":
    main()
