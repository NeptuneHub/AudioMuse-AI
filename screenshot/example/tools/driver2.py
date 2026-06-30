import os
import traceback

try:
    import driver
except ModuleNotFoundError:
    from screenshot.example.tools import driver
from playwright.sync_api import sync_playwright

BASE = os.environ.get("AUDIOMUSE_BASE", "http://YOUR-SERVER:8000")
OUT = os.environ.get("AUDIOMUSE_OUT", "screenshot/example")
USER = os.environ.get("AUDIOMUSE_USER", "admin")
PW = os.environ.get("AUDIOMUSE_PW", "admin")

URLS = {
    'similarity': '/similarity',
    'artist_similarity': '/artist_similarity',
    'path': '/path',
    'waveform': '/waveform',
    'lyrics_search': '/lyrics_search',
    'alchemy': '/alchemy',
}

FLOWS = {
    "similarity": [
        {
            "label": "result",
            "results_wait_selector": "#results-table-wrapper .song-result-list .result-item",
            "steps": [
                {"action": "type", "selector": "#search_query", "value": "ar"},
                {
                    "action": "wait_selector",
                    "selector": "#autocomplete-results:not(.hidden) .autocomplete-item",
                },
                {"action": "wait_ms", "selector": "body", "value": "400"},
                {"action": "click_first", "selector": "#autocomplete-results .autocomplete-item"},
                {"action": "click", "selector": "#similarity-form button[type=\"submit\"]"},
                {
                    "action": "wait_selector",
                    "selector": "#results-table-wrapper .song-result-list .result-item",
                },
            ],
        }
    ],
    "artist_similarity": [
        {
            "label": "result",
            "results_wait_selector": "#results-table-wrapper .results-table",
            "steps": [
                {"action": "type", "selector": "#artist_search", "value": "a"},
                {"action": "type", "selector": "#artist_search", "value": "ar"},
                {"action": "wait_selector", "selector": "#autocomplete-results .autocomplete-item"},
                {"action": "click_first", "selector": "#autocomplete-results .autocomplete-item"},
                {"action": "click", "selector": "#find-artists-btn"},
                {"action": "wait_selector", "selector": "#results-table-wrapper .results-table"},
            ],
        }
    ],
    "path": [
        {
            "label": "result",
            "results_wait_selector": "#results-table-wrapper .result-item",
            "steps": [
                {"action": "type", "selector": "#start_search", "value": "ar"},
                {
                    "action": "wait_selector",
                    "selector": "#start-autocomplete-results .autocomplete-item",
                },
                {
                    "action": "click_first",
                    "selector": "#start-autocomplete-results .autocomplete-item",
                },
                {"action": "type", "selector": "#end_search", "value": "a"},
                {
                    "action": "wait_selector",
                    "selector": "#end-autocomplete-results .autocomplete-item",
                },
                {
                    "action": "click_first",
                    "selector": "#end-autocomplete-results .autocomplete-item",
                },
                {"action": "click", "selector": "#path-form button[type=\"submit\"]"},
                {"action": "wait_selector", "selector": "#results-table-wrapper .result-item"},
            ],
        }
    ],
    "waveform": [
        {
            "label": "result",
            "results_wait_selector": "#waveform-container:not(.hidden)",
            "steps": [
                {"action": "fill", "selector": "#search_query", "value": "Endless Skyline"},
                {"action": "wait_ms", "selector": "body", "value": "300"},
                {
                    "action": "eval",
                    "selector": "body",
                    "value": "(()=>{const ac=document.getElementById('autocomplete-results');if(ac)ac.classList.add('hidden');const i=document.getElementById('selected_item_id');i.value='id-1234';const t=document.getElementById('title_display');if(t)t.textContent='Title: Endless Skyline - The Ember Echo';const f=document.getElementById('waveform-form');if(f.requestSubmit)f.requestSubmit();else f.dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));})()",
                },
                {"action": "wait_selector", "selector": "#waveform-container:not(.hidden)"},
                {"action": "wait_ms", "selector": "body", "value": "1000"},
            ],
        }
    ],
    "lyrics_search": [
        {
            "label": "axis",
            "results_wait_selector": "#results-list .result-item",
            "steps": [
                {"action": "click", "selector": ".tab-btn[data-tab=\"axes\"]"},
                {"action": "wait_selector", "selector": "#tab-axes.active"},
                {
                    "action": "eval",
                    "selector": "body",
                    "value": "(()=>{const ss=[...document.querySelectorAll('#axes-container select')];for(const s of ss){for(const o of s.options){if(o.value){s.value=o.value;s.dispatchEvent(new Event('change',{bubbles:true}));return;}}}})()",
                },
                {"action": "wait_ms", "selector": "body", "value": "300"},
                {"action": "click", "selector": "#axis-form button[type=\"submit\"]"},
                {"action": "wait_selector", "selector": "#results-list .result-item"},
            ],
        },
        {
            "label": "text",
            "results_wait_selector": "#results-list .result-item",
            "steps": [
                {"action": "click", "selector": ".tab-btn[data-tab=\"text\"]"},
                {"action": "wait_selector", "selector": "#tab-text.active"},
                {
                    "action": "fill",
                    "selector": "#search-query",
                    "value": "heartbreak in the city at night",
                },
                {"action": "click", "selector": "#search-form button[type=\"submit\"]"},
                {"action": "wait_selector", "selector": "#results-list .result-item"},
            ],
        },
        {
            "label": "song",
            "results_wait_selector": "#results-list .result-item",
            "steps": [
                {"action": "click", "selector": ".tab-btn[data-tab=\"song\"]"},
                {"action": "wait_selector", "selector": "#tab-song.active"},
                {"action": "type", "selector": "#sg-search-query", "value": "a"},
                {"action": "wait_ms", "selector": "body", "value": "400"},
                {"action": "type", "selector": "#sg-search-query", "value": "ar"},
                {
                    "action": "wait_selector",
                    "selector": "#sg-autocomplete-results:not(.hidden) .autocomplete-item",
                },
                {
                    "action": "click_first",
                    "selector": "#sg-autocomplete-results .autocomplete-item",
                },
                {"action": "click", "selector": "#song-form button[type=\"submit\"]"},
                {"action": "wait_selector", "selector": "#results-list .result-item"},
            ],
        },
    ],
    "alchemy": [
        {
            "label": "result",
            "results_wait_selector": "#results-table-wrapper .result-item",
            "steps": [
                {"action": "wait_selector", "selector": ".alchemy-card .song"},
                {"action": "type", "selector": ".alchemy-card .song", "value": "a"},
                {"action": "type", "selector": ".alchemy-card .song", "value": "ar"},
                {"action": "wait_ms", "selector": "body", "value": "500"},
                {"action": "wait_selector", "selector": ".autocomplete-results .autocomplete-item"},
                {"action": "click_first", "selector": ".autocomplete-results .autocomplete-item"},
                {"action": "wait_ms", "selector": "body", "value": "300"},
                {"action": "click", "selector": "#run-alchemy"},
                {"action": "wait_selector", "selector": "#results-table-wrapper .result-item"},
            ],
        }
    ],
}


def out_name(key, label, i):
    pref = {
        'similarity': '04_similarity_4',
        'artist_similarity': '05_artist_similarity_2',
        'path': '06_path_3',
        'waveform': '12_waveform_2',
        'alchemy': '07_alchemy_4',
    }
    if key == 'lyrics_search':
        return "09_lyrics_search_%d_%s_result" % (4 + i, label)
    return "%s_result" % pref[key]


def run_steps(page, steps):
    for s in steps:
        a = s.get('action')
        sel = s.get('selector') or 'body'
        val = s.get('value')
        try:
            if a == 'fill':
                page.fill(sel, val or '', timeout=5000)
            elif a == 'type':
                try:
                    page.click(sel, timeout=3000)
                except Exception:
                    pass
                page.fill(sel, '', timeout=3000)
                page.type(sel, val or 'a', delay=70)
            elif a == 'click':
                try:
                    page.click(sel, timeout=6000)
                except Exception:
                    page.click(sel, timeout=3000, force=True)
            elif a == 'click_first':
                page.locator(sel).first.click(timeout=6000)
            elif a == 'select_option':
                page.select_option(sel, val, timeout=5000)
            elif a == 'press_enter':
                page.press(sel, 'Enter')
            elif a == 'wait_selector':
                page.wait_for_selector(sel, timeout=7000)
            elif a == 'wait_ms':
                page.wait_for_timeout(int(val or '500'))
            elif a == 'eval':
                page.evaluate(val or '')
        except Exception as e:
            print("STEP_FAIL", a, sel, repr(e), flush=True)


def shot(page, name):
    try:
        page.screenshot(path="%s/%s.png" % (OUT, name), full_page=True)
        print("SHOT", name, flush=True)
    except Exception as e:
        print("SHOT_FAIL", name, repr(e), flush=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        ctx = browser.new_context(viewport={'width': 1440, 'height': 900}, ignore_https_errors=True)
        ctx.add_init_script(driver.INIT_JS)
        ctx.route("**/api/**", driver.handle_route)
        page = ctx.new_page()

        page.goto(BASE + "/login", wait_until='domcontentloaded', timeout=30000)
        page.fill('#login-user', USER)
        page.fill('#login-password', PW)
        page.click('button[type="submit"]')
        try:
            page.wait_for_url(lambda u: '/login' not in u, timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(1200)
        print("LOGIN_OK", page.url, flush=True)

        run_only = [k for k in os.environ.get('RUN_ONLY', '').split(',') if k]
        for key, flows in FLOWS.items():
            if run_only and key not in run_only:
                continue
            url = BASE + URLS[key]
            for i, flow in enumerate(flows):
                label = flow.get('label') or 'result'
                print("FLOW", key, label, flush=True)
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    page.wait_for_timeout(2200)
                    run_steps(page, flow.get('steps') or [])
                    rw = flow.get('results_wait_selector')
                    if rw:
                        try:
                            page.wait_for_selector(rw, timeout=8000)
                        except Exception:
                            pass
                    page.wait_for_timeout(1500)
                    shot(page, out_name(key, label, i))
                except Exception as e:
                    print("FLOW_ERR", key, label, repr(e), flush=True)
                    traceback.print_exc()

        browser.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
