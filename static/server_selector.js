// AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
// Copyright (C) 2025 NeptuneHub
// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared media-server selector. Renders a dropdown of configured servers (when
// multi-server is enabled and more than one exists) and centrally appends the
// selected server id to same-origin API requests, so any page's playlist and
// search calls target the chosen server. Selecting the default (or when only one
// server exists) leaves every request exactly as before.

(function () {
    var STORAGE_KEY = 'audiomuse_selected_server';
    var state = { defaultId: null, servers: [], enabled: false, loaded: false };

    function selectedId() {
        return localStorage.getItem(STORAGE_KEY) || '';
    }

    function isNonDefaultSelection(id) {
        if (!id || id === state.defaultId) {
            return false;
        }
        if (state.servers.length === 0) {
            // Optimistic only while the server list is still loading; once the
            // load has completed (or failed) an unknown id must not be injected,
            // or a stale/deleted selection would 400 every request forever.
            return !state.loaded;
        }
        return state.servers.some(function (s) { return s.server_id === id; });
    }

    // The dashboard is deliberately NOT server-scoped: it reports the catalogue
    // (the union of every server) alongside an explicit per-server section, and
    // each number says which of the two it is. Injecting ?server= there appended
    // a parameter the endpoint ignores, so the picker silently did nothing.
    // /api/playlists is the same shape: it always returns EVERY server's last
    // clustering run grouped per server, so the picker must not scope it.
    function shouldInject(pathname) {
        if (pathname.indexOf('/api/servers') !== -1
            || pathname.indexOf('/api/dashboard/') !== -1
            || pathname.indexOf('/api/playlists') !== -1) {
            return false;
        }
        return pathname.indexOf('/api/') !== -1 || pathname.indexOf('/chat/') !== -1;
    }

    function selectedServer(id) {
        return state.servers.filter(function (s) { return s.server_id === id; })[0] || null;
    }

    function selectedName(id) {
        var match = selectedServer(id);
        return match ? match.name : null;
    }

    var chainedFetch = window.fetch;
    window.fetch = function (input, init) {
        var injected = null;
        try {
            var id = selectedId();
            if (isNonDefaultSelection(id) && (typeof input === 'string' || input instanceof Request)) {
                var name = selectedName(id) || (state.servers.length === 0 ? id : null);
                var rawUrl = input instanceof Request ? input.url : input;
                var u = new URL(rawUrl, window.location.origin);
                if (name && u.origin === window.location.origin && shouldInject(u.pathname) && !u.searchParams.has('server')) {
                    u.searchParams.set('server', name);
                    injected = id;
                    input = input instanceof Request
                        ? new Request(u.toString(), input)
                        : u.pathname + u.search + u.hash;
                }
            }
        } catch (e) {
            // Never let selection logic break a request.
        }
        return chainedFetch.call(this, input, init).then(function (response) {
            if (injected && response.status === 400) {
                response.clone().json().then(function (body) {
                    if (body && typeof body.error === 'string' && /unknown server/i.test(body.error)) {
                        forgetStaleSelection(injected);
                    }
                }).catch(function () { /* not JSON: leave the selection alone */ });
            }
            return response;
        });
    };

    function escapeHtml(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function render() {
        var mount = document.getElementById('server-selector-nav');
        if (!mount) {
            return;
        }
        if (!state.enabled || state.servers.length < 2) {
            mount.classList.remove('active');
            mount.innerHTML = '';
            return;
        }
        var current = selectedId() || state.defaultId || '';
        var html = '<select id="server-selector" class="server-selector" aria-label="Music server">';
        state.servers.forEach(function (s) {
            var label = s.name + (s.is_default ? ' (default)' : '');
            var sel = (s.server_id === current) ? ' selected' : '';
            html += '<option value="' + escapeHtml(s.server_id) + '"' + sel + '>' + escapeHtml(label) + '</option>';
        });
        html += '</select>';
        mount.innerHTML = html;
        mount.classList.add('active');
        var select = document.getElementById('server-selector');
        select.addEventListener('change', function () {
            if (this.value === state.defaultId) {
                localStorage.removeItem(STORAGE_KEY);
            } else {
                localStorage.setItem(STORAGE_KEY, this.value);
            }
            // Pages render server-specific state (the Sonic Fingerprint form's
            // credential fields, cached results, counts). Reload so the whole
            // page speaks to the newly selected server instead of keeping the
            // previous one's view.
            window.location.reload();
        });
    }

    // Page scope chip, pinned to the top-right corner of the page's TITLE CARD.
    //
    // Shown whenever at least one server is configured, so every page always
    // carries its CATALOGUE / PER SERVER label. Positioned absolutely inside the
    // card so it cannot reflow the page it annotates.
    function renderScopeChip() {
        var existing = document.getElementById('page-scope-chip');
        if (existing) {
            existing.remove();
        }
        if (state.servers.length < 1) {
            return;
        }
        var root = document.querySelector('.container[data-page-scope]');
        if (!root) {
            return;
        }
        var scope = root.getAttribute('data-page-scope');
        if (scope !== 'catalogue' && scope !== 'server') {
            return;
        }
        var h1 = root.querySelector('h1');
        if (!h1) {
            return;
        }
        // The title card is the `section` panel the <h1> sits in; fall back to the
        // heading's own wrapper when a page does not use one.
        var card = h1.closest('section') || h1.parentElement;
        if (!card) {
            return;
        }
        var catalogue = scope === 'catalogue';
        var chip = document.createElement('span');
        chip.id = 'page-scope-chip';
        chip.className = 'scope-chip scope-chip-corner '
            + (catalogue ? 'scope-catalog' : 'scope-server');
        chip.textContent = catalogue ? 'Catalogue' : 'Per server';
        chip.title = catalogue
            ? 'This page acts on the whole catalogue: every analyzed song across all your music servers, deduplicated. The server picker does not change it.'
            : 'This page acts on the one music server selected in the sidebar. Switch servers there to change what it returns.';
        card.classList.add('scope-chip-host');
        card.appendChild(chip);
    }

    function forgetStaleSelection(id) {
        // The selected server was deleted (in another tab, or by another admin):
        // every request would keep 400ing with it. Drop it and show the default.
        if (!id || localStorage.getItem(STORAGE_KEY) !== id) {
            return;
        }
        localStorage.removeItem(STORAGE_KEY);
        window.location.reload();
    }

    function load() {
        chainedFetch.call(window, '/api/servers', { headers: { 'Accept': 'application/json' } })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                state.loaded = true;
                if (!data) {
                    return;
                }
                state.enabled = !!data.multi_server_enabled;
                state.servers = data.servers || [];
                state.defaultId = data.default_id;
                var current = selectedId();
                if (current && !state.servers.some(function (s) { return s.server_id === current; })) {
                    localStorage.removeItem(STORAGE_KEY);
                }
                render();
                renderScopeChip();
            })
            .catch(function () {
                // Not authenticated or endpoint unavailable; leave UI untouched
                // and stop the optimistic pre-load injection (fail-safe).
                state.loaded = true;
            });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', load);
    } else {
        load();
    }
})();
