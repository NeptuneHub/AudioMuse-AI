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
    var state = { defaultId: null, servers: [], enabled: false };

    function selectedId() {
        return localStorage.getItem(STORAGE_KEY) || '';
    }

    function isNonDefaultSelection(id) {
        if (!id || id === state.defaultId) {
            return false;
        }
        return state.servers.some(function (s) { return s.server_id === id; });
    }

    function shouldInject(pathname) {
        if (pathname.indexOf('/api/servers') !== -1) {
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
        try {
            var id = selectedId();
            if (isNonDefaultSelection(id) && typeof input === 'string') {
                var name = selectedName(id);
                var u = new URL(input, window.location.origin);
                if (name && u.origin === window.location.origin && shouldInject(u.pathname) && !u.searchParams.has('server')) {
                    u.searchParams.set('server', name);
                    input = u.pathname + u.search + u.hash;
                }
            }
        } catch (e) {
            // Never let selection logic break a request.
        }
        return chainedFetch.call(this, input, init);
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
            var label = s.name
                + (s.is_default ? ' (default)' : '')
                + (s.enabled ? '' : ' [disabled]');
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
        });
    }

    function load() {
        chainedFetch.call(window, '/api/servers', { headers: { 'Accept': 'application/json' } })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
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
            })
            .catch(function () {
                // Not authenticated or endpoint unavailable; leave UI untouched.
            });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', load);
    } else {
        load();
    }
})();
