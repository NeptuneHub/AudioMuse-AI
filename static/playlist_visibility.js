// AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
// Copyright (C) 2025 NeptuneHub
// SPDX-License-Identifier: AGPL-3.0-only
//
// Playlist visibility choice for reverse-proxy users. When AudioMuse acts as the
// logged-in user on Navidrome (NAVIDROME_EXTAUTH_ENABLED), a playlist created in
// the UI belongs to that user and is private by default. This script adds a
// "Share with all users" checkbox next to every playlist-name field (and next to
// any element marked data-playlist-visibility-anchor) and, while it is ticked,
// centrally appends playlist_public=true to same-origin playlist-creation POSTs,
// the same way server_selector.js appends the selected server. The checkboxes on
// one page share a single state. Without the body flag nothing is rendered or
// changed, so service-account behaviour stays exactly as before.

(function () {
    if (!document.body || document.body.getAttribute('data-playlist-visibility') !== 'true') {
        return;
    }

    var ANCHOR_SELECTOR = 'input[name="playlist_name"], input.tree-leaf-name, [data-playlist-visibility-anchor]';
    var state = { shared: false };
    var boxes = [];

    function sync() {
        boxes.forEach(function (box) { box.checked = state.shared; });
    }

    function attach(anchor) {
        if (anchor.getAttribute('data-playlist-visibility-attached') === '1') {
            return;
        }
        anchor.setAttribute('data-playlist-visibility-attached', '1');
        var label = document.createElement('label');
        label.className = 'playlist-visibility';
        label.title = 'Unticked: only you see the playlist in Navidrome.';
        var box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = state.shared;
        box.addEventListener('change', function () {
            state.shared = box.checked;
            sync();
        });
        boxes.push(box);
        label.appendChild(box);
        label.appendChild(document.createTextNode(' Share with all users'));
        if (anchor.tagName === 'INPUT') {
            anchor.insertAdjacentElement('afterend', label);
        } else {
            anchor.insertAdjacentElement('beforebegin', label);
        }
    }

    function scan(root) {
        if (root.matches && root.matches(ANCHOR_SELECTOR)) {
            attach(root);
        }
        if (root.querySelectorAll) {
            Array.prototype.forEach.call(root.querySelectorAll(ANCHOR_SELECTOR), attach);
        }
    }

    function isPlaylistCreation(init) {
        if (!init || !init.method || String(init.method).toUpperCase() !== 'POST') {
            return false;
        }
        return typeof init.body === 'string' && init.body.indexOf('"playlist_name"') !== -1;
    }

    var chainedFetch = window.fetch;
    window.fetch = function (input, init) {
        try {
            if (state.shared && typeof input === 'string' && isPlaylistCreation(init)) {
                var u = new URL(input, window.location.origin);
                if (u.origin === window.location.origin) {
                    u.searchParams.set('playlist_public', 'true');
                    input = u.pathname + u.search + u.hash;
                }
            }
        } catch (e) {
            // Never let the visibility choice break a request.
        }
        return chainedFetch.call(this, input, init);
    };

    scan(document);
    new MutationObserver(function (mutations) {
        mutations.forEach(function (m) {
            Array.prototype.forEach.call(m.addedNodes, function (node) {
                if (node.nodeType === 1) {
                    scan(node);
                }
            });
        });
    }).observe(document.body, { childList: true, subtree: true });
})();
