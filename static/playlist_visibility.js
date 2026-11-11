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
    'use strict';

    if (!document.body || document.body.dataset.playlistVisibility !== 'true') {
        return;
    }

    const ANCHOR_SELECTOR = 'input[name="playlist_name"], input.tree-leaf-name, [data-playlist-visibility-anchor]';
    const state = { shared: false };
    const boxes = [];

    function sync() {
        for (const box of boxes) {
            box.checked = state.shared;
        }
    }

    function attach(anchor) {
        if (anchor.dataset.playlistVisibilityAttached === '1') {
            return;
        }
        anchor.dataset.playlistVisibilityAttached = '1';
        const label = document.createElement('label');
        label.className = 'playlist-visibility';
        label.title = 'Unticked: only you see the playlist in Navidrome.';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = state.shared;
        box.addEventListener('change', () => {
            state.shared = box.checked;
            sync();
        });
        boxes.push(box);
        label.append(box, ' Share with all users');
        if (anchor.tagName === 'INPUT') {
            anchor.after(label);
        } else {
            anchor.before(label);
        }
    }

    function scan(root) {
        if (root.matches?.(ANCHOR_SELECTOR)) {
            attach(root);
        }
        if (root.querySelectorAll) {
            for (const anchor of root.querySelectorAll(ANCHOR_SELECTOR)) {
                attach(anchor);
            }
        }
    }

    function isPlaylistCreation(init) {
        if (!init?.method || String(init.method).toUpperCase() !== 'POST') {
            return false;
        }
        return typeof init.body === 'string' && init.body.includes('"playlist_name"');
    }

    function withSharedFlag(input, init) {
        if (!state.shared || typeof input !== 'string' || !isPlaylistCreation(init)) {
            return input;
        }
        const url = new URL(input, window.location.origin);
        if (url.origin !== window.location.origin) {
            return input;
        }
        url.searchParams.set('playlist_public', 'true');
        return url.pathname + url.search + url.hash;
    }

    const chainedFetch = window.fetch;
    window.fetch = function (input, init) {
        let target = input;
        try {
            target = withSharedFlag(input, init);
        } catch (error) {
            console.warn('Playlist visibility flag not applied:', error);
        }
        return chainedFetch.call(this, target, init);
    };

    scan(document);
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    scan(node);
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
