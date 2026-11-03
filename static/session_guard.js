/* AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License v3.0. See the LICENSE file
 * in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
 *
 * Sends the page to the login form the moment the session is gone.
 *
 * Loaded once from layout.html, only when login is enabled, so no page needs its
 * own check. It wraps fetch() the same way server_selector.js does.
 *
 * Main Features:
 * - A same-origin answer that the auth barrier rejected (HTTP 401 with error 1005,
 *   or a non-/api call redirected to /login) navigates to the login page and never
 *   settles, so no page can show it as the result of whatever the user clicked
 * - Once that navigation starts every later answer is held too, so parallel calls
 *   cannot flash their own error before the page goes away
 * - /api/version is probed when the tab is shown again, when Back/Forward shows
 *   the page (bfcache, or the HTTP cache, which never asks the server), and on the
 *   first click or key press after a quiet minute, so an expired session is caught
 *   before the user fills a form
 * - A 401 from anywhere else (another origin, a proxy in front, a media server
 *   error with its own code) is passed through untouched
 */
(function () {
    var UNAUTHORIZED = 1005;
    var CHECK_INTERVAL_MS = 60000;
    var loginPath = window.appUrl('/login');
    var leaving = false;
    var lastCheck = Date.now();

    function never() {
        return new Promise(function () {});
    }

    function toLogin() {
        if (!leaving) {
            leaving = true;
            window.appRedirect('/login');
        }
        return never();
    }

    function sameOriginPath(url) {
        try {
            var u = new URL(url, window.location.href);
            return u.origin === window.location.origin ? u.pathname : null;
        } catch (e) {
            return null;
        }
    }

    var chainedFetch = window.fetch;
    window.fetch = function (input, init) {
        return chainedFetch.call(this, input, init).then(function (response) {
            if (leaving) {
                return never();
            }
            var path = sameOriginPath(response.url);
            if (path === null) {
                return response;
            }
            if (response.redirected && path === loginPath) {
                return toLogin();
            }
            if (response.status !== 401) {
                return response;
            }
            return response.clone().json().then(function (body) {
                return body && body.error_code === UNAUTHORIZED ? toLogin() : response;
            }, function () {
                return response;
            });
        });
    };

    function check(force) {
        if (leaving || (!force && Date.now() - lastCheck < CHECK_INTERVAL_MS)) {
            return;
        }
        lastCheck = Date.now();
        window.fetch('/api/version', { cache: 'no-store' }).catch(function () {});
    }

    window.addEventListener('pageshow', function (event) {
        var nav = window.performance.getEntriesByType('navigation')[0];
        if (event.persisted || (nav && nav.type === 'back_forward')) {
            check(true);
        }
    });
    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'visible') {
            check(false);
        }
    });
    window.addEventListener('focus', function () { check(false); });
    window.addEventListener('pointerdown', function () { check(false); }, true);
    window.addEventListener('keydown', function () { check(false); }, true);
})();
