/* AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License v3.0. See the LICENSE file
 * in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
 *
 * Waiting for the server to come back after it restarts itself.
 *
 * Several flows end by restarting Flask under the user's feet. Each of them used
 * to guess how long that takes with its own hardcoded countdown - twenty seconds -
 * which is both far too long when the restart takes two, and not long enough on a
 * slow host. This probes instead: it counts down a short floor so the user sees
 * something happening and the restart has actually begun, then polls until the
 * server answers again and redirects the moment it does.
 *
 * Deliberately NOT a list of call sites here: the previous version enumerated
 * three and was already wrong when it was written, because the plugins page was a
 * fourth and still had its own countdown. This is loaded from layout.html, so
 * every page has it and none needs a guard around the call.
 *
 * ``restartScheduled: false`` skips only the PROBING, never the countdown. The
 * server already knows whether it scheduled a restart - schedule_flask_restart
 * returns false outright when DISABLE_FLASK_RESTART is set or the process is not
 * the Flask service - and the endpoints report it. Without that the page
 * announced "Restarting services" and probed a server that was never going to go
 * away, which only appeared to work because the first probe hit the
 * still-running process. It used to redirect INSTANTLY in that case, which is
 * why a restore, a migration or a wizard save could finish with no visible
 * countdown at all: the flow completed and the page simply vanished from under
 * the user. Every caller now gets the same visible countdown and differs only in
 * whether a probe follows it.
 *
 * The redirect carries a cache buster. The point of these flows is that the
 * configuration or the whole database just changed, so serving the target page
 * from the browser's heuristic cache would show the user exactly the stale state
 * the reload exists to replace; no HTML response in this app sets no-store.
 *
 * Any response below HTTP 500 counts as "up", including a login redirect or a
 * 403, because the question is whether Flask is serving, not whether this page
 * is still authorized.
 *
 * ``DEFAULT_FLOOR_SECONDS`` is the ONE countdown every caller shows. It is a
 * floor, not a wait: once it reaches zero the probing starts and the page
 * reloads the moment the server answers, so a restart that takes two seconds
 * does not cost twenty. Only the ceiling is ever overridden, by the restore,
 * which replaces the whole database before the services come back.
 *
 * ``until`` is watched from the moment ``waitAndGo`` is called, not from the
 * end of the countdown. A request that fails usually fails while the countdown
 * is still ticking, and the tick then painted a reassuring "restarting in N
 * seconds" straight over the error - the caller's error, or this module's -
 * once per second until the floor expired, so the user was told everything was
 * fine right up to the redirect that never came. A rejection therefore stops
 * the countdown dead and nothing renders over the failure afterwards; a caller
 * that renders its own, better-worded message is free to do so and it stays.
 * Resolution still changes nothing until the floor is over: the countdown
 * always runs to completion on the success path.
 */
(function () {
    var DEFAULT_FLOOR_SECONDS = 20;
    var DEFAULT_MAX_SECONDS = 120;
    var PROBE_INTERVAL_MS = 500;

    function redirect(target) {
        var fresh = target + (target.includes('?') ? '&' : '?') + '_reloaded=' + Date.now();
        if (window.appRedirect) {
            window.appRedirect(fresh);
        } else {
            window.location.href = fresh;
        }
    }

    function probe(target) {
        var url = (window.appUrl ? window.appUrl(target) : target);
        var bust = url + (url.includes('?') ? '&' : '?') + '_probe=' + Date.now();
        return fetch(bust, { method: 'GET', cache: 'no-store', redirect: 'follow' })
            .then(function (resp) { return resp.status < 500; })
            .catch(function () { return false; });
    }

    function waitAndGo(options) {
        var opts = options || {};
        var element = opts.element || null;
        var prefix = opts.prefix || '';
        var target = opts.target || '/';
        var floorSeconds = typeof opts.floorSeconds === 'number'
            ? opts.floorSeconds : DEFAULT_FLOOR_SECONDS;
        var maxSeconds = typeof opts.maxSeconds === 'number'
            ? opts.maxSeconds : DEFAULT_MAX_SECONDS;
        var deadline = Date.now() + maxSeconds * 1000;
        var remaining = floorSeconds;

        function render(text) {
            if (element) { element.textContent = text; }
        }

        var willRestart = opts.restartScheduled !== false;

        function reconnecting() {
            render((prefix ? prefix + ' ' : '') + 'Restarting services - reconnecting...');
            (function attempt() {
                if (Date.now() >= deadline) {
                    redirect(target);
                    return;
                }
                probe(target).then(function (up) {
                    if (up) {
                        render((prefix ? prefix + ' ' : '') + 'Back up - redirecting...');
                        redirect(target);
                    } else {
                        window.setTimeout(attempt, PROBE_INTERVAL_MS);
                    }
                });
            })();
        }

        function finish() {
            if (willRestart) {
                reconnecting();
                return;
            }
            render((prefix ? prefix + ' ' : '') + 'Reloading...');
            redirect(target);
        }

        var untilResolved = false;
        var untilFailed = false;
        var floorReached = false;

        function untilSucceeded() {
            untilResolved = true;
            if (floorReached) { finish(); }
        }

        function untilRejected(err) {
            untilFailed = true;
            render(err?.message || 'That did not work. Check the logs.');
        }

        // `until` lets the countdown start the INSTANT the user acts, instead of
        // after the request that triggers the restart comes back. That request is
        // held open for the whole restart, so waiting for it meant the page sat
        // frozen for minutes and the countdown only appeared once there was nothing
        // left to count. Nothing is redirected until it resolves, so a request that
        // fails still reports its error instead of navigating away.
        if (opts.until) {
            opts.until.then(untilSucceeded, untilRejected);
        }

        function tick() {
            if (untilFailed) {
                return;
            }
            if (remaining > 0) {
                render(
                    (prefix ? prefix + ' ' : '')
                    + (willRestart ? 'Restarting services - reconnecting in ' : 'Reloading in ')
                    + remaining + ' second' + (remaining === 1 ? '' : 's') + '...'
                );
                remaining -= 1;
                window.setTimeout(tick, 1000);
                return;
            }
            floorReached = true;
            if (!opts.until || untilResolved) {
                finish();
            }
        }

        tick();
    }

    window.AudioMuseRestart = { waitAndGo: waitAndGo, probe: probe };
})();
