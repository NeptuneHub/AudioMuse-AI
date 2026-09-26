/* AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License v3.0. See the LICENSE file
 * in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
 *
 * The one search-as-you-type picker behind every song, artist and playlist box.
 *
 * Every box searches from the first typed character, shows the first page and
 * loads the next one when its list is scrolled to the end, so no page can drift
 * from that contract again. The page size and the minimum come from the server
 * (app_helper.SEARCH_PAGE_SIZE / SEARCH_MIN_QUERY_LENGTH, rendered on this
 * script tag by templates/includes/layout.html), and whether another page exists
 * comes from the endpoint's X-Search-Has-More header, so rows the server drops
 * after its SQL page never end the list early. Superseded requests are aborted,
 * rows are rendered as text (never HTML), and the input behaves as an ARIA
 * combobox (arrows, Enter, Escape).
 */
(function () {
    var settings = document.currentScript.dataset;
    var PAGE_SIZE = Number(settings.pageSize);
    var MIN_CHARS = Number(settings.minChars);
    var DEBOUNCE_MS = 300;
    var instances = 0;

    function songRow(track) {
        return [
            ['title', track.title || 'N/A'],
            ['artist', track.author || 'N/A'],
            ['album', 'Album: ' + (track.album || 'Unknown')]
        ];
    }

    function artistRow(artist) {
        return [
            ['title', artist.artist || 'N/A'],
            ['artist', (artist.track_count || 0) + ' tracks']
        ];
    }

    function playlistRow(playlist) {
        return [
            ['title', playlist.name || 'N/A'],
            ['artist', playlist.count === null || playlist.count === undefined ? 'Playlist' : playlist.count + ' tracks']
        ];
    }

    function messageRow(text) {
        var row = document.createElement('div');
        row.className = 'autocomplete-item';
        row.setAttribute('role', 'presentation');
        var em = document.createElement('em');
        em.textContent = text;
        row.appendChild(em);
        return row;
    }

    function errorText(err) {
        return (err && err.message) ? err.message : 'Search failed';
    }

    function attach(cfg) {
        var input = cfg.input;
        var box = cfg.box;
        var queryParam = cfg.queryParam || 'search_query';
        var render = cfg.render || songRow;
        var emptyText = cfg.emptyText || 'No results found';
        var enabled = cfg.enabled || function () { return true; };
        var container = cfg.container || input.closest('.autocomplete-container') || box.parentElement;
        var prefix = 'ac' + (++instances) + '-';
        var run = 0;
        var query = '';
        var offset = 0;
        var shownQuery = null;
        var timer = null;
        var observer = null;
        var controller = null;
        var active = -1;
        var optionSeq = 0;
        var dismissed = false;

        if (!box.id) box.id = prefix + 'list';
        box.setAttribute('role', 'listbox');
        input.setAttribute('role', 'combobox');
        input.setAttribute('aria-autocomplete', 'list');
        input.setAttribute('aria-controls', box.id);
        input.setAttribute('aria-expanded', 'false');
        input.setAttribute('autocomplete', 'off');

        function options() {
            return box.querySelectorAll('[role="option"]');
        }

        function listMatchesInput() {
            return shownQuery !== null && shownQuery === input.value.trim();
        }

        function markActive() {
            options().forEach(function (row, i) {
                row.classList.toggle('active', i === active);
                row.setAttribute('aria-selected', i === active ? 'true' : 'false');
            });
        }

        function setActive(index) {
            var list = options();
            if (!list.length) return;
            active = ((index % list.length) + list.length) % list.length;
            markActive();
            list[active].scrollIntoView({ block: 'nearest' });
            input.setAttribute('aria-activedescendant', list[active].id);
        }

        function clearActive() {
            active = -1;
            input.removeAttribute('aria-activedescendant');
            markActive();
        }

        function show() {
            box.classList.remove('hidden');
            input.setAttribute('aria-controls', box.id);
            input.setAttribute('aria-expanded', 'true');
        }

        function hide() {
            box.classList.add('hidden');
            input.setAttribute('aria-expanded', 'false');
            clearActive();
        }

        function dismiss() {
            dismissed = true;
            hide();
        }

        function stopObserver() {
            if (observer) {
                observer.disconnect();
                observer = null;
            }
        }

        function abortPending() {
            if (controller) {
                controller.abort();
                controller = null;
            }
        }

        function cancel() {
            run++;
            clearTimeout(timer);
            stopObserver();
            abortPending();
        }

        function reset() {
            cancel();
            query = '';
            offset = 0;
            shownQuery = null;
            box.innerHTML = '';
            hide();
        }

        async function fetchPage(text, start) {
            var extra = typeof cfg.params === 'function' ? cfg.params() : cfg.params;
            var params = new URLSearchParams(extra || {});
            params.set(queryParam, text);
            params.set('start', String(start));
            params.set('end', String(start + PAGE_SIZE));
            abortPending();
            controller = new AbortController();
            var response = await fetch(cfg.url + (cfg.url.indexOf('?') === -1 ? '?' : '&') + params.toString(), { signal: controller.signal });
            var body = await readJsonBody(response);
            if (!response.ok || !Array.isArray(body)) {
                throw new Error(apiErrorText(body, 'Search failed (HTTP ' + response.status + ')'));
            }
            var more = response.headers.get('X-Search-Has-More');
            return { rows: body, more: more === null ? body.length >= PAGE_SIZE : more === '1' };
        }

        function select(item) {
            if (cfg.onSelect(item) === false) return;
            reset();
        }

        function appendRows(rows) {
            rows.forEach(function (item) {
                var row = document.createElement('div');
                row.className = 'autocomplete-item';
                row.setAttribute('role', 'option');
                row.id = prefix + (optionSeq++);
                row.acItem = item;
                render(item).forEach(function (part) {
                    var line = document.createElement('div');
                    line.className = part[0];
                    line.textContent = part[1];
                    row.appendChild(line);
                });
                row.addEventListener('click', function () { select(item); });
                box.appendChild(row);
            });
        }

        function appendPage(page, start, token) {
            offset = start + PAGE_SIZE;
            appendRows(page.rows);
            if (!page.more) return;
            var sentinel = messageRow('Loading more...');
            sentinel.classList.add('autocomplete-load-more');
            box.appendChild(sentinel);
            observer = new IntersectionObserver(function (entries) {
                if (!entries.some(function (entry) { return entry.isIntersecting; })) return;
                stopObserver();
                loadMore(token, sentinel);
            }, { root: box });
            observer.observe(sentinel);
        }

        async function loadMore(token, sentinel) {
            var start = offset;
            try {
                var page = await fetchPage(query, start);
                if (token !== run) return;
                sentinel.remove();
                appendPage(page, start, token);
            } catch (err) {
                if (token !== run) return;
                console.error('Search failed:', err);
                sentinel.firstChild.textContent = errorText(err);
            }
        }

        async function search(text) {
            cancel();
            var token = run;
            query = text;
            offset = 0;
            try {
                var page = await fetchPage(text, 0);
                if (token !== run) return;
                box.innerHTML = '';
                clearActive();
                shownQuery = text;
                appendPage(page, 0, token);
                if (!box.childElementCount) box.appendChild(messageRow(emptyText));
                if (!dismissed) show();
            } catch (err) {
                if (token !== run) return;
                console.error('Search failed:', err);
                box.innerHTML = '';
                clearActive();
                shownQuery = text;
                box.appendChild(messageRow(errorText(err)));
                if (!dismissed) show();
            }
        }

        function onInput() {
            if (!enabled()) return;
            cancel();
            dismissed = false;
            clearActive();
            var text = input.value.trim();
            if (text.length < MIN_CHARS) {
                reset();
                return;
            }
            timer = setTimeout(function () { search(text); }, DEBOUNCE_MS);
        }

        function onKeydown(e) {
            if (!enabled()) return;
            var open = !box.classList.contains('hidden');
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                if (!open || !listMatchesInput()) return;
                e.preventDefault();
                if (e.key === 'ArrowDown') {
                    setActive(active + 1);
                } else {
                    setActive(active < 0 ? -1 : active - 1);
                }
            } else if (e.key === 'Enter') {
                var list = options();
                if (open && active >= 0 && list[active]) {
                    e.preventDefault();
                    select(list[active].acItem);
                }
            } else if (e.key === 'Escape') {
                if (open) e.preventDefault();
                dismiss();
            } else if (e.key === 'Tab') {
                dismiss();
            }
        }

        function onFocus() {
            if (!enabled()) return;
            dismissed = false;
            if (listMatchesInput() && box.childElementCount) show();
        }

        function onDocumentClick(e) {
            if (!container.contains(e.target) && !box.contains(e.target)) dismiss();
        }

        function destroy() {
            reset();
            input.removeEventListener('input', onInput);
            input.removeEventListener('keydown', onKeydown);
            input.removeEventListener('focus', onFocus);
            document.removeEventListener('click', onDocumentClick);
        }

        input.addEventListener('input', onInput);
        input.addEventListener('keydown', onKeydown);
        input.addEventListener('focus', onFocus);
        document.addEventListener('click', onDocumentClick);

        return { reset: reset, hide: hide, destroy: destroy };
    }

    window.AudioMuseAutocomplete = {
        PAGE_SIZE: PAGE_SIZE,
        MIN_CHARS: MIN_CHARS,
        DEBOUNCE_MS: DEBOUNCE_MS,
        attach: attach,
        songRow: songRow,
        artistRow: artistRow,
        playlistRow: playlistRow
    };
})();
