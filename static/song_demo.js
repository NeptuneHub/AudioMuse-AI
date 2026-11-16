/* AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License v3.0. See the LICENSE file
 * in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
 *
 * The one place a page asks for song demos (POST /api/demo, app_demo.py).
 *
 * AudioMuseDemo.create({url, seconds}) returns sections(ids), a Map from each id
 * to its {start_seconds, end_seconds, duration_seconds} row, and queue(songs),
 * which gives every song that has a demo its start, end and duration so
 * static/song_player.js plays only that part. The URL defaults to the data-url
 * of this script's tag (templates/includes/layout.html). Each id is asked once
 * per page and the answer is kept, so playing the demos again or from another
 * row makes no new request. Songs without a demo (no neural fingerprint) are
 * left out of the queue. seconds is optional: the API default applies when it
 * is not given.
 */
(function () {
    const defaults = document.currentScript ? document.currentScript.dataset : {};

    function create(settings) {
        const options = Object.assign({ url: defaults.url }, settings || {});
        const known = new Map();

        async function sections(ids) {
            const missing = [...new Set(ids)].filter(id => !known.has(id));
            if (missing.length) {
                const body = { item_ids: missing };
                if (typeof options.seconds === 'number') body.seconds = options.seconds;
                const response = await fetch(options.url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await readJsonBody(response);
                if (!response.ok || data === null) throw new Error(apiErrorText(data, `Demo failed (HTTP ${response.status})`));
                missing.forEach((id, n) => known.set(id, (data.results || [])[n] || null));
            }
            return new Map(ids.map(id => [id, known.get(id) || null]));
        }

        async function queue(songs) {
            const found = await sections(songs.map(song => song.item_id));
            return songs.flatMap(song => {
                const section = found.get(song.item_id);
                if (!section || typeof section.start_seconds !== 'number' || typeof section.end_seconds !== 'number') return [];
                return [Object.assign({}, song, { start: section.start_seconds, end: section.end_seconds, duration: section.duration_seconds })];
            });
        }

        return { sections, queue };
    }

    window.AudioMuseDemo = { create };
})();
