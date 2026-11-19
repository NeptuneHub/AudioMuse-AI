/* AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
 * Copyright (C) 2025 NeptuneHub
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License v3.0. See the LICENSE file
 * in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>
 *
 * The one song player of every page that answers a list of songs: its whole
 * interface lives in this file, so a change here reaches every page.
 *
 * templates/includes/layout.html loads it (not on the map and clustering pages)
 * with the play, smart fade and transition URLs and the neural fingerprint flag
 * on its script tag. Each .song-player-options element gets the Smart fade and
 * Demo checkboxes, only with the neural fingerprint on; every copy on a page stays in step and
 * ticking one unticks the other, so a search plays whole songs, only demos or
 * only smart fade. They are read when a search runs. Every result list gets, right above
 * it, a bar with Play, Skip to the fade (shown only with Smart fade) and Stop, a
 * clock and a status line. A page calls, around its own rendering:
 *   rows = await AudioMusePlayer.arrange(list, rows, {seed, keepOrder}) before
 *     rendering: with Smart fade the rows come back in smart fade order with
 *     where each one starts and ends (keepOrder keeps a path or an album in its order);
 *   AudioMusePlayer.show(list) once renderSongList filled the list;
 *   AudioMusePlayer.hide(list) when a new search starts.
 * The songs played are the list's .result-item rows (data-item-id, data-title),
 * after the seed when the page gives one. Play plays whole songs; with Demo only
 * each song's demo (static/song_demo.js) with a 4 s crossfade, the last one
 * stopping at its end; with Smart fade each song from its fade-in to its
 * fade-out, mixed into the next by the transition API, which the page only
 * executes, nothing is computed here: each song plays at its gain (the same
 * loudness for all); from start, for duration, both songs follow their
 * position lists (a beat mix stretches them along these, pitch kept, so the
 * tempo glides), their faders follow their volume lists (already shaped to a
 * steady level and clip-safe) and their high shelf, mid peak and low shelf
 * (or the next song's high-pass) follow high_db, mid_db, bass_db and
 * highpass_hz, flat on every source until a mix; the next song then plays on alone. The first
 * song of a play waits for its own transition, which carries its gain. Without a
 * plan (an error, a late answer, no times) the page plays a plain crossfade
 * over SMART_CROSSFADE_SECONDS at the order's times: S-curves (FADE_IN,
 * FADE_OUT, slow at both ends) whose two volumes always add up to 1, so it
 * cannot clip; it is labelled "Default crossfade" like the planned one. The
 * demo crossfade and the last demo's fade out use the same curves. A row
 * click plays from that row.
 *
 * With Smart fade the page asks for a transition plan only for the song that
 * plays, as it plays (requestPlan keeps each answer; play takes a fresh copy),
 * and the next row's bar starts with an icon (MARKS) for how that song comes
 * in: a dot while it is planned, an equals sign for a beat-matched mix, rising
 * lines for a rise, a split square for the default crossfade; the player bar
 * counts the transitions planned so far.
 *
 * During a transition the highlight and the clock stay on the louder song:
 * the ending one until the middle of the mix or crossfade, then they jump to
 * the starting one (swapView), not when the transition starts or ends.
 *
 * Skip to the fade pressed while the next song or its plan is still loading
 * waits for them, so the planned mix is not missed for the default crossfade.
 *
 * Songs are fetched whole and decoded into Web Audio buffers; every cut, fade
 * and stretch is scheduled on one AudioContext clock made in the first Play
 * click, never on an audio element (Safari refuses play() outside a click and
 * seeks VBR MP3 inaccurately). Every source starts on a whole sample of that
 * clock at a whole-sample offset (onGrid): Chrome and Safari interpolate a
 * source started between two samples for its whole length, which dulls the
 * highs. After a mix the next song goes on from a fresh source with flat
 * filters and the mix's sources are muted (no filter tail rings on), so
 * outside its transitions a song plays exactly as decoded.
 * One list plays at a time. The playing row fills
 * with the progress and a click or drag on it moves in the song, up to the line
 * where it fades.
 */
(function () {
    const CROSSFADE_SECONDS = 4;
    const SMART_CROSSFADE_SECONDS = 10;
    const FADE_POINTS = 64;
    const FADE_IN = Float32Array.from({ length: FADE_POINTS }, (_, k) => 0.5 - 0.5 * Math.cos(Math.PI * k / (FADE_POINTS - 1)));
    const FADE_OUT = FADE_IN.slice().reverse();
    const settings = document.currentScript ? document.currentScript.dataset : {};
    const URLS = { play: settings.playUrl || '', smartFade: settings.smartFadeUrl || '', transition: settings.transitionUrl || '' };
    const READY = !!URLS.play;
    const NEURAL = settings.neural === 'true' && !!URLS.smartFade;
    const OPTIONS = {
        smart: {
            label: 'Smart fade',
            help: 'Orders the songs so each one flows into the next (a path or an album keeps its order), plays them all at the same loudness, each from where it should start to where it should end with a beat-matched mix of at least 15 seconds that hands the bass, mids and highs over one after the other (only when the beats are confirmed to lock; otherwise the default 10 second crossfade, started on a beat and into the next song\'s section start), and shows Skip to the fade. As each transition is planned, a small icon on the next row says how that song comes in: an equals sign for a beat-matched mix, rising lines for a beat-matched rise, a split square for the default crossfade; the player bar counts them. Set it before the search.'
        },
        demo: {
            label: 'Demo',
            help: 'Plays only the demo of each song, its chorus or drop, one after the other with a short crossfade. Set it before the search.'
        }
    };
    const picked = { smart: false, demo: false };
    const instances = new Map();
    let audioCtx = null;
    let demoClient = null;

    function demos() {
        if (!demoClient) demoClient = window.AudioMuseDemo.create();
        return demoClient;
    }

    function clock(seconds) {
        const total = Math.max(0, Math.round(seconds));
        return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    const MARKS = {
        pending: '<svg viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="1.6" fill="currentColor"/></svg>',
        crossfade: '<svg viewBox="0 0 12 12" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.2"/><path d="M10.5 1.5v9h-9z" fill="currentColor"/></svg>',
        blend: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 4.5h8M2 7.5h8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
        rise: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 7l8-4M2 10.5l8-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>'
    };

    function markHtml(mark) {
        const kind = mark ? mark.kind : 'none';
        const title = mark && mark.title ? ` title="${escapeHtml(mark.title)}"` : '';
        return `<span class="play-mark" data-mark="${kind}"${title}>${MARKS[kind] || ''}</span>`;
    }

    function showSpans(list, songs, label, markOf) {
        list.querySelectorAll('.play-span').forEach(span => span.remove());
        const items = list.querySelectorAll('.result-item');
        songs.forEach(song => {
            const item = song.row >= 0 ? items[song.row] : null;
            if (!item || typeof song.start !== 'number' || typeof song.end !== 'number') return;
            const length = Math.max(song.duration || 0, song.end, 1);
            const left = Math.min(100, 100 * song.start / length);
            const width = Math.max(1, Math.min(100 - left, 100 * (song.end - song.start) / length));
            const start = clock(song.start);
            const end = clock(song.end);
            (item.querySelector('.result-main') || item).insertAdjacentHTML('beforeend',
                `<div class="play-span" title="${escapeHtml(label)}: start at ${start}, end at ${end} of ${clock(length)}">`
                + (markOf ? markHtml(markOf(song)) : '')
                + `<span class="play-span-track"><span class="play-span-fill" style="left:${left.toFixed(1)}%;width:${width.toFixed(1)}%"></span></span>`
                + `<span>${start} - ${end}</span></div>`);
        });
    }

    function unlockAudio() {
        if (!audioCtx && window.AudioContext) {
            if (navigator.audioSession) navigator.audioSession.type = 'playback';
            audioCtx = new AudioContext();
        }
        if (audioCtx) audioCtx.resume().catch(() => {});
        return !!audioCtx;
    }

    function onGrid(seconds) {
        return Math.round(seconds * audioCtx.sampleRate) / audioCtx.sampleRate;
    }

    async function decode(data) {
        try {
            return await audioCtx.decodeAudioData(data);
        } catch (error) {
            throw new Error('This browser cannot play this song.');
        }
    }

    const levels = new WeakMap();

    function levelOf(buffer) {
        return levels.get(buffer) || 1;
    }

    function along(points, step) {
        return t => {
            const x = Math.min(points.length - 1, Math.max(0, t / step));
            const k = Math.min(points.length - 2, Math.floor(x));
            return points[k] + (points[k + 1] - points[k]) * (x - k);
        };
    }

    function follow(param, values, from, length) {
        if (values) param.setValueCurveAtTime(Float32Array.from(values), from, length);
    }

    function stretchSegment(buffer, inputAt, seconds) {
        const rate = buffer.sampleRate;
        const size = 2 * Math.round(0.023 * rate);
        const hop = size / 2;
        const slack = hop >> 1;
        const count = Math.max(hop, Math.round(seconds * rate));
        const start = Math.round(inputAt(0) * rate);
        const base = Math.max(0, start - slack);
        const offset = start - base;
        const resume = Math.round(inputAt(count / rate) * rate);
        const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
        const mono = new Float32Array(Math.max(size, Math.min(buffer.length, resume + 2 * size + slack) - base));
        channels.forEach(data => {
            for (let i = 0; i < mono.length && base + i < data.length; i++) mono[i] += data[base + i];
        });
        const shape = Float32Array.from({ length: size }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / size));
        const similarity = (p, q, step) => {
            let dot = 0;
            let energy = 1e-9;
            for (let i = 0; i < size; i += step) {
                dot += mono[p + i] * mono[q + i];
                energy += mono[p + i] * mono[p + i];
            }
            return dot / Math.sqrt(energy);
        };
        const search = (low, high, step, q) => {
            let best = low;
            let score = -Infinity;
            for (let p = low; p <= high; p += step) {
                const value = similarity(p, q, step);
                if (value > score) {
                    score = value;
                    best = p;
                }
            }
            return best;
        };
        const out = audioCtx.createBuffer(channels.length, count, rate);
        const targets = channels.map((_, c) => out.getChannelData(c));
        channels.forEach((data, c) => {
            for (let i = 0; i < hop && i < count; i++) targets[c][i] += (data[start + i] || 0) * shape[hop + i];
        });
        let previous = offset;
        for (let k = 0; k * hop < count; k++) {
            const ideal = Math.min(mono.length - size, Math.round(inputAt(k * hop / rate) * rate) - base);
            let pick = Math.max(0, ideal);
            const natural = previous + hop;
            if (k && natural + size <= mono.length) {
                const low = Math.max(0, ideal - slack);
                const high = Math.max(low, Math.min(mono.length - size, ideal + slack));
                const coarse = search(low, high, 4, natural);
                pick = search(Math.max(low, coarse - 3), Math.min(high, coarse + 3), 1, natural);
            }
            channels.forEach((data, c) => {
                for (let i = 0; i < size && k * hop + i < count; i++) targets[c][k * hop + i] += (data[base + pick + i] || 0) * shape[i];
            });
            previous = pick;
        }
        channels.forEach((data, c) => {
            for (let i = 0; i < hop; i++) {
                const blend = (i + 1) / hop;
                targets[c][count - hop + i] = targets[c][count - hop + i] * (1 - blend) + (data[resume - hop + i] || 0) * blend;
            }
        });
        return out;
    }

    function create(list) {
        const bar = document.createElement('div');
        bar.className = 'song-player hidden';
        bar.innerHTML = '<button type="button" class="btn btn-primary play-btn">Play</button>'
            + '<button type="button" class="btn btn-ghost play-skip-btn hidden" disabled>Skip to the fade</button>'
            + '<button type="button" class="btn btn-ghost play-stop-btn" disabled>Stop</button>'
            + '<span class="play-clock"></span>'
            + '<span class="play-summary" title="Transitions planned for this list: beat-matched mixes, then default crossfades"></span>'
            + '<span class="now-playing" aria-live="polite"></span>';
        list.parentNode.insertBefore(bar, list);
        list.classList.add('playable-list');
        const playBtn = bar.querySelector('.play-btn');
        const skipBtn = bar.querySelector('.play-skip-btn');
        const stopBtn = bar.querySelector('.play-stop-btn');
        const playClock = bar.querySelector('.play-clock');
        const playSummary = bar.querySelector('.play-summary');
        const nowPlaying = bar.querySelector('.now-playing');

        let choice = null;
        let shown = 0;
        let playQueue = [];
        let playMode = { songs: [], beatMatch: false, stopAtEnd: false, label: '' };
        let playToken = 0;
        let playCurrent = null;
        let playTicker = null;
        let scrub = null;
        const playNodes = new Set();
        let swapView = null;
        const planCache = new Map();
        const planMarks = new Map();
        let listPairs = [];

        function listSongs() {
            const songs = [...list.querySelectorAll('.result-item')]
                .map((item, row) => ({ item_id: item.dataset.itemId, title: item.dataset.title || 'Unknown', row }))
                .filter(song => song.item_id);
            if (choice && choice.seed) songs.unshift({ item_id: choice.seed.item_id, title: choice.seed.title, row: -1 });
            return songs;
        }

        function withFade(song) {
            const fade = choice && choice.fades.get(song.item_id);
            return fade ? Object.assign({}, song, { start: fade.fade_in_seconds, end: fade.fade_out_seconds, duration: fade.duration_seconds }) : song;
        }

        async function buildQueue() {
            const songs = listSongs();
            if (choice && choice.demo) return { songs: await demos().queue(songs), beatMatch: false, stopAtEnd: true, label: 'Demo' };
            if (choice && choice.smart) return { songs: songs.map(withFade), beatMatch: true, stopAtEnd: false, label: 'Smart fade' };
            return { songs, beatMatch: false, stopAtEnd: false, label: '' };
        }

        async function loadSong(index) {
            const response = await fetch(URLS.play.replace('__ID__', encodeURIComponent(playQueue[index].item_id)));
            if (!response.ok) throw new Error(apiErrorText(await readJsonBody(response), `Could not load the song (HTTP ${response.status})`));
            return decode(await response.arrayBuffer());
        }

        function markPlaying(index) {
            const row = index >= 0 && playQueue[index] ? playQueue[index].row : -1;
            list.querySelectorAll('.result-item').forEach((item, n) => {
                const playing = n === row;
                item.classList.toggle('playing', playing);
                if (playing) item.title = 'Click or drag to move in the song, up to the line where it fades';
                else item.removeAttribute('title');
            });
        }

        function crossfadeSeconds() {
            return playMode.beatMatch ? SMART_CROSSFADE_SECONDS : CROSSFADE_SECONDS;
        }

        function seekLimit(state) {
            if (state.mix) return state.mix.start;
            return state.fadeEnd !== null ? state.fadeEnd - crossfadeSeconds() : state.buffer.duration;
        }

        function showProgress(item, second, state) {
            const length = state.buffer.duration || 1;
            item.style.setProperty('--play-progress', `${(100 * Math.min(1, second / length)).toFixed(2)}%`);
            item.style.setProperty('--play-limit', `${(100 * Math.min(1, seekLimit(state) / length)).toFixed(2)}%`);
        }

        function tick() {
            const now = audioCtx.currentTime;
            const view = swapView && now >= swapView.from ? swapView : null;
            const state = view ? view.state : playCurrent;
            if (!state) return;
            if (view) list.querySelectorAll('.result-item').forEach((item, n) => item.classList.toggle('playing', n === view.row));
            const second = view ? view.at(now) : Math.min(state.buffer.duration, state.node.offset + Math.max(0, now - state.node.when));
            playClock.textContent = `${clock(second)} / ${clock(state.buffer.duration)}`;
            const item = list.querySelector('.result-item.playing');
            if (item && !scrub) showProgress(item, second, state);
        }

        function canSeek(state) {
            return !!state && state === playCurrent && audioCtx.currentTime < state.handoverAt - 0.1;
        }

        function seekTo(state, second) {
            halt(state.node);
            state.node = startSource(state.buffer, audioCtx.currentTime + 0.05, Math.max(0, second), false);
            state.plan();
            tick();
        }

        function skipToFade(state) {
            if (!canSeek(state)) return;
            if (!state.ready) {
                state.skipWhenReady = true;
                nowPlaying.textContent = 'Preparing the mix...';
                return;
            }
            if (state.skipWhenReady) {
                state.skipWhenReady = false;
                nowPlaying.textContent = state.label;
            }
            const target = seekLimit(state) - 6;
            if (target > state.node.offset + audioCtx.currentTime - state.node.when) seekTo(state, target);
        }

        function startSource(buffer, when, offset, fadeIn, filter) {
            const source = audioCtx.createBufferSource();
            const air = audioCtx.createBiquadFilter();
            const tone = audioCtx.createBiquadFilter();
            const mid = audioCtx.createBiquadFilter();
            const high = audioCtx.createBiquadFilter();
            const gain = audioCtx.createGain();
            const level = audioCtx.createGain();
            const node = { source, air, tone, mid, high, gain, level, when: onGrid(when), offset: onGrid(Math.max(0, Math.min(offset, buffer.duration))) };
            air.type = 'lowpass';
            air.frequency.value = audioCtx.sampleRate / 2;
            tone.type = filter || 'lowshelf';
            tone.gain.value = 0;
            mid.type = 'peaking';
            mid.gain.value = 0;
            high.type = 'highshelf';
            high.gain.value = 0;
            source.buffer = buffer;
            source.connect(air);
            air.connect(tone);
            tone.connect(mid);
            mid.connect(high);
            high.connect(gain);
            level.gain.value = levelOf(buffer);
            gain.connect(level);
            level.connect(audioCtx.destination);
            if (fadeIn) gain.gain.setValueCurveAtTime(FADE_IN, node.when, fadeIn);
            source.onended = () => playNodes.delete(node);
            source.start(node.when, node.offset);
            playNodes.add(node);
            return node;
        }

        function halt(node) {
            playNodes.delete(node);
            node.gain.disconnect();
            try {
                node.source.stop();
            } catch (error) {
                node.source.disconnect();
            }
        }

        function stopPlayback() {
            playToken += 1;
            [...playNodes].forEach(halt);
            swapView = null;
            clearInterval(playTicker);
            playCurrent = null;
            scrub = null;
            markPlaying(-1);
            playClock.textContent = '';
            nowPlaying.textContent = '';
            skipBtn.disabled = true;
            stopBtn.disabled = true;
        }

        async function loadTransition(index) {
            const from = playQueue[index];
            const to = playQueue[index + 1];
            if (!playMode.beatMatch || !URLS.transition || !to || typeof from.end !== 'number' || typeof to.start !== 'number') return null;
            const plan = await requestPlan(from, to);
            return plan ? Object.assign({}, plan) : null;
        }

        function pairKey(from, to) {
            return `${from.item_id}>${to.item_id}@${from.end}/${to.start}`;
        }

        function markFor(plan, to) {
            const tempo = plan ? plan.info : {};
            if (plan && plan.type === 'beat_mix') return { kind: 'blend', title: `Beat-matched mix with a bass swap into ${to.title} (${tempo.tempo_from} to ${tempo.tempo_to} BPM)` };
            if (plan && plan.type === 'beat_rise') return { kind: 'rise', title: `Beat-matched rise into ${to.title} (${tempo.tempo_from} to ${tempo.tempo_to} BPM)` };
            if (!tempo.tempo_from) return { kind: 'crossfade', title: `Default crossfade into ${to.title}` };
            return { kind: 'crossfade', title: `Default crossfade into ${to.title} (${tempo.tempo_from} and ${tempo.tempo_to} BPM)` };
        }

        function requestPlan(from, to) {
            const key = pairKey(from, to);
            if (!planCache.has(key)) {
                const query = new URLSearchParams({ from_id: from.item_id, to_id: to.item_id, out: from.end, in: to.start });
                planMarks.set(key, { kind: 'pending', title: 'Planning this transition...' });
                paintMark(to, planMarks.get(key));
                planCache.set(key, fetch(`${URLS.transition}?${query}`)
                    .then(async response => {
                        const plan = await readJsonBody(response);
                        return response.ok && plan && plan.duration > 0 ? plan : null;
                    })
                    .catch(() => {
                        planCache.delete(key);
                        return null;
                    })
                    .then(plan => {
                        planMarks.set(key, markFor(plan, to));
                        paintMark(to, planMarks.get(key));
                        showSummary();
                        return plan;
                    }));
            }
            return planCache.get(key);
        }

        function paintMark(song, mark) {
            const item = song.row >= 0 ? list.querySelectorAll('.result-item')[song.row] : null;
            const old = item && item.querySelector('.play-mark');
            if (old) old.outerHTML = markHtml(mark);
        }

        function smartSpans(songs) {
            const marks = new Map(songs.map((song, k) => {
                if (!k) return [song, { kind: 'none' }];
                const from = songs[k - 1];
                if (!URLS.transition || typeof from.end !== 'number' || typeof song.start !== 'number') return [song, markFor(null, song)];
                return [song, planMarks.get(pairKey(from, song)) || { kind: 'none' }];
            }));
            showSpans(list, songs, 'Smart fade', song => marks.get(song));
        }

        function showSummary() {
            const kinds = listPairs.map(key => (planMarks.get(key) || { kind: 'none' }).kind);
            const special = kinds.filter(kind => kind === 'blend' || kind === 'rise').length;
            const crossfades = kinds.filter(kind => kind === 'crossfade').length;
            playSummary.innerHTML = special + crossfades
                ? `<span class="play-mark" data-mark="blend">${MARKS.blend}</span>${special} <span class="play-mark" data-mark="crossfade">${MARKS.crossfade}</span>${crossfades}`
                : '';
        }

        function trackPairs(songs) {
            listPairs = [];
            for (let k = 1; k < songs.length; k++) {
                if (URLS.transition && typeof songs[k - 1].end === 'number' && typeof songs[k].start === 'number') listPairs.push(pairKey(songs[k - 1], songs[k]));
            }
            showSummary();
        }

        function entryOffset(index) {
            const song = playQueue[index];
            return typeof song.start === 'number' ? song.start : 0;
        }

        function fadeEndOf(index) {
            const song = playQueue[index];
            return typeof song.end === 'number' && (index + 1 < playQueue.length || playMode.stopAtEnd) ? song.end : null;
        }

        function begin(index, buffer, node) {
            const token = playToken;
            const song = playQueue[index];
            const hasNext = index + 1 < playQueue.length;
            const fadeEnd = fadeEndOf(index);
            swapView = null;
            const state = { index, buffer, node, fadeEnd, next: null, mix: null, failed: null, ready: !hasNext, handoverAt: Infinity, timers: [], scheduled: [] };
            const after = (time, action) => state.timers.push(setTimeout(() => {
                if (token === playToken && playCurrent === state) action();
            }, Math.max(0, (time - audioCtx.currentTime) * 1000)));
            state.plan = () => {
                if (token !== playToken || playCurrent !== state || !state.ready) return;
                state.timers.forEach(clearTimeout);
                state.timers = [];
                state.scheduled.forEach(halt);
                state.scheduled = [];
                swapView = null;
                const now = audioCtx.currentTime;
                const at = second => onGrid(state.node.when + second - state.node.offset);
                const mix = state.mix;
                if (!state.next) {
                    const ending = fadeEnd !== null && playMode.stopAtEnd;
                    state.handoverAt = ending ? Math.max(now + 0.05, at(Math.min(fadeEnd, buffer.duration) - CROSSFADE_SECONDS)) : at(buffer.duration);
                    if (ending) {
                        state.node.gain.gain.setValueCurveAtTime(FADE_OUT, state.handoverAt, CROSSFADE_SECONDS);
                        state.node.source.stop(state.handoverAt + CROSSFADE_SECONDS);
                    }
                    after(state.handoverAt + (ending ? CROSSFADE_SECONDS : 0), () => {
                        if (state.failed) nowPlaying.textContent = state.failed;
                        else stopPlayback();
                    });
                } else if (mix && at(mix.start) > now + 0.1) {
                    state.handoverAt = at(mix.start);
                    const direct = !mix.buffer;
                    const resume = onGrid(state.handoverAt + mix.duration);
                    const length = resume - state.handoverAt;
                    const outgoing = direct ? state.node : startSource(mix.lead, state.handoverAt, 0, false);
                    if (!direct) state.node.source.stop(state.handoverAt);
                    outgoing.source.stop(resume);
                    const blend = direct ? startSource(state.next, state.handoverAt, mix.to.position[0], false) : startSource(mix.buffer, state.handoverAt, 0, false, mix.to.highpass_hz ? 'highpass' : 'lowshelf');
                    if (direct) blend.source.stop(resume);
                    [[outgoing, mix.from], [blend, mix.to]].forEach(([node, song]) => {
                        node.tone.frequency.value = mix.filters.bass_hz;
                        node.mid.frequency.value = mix.filters.mid_hz;
                        node.mid.Q.value = mix.filters.mid_q;
                        node.high.frequency.value = mix.filters.high_hz;
                        follow(node.gain.gain, song.volume, state.handoverAt, length);
                        follow(node.high.gain, song.high_db, state.handoverAt, length);
                        follow(node.mid.gain, song.mid_db, state.handoverAt, length);
                        follow(node.tone.gain, song.bass_db, state.handoverAt, length);
                    });
                    follow(blend.tone.frequency, mix.to.highpass_hz, state.handoverAt, length);
                    [outgoing, blend].forEach(node => node.gain.gain.setValueAtTime(0, resume + 1 / audioCtx.sampleRate));
                    const entry = startSource(state.next, resume, direct ? blend.offset + length : mix.follow(mix.buffer.length / mix.buffer.sampleRate), false);
                    state.scheduled = direct ? [blend, entry] : [outgoing, blend, entry];
                    swapView = {
                        from: state.handoverAt + mix.middle,
                        row: playQueue[index + 1].row,
                        state: { buffer: state.next, mix: null, fadeEnd: fadeEndOf(index + 1) },
                        at: direct ? t => mix.to.position[0] + t - state.handoverAt : t => mix.follow(t - state.handoverAt)
                    };
                    after(state.handoverAt, () => {
                        const kind = direct ? 'Default crossfade' : `Beat-matched ${mix.type === 'beat_rise' ? 'rise' : 'mix with a bass swap'}`;
                        const tempo = mix.info;
                        const tempos = !tempo.tempo_from ? '' : direct ? ` (${tempo.tempo_from} and ${tempo.tempo_to} BPM)` : ` (tempo gliding from ${tempo.tempo_from} to ${tempo.tempo_to} BPM)`;
                        nowPlaying.textContent = `${kind} into ${playQueue[index + 1].title}${tempos}`;
                    });
                    after(resume, () => begin(index + 1, state.next, state.scheduled[state.scheduled.length - 1]));
                } else {
                    const fading = fadeEnd !== null;
                    const span = crossfadeSeconds();
                    state.handoverAt = Math.max(now + 0.05, at(fading ? Math.min(fadeEnd, buffer.duration) - span : buffer.duration));
                    if (fading) {
                        state.node.gain.gain.setValueCurveAtTime(FADE_OUT, state.handoverAt, span);
                        state.node.source.stop(state.handoverAt + span);
                    }
                    const entry = startSource(state.next, state.handoverAt, entryOffset(index + 1), fading ? span : 0);
                    state.scheduled = [entry];
                    if (fading && playMode.beatMatch) after(state.handoverAt, () => { nowPlaying.textContent = `Default crossfade into ${playQueue[index + 1].title}`; });
                    after(state.handoverAt + (fading ? span / 2 : 0), () => begin(index + 1, state.next, state.scheduled[0]));
                }
            };
            playCurrent = state;
            markPlaying(index);
            tick();
            state.label = `Playing ${song.title} from ${clock(node.offset)}` + (fadeEnd !== null ? ` until ${clock(fadeEnd)}` : '');
            nowPlaying.textContent = state.label;
            if (hasNext) {
                Promise.all([loadSong(index + 1).catch(error => { state.failed = error.message; return null; }), loadTransition(index)])
                    .then(([next, mix]) => {
                        state.next = next;
                        const planned = next && mix;
                        if (planned) levels.set(next, mix.to.gain);
                        const stretched = planned && mix.type !== 'crossfade';
                        const lead = stretched ? along(mix.from.position, mix.step) : null;
                        const follower = stretched ? along(mix.to.position, mix.step) : null;
                        state.mix = planned ? Object.assign(mix, stretched ? { follow: follower, lead: stretchSegment(buffer, lead, mix.duration), buffer: stretchSegment(next, follower, mix.duration) } : { buffer: null }) : null;
                        if (state.mix && state.mix.buffer) {
                            levels.set(state.mix.lead, levelOf(buffer));
                            levels.set(state.mix.buffer, levelOf(next));
                        }
                        state.ready = true;
                        state.plan();
                        if (state.skipWhenReady) skipToFade(state);
                    })
                    .catch(error => {
                        console.error('Could not prepare the beat-matched mix; crossfading into the next song instead.', error);
                        state.mix = null;
                        state.ready = true;
                        state.plan();
                        if (state.skipWhenReady) skipToFade(state);
                    });
            }
            state.plan();
        }

        async function playSong(index) {
            const token = playToken;
            nowPlaying.textContent = `Loading ${playQueue[index].title}...`;
            try {
                const [buffer, plan] = await Promise.all([loadSong(index), loadTransition(index)]);
                if (plan) levels.set(buffer, plan.from.gain);
                if (token === playToken) begin(index, buffer, startSource(buffer, audioCtx.currentTime + 0.05, entryOffset(index), false));
            } catch (error) {
                if (token === playToken) nowPlaying.textContent = error.message;
            }
        }

        async function startPlayback(row) {
            const ready = unlockAudio();
            stopAll();
            if (!ready) {
                nowPlaying.textContent = 'This browser cannot play songs here.';
                return;
            }
            const token = playToken;
            const demoMode = !!(choice && choice.demo);
            let queue;
            try {
                if (demoMode) nowPlaying.textContent = 'Finding the demos...';
                queue = await buildQueue();
            } catch (error) {
                if (token === playToken) nowPlaying.textContent = error.message;
                return;
            }
            if (token !== playToken) return;
            playMode = queue;
            playQueue = queue.songs;
            if (queue.beatMatch) smartSpans(playQueue);
            else if (queue.label) showSpans(list, playQueue, queue.label);
            const index = row === undefined ? 0 : playQueue.findIndex(song => song.row >= row);
            if (index < 0) {
                nowPlaying.textContent = demoMode ? 'No demo for this song.' : 'Nothing to play.';
                return;
            }
            skipBtn.disabled = false;
            stopBtn.disabled = false;
            playTicker = setInterval(tick, 250);
            playSong(index);
        }

        playBtn.addEventListener('click', () => startPlayback());
        stopBtn.addEventListener('click', stopPlayback);
        skipBtn.addEventListener('click', () => skipToFade(playCurrent));
        list.addEventListener('click', event => {
            const item = event.target.closest('.result-item');
            if (!item || item.classList.contains('playing') || bar.classList.contains('hidden')) return;
            startPlayback([...list.querySelectorAll('.result-item')].indexOf(item));
        });

        function scrubSecond(event) {
            const rect = scrub.item.getBoundingClientRect();
            const share = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
            return Math.max(0, Math.min(share * scrub.state.buffer.duration, seekLimit(scrub.state) - 1));
        }

        list.addEventListener('pointerdown', event => {
            const item = event.target.closest('.result-item.playing');
            if (!item || event.button !== 0 || !canSeek(playCurrent)) return;
            scrub = { item, state: playCurrent };
            item.setPointerCapture(event.pointerId);
            showProgress(item, scrubSecond(event), scrub.state);
        });
        list.addEventListener('pointermove', event => {
            if (scrub) showProgress(scrub.item, scrubSecond(event), scrub.state);
        });
        list.addEventListener('pointerup', event => {
            if (!scrub) return;
            const { state } = scrub;
            const second = scrubSecond(event);
            scrub = null;
            if (canSeek(state)) seekTo(state, second);
        });
        list.addEventListener('pointercancel', () => {
            scrub = null;
            tick();
        });

        async function arrange(rows, options) {
            stopPlayback();
            shown += 1;
            planCache.clear();
            planMarks.clear();
            listPairs = [];
            showSummary();
            const seed = options.seed && options.seed.item_id ? { item_id: String(options.seed.item_id), title: options.seed.title || 'Unknown' } : null;
            choice = { smart: NEURAL && picked.smart && !picked.demo, demo: NEURAL && picked.demo && !picked.smart, seed, fades: new Map(), note: '' };
            if (!choice.smart || !rows.length) return rows;
            try {
                const body = { item_ids: [...(seed ? [seed.item_id] : []), ...rows.map(row => String(row.item_id))] };
                if (options.keepOrder) body.keep_order = true;
                const response = await fetch(URLS.smartFade, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                const data = await readJsonBody(response);
                if (!response.ok || data === null) throw new Error(apiErrorText(data, `Smart fade failed (HTTP ${response.status})`));
                const answer = data.results || [];
                answer.forEach(row => choice.fades.set(row.item_id, row));
                if (options.keepOrder) return rows;
                const waiting = new Map();
                rows.forEach(row => {
                    const key = String(row.item_id);
                    if (!waiting.has(key)) waiting.set(key, []);
                    waiting.get(key).push(row);
                });
                if (seed && waiting.has(seed.item_id)) answer.shift();
                const ordered = [];
                answer.forEach(row => {
                    const bucket = waiting.get(row.item_id);
                    if (bucket && bucket.length) ordered.push(bucket.shift());
                });
                waiting.forEach(bucket => ordered.push(...bucket));
                return ordered;
            } catch (error) {
                choice.smart = false;
                choice.note = `Smart fade is off for these results: ${error.message}`;
                return rows;
            }
        }

        function show() {
            if (!choice) choice = { smart: false, demo: false, seed: null, fades: new Map(), note: '' };
            const token = ++shown;
            bar.classList.remove('hidden');
            skipBtn.classList.toggle('hidden', !choice.smart);
            nowPlaying.textContent = choice.note;
            if (choice.demo) {
                demos().queue(listSongs())
                    .then(songs => { if (token === shown) showSpans(list, songs, 'Demo'); })
                    .catch(error => { if (token === shown && !playCurrent) nowPlaying.textContent = error.message; });
            } else if (choice.smart) {
                const songs = listSongs().map(withFade);
                smartSpans(songs);
                trackPairs(songs);
            }
        }

        function hide() {
            stopPlayback();
            shown += 1;
            choice = null;
            listPairs = [];
            showSummary();
            bar.classList.add('hidden');
        }

        return { arrange, show, hide, stop: stopPlayback };
    }

    function instance(list) {
        if (!instances.has(list)) instances.set(list, create(list));
        return instances.get(list);
    }

    function stopAll() {
        instances.forEach(player => player.stop());
    }

    function renderOptions(box) {
        if (!READY || !NEURAL || box.dataset.ready) return;
        box.dataset.ready = 'true';
        box.innerHTML = Object.entries(OPTIONS).map(([name, option]) => '<span class="song-player-option-group">'
            + `<label class="song-player-option"><input type="checkbox" data-option="${name}"> ${option.label}</label>`
            + `<span class="info-tooltip" tabindex="0"><span class="info-icon"></span><span class="tooltip-text">${option.help}</span></span>`
            + '</span>').join('');
        box.querySelectorAll('input[data-option]').forEach(input => {
            input.checked = picked[input.dataset.option];
            input.addEventListener('change', () => {
                const name = input.dataset.option;
                picked[name] = input.checked;
                if (input.checked) Object.keys(picked).forEach(other => { if (other !== name) picked[other] = false; });
                document.querySelectorAll('.song-player-options input[data-option]').forEach(other => { other.checked = picked[other.dataset.option]; });
            });
        });
    }

    function renderAllOptions() {
        document.querySelectorAll('.song-player-options').forEach(renderOptions);
    }

    renderAllOptions();
    document.addEventListener('DOMContentLoaded', renderAllOptions);

    window.AudioMusePlayer = {
        arrange: (list, rows, options) => READY ? instance(list).arrange(rows || [], options || {}) : Promise.resolve(rows || []),
        show: list => { if (READY) instance(list).show(); },
        hide: list => { if (instances.has(list)) instances.get(list).hide(); },
        clock,
        showSpans
    };
})();
