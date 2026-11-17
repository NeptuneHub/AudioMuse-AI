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
 * fade-out, mixed into the next on the beat by the transition API's plan. A row
 * click plays from that row.
 *
 * Songs are fetched whole and decoded into Web Audio buffers; every cut, fade
 * and stretch is scheduled on one AudioContext clock made in the first Play
 * click, never on an audio element (Safari refuses play() outside a click and
 * seeks VBR MP3 inaccurately). One list plays at a time. The playing row fills
 * with the progress and a click or drag on it moves in the song, up to the line
 * where it fades.
 */
(function () {
    const CROSSFADE_SECONDS = 4;
    const settings = document.currentScript ? document.currentScript.dataset : {};
    const URLS = { play: settings.playUrl || '', smartFade: settings.smartFadeUrl || '', transition: settings.transitionUrl || '' };
    const READY = !!URLS.play;
    const NEURAL = settings.neural === 'true' && !!URLS.smartFade;
    const OPTIONS = {
        smart: {
            label: 'Smart fade',
            help: 'Orders the songs so each one flows into the next (a path or an album keeps its order), plays each from where it should start to where it should end with a beat-matched mix, and shows Skip to the fade. Set it before the search.'
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

    function showSpans(list, songs, label) {
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

    async function decode(data) {
        try {
            return await audioCtx.decodeAudioData(data);
        } catch (error) {
            throw new Error('This browser cannot play this song.');
        }
    }

    function stretchSegment(buffer, from, to, seconds) {
        const rate = buffer.sampleRate;
        const size = 2 * Math.round(0.023 * rate);
        const hop = size / 2;
        const slack = hop >> 1;
        const count = Math.max(hop, Math.round(seconds * rate));
        const base = Math.max(0, Math.round(from * rate) - slack);
        const offset = Math.round(from * rate) - base;
        const resume = Math.round(to * rate);
        const speed = (resume - Math.round(from * rate)) / count;
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
        let previous = offset;
        for (let k = 0; k * hop < count; k++) {
            const ideal = Math.min(mono.length - size, offset + Math.round(k * hop * speed));
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

    function powerCurve(rising) {
        return Float32Array.from({ length: 256 }, (_, i) => (rising ? Math.sin : Math.cos)(Math.PI / 2 * i / 255));
    }

    function create(list) {
        const bar = document.createElement('div');
        bar.className = 'song-player hidden';
        bar.innerHTML = '<button type="button" class="btn btn-primary play-btn">Play</button>'
            + '<button type="button" class="btn btn-ghost play-skip-btn hidden" disabled>Skip to the fade</button>'
            + '<button type="button" class="btn btn-ghost play-stop-btn" disabled>Stop</button>'
            + '<span class="play-clock"></span>'
            + '<span class="now-playing" aria-live="polite"></span>';
        list.parentNode.insertBefore(bar, list);
        list.classList.add('playable-list');
        const playBtn = bar.querySelector('.play-btn');
        const skipBtn = bar.querySelector('.play-skip-btn');
        const stopBtn = bar.querySelector('.play-stop-btn');
        const playClock = bar.querySelector('.play-clock');
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

        function seekLimit(state) {
            if (state.mix) return state.mix.mix_at;
            return state.fadeEnd !== null ? state.fadeEnd - CROSSFADE_SECONDS : state.buffer.duration;
        }

        function showProgress(item, second, state) {
            const length = state.buffer.duration || 1;
            item.style.setProperty('--play-progress', `${(100 * Math.min(1, second / length)).toFixed(2)}%`);
            item.style.setProperty('--play-limit', `${(100 * Math.min(1, seekLimit(state) / length)).toFixed(2)}%`);
        }

        function tick() {
            const state = playCurrent;
            if (!state) return;
            const second = Math.min(state.buffer.duration, state.node.offset + Math.max(0, audioCtx.currentTime - state.node.when));
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

        function startSource(buffer, when, offset, fadeIn) {
            const source = audioCtx.createBufferSource();
            const gain = audioCtx.createGain();
            const node = { source, gain, when, offset: Math.max(0, Math.min(offset, buffer.duration)) };
            source.buffer = buffer;
            source.connect(gain);
            gain.connect(audioCtx.destination);
            if (fadeIn) {
                gain.gain.setValueAtTime(0, when);
                gain.gain.linearRampToValueAtTime(1, when + CROSSFADE_SECONDS);
            }
            source.onended = () => playNodes.delete(node);
            source.start(when, node.offset);
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
            const query = new URLSearchParams({ from_id: from.item_id, to_id: to.item_id, out: from.end, in: to.start });
            try {
                const response = await fetch(`${URLS.transition}?${query}`);
                const plan = await readJsonBody(response);
                return response.ok && plan && plan.mix_seconds > 0 ? plan : null;
            } catch (error) {
                return null;
            }
        }

        function entryOffset(index) {
            const song = playQueue[index];
            return typeof song.start === 'number' ? song.start : 0;
        }

        function begin(index, buffer, node) {
            const token = playToken;
            const song = playQueue[index];
            const hasNext = index + 1 < playQueue.length;
            const fadeEnd = typeof song.end === 'number' && (hasNext || playMode.stopAtEnd) ? song.end : null;
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
                const now = audioCtx.currentTime;
                const at = second => state.node.when + second - state.node.offset;
                const mix = state.mix;
                if (!state.next) {
                    const ending = fadeEnd !== null && playMode.stopAtEnd;
                    state.handoverAt = ending ? Math.max(now + 0.05, at(Math.min(fadeEnd, buffer.duration) - CROSSFADE_SECONDS)) : at(buffer.duration);
                    if (ending) {
                        state.node.gain.gain.setValueAtTime(1, state.handoverAt);
                        state.node.gain.gain.linearRampToValueAtTime(0, state.handoverAt + CROSSFADE_SECONDS);
                        state.node.source.stop(state.handoverAt + CROSSFADE_SECONDS);
                    }
                    after(state.handoverAt + (ending ? CROSSFADE_SECONDS : 0), () => {
                        if (state.failed) nowPlaying.textContent = state.failed;
                        else stopPlayback();
                    });
                } else if (mix && at(mix.mix_at) > now + 0.1) {
                    state.handoverAt = at(mix.mix_at);
                    const resume = state.handoverAt + mix.mix_seconds;
                    state.node.gain.gain.setValueCurveAtTime(powerCurve(false), state.handoverAt, mix.mix_seconds);
                    state.node.source.stop(resume);
                    const blend = startSource(mix.buffer, state.handoverAt, 0, false);
                    blend.gain.gain.setValueCurveAtTime(powerCurve(true), state.handoverAt, mix.mix_seconds);
                    state.scheduled = [blend, startSource(state.next, resume, mix.resume_at, false)];
                    after(state.handoverAt, () => {
                        nowPlaying.textContent = `Beat-matched mix into ${playQueue[index + 1].title} (${mix.tempo_from} to ${mix.tempo_to} BPM, stretch ${mix.stretch})`;
                    });
                    after(resume, () => begin(index + 1, state.next, state.scheduled[1]));
                } else {
                    const fading = fadeEnd !== null;
                    state.handoverAt = Math.max(now + 0.05, at(fading ? Math.min(fadeEnd, buffer.duration) - CROSSFADE_SECONDS : buffer.duration));
                    if (fading) {
                        state.node.gain.gain.setValueAtTime(1, state.handoverAt);
                        state.node.gain.gain.linearRampToValueAtTime(0, state.handoverAt + CROSSFADE_SECONDS);
                        state.node.source.stop(state.handoverAt + CROSSFADE_SECONDS);
                    }
                    state.scheduled = [startSource(state.next, state.handoverAt, entryOffset(index + 1), fading)];
                    after(state.handoverAt, () => begin(index + 1, state.next, state.scheduled[0]));
                }
            };
            playCurrent = state;
            markPlaying(index);
            nowPlaying.textContent = `Playing ${song.title} from ${clock(node.offset)}` + (fadeEnd !== null ? ` until ${clock(fadeEnd)}` : '');
            if (hasNext) {
                Promise.all([loadSong(index + 1).catch(error => { state.failed = error.message; return null; }), loadTransition(index)])
                    .then(([next, mix]) => {
                        state.next = next;
                        state.mix = next && mix ? Object.assign(mix, { buffer: stretchSegment(next, mix.in_at, mix.resume_at, mix.mix_seconds) }) : null;
                        state.ready = true;
                        state.plan();
                    })
                    .catch(error => {
                        console.error('Could not prepare the beat-matched mix; crossfading into the next song instead.', error);
                        state.mix = null;
                        state.ready = true;
                        state.plan();
                    });
            }
            state.plan();
        }

        async function playSong(index) {
            const token = playToken;
            nowPlaying.textContent = `Loading ${playQueue[index].title}...`;
            try {
                const buffer = await loadSong(index);
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
            if (queue.label) showSpans(list, playQueue, queue.label);
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
        skipBtn.addEventListener('click', () => {
            const state = playCurrent;
            if (!canSeek(state)) return;
            const target = seekLimit(state) - 6;
            if (target > state.node.offset + audioCtx.currentTime - state.node.when) seekTo(state, target);
        });
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
                showSpans(list, listSongs().map(withFade), 'Smart fade');
            }
        }

        function hide() {
            stopPlayback();
            shown += 1;
            choice = null;
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
