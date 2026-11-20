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
 * clock and a status line. While a list plays, Play is Pause: it suspends the
 * AudioContext, so every scheduled source and fade waits with its clock, and
 * the timers (atTime) wait for that clock too; Play resumes it. A page calls,
 * around its own rendering:
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
 * loudness for all); from start, for duration, both songs play their own
 * audio at their speed lists (playbackRate: in a beat mix the tempo glides
 * like a DJ deck without key lock, no time-stretch artifacts; a crossfade
 * stays at 1), their faders follow their volume lists (already shaped to a
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
 * The AudioContext runs at the output device's own rate, the most it can play
 * (a 192 kHz song on a 192 kHz device stays 192 kHz). A song is never decoded
 * whole: it plays in pieces (the play route with start and seconds), each its
 * own samples cut on the sample by the server, PIECE_SECONDS long on a grid of
 * whole seconds with PIECE_PAD_SECONDS more on each side (where the browser's
 * resampling of a piece differs from the whole song's). Every song has one
 * filter chain (voice); its pieces feed it one after the other, each joined to
 * the next inside the pad, JOIN_LEAD_SECONDS ahead, through a JOIN_SECONDS blend
 * of identical audio; the next piece is fetched as one starts and a played one
 * is dropped, so a 15 minute 192 kHz song holds about 100 MB, not 1.4 GB. A
 * transition plays one piece per song that covers the whole mix (its speed
 * curve never crosses a join), loaded with the plan and scheduled
 * COMMIT_LEAD_SECONDS before it starts. A late piece is joined inside the pad,
 * or the song waits for it and every time is planned again.
 * A song the browser cannot fetch or decode is announced ("Song not supported
 * in the frontend") and taken out of the queue; playback goes on with the next.
 * The song a click starts plays once its first piece (FIRST_PIECE_SECONDS)
 * came; its transition is awaited only within START_SECONDS, a later answer
 * glides it to its gain.
 * Songs are decoded into Web Audio buffers; every cut, fade
 * and stretch is scheduled on one AudioContext clock made in the first Play
 * click, never on an audio element (Safari refuses play() outside a click and
 * seeks VBR MP3 inaccurately). Every source starts on a whole sample of that
 * clock at a whole-sample offset (onGrid): Chrome and Safari interpolate a
 * source started between two samples for its whole length, which dulls the
 * highs. After a mix the next song goes on from a fresh source at
 * continue_from with flat filters, blended in over JOIN_SECONDS while the
 * mix's source fades out (no filter tail rings on, no click), so outside its
 * transitions a song plays exactly as decoded.
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
    const JOIN_SECONDS = 0.023;
    const START_SECONDS = 4;
    const PIECE_SECONDS = 30;
    const PIECE_PAD_SECONDS = 1;
    const FIRST_PIECE_SECONDS = 8;
    const JOIN_LEAD_SECONDS = 5;
    const COMMIT_LEAD_SECONDS = 5;
    const GAIN_GLIDE_SECONDS = 0.3;
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

    function atTime(timers, time, action) {
        const fire = () => {
            const left = time - audioCtx.currentTime;
            if (audioCtx.state === 'suspended' && left > 0.02) timers.push(setTimeout(fire, Math.max(250, left * 1000)));
            else action();
        };
        timers.push(setTimeout(fire, Math.max(0, (time - audioCtx.currentTime) * 1000)));
    }

    function onGrid(seconds) {
        return Math.round(seconds * audioCtx.sampleRate) / audioCtx.sampleRate;
    }

    async function decode(data) {
        try {
            return await audioCtx.decodeAudioData(data);
        } catch (error) {
            throw new Error('Song not supported in the frontend.');
        }
    }

    function skipNotice(song, error) {
        return `Skipped ${song.title}: ${error.message}`;
    }

    const levels = new WeakMap();

    function levelOf(track) {
        return levels.get(track) || 1;
    }

    function travelled(speed, step) {
        let seconds = 0;
        for (let k = 1; k < speed.length; k++) seconds += (speed[k - 1] + speed[k]) / 2 * step;
        return seconds;
    }

    function follow(param, values, from, length) {
        if (values) param.setValueCurveAtTime(Float32Array.from(values), from, length);
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

        function makeTrack(song) {
            const seconds = typeof song.duration === 'number' && song.duration > 0 ? song.duration : null;
            return { song, seconds, pieces: new Map(), ready: new Map(), onLength: null };
        }

        function learnLength(track, seconds) {
            if (!(seconds > 0) || (track.seconds !== null && Math.abs(track.seconds - seconds) < 0.01)) return;
            track.seconds = seconds;
            if (track.onLength) track.onLength();
        }

        function loadPiece(track, start, seconds) {
            const key = `${start}/${seconds}`;
            if (!track.pieces.has(key)) {
                const url = `${URLS.play.replace('__ID__', encodeURIComponent(track.song.item_id))}?start=${start}&seconds=${seconds}`;
                const request = fetch(url).then(async response => {
                    if (!response.ok) throw new Error(apiErrorText(await readJsonBody(response), `Could not load the song (HTTP ${response.status})`));
                    const length = Number(response.headers.get('X-Song-Seconds'));
                    const buffer = await decode(await response.arrayBuffer());
                    const end = start + buffer.duration;
                    const last = buffer.length < Math.round(seconds * audioCtx.sampleRate) - 1;
                    const piece = { key, start, buffer, end, last, until: last ? end : start + seconds - PIECE_PAD_SECONDS };
                    track.ready.set(key, piece);
                    learnLength(track, length > 0 ? length : last ? end : null);
                    return piece;
                });
                request.catch(() => track.pieces.delete(key));
                track.pieces.set(key, request);
            }
            return track.pieces.get(key);
        }

        function pieceAt(track, second) {
            const k = Math.floor(second / PIECE_SECONDS);
            const start = Math.max(0, k * PIECE_SECONDS - PIECE_PAD_SECONDS);
            return loadPiece(track, start, (k + 1) * PIECE_SECONDS + PIECE_PAD_SECONDS - start);
        }

        function pieceAround(track, from, to) {
            const start = Math.max(0, Math.floor(from) - PIECE_PAD_SECONDS);
            return loadPiece(track, start, Math.ceil(to) + PIECE_PAD_SECONDS - start);
        }

        function coveringPiece(track, second) {
            let best = null;
            track.ready.forEach(piece => {
                if (piece.start <= second && second < piece.end - 0.5 && (!best || piece.until > best.until)) best = piece;
            });
            return best;
        }

        function prune(track, second) {
            track.ready.forEach((piece, key) => {
                if (piece.until <= second) {
                    track.ready.delete(key);
                    track.pieces.delete(key);
                }
            });
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

        function lengthOf(state) {
            return state.track.seconds || 0;
        }

        function seekLimit(state) {
            if (state.mix) return state.mix.start;
            return state.fadeEnd !== null ? state.fadeEnd - crossfadeSeconds() : lengthOf(state);
        }

        function showProgress(item, second, state) {
            const length = lengthOf(state) || 1;
            item.style.setProperty('--play-progress', `${(100 * Math.min(1, second / length)).toFixed(2)}%`);
            item.style.setProperty('--play-limit', `${(100 * Math.min(1, seekLimit(state) / length)).toFixed(2)}%`);
        }

        function tick() {
            const now = audioCtx.currentTime;
            const view = swapView && now >= swapView.from ? swapView : null;
            const state = view ? view.state : playCurrent;
            if (!state) return;
            if (view) list.querySelectorAll('.result-item').forEach((item, n) => item.classList.toggle('playing', n === view.row));
            const second = view ? view.at(now) : Math.min(lengthOf(state) || Infinity, state.node.offset + Math.max(0, now - state.node.when));
            playClock.textContent = `${clock(second)} / ${clock(lengthOf(state))}`;
            const item = list.querySelector('.result-item.playing');
            if (item && !scrub) showProgress(item, second, state);
        }

        function isLive(state) {
            return !!state && state === playCurrent && audioCtx.currentTime < state.handoverAt - 0.1;
        }

        function canSeek(state) {
            return isLive(state) && !state.committed;
        }

        function glide(state, gain) {
            if (state !== playCurrent) return;
            levels.set(state.track, gain);
            state.node.level.gain.setTargetAtTime(gain, audioCtx.currentTime, GAIN_GLIDE_SECONDS);
        }

        function seekTo(state, second) {
            const target = Math.max(0, second);
            const seek = state.seek = {};
            pieceAt(state.track, target).then(() => {
                if (state.seek !== seek || !canSeek(state)) return;
                halt(state.node);
                state.node = startSource(state.track, audioCtx.currentTime + 0.05, target, false);
                state.node.onMove = state.plan;
                state.plan();
                tick();
            }, error => console.error('Could not move in the song.', error));
        }

        function skipToFade(state) {
            if (!isLive(state)) return;
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

        function makeVoice(track, filter, when, offset) {
            const air = audioCtx.createBiquadFilter();
            const tone = audioCtx.createBiquadFilter();
            const mid = audioCtx.createBiquadFilter();
            const high = audioCtx.createBiquadFilter();
            const gain = audioCtx.createGain();
            const level = audioCtx.createGain();
            const voice = {
                track, air, tone, mid, high, gain, level, when: onGrid(when), offset: onGrid(Math.max(0, offset)),
                feeds: [], timers: [], chained: null, single: null, pending: false, halted: false,
                stopAt: Infinity, noJoinAfter: Infinity, onMove: null
            };
            air.type = 'lowpass';
            air.frequency.value = audioCtx.sampleRate / 2;
            tone.type = filter || 'lowshelf';
            tone.gain.value = 0;
            mid.type = 'peaking';
            mid.gain.value = 0;
            high.type = 'highshelf';
            high.gain.value = 0;
            air.connect(tone);
            tone.connect(mid);
            mid.connect(high);
            high.connect(gain);
            level.gain.value = levelOf(track);
            gain.connect(level);
            level.connect(audioCtx.destination);
            playNodes.add(voice);
            return voice;
        }

        function endFeed(item, at) {
            item.end = Math.min(item.end, at);
            item.source.stop(Math.max(item.end, item.when));
        }

        function feed(voice, piece, when, second, fadeIn) {
            const source = audioCtx.createBufferSource();
            const fader = audioCtx.createGain();
            const item = { source, fader, piece, when, end: Infinity };
            source.buffer = piece.buffer;
            source.connect(fader);
            fader.connect(voice.air);
            if (fadeIn) fader.gain.setValueCurveAtTime(FADE_IN, when, fadeIn);
            source.onended = () => {
                fader.disconnect();
                voice.feeds = voice.feeds.filter(other => other !== item);
                if (!voice.feeds.length && !voice.pending) playNodes.delete(voice);
            };
            source.start(when, onGrid(Math.max(0, second - piece.start)));
            voice.feeds.push(item);
            if (voice.stopAt < Infinity) endFeed(item, voice.stopAt);
            return item;
        }

        function chain(voice, piece, when, second, fadeIn) {
            const item = feed(voice, piece, when, second, fadeIn);
            voice.chained = item;
            voice.pending = !piece.last;
            if (piece.last) return;
            const joinAt = onGrid(when + piece.until - second);
            const next = pieceAt(voice.track, piece.until);
            const join = () => next.then(following => {
                if (voice.halted || voice.chained !== item) return;
                if (joinAt >= voice.stopAt || joinAt >= voice.noJoinAfter) {
                    voice.pending = false;
                    return;
                }
                const at = Math.max(joinAt, onGrid(audioCtx.currentTime + 0.05));
                if (at + JOIN_SECONDS <= when + piece.end - second) {
                    item.fader.gain.setValueCurveAtTime(FADE_OUT, at, JOIN_SECONDS);
                    endFeed(item, at + JOIN_SECONDS);
                    chain(voice, following, at, second + (at - when), JOIN_SECONDS);
                } else {
                    voice.when = onGrid(audioCtx.currentTime + 0.05);
                    voice.offset = onGrid(piece.end);
                    chain(voice, following, voice.when, voice.offset, JOIN_SECONDS);
                    if (voice.onMove) voice.onMove();
                }
                prune(voice.track, piece.until);
            }, error => {
                voice.pending = false;
                console.error('The rest of the song did not load; it plays as far as it came.', error);
                learnLength(voice.track, piece.end);
            });
            atTime(voice.timers, joinAt - JOIN_LEAD_SECONDS, join);
        }

        function startSource(track, when, offset, fadeIn, filter) {
            const voice = makeVoice(track, filter, when, offset);
            if (fadeIn) voice.gain.gain.setValueCurveAtTime(FADE_IN, voice.when, fadeIn);
            const piece = coveringPiece(track, voice.offset);
            if (piece) {
                chain(voice, piece, voice.when, voice.offset, 0);
                return voice;
            }
            voice.pending = true;
            pieceAt(track, voice.offset).then(found => {
                if (voice.halted) return;
                const late = onGrid(audioCtx.currentTime + 0.05);
                if (late > voice.when) {
                    voice.offset = onGrid(voice.offset);
                    voice.when = late;
                }
                chain(voice, found, voice.when, voice.offset, 0);
                if (voice.onMove) voice.onMove();
            }, error => {
                voice.pending = false;
                console.error('This part of the song did not load.', error);
            });
            return voice;
        }

        function startPieceVoice(track, piece, when, second, filter) {
            const voice = makeVoice(track, filter, when, second);
            voice.single = feed(voice, piece, voice.when, voice.offset, 0);
            return voice;
        }

        function switchToPiece(voice, piece, at) {
            const second = voice.offset + (at - voice.when);
            voice.timers.forEach(clearTimeout);
            voice.timers = [];
            voice.chained = null;
            voice.feeds.forEach(item => {
                if (item.when < at) {
                    item.fader.gain.cancelScheduledValues(at);
                    item.fader.gain.setValueCurveAtTime(FADE_OUT, at, JOIN_SECONDS);
                    endFeed(item, at + JOIN_SECONDS);
                } else {
                    endFeed(item, item.when);
                }
            });
            voice.single = feed(voice, piece, at, second, JOIN_SECONDS);
            voice.pending = false;
            return voice.single.source;
        }

        function stopVoice(voice, at) {
            voice.stopAt = Math.min(voice.stopAt, at);
            voice.feeds.forEach(item => endFeed(item, voice.stopAt));
        }

        function halt(voice) {
            voice.halted = true;
            voice.timers.forEach(clearTimeout);
            playNodes.delete(voice);
            voice.gain.disconnect();
            voice.feeds.forEach(item => {
                try {
                    item.source.stop();
                } catch (error) {
                    item.source.disconnect();
                }
            });
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
            playBtn.textContent = 'Play';
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

        function begin(index, track, node) {
            const token = playToken;
            const song = playQueue[index];
            const hasNext = index + 1 < playQueue.length;
            const fadeEnd = fadeEndOf(index);
            swapView = null;
            const state = {
                index, track, node, fadeEnd, next: null, mix: null, pieces: null, ready: !hasNext, committed: false,
                handoverAt: Infinity, timers: [], scheduled: []
            };
            const after = (time, action) => atTime(state.timers, time, () => {
                if (token === playToken && playCurrent === state) action();
            });
            state.plan = () => {
                if (token !== playToken || playCurrent !== state || !state.ready || state.committed) return;
                state.timers.forEach(clearTimeout);
                state.timers = [];
                state.scheduled.forEach(halt);
                state.scheduled = [];
                swapView = null;
                state.node.noJoinAfter = Infinity;
                const now = audioCtx.currentTime;
                const at = second => onGrid(state.node.when + second - state.node.offset);
                const length = lengthOf(state);
                const mix = state.mix;
                let commit;
                if (!state.next) {
                    if (!length) {
                        state.handoverAt = Infinity;
                        return;
                    }
                    const ending = fadeEnd !== null && playMode.stopAtEnd;
                    state.handoverAt = ending ? Math.max(now + 0.05, at(Math.min(fadeEnd, length) - CROSSFADE_SECONDS)) : at(length);
                    commit = () => {
                        if (ending) {
                            state.node.gain.gain.setValueCurveAtTime(FADE_OUT, state.handoverAt, CROSSFADE_SECONDS);
                            stopVoice(state.node, state.handoverAt + CROSSFADE_SECONDS);
                        }
                        after(state.handoverAt + (ending ? CROSSFADE_SECONDS : 0), stopPlayback);
                    };
                } else if (mix && state.pieces && at(mix.start) > now + 0.1) {
                    state.handoverAt = at(mix.start);
                    state.node.noJoinAfter = state.handoverAt - 2 * JOIN_SECONDS;
                    commit = () => {
                        const plain = mix.type === 'crossfade';
                        const resume = onGrid(state.handoverAt + mix.duration);
                        const span = resume - state.handoverAt;
                        const outgoing = state.node;
                        const outgoingSource = switchToPiece(outgoing, state.pieces.from, state.handoverAt);
                        stopVoice(outgoing, resume);
                        const blend = startPieceVoice(state.next, state.pieces.to, state.handoverAt, mix.to.start, mix.to.highpass_hz ? 'highpass' : 'lowshelf');
                        stopVoice(blend, resume + JOIN_SECONDS);
                        [[outgoing, mix.from, outgoingSource], [blend, mix.to, blend.single.source]].forEach(([node, side, source]) => {
                            node.tone.frequency.value = mix.filters.bass_hz;
                            node.mid.frequency.value = mix.filters.mid_hz;
                            node.mid.Q.value = mix.filters.mid_q;
                            node.high.frequency.value = mix.filters.high_hz;
                            follow(source.playbackRate, side.speed, state.handoverAt, span);
                            follow(node.gain.gain, side.volume, state.handoverAt, span);
                            follow(node.high.gain, side.high_db, state.handoverAt, span);
                            follow(node.mid.gain, side.mid_db, state.handoverAt, span);
                            follow(node.tone.gain, side.bass_db, state.handoverAt, span);
                        });
                        follow(blend.tone.frequency, mix.to.highpass_hz, state.handoverAt, span);
                        outgoing.gain.gain.setValueAtTime(0, resume + 1 / audioCtx.sampleRate);
                        blend.gain.gain.setValueCurveAtTime(FADE_OUT, resume + 1 / audioCtx.sampleRate, JOIN_SECONDS);
                        const entry = startSource(state.next, resume, mix.to.continue_from, JOIN_SECONDS);
                        state.scheduled = [blend, entry];
                        const played = mix.to.continue_from - mix.to.start;
                        swapView = {
                            from: state.handoverAt + mix.middle,
                            row: playQueue[index + 1].row,
                            state: { track: state.next, mix: null, fadeEnd: fadeEndOf(index + 1) },
                            at: t => mix.to.start + played * Math.min(1, Math.max(0, (t - state.handoverAt) / span))
                        };
                        after(state.handoverAt, () => {
                            const kind = plain ? 'Default crossfade' : `Beat-matched ${mix.type === 'beat_rise' ? 'rise' : 'mix with a bass swap'}`;
                            const tempo = mix.info;
                            const tempos = !tempo.tempo_from ? '' : plain ? ` (${tempo.tempo_from} and ${tempo.tempo_to} BPM)` : ` (tempo gliding from ${tempo.tempo_from} to ${tempo.tempo_to} BPM)`;
                            nowPlaying.textContent = `${kind} into ${playQueue[index + 1].title}${tempos}`;
                        });
                        after(resume, () => begin(index + 1, state.next, entry));
                    };
                } else {
                    if (!length) {
                        state.handoverAt = Infinity;
                        return;
                    }
                    const fading = fadeEnd !== null;
                    const span = crossfadeSeconds();
                    state.handoverAt = Math.max(now + 0.05, at(fading ? Math.min(fadeEnd, length) - span : length));
                    commit = () => {
                        if (fading) {
                            state.node.gain.gain.setValueCurveAtTime(FADE_OUT, state.handoverAt, span);
                            stopVoice(state.node, state.handoverAt + span);
                        }
                        const entry = startSource(state.next, state.handoverAt, entryOffset(index + 1), fading ? span : 0);
                        state.scheduled = [entry];
                        if (fading && playMode.beatMatch) after(state.handoverAt, () => { nowPlaying.textContent = `Default crossfade into ${playQueue[index + 1].title}`; });
                        after(state.handoverAt + (fading ? span / 2 : 0), () => begin(index + 1, state.next, entry));
                    };
                }
                after(state.handoverAt - COMMIT_LEAD_SECONDS, () => {
                    state.committed = true;
                    commit();
                });
            };
            playCurrent = state;
            node.onMove = state.plan;
            track.onLength = state.plan;
            markPlaying(index);
            tick();
            state.label = `Playing ${song.title} from ${clock(node.offset)}` + (fadeEnd !== null ? ` until ${clock(fadeEnd)}` : '');
            nowPlaying.textContent = state.label;
            if (hasNext) {
                prepareNext(state, token)
                    .then(([next, mix, pieces]) => {
                        state.next = next;
                        const planned = next && mix && pieces;
                        if (planned) levels.set(next, mix.to.gain);
                        state.mix = planned ? mix : null;
                        state.pieces = planned ? pieces : null;
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
            return state;
        }

        async function mixPieces(state, next, mix) {
            if (!mix) return null;
            try {
                const ending = mix.start + travelled(mix.from.speed, mix.step);
                const [from, to] = await Promise.all([
                    pieceAround(state.track, mix.start, ending),
                    pieceAround(next, mix.to.start, mix.to.continue_from),
                    pieceAt(next, mix.to.continue_from)
                ]);
                return { from, to };
            } catch (error) {
                console.error('Could not load the pieces of the beat-matched mix; crossfading into the next song instead.', error);
                return null;
            }
        }

        async function prepareNext(state, token) {
            while (state.index + 1 < playQueue.length) {
                const song = playQueue[state.index + 1];
                const next = makeTrack(song);
                let notice = '';
                const [entry, mix] = await Promise.all([
                    pieceAt(next, entryOffset(state.index + 1)).catch(error => { notice = skipNotice(song, error); return null; }),
                    loadTransition(state.index)
                ]);
                if (token !== playToken || playCurrent !== state) return [null, null, null];
                if (entry) return [next, mix, await mixPieces(state, next, mix)];
                playQueue.splice(state.index + 1, 1);
                nowPlaying.textContent = `${state.label}. ${notice}`;
            }
            return [null, null, null];
        }

        async function playSong(index) {
            const token = playToken;
            const clicked = performance.now();
            let notice = '';
            try {
                while (index < playQueue.length) {
                    const song = playQueue[index];
                    nowPlaying.textContent = `${notice ? notice + ' ' : ''}Loading ${song.title}...`;
                    const track = makeTrack(song);
                    const entry = entryOffset(index);
                    const planned = loadTransition(index);
                    const first = await pieceAround(track, entry, entry + FIRST_PIECE_SECONDS).catch(error => { notice = skipNotice(song, error); return null; });
                    if (token !== playToken) return;
                    if (!first) {
                        playQueue.splice(index, 1);
                        continue;
                    }
                    const left = Math.max(0, START_SECONDS - (performance.now() - clicked) / 1000);
                    const plan = await Promise.race([planned, new Promise(resolve => setTimeout(resolve, left * 1000))]);
                    if (token !== playToken) return;
                    if (plan) levels.set(track, plan.from.gain);
                    const state = begin(index, track, startSource(track, audioCtx.currentTime + 0.05, entry, false));
                    if (plan === undefined) planned.then(late => { if (late && token === playToken) glide(state, late.from.gain); });
                    if (notice) nowPlaying.textContent = `${state.label}. ${notice}`;
                    return;
                }
                nowPlaying.textContent = notice || 'Nothing to play.';
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
            playBtn.textContent = 'Pause';
            playTicker = setInterval(tick, 250);
            playSong(index);
        }

        playBtn.addEventListener('click', () => {
            if (stopBtn.disabled) return startPlayback();
            const pause = playBtn.textContent === 'Pause';
            (pause ? audioCtx.suspend() : audioCtx.resume()).catch(() => {});
            playBtn.textContent = pause ? 'Play' : 'Pause';
        });
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
            return Math.max(0, Math.min(share * lengthOf(scrub.state), seekLimit(scrub.state) - 1));
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
