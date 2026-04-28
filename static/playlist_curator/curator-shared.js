/* ============================================================
   Playlist Curator — shared client logic
   Loaded on BOTH the Smart Search and Playlist Extender pages.
   Owns:
     - Workbench state (in-memory + localStorage), DOM rendering
     - Sticky web player (audio element + transport)
     - Save playlist flow (POST /api/curator/save_playlist)
     - Find duplicates flow (POST /api/curator/find_duplicates)
     - Toast + status helpers shared between pages

   Page-specific scripts (curator-search.js / curator-extender.js)
   build on top of this module via window.* hooks.
   ============================================================ */
(function () {
    'use strict';

    // ---------- Inline SVG icons ----------
    const ICONS = {
        list: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',
        link: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
        save: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/></svg>',
        copies: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
        trash: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
        x: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
        caret: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
        play: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
        pause: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>',
        check: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
        warn: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"/></svg>',
        plus: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>',
        music: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
    };
    window.CURATOR_ICONS = ICONS;

    // ---------- Helpers ----------
    function escHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }
    window.escHtml = escHtml;

    const INFLUENCE = [
        { level: 0, label: 'Normal', tip: 'Same influence as all other tracks' },
        { level: 1, label: 'Boost',  tip: '~5% influence — nudges results toward this track' },
        { level: 2, label: 'Strong', tip: '~15% influence — noticeably shapes results' },
        { level: 3, label: 'Focus',  tip: '~30% influence — dominates the search direction' },
    ];
    function getInfluenceInfo(level) { return INFLUENCE[level] || INFLUENCE[0]; }
    window.CURATOR_INFLUENCE = INFLUENCE;
    window.getInfluenceInfo = getInfluenceInfo;

    // ---------- Workbench state ----------
    const STORAGE_KEY = 'audiomuse:curator:workbench';
    let workbench = { tracks: [] };

    function loadWorkbench() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return { tracks: [] };
            const parsed = JSON.parse(raw);
            if (!parsed || !Array.isArray(parsed.tracks)) return { tracks: [] };
            return parsed;
        } catch (e) {
            console.warn('Failed to load workbench from localStorage:', e);
            return { tracks: [] };
        }
    }

    function saveWorkbench() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(workbench));
        } catch (e) {
            console.warn('Failed to save workbench:', e);
        }
    }

    // Trailing-debounced save: in-memory state mutates immediately, but the
    // expensive JSON.stringify + localStorage.setItem is deferred. flushSync()
    // runs the pending write right now (used before navigation to avoid races).
    let saveTimer = null;
    function scheduleSave() {
        if (saveTimer !== null) return;
        saveTimer = setTimeout(() => { saveTimer = null; saveWorkbench(); }, 250);
    }
    function flushSync() {
        if (saveTimer !== null) { clearTimeout(saveTimer); saveTimer = null; }
        saveWorkbench();
    }
    window.addEventListener('beforeunload', flushSync);

    function getWorkbench() { return workbench; }

    // Public mutators — page scripts call these
    function workbenchAdd(track, source) {
        if (!track || !track.item_id) return;
        if (workbench.tracks.some(t => t.item_id === track.item_id)) return;
        workbench.tracks.push(Object.assign({}, track, { source: source || 'search', influence: 0 }));
        commit({ changedIds: [track.item_id] });
    }

    function workbenchAddBulk(tracks, source) {
        if (!Array.isArray(tracks) || tracks.length === 0) return 0;
        const existing = new Set(workbench.tracks.map(t => t.item_id));
        let added = 0;
        const addedIds = [];
        tracks.forEach(t => {
            if (!t || !t.item_id || existing.has(t.item_id)) return;
            workbench.tracks.push(Object.assign({}, t, { source: source || 'search', influence: 0 }));
            existing.add(t.item_id);
            addedIds.push(t.item_id);
            added++;
        });
        if (added > 0) commit({ changedIds: addedIds });
        return added;
    }

    function workbenchRemove(itemId) {
        const before = workbench.tracks.length;
        workbench.tracks = workbench.tracks.filter(t => t.item_id !== itemId);
        if (workbench.tracks.length !== before) commit({ changedIds: [itemId] });
    }

    function workbenchSetInfluence(itemId, level) {
        const track = workbench.tracks.find(t => t.item_id === itemId);
        if (!track) return;
        track.influence = ((level % 4) + 4) % 4;
        commit({ changedIds: [itemId] });
    }

    function workbenchClear() {
        if (workbench.tracks.length === 0) return;
        workbench = { tracks: [] };
        commit({ changedIds: null });
    }

    function workbenchHas(itemId) {
        return workbench.tracks.some(t => t.item_id === itemId);
    }

    function workbenchGetInfluence(itemId) {
        const t = workbench.tracks.find(x => x.item_id === itemId);
        return t ? (t.influence || 0) : 0;
    }

    function commit(detail) {
        scheduleSave();
        renderWorkbench();
        // Let pages react (e.g. re-paint result rows that should turn green).
        // detail.changedIds is null for "everything changed" (clear, cross-tab sync).
        document.dispatchEvent(new CustomEvent('curator:workbench:changed', {
            detail: detail || { changedIds: null }
        }));
    }

    // Cross-tab sync
    window.addEventListener('storage', (e) => {
        if (e.key !== STORAGE_KEY) return;
        workbench = loadWorkbench();
        renderWorkbench();
        document.dispatchEvent(new CustomEvent('curator:workbench:changed', {
            detail: { changedIds: null }
        }));
    });

    window.workbenchAdd = workbenchAdd;
    window.workbenchAddBulk = workbenchAddBulk;
    window.workbenchRemove = workbenchRemove;
    window.workbenchSetInfluence = workbenchSetInfluence;
    window.workbenchClear = workbenchClear;
    window.workbenchHas = workbenchHas;
    window.workbenchGetInfluence = workbenchGetInfluence;
    window.workbenchFlushSync = flushSync;
    window.getWorkbench = getWorkbench;

    // ---------- Workbench rendering ----------
    function sourceCounts() {
        let s = 0, e = 0;
        workbench.tracks.forEach(t => { if (t.source === 'search') s++; else e++; });
        return { search: s, extend: e };
    }

    function renderRail() {
        const list = document.getElementById('curator-wb-list');
        const totalEl = document.getElementById('curator-wb-total');
        const searchEl = document.getElementById('curator-wb-from-search');
        const extendEl = document.getElementById('curator-wb-from-extend');
        const saveBtn = document.getElementById('curator-wb-save-btn');
        const findDupsBtn = document.getElementById('curator-wb-finddups-btn');
        const clearBtn = document.getElementById('curator-wb-clear-btn');
        const nameInput = document.getElementById('curator-wb-name');
        if (!list) return;

        const total = workbench.tracks.length;
        const counts = sourceCounts();

        if (totalEl) totalEl.textContent = total;
        if (searchEl) searchEl.textContent = counts.search;
        if (extendEl) extendEl.textContent = counts.extend > 0 ? '+ ' + counts.extend : counts.extend;

        // List
        if (total === 0) {
            const onSearchPage = document.body.dataset.curatorPage === 'search';
            const msg = onSearchPage
                ? "Add tracks from your search results — they'll show up here."
                : "No seed yet. Switch to Smart Search to find tracks first.";
            list.innerHTML = '<div class="curator-wb-empty">' + escHtml(msg) + '</div>';
        } else {
            list.innerHTML = workbench.tracks.map(t => {
                const tone = t.source === 'search' ? 'primary' : 'success';
                const lbl = t.source === 'search' ? 'SRCH' : 'EXT';
                const dot = t.influence > 0
                    ? `<span class="curator-wb-influence-dot" data-level="${t.influence}" title="Influence: ${escHtml(getInfluenceInfo(t.influence).label)}"></span>`
                    : '';
                return `<div class="curator-wb-item">
                    <span class="curator-pill" data-tone="${tone}" style="font-size:9px;flex-shrink:0;">${lbl}</span>
                    <div class="curator-wb-item-meta">
                        <div class="curator-wb-item-title">${escHtml(t.title)}</div>
                        <div class="curator-wb-item-sub">${escHtml(t.author || '')}</div>
                    </div>
                    ${dot}
                    <button type="button" class="curator-remove-x" data-wb-remove="${escHtml(t.item_id)}" title="Remove">${ICONS.x}</button>
                </div>`;
            }).join('');
        }

        // Save / actions enable
        const canSave = total > 0 && nameInput && nameInput.value.trim().length > 0;
        if (saveBtn) saveBtn.disabled = !canSave;
        if (saveBtn) {
            const lbl = `Save ${total} ${total === 1 ? 'track' : 'tracks'}`;
            saveBtn.innerHTML = `${ICONS.save} <span>${escHtml(lbl)}</span>`;
        }
        if (findDupsBtn) findDupsBtn.disabled = total < 2;
        if (clearBtn) clearBtn.disabled = total === 0;
    }

    function renderSheet() {
        const sheet = document.getElementById('curator-workbench-sheet');
        if (!sheet) return;
        const total = workbench.tracks.length;
        const counts = sourceCounts();

        const handle = document.getElementById('curator-sheet-handle');
        const title = document.getElementById('curator-sheet-title');
        const sub = document.getElementById('curator-sheet-sub');
        const list = document.getElementById('curator-sheet-list');
        const saveBtn = document.getElementById('curator-sheet-save-btn');
        const findDupsBtn = document.getElementById('curator-sheet-finddups-btn');
        const clearBtn = document.getElementById('curator-sheet-clear-btn');
        const nameInput = document.getElementById('curator-sheet-name');

        if (handle) handle.classList.toggle('has-tracks', total > 0);
        if (title) title.textContent = `Workbench · ${total} ${total === 1 ? 'track' : 'tracks'}`;
        if (sub) sub.textContent = `${counts.search} from Search · ${counts.extend} from Extender`;

        if (list) {
            if (total === 0) {
                list.innerHTML = '<div class="curator-wb-empty">Empty. Add tracks from Smart Search to get started.</div>';
            } else {
                list.innerHTML = workbench.tracks.map(t => {
                    const tone = t.source === 'search' ? 'primary' : 'success';
                    const lbl = t.source === 'search' ? 'SRCH' : 'EXT';
                    return `<div class="curator-wb-item">
                        <span class="curator-pill" data-tone="${tone}" style="font-size:9px;flex-shrink:0;">${lbl}</span>
                        <div class="curator-wb-item-meta">
                            <div class="curator-wb-item-title">${escHtml(t.title)}</div>
                            <div class="curator-wb-item-sub">${escHtml(t.author || '')}</div>
                        </div>
                        <button type="button" class="curator-remove-x" data-wb-remove="${escHtml(t.item_id)}" title="Remove">${ICONS.x}</button>
                    </div>`;
                }).join('');
            }
        }
        const canSave = total > 0 && nameInput && nameInput.value.trim().length > 0;
        if (saveBtn) {
            saveBtn.disabled = !canSave;
            saveBtn.innerHTML = `${ICONS.save} <span>Save</span>`;
        }
        if (findDupsBtn) findDupsBtn.disabled = total < 2;
        if (clearBtn) clearBtn.disabled = total === 0;
    }

    function renderWorkbench() {
        renderRail();
        renderSheet();
    }
    window.renderWorkbench = renderWorkbench;

    // ---------- Sticky web player ----------
    let player = {
        audio: null,
        title: null,
        artist: null,
        playPauseBtn: null,
        currentTime: null,
        duration: null,
        progressFill: null,
        progressBar: null,
        currentId: null,
    };

    function initPlayer() {
        player.audio = document.getElementById('curator-audio');
        player.title = document.getElementById('curator-player-title');
        player.artist = document.getElementById('curator-player-artist');
        player.playPauseBtn = document.getElementById('curator-player-playpause');
        player.currentTime = document.getElementById('curator-player-current');
        player.duration = document.getElementById('curator-player-duration');
        player.progressFill = document.getElementById('curator-player-bar-fill');
        player.progressBar = document.getElementById('curator-player-bar');
        if (!player.audio) return;

        player.audio.addEventListener('timeupdate', updateProgress);
        player.audio.addEventListener('ended', () => {
            if (player.playPauseBtn) player.playPauseBtn.innerHTML = ICONS.play;
            updateRowPlayingIndicators();
        });
        player.audio.addEventListener('play', () => {
            if (player.playPauseBtn) player.playPauseBtn.innerHTML = ICONS.pause;
            updateRowPlayingIndicators();
        });
        player.audio.addEventListener('pause', () => {
            if (player.playPauseBtn) player.playPauseBtn.innerHTML = ICONS.play;
            updateRowPlayingIndicators();
        });

        if (player.progressBar) {
            player.progressBar.addEventListener('click', (e) => {
                if (!player.audio.duration) return;
                const rect = player.progressBar.getBoundingClientRect();
                const pct = (e.clientX - rect.left) / rect.width;
                player.audio.currentTime = pct * player.audio.duration;
            });
        }
    }

    function formatTime(s) {
        if (!s || isNaN(s)) return '0:00';
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${sec.toString().padStart(2, '0')}`;
    }

    function updateProgress() {
        if (!player.audio || isNaN(player.audio.duration)) return;
        if (player.currentTime) player.currentTime.textContent = formatTime(player.audio.currentTime);
        if (player.duration) player.duration.textContent = formatTime(player.audio.duration);
        if (player.progressFill) {
            const pct = (player.audio.currentTime / player.audio.duration) * 100;
            player.progressFill.style.width = pct + '%';
        }
    }

    function updateRowPlayingIndicators() {
        const isPlaying = player.audio && !player.audio.paused;
        document.querySelectorAll('.curator-icon-btn[data-stream]').forEach(btn => {
            const btnId = btn.dataset.itemId;
            const active = isPlaying && btnId === player.currentId;
            btn.classList.toggle('playing', active);
            btn.innerHTML = active ? ICONS.pause : ICONS.play;
        });
    }

    function playSong(itemId, url, title, artist) {
        if (!url || !player.audio) return;
        const playerEl = document.getElementById('curator-sticky-player');
        if (playerEl) playerEl.classList.add('visible');
        if (player.currentId === itemId && !player.audio.paused) {
            player.audio.pause();
            return;
        }
        if (player.currentId === itemId && player.audio.paused) {
            player.audio.play().catch(() => {});
            return;
        }
        player.currentId = itemId;
        if (player.title) player.title.textContent = title || 'Unknown';
        if (player.artist) player.artist.textContent = artist || '';
        player.audio.src = url;
        player.audio.play().catch(() => {});
    }
    window.curatorPlay = playSong;

    function togglePlay() {
        if (!player.audio) return;
        if (player.audio.paused) player.audio.play().catch(() => {});
        else player.audio.pause();
    }
    function stopPlayer() {
        if (!player.audio) return;
        player.audio.pause();
        player.audio.currentTime = 0;
        player.currentId = null;
        const playerEl = document.getElementById('curator-sticky-player');
        if (playerEl) playerEl.classList.remove('visible');
        updateRowPlayingIndicators();
    }
    function seek(seconds) {
        if (!player.audio) return;
        player.audio.currentTime += seconds;
    }
    window.curatorTogglePlay = togglePlay;
    window.curatorStopPlayer = stopPlayer;
    window.curatorSeek = seek;

    // ---------- Toast ----------
    function toast(msg, kind) {
        let stack = document.getElementById('curator-toast-stack');
        if (!stack) {
            stack = document.createElement('div');
            stack.id = 'curator-toast-stack';
            stack.className = 'curator-toast-stack';
            document.body.appendChild(stack);
        }
        const t = document.createElement('div');
        t.className = 'curator-toast ' + (kind || 'success');
        t.textContent = msg;
        stack.appendChild(t);
        setTimeout(() => { t.remove(); }, 2800);
    }
    window.curatorToast = toast;

    // ---------- Status helpers (for the seed/search-loading line) ----------
    function setStatus(targetId, msg, kind) {
        const el = document.getElementById(targetId);
        if (!el) return;
        el.className = 'curator-status' + (kind ? ' ' + kind : '');
        if (kind === 'loading') {
            el.innerHTML = '<span class="curator-spinner"></span>' + escHtml(msg);
        } else {
            el.textContent = msg || '';
        }
    }
    window.curatorSetStatus = setStatus;

    // ---------- Save playlist ----------
    async function savePlaylist(name, sourceId) {
        const wb = getWorkbench();
        if (!name || !name.trim()) { toast('Please enter a playlist name.', 'error'); return false; }
        if (wb.tracks.length === 0) { toast('Workbench is empty.', 'error'); return false; }

        const trackIds = wb.tracks.map(t => t.item_id);
        const payload = { new_playlist_name: name.trim(), track_ids: trackIds };
        try {
            const res = await fetch('/api/curator/save_playlist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Save failed');
            toast(`Saved "${name.trim()}" — ${trackIds.length} tracks`, 'success');
            workbenchClear();
            return true;
        } catch (e) {
            toast(e.message || 'Save failed', 'error');
            return false;
        }
    }
    window.curatorSavePlaylist = savePlaylist;

    // ---------- Find duplicates panel ----------
    let dedupGroups = [];

    function renderDedupGroups() {
        const container = document.getElementById('curator-dedup-groups');
        if (!container) return;
        let html = '';
        dedupGroups.forEach((group, gi) => {
            html += `<div class="curator-dedup-group">`;
            html += `<div class="curator-dedup-group-header">Group ${gi + 1}</div>`;
            group.tracks.forEach((track, ti) => {
                const isKeeper = ti === 0;
                const keepState = track._keep !== undefined ? track._keep : isKeeper;
                const stateClass = keepState ? 'keep' : 'remove';
                const badgeText = keepState ? 'KEEP' : 'REMOVE';
                const artist = escHtml(track.author || 'Unknown');
                const title = escHtml(track.title || 'Unknown');
                const album = escHtml(track.album || '');
                const year = track.year ? ` (${track.year})` : '';
                const rating = track.rating ? ' ' + '★'.repeat(Math.round(track.rating)) : '';
                html += `<div class="curator-dedup-track ${stateClass}" data-dg="${gi}" data-dt="${ti}">
                    <span class="curator-dedup-badge ${stateClass}">${badgeText}</span>
                    <div class="curator-dedup-track-info">
                        <div class="curator-dedup-track-title">${title}</div>
                        <div class="curator-dedup-track-meta">${artist} — ${album}${year}${rating}</div>
                    </div>
                    <span style="font-size:11px;color:var(--text-muted);">score: ${escHtml(track.score)}</span>
                </div>`;
            });
            html += `</div>`;
        });
        container.innerHTML = html;
    }

    function attachDedupHandlers() {
        const container = document.getElementById('curator-dedup-groups');
        if (!container || container._handlerAttached) return;
        container._handlerAttached = true;
        container.addEventListener('click', (e) => {
            const row = e.target.closest('.curator-dedup-track');
            if (!row) return;
            const gi = parseInt(row.dataset.dg, 10);
            const ti = parseInt(row.dataset.dt, 10);
            const group = dedupGroups[gi];
            if (!group) return;
            const clicked = group.tracks[ti];
            const currentState = clicked._keep !== undefined ? clicked._keep : (ti === 0);
            clicked._keep = !currentState;
            const anyOtherKeep = group.tracks.some((t, i) =>
                i !== ti && (t._keep !== undefined ? t._keep : (i === 0))
            );
            if (!anyOtherKeep && !clicked._keep) {
                clicked._keep = true; // keep at least one
            }
            renderDedupGroups();
        });
    }

    async function findDuplicates() {
        const wb = getWorkbench();
        if (wb.tracks.length < 2) { toast('Need at least 2 tracks to scan.', 'error'); return; }

        const panel = document.getElementById('curator-dedup-panel');
        const container = document.getElementById('curator-dedup-groups');
        const titleEl = document.getElementById('curator-dedup-title');
        const sliderEl = document.getElementById('curator-dedup-threshold');
        const threshold = sliderEl ? parseFloat(sliderEl.value) : 0.010;

        if (!panel || !container) return;
        panel.classList.remove('hidden');
        container.innerHTML = '<p class="curator-status loading"><span class="curator-spinner"></span>Scanning for duplicates...</p>';
        attachDedupHandlers();

        try {
            const res = await fetch('/api/curator/find_duplicates', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ track_ids: wb.tracks.map(t => t.item_id), threshold }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed');
            dedupGroups = data.groups || [];
            if (data.total_groups === 0) {
                container.innerHTML = '<p class="curator-empty-state">No duplicates found at this sensitivity level.</p>';
                if (titleEl) titleEl.textContent = 'No Duplicates Found';
                setTimeout(() => panel.classList.add('hidden'), 3000);
                return;
            }
            const removable = data.total_duplicate_tracks - data.total_groups;
            if (titleEl) titleEl.textContent = `Duplicates Found: ${data.total_groups} group${data.total_groups > 1 ? 's' : ''} (${removable} to remove)`;
            renderDedupGroups();
            panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (e) {
            container.innerHTML = `<p class="curator-status error">${escHtml(e.message)}</p>`;
        }
    }
    window.curatorFindDuplicates = findDuplicates;

    function removeAllMarkedDuplicates() {
        const idsToRemove = [];
        dedupGroups.forEach(group => {
            group.tracks.forEach((track, idx) => {
                const isKeeper = track._keep !== undefined ? track._keep : (idx === 0);
                if (!isKeeper) idsToRemove.push(track.item_id);
            });
        });
        if (idsToRemove.length === 0) return closeDedupPanel();
        idsToRemove.forEach(id => workbenchRemove(id));
        closeDedupPanel();
        toast(`Removed ${idsToRemove.length} duplicate${idsToRemove.length > 1 ? 's' : ''} from Workbench.`, 'success');
    }
    function closeDedupPanel() {
        const panel = document.getElementById('curator-dedup-panel');
        if (panel) panel.classList.add('hidden');
        dedupGroups = [];
    }
    window.curatorRemoveAllMarkedDuplicates = removeAllMarkedDuplicates;
    window.curatorCloseDedupPanel = closeDedupPanel;

    // ---------- Init ----------
    function attachWorkbenchHandlers() {
        // Rail handlers
        document.addEventListener('click', (e) => {
            const removeBtn = e.target.closest('[data-wb-remove]');
            if (removeBtn) {
                e.preventDefault();
                workbenchRemove(removeBtn.dataset.wbRemove);
                return;
            }
            const playBtn = e.target.closest('[data-stream]');
            if (playBtn) {
                e.preventDefault();
                playSong(playBtn.dataset.itemId, playBtn.dataset.stream, playBtn.dataset.title, playBtn.dataset.artist);
                return;
            }
        });

        // Save buttons (rail + sheet)
        const railSaveBtn = document.getElementById('curator-wb-save-btn');
        const railNameInput = document.getElementById('curator-wb-name');
        if (railSaveBtn && railNameInput) {
            railSaveBtn.addEventListener('click', async () => {
                const name = railNameInput.value;
                const ok = await savePlaylist(name);
                if (ok) railNameInput.value = '';
            });
            railNameInput.addEventListener('input', renderRail);
            railNameInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); railSaveBtn.click(); }
            });
        }
        const sheetSaveBtn = document.getElementById('curator-sheet-save-btn');
        const sheetNameInput = document.getElementById('curator-sheet-name');
        if (sheetSaveBtn && sheetNameInput) {
            sheetSaveBtn.addEventListener('click', async () => {
                const name = sheetNameInput.value;
                const ok = await savePlaylist(name);
                if (ok) {
                    sheetNameInput.value = '';
                    closeSheet();
                }
            });
            sheetNameInput.addEventListener('input', renderSheet);
            sheetNameInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); sheetSaveBtn.click(); }
            });
        }

        // Find duplicates buttons
        ['curator-wb-finddups-btn', 'curator-sheet-finddups-btn'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.addEventListener('click', findDuplicates);
        });

        // Clear buttons
        ['curator-wb-clear-btn', 'curator-sheet-clear-btn'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.addEventListener('click', () => {
                if (workbench.tracks.length === 0) return;
                if (confirm('Clear the entire Workbench?')) workbenchClear();
            });
        });

        // Sheet expand/collapse
        const handle = document.getElementById('curator-sheet-handle');
        const backdrop = document.getElementById('curator-sheet-backdrop');
        const sheet = document.getElementById('curator-workbench-sheet');
        if (handle && sheet) {
            handle.addEventListener('click', () => sheet.classList.toggle('open'));
        }
        if (backdrop && sheet) {
            backdrop.addEventListener('click', () => sheet.classList.remove('open'));
        }

        // Dedup close + remove-all
        const dedupClose = document.getElementById('curator-dedup-close');
        const dedupRemoveAll = document.getElementById('curator-dedup-removeall');
        if (dedupClose) dedupClose.addEventListener('click', closeDedupPanel);
        if (dedupRemoveAll) dedupRemoveAll.addEventListener('click', removeAllMarkedDuplicates);

        // Player buttons
        const ppBtn = document.getElementById('curator-player-playpause');
        if (ppBtn) ppBtn.addEventListener('click', togglePlay);
        const back10 = document.getElementById('curator-player-back');
        if (back10) back10.addEventListener('click', () => seek(-10));
        const fwd10 = document.getElementById('curator-player-fwd');
        if (fwd10) fwd10.addEventListener('click', () => seek(10));
        const stopBtn = document.getElementById('curator-player-stop');
        if (stopBtn) stopBtn.addEventListener('click', stopPlayer);
    }

    function closeSheet() {
        const sheet = document.getElementById('curator-workbench-sheet');
        if (sheet) sheet.classList.remove('open');
    }

    function init() {
        workbench = loadWorkbench();
        initPlayer();
        attachWorkbenchHandlers();
        renderWorkbench();
        // Topbar count badge
        document.addEventListener('curator:workbench:changed', () => {
            const badge = document.getElementById('curator-topbar-count-num');
            if (badge) badge.textContent = workbench.tracks.length;
            const badgeWrap = document.getElementById('curator-topbar-count');
            if (badgeWrap) badgeWrap.style.display = workbench.tracks.length > 0 ? '' : 'none';
        });
        // Initial badge
        const badge = document.getElementById('curator-topbar-count-num');
        if (badge) badge.textContent = workbench.tracks.length;
        const badgeWrap = document.getElementById('curator-topbar-count');
        if (badgeWrap) badgeWrap.style.display = workbench.tracks.length > 0 ? '' : 'none';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
