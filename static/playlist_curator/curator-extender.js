/* ============================================================
   Playlist Curator — Playlist Extender page
   - Seed dropdown: Workbench (default if non-empty) + AudioMuse
     cluster playlists + media-server playlists.
   - Tune-the-search panel (collapsible): similarity, min rating,
     max songs, year range, dup-mode + sensitivity.
   - "Find similar songs" -> POST /api/curator/search
       * payload shape preserved from existing implementation.
   - Render results with Distance + Influence columns; cycling
     Influence updates the workbench (track must already be in WB).
   - Near-duplicate rows highlighted yellow (when dup-mode='mark').
   ============================================================ */
(function () {
    'use strict';

    const ICONS = window.CURATOR_ICONS;
    const escHtml = window.escHtml;
    const getInfluenceInfo = window.getInfluenceInfo;

    const SEED_WORKBENCH = '__workbench__';

    let lastResults = [];
    let yearMin = null;
    let yearMax = null;
    let serverPlaylistsLoaded = false;
    let renderToken = 0;

    // Cache cluster-playlist tracks fetched on init so we can populate the
    // Workbench when the user picks one as a seed (avoids a second round trip).
    const clusterPlaylistsCache = {};

    function renderInChunks(container, rows, rowFn, opts) {
        const opt = opts || {};
        const firstChunk = opt.firstChunk || 500;
        const chunkSize = opt.chunkSize || 200;
        const myToken = ++renderToken;
        let i = 0;
        const renderSlice = (n) => {
            if (myToken !== renderToken || !container.isConnected) return;
            const slice = rows.slice(i, Math.min(i + n, rows.length));
            if (slice.length === 0) return;
            container.insertAdjacentHTML('beforeend', slice.map(rowFn).join(''));
            i += slice.length;
            if (i < rows.length) requestAnimationFrame(() => renderSlice(chunkSize));
        };
        renderSlice(firstChunk);
    }

    function debounce(fn, ms) {
        let h = null;
        return function () {
            if (h !== null) clearTimeout(h);
            const args = arguments;
            h = setTimeout(() => { h = null; fn.apply(null, args); }, ms);
        };
    }

    // Toggle dup-warn / hidden state on existing rows without rebuilding HTML.
    // Only touches rows that have data-dup-distance (i.e. actual near-dup candidates).
    function applyDupVisibility() {
        const dupModeEl = document.getElementById('curator-dup-mode');
        const dupSliderEl = document.getElementById('curator-dup-threshold');
        if (!dupModeEl || !dupSliderEl) return;
        const dupMode = dupModeEl.value;
        const dupSlider = parseFloat(dupSliderEl.value);
        const flagDups = dupMode !== 'off';
        const hideDups = dupMode === 'hide';

        const wrap = document.getElementById('curator-results-table-wrap');
        const cards = document.getElementById('curator-results-cards');
        const containers = [wrap, cards].filter(Boolean);
        let visibleCount = 0;
        // Count non-dup rows once (cheap: querySelectorAll on table only)
        const tbody = wrap ? wrap.querySelector('tbody') : null;
        const allRows = tbody ? tbody.querySelectorAll('tr[data-row-id]') : [];
        allRows.forEach(tr => {
            const hasDup = tr.hasAttribute('data-dup-distance');
            if (!hasDup) { visibleCount++; return; }
            const dist = parseFloat(tr.getAttribute('data-dup-distance'));
            const isDup = flagDups && dist < dupSlider;
            const isInWb = tr.classList.contains('in-wb');
            if (!isInWb) tr.classList.toggle('dup-warn', isDup);
            const hide = hideDups && isDup;
            tr.style.display = hide ? 'none' : '';
            if (!hide) visibleCount++;
            const badge = tr.querySelector('.curator-dup-badge');
            if (badge) badge.style.display = isDup ? '' : 'none';
        });
        // Mirror state to card list
        if (cards) {
            cards.querySelectorAll('.curator-track-card[data-row-id]').forEach(card => {
                const hasDup = card.hasAttribute('data-dup-distance');
                if (!hasDup) return;
                const dist = parseFloat(card.getAttribute('data-dup-distance'));
                const isDup = flagDups && dist < dupSlider;
                const isInWb = card.classList.contains('in-wb');
                if (!isInWb) card.classList.toggle('dup-warn', isDup);
                card.style.display = (hideDups && isDup) ? 'none' : '';
                const badge = card.querySelector('.curator-dup-badge');
                if (badge) badge.style.display = isDup ? '' : 'none';
            });
        }
        const headCount = document.getElementById('curator-results-count');
        if (headCount) headCount.textContent = visibleCount;
    }
    const applyDupVisibilityDebounced = debounce(applyDupVisibility, 120);

    // ---------- Initial state from form controls ----------
    function readControl(id, fallback) {
        const el = document.getElementById(id);
        return el ? el.value : fallback;
    }

    // ---------- Load filter options (year range, etc.) ----------
    async function loadFilterOptions() {
        try {
            const res = await fetch('/api/curator/filter_options');
            const data = await res.json();
            if (data && data.year_min && data.year_max) {
                yearMin = data.year_min;
                yearMax = data.year_max;
                const minSlider = document.getElementById('curator-year-min');
                const maxSlider = document.getElementById('curator-year-max');
                if (minSlider && maxSlider) {
                    minSlider.min = yearMin; minSlider.max = yearMax; minSlider.value = yearMin;
                    maxSlider.min = yearMin; maxSlider.max = yearMax; maxSlider.value = yearMax;
                    updateYearDisplay();
                }
            }
        } catch (e) {
            console.warn('Failed to load filter options:', e);
        }
    }

    // ---------- Load cluster + server playlists ----------
    async function loadClusterPlaylists() {
        try {
            const res = await fetch('/api/playlists');
            const playlists = await res.json();
            const select = document.getElementById('curator-seed-select');
            if (!select) return;
            // Insert cluster options after Workbench
            const group = document.createElement('optgroup');
            group.label = 'AudioMuse Playlists';
            group.dataset.kind = 'cluster';
            for (const [name, tracks] of Object.entries(playlists || {})) {
                clusterPlaylistsCache[name] = tracks;
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = `${name} (${tracks.length} songs)`;
                group.appendChild(opt);
            }
            if (group.children.length > 0) select.appendChild(group);
        } catch (e) {
            console.warn('Failed to load cluster playlists:', e);
        }
    }

    async function loadServerPlaylists() {
        if (serverPlaylistsLoaded) return;
        const statusEl = document.getElementById('curator-seed-status');
        if (statusEl) window.curatorSetStatus('curator-seed-status', 'Loading media server playlists…', 'loading');
        try {
            const res = await fetch('/api/curator/server_playlists');
            const playlists = await res.json();
            if (!res.ok) throw new Error(playlists.error || 'Failed');

            const select = document.getElementById('curator-seed-select');
            if (!select) return;
            select.querySelectorAll('optgroup[data-kind="server"]').forEach(o => o.remove());

            if (playlists.length > 0) {
                const group = document.createElement('optgroup');
                group.label = 'Media Server';
                group.dataset.kind = 'server';
                playlists.forEach(pl => {
                    const opt = document.createElement('option');
                    opt.value = '__server__' + pl.playlist_id;
                    opt.textContent = `${pl.playlist_name} (${pl.song_count} songs)`;
                    group.appendChild(opt);
                });
                select.appendChild(group);
            }
            window.curatorSetStatus('curator-seed-status',
                playlists.length > 0 ? `${playlists.length} media server playlist${playlists.length > 1 ? 's' : ''} loaded.` : 'No media server playlists.',
                playlists.length > 0 ? 'success' : '');
            setTimeout(() => window.curatorSetStatus('curator-seed-status', '', ''), 2500);
            serverPlaylistsLoaded = true;
        } catch (e) {
            window.curatorSetStatus('curator-seed-status', e.message, 'error');
        }
    }

    async function fetchServerPlaylistTracks(seedValue) {
        const playlistId = seedValue.replace('__server__', '');
        try {
            const res = await fetch('/api/curator/server_playlist_tracks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ playlist_id: playlistId }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load playlist tracks');
            return data;
        } catch (e) {
            window.curatorSetStatus('curator-seed-status', e.message, 'error');
            return null;
        }
    }

    // ---------- Auto-load seed playlist into Workbench ----------
    // When the user picks a server or cluster playlist as the seed, copy its
    // tracks into the Workbench (skipping duplicates) and switch the seed to
    // Workbench so per-track influence controls become available in the rail.
    async function loadSeedIntoWorkbench(seedValue) {
        if (!seedValue || seedValue === SEED_WORKBENCH) return;
        const statusId = 'curator-extender-status';

        let tracks = null;
        if (seedValue.startsWith('__server__')) {
            window.curatorSetStatus(statusId, 'Loading playlist tracks…', 'loading');
            const data = await fetchServerPlaylistTracks(seedValue);
            if (!data) return;
            tracks = data.tracks || [];
        } else if (clusterPlaylistsCache[seedValue]) {
            tracks = clusterPlaylistsCache[seedValue];
        }

        if (!tracks || tracks.length === 0) {
            window.curatorSetStatus(statusId, 'Playlist is empty.', 'error');
            return;
        }

        const added = window.workbenchAddBulk(tracks, 'extend');
        const select = document.getElementById('curator-seed-select');
        if (select) {
            refreshWorkbenchOption();
            select.value = SEED_WORKBENCH;
        }
        const skipped = tracks.length - added;
        const msg = added > 0
            ? `Loaded ${added} track${added === 1 ? '' : 's'} into Workbench${skipped > 0 ? ` (${skipped} already there)` : ''}. Tune influence on the right →`
            : 'All tracks were already in the Workbench.';
        window.curatorSetStatus(statusId, msg, 'success');
        setTimeout(() => window.curatorSetStatus(statusId, '', ''), 4000);
    }

    // ---------- Seed dropdown management ----------
    function refreshWorkbenchOption() {
        const select = document.getElementById('curator-seed-select');
        if (!select) return;
        const wb = window.getWorkbench();
        let opt = select.querySelector('option[value="' + SEED_WORKBENCH + '"]');
        if (wb.tracks.length > 0) {
            if (!opt) {
                opt = document.createElement('option');
                opt.value = SEED_WORKBENCH;
                select.insertBefore(opt, select.firstChild);
            }
            opt.textContent = `Workbench · ${wb.tracks.length} track${wb.tracks.length === 1 ? '' : 's'}`;
            // Auto-select Workbench if no other choice has been made yet
            if (!select.value || select.value === '') {
                select.value = SEED_WORKBENCH;
            }
        } else if (opt) {
            const wasSelected = select.value === SEED_WORKBENCH;
            opt.remove();
            if (wasSelected) select.value = '';
        }
    }

    // ---------- Year range display ----------
    function updateYearDisplay() {
        const minSlider = document.getElementById('curator-year-min');
        const maxSlider = document.getElementById('curator-year-max');
        const label = document.getElementById('curator-year-value');
        const track = document.getElementById('curator-year-track');
        if (!minSlider || !maxSlider) return;
        const minV = parseInt(minSlider.value);
        const maxV = parseInt(maxSlider.value);
        const dbMin = parseInt(minSlider.min);
        const dbMax = parseInt(maxSlider.max);
        if (label) label.textContent = (minV === dbMin && maxV === dbMax) ? 'Off' : `${minV} – ${maxV}`;
        if (track && (dbMax - dbMin) > 0) {
            const lp = ((minV - dbMin) / (dbMax - dbMin)) * 100;
            const rp = ((maxV - dbMin) / (dbMax - dbMin)) * 100;
            track.style.left = lp + '%';
            track.style.width = (rp - lp) + '%';
        }
    }

    // ---------- Run extend ----------
    async function runExtend() {
        const select = document.getElementById('curator-seed-select');
        const seedValue = select ? select.value : '';
        const statusId = 'curator-extender-status';

        if (!seedValue) {
            window.curatorSetStatus(statusId, 'Please choose a seed.', 'error');
            return;
        }

        const wb = window.getWorkbench();
        const maxSongs = parseInt(readControl('curator-max-songs', '50'), 10);
        const threshold = parseFloat(readControl('curator-similarity', '0.5'));
        const minRatingVal = parseFloat(readControl('curator-min-rating', '0'));
        const minRating = minRatingVal > 0 ? minRatingVal : null;

        const yMinEl = document.getElementById('curator-year-min');
        const yMaxEl = document.getElementById('curator-year-max');
        const yMinV = yMinEl ? parseInt(yMinEl.value) : null;
        const yMaxV = yMaxEl ? parseInt(yMaxEl.value) : null;
        const yDbMin = yMinEl ? parseInt(yMinEl.min) : null;
        const yDbMax = yMaxEl ? parseInt(yMaxEl.max) : null;
        const yearMinPayload = (yMinV != null && (yMinV !== yDbMin || yMaxV !== yDbMax)) ? yMinV : null;
        const yearMaxPayload = (yMaxV != null && (yMinV !== yDbMin || yMaxV !== yDbMax)) ? yMaxV : null;

        const dupMode = document.getElementById('curator-dup-mode').value;
        const flagDups = dupMode !== 'off';
        const dupSliderMax = parseFloat(document.getElementById('curator-dup-threshold').max);
        const duplicateThreshold = flagDups ? dupSliderMax : -1;

        const payload = {
            max_songs: maxSongs,
            similarity_threshold: threshold,
            min_rating: minRating,
            year_min: yearMinPayload,
            year_max: yearMaxPayload,
            included_ids: [],
            excluded_ids: [],
            search_only: false,
            source_weights: {},
            included_weights: {},
            duplicate_threshold: duplicateThreshold,
        };

        if (seedValue === SEED_WORKBENCH) {
            if (wb.tracks.length === 0) {
                window.curatorSetStatus(statusId, 'Workbench is empty. Add tracks from Smart Search first.', 'error');
                return;
            }
            payload.source_ids = wb.tracks.map(t => t.item_id);
            wb.tracks.forEach(t => { payload.source_weights[t.item_id] = t.influence || 0; });
        } else if (seedValue.startsWith('__server__')) {
            window.curatorSetStatus(statusId, 'Loading playlist tracks from media server…', 'loading');
            const data = await fetchServerPlaylistTracks(seedValue);
            if (!data) return;
            payload.source_ids = data.tracks.map(t => t.item_id);
        } else {
            payload.playlist_name = seedValue;
        }

        const runBtn = document.getElementById('curator-extender-run');
        if (runBtn) runBtn.disabled = true;
        window.curatorSetStatus(statusId, 'Finding similar songs…', 'loading');

        try {
            const res = await fetch('/api/curator/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Extend failed');
            lastResults = Array.isArray(data.results) ? data.results : [];
            renderResults();
            window.curatorSetStatus(statusId, '', '');
        } catch (e) {
            window.curatorSetStatus(statusId, e.message || 'Extend failed', 'error');
        } finally {
            if (runBtn) runBtn.disabled = false;
        }
    }

    // ---------- Render results ----------
    function renderResults() {
        const section = document.getElementById('curator-results-section');
        if (!section) return;
        section.classList.remove('hidden');

        const dupMode = document.getElementById('curator-dup-mode').value;
        const dupSlider = parseFloat(document.getElementById('curator-dup-threshold').value);
        const flagDups = dupMode !== 'off';

        const wrap = document.getElementById('curator-results-table-wrap');
        const cards = document.getElementById('curator-results-cards');

        if (lastResults.length === 0) {
            renderToken++;
            const empty = `<div class="curator-empty-state">No similar tracks found. Try raising the threshold.</div>`;
            if (wrap) wrap.innerHTML = empty;
            if (cards) cards.innerHTML = empty;
            const headCount = document.getElementById('curator-results-count');
            if (headCount) headCount.textContent = 0;
            return;
        }

        // Render all rows in chunks; applyDupVisibility hides flagged rows and
        // updates the count header. Streaming via rAF keeps the main thread free.
        const rowFn = t => rowHtml(t, dupSlider, flagDups);
        const cardFn = t => cardHtml(t, dupSlider, flagDups);
        if (wrap) {
            wrap.innerHTML = renderTableShell();
            const tbody = wrap.querySelector('tbody');
            if (tbody) renderInChunks(tbody, lastResults, rowFn);
        }
        if (cards) {
            cards.innerHTML = '';
            renderInChunks(cards, lastResults, cardFn);
        }
        applyDupVisibility();
    }

    function renderTableShell() {
        return `<div class="curator-card flush">
            <table class="curator-table">
                <thead><tr>
                    <th class="col-play"></th>
                    <th>Title / Artist</th>
                    <th>Album</th>
                    <th class="col-distance">Distance</th>
                    <th class="col-influence">Influence</th>
                    <th class="col-actions">Action</th>
                </tr></thead>
                <tbody></tbody>
            </table>
        </div>`;
    }

    function rowMeta(t) {
        const id = escHtml(t.item_id);
        const artist = escHtml(t.song_artist || t.author || 'Unknown');
        const title = escHtml(t.title || 'Unknown');
        const album = escHtml(t.album || '-');
        const stream = '/api/curator/stream/' + encodeURIComponent(t.item_id);
        const distance = (t.distance != null) ? t.distance.toFixed(3) : 'N/A';
        return { id, artist, title, album, stream, distance };
    }

    // Build a plain-text tooltip describing which source track a result duplicates.
    // Used as the title="" attr on the warn pill so hovering reveals the source.
    function buildDupTooltip(dupOf) {
        if (!dupOf) return '';
        const title = dupOf.title || 'Unknown';
        const author = dupOf.author || 'Unknown';
        const album = dupOf.album ? ` · ${dupOf.album}` : '';
        const dist = (typeof dupOf.distance === 'number') ? ` (distance ${dupOf.distance.toFixed(3)})` : '';
        return `Duplicate of "${title}" by ${author}${album}${dist}`;
    }

    function rowHtml(t, dupSlider, flagDups) {
        const inWb = window.workbenchHas(t.item_id);
        const dupDist = (t.duplicate_of && typeof t.duplicate_of.distance === 'number') ? t.duplicate_of.distance : null;
        const isDup = flagDups && dupDist !== null && dupDist < dupSlider;
        const rowCls = inWb ? 'in-wb' : (isDup ? 'dup-warn' : '');
        const m = rowMeta(t);
        const inf = window.workbenchGetInfluence(t.item_id);
        const infInfo = getInfluenceInfo(inf);
        // Always emit the badge for rows with duplicate_of so the slider can
        // toggle visibility without rebuilding the row HTML.
        const dupTooltip = dupDist !== null ? buildDupTooltip(t.duplicate_of) : '';
        const dupBadge = dupDist !== null
            ? `<div class="curator-dup-badge" style="margin-top:4px;${isDup ? '' : 'display:none;'}"><span class="curator-pill" data-tone="warn" title="${escHtml(dupTooltip)}">${ICONS.warn} Near-duplicate (${dupDist.toFixed(3)})</span></div>`
            : '';
        const influenceCell = inWb
            ? `<button type="button" class="curator-influence-btn" data-level="${inf}" data-influence-id="${m.id}" title="${escHtml(infInfo.tip)}">${escHtml(infInfo.label)}</button>`
            : `<span style="color:var(--text-muted);font-size:11px;">—</span>`;
        const actionHtml = inWb ? `
            <div class="curator-row-actions">
                <span class="curator-pill" data-tone="success" title="In Workbench">${ICONS.check} Added</span>
                <button type="button" class="curator-remove-x" data-wb-remove="${m.id}" title="Remove from Workbench">${ICONS.x}</button>
            </div>` : `
            <div class="curator-row-actions">
                <button type="button" class="curator-btn" data-kind="success" data-size="sm" data-extend-add="${m.id}">${ICONS.plus} Add</button>
            </div>`;
        const dupAttr = dupDist !== null ? ` data-dup-distance="${dupDist}"` : '';
        return `<tr class="${rowCls}" data-row-id="${m.id}"${dupAttr}>
            <td class="col-play">
                <button type="button" class="curator-icon-btn" data-stream="${escHtml(m.stream)}" data-item-id="${m.id}" data-title="${m.title}" data-artist="${m.artist}">${ICONS.play}</button>
            </td>
            <td>
                <div class="curator-track-cell-title">${m.title}</div>
                <div class="curator-track-cell-sub">${m.artist}</div>
                ${dupBadge}
            </td>
            <td class="col-album">${m.album}</td>
            <td class="col-distance">${m.distance}</td>
            <td class="col-influence">${influenceCell}</td>
            <td class="col-actions">${actionHtml}</td>
        </tr>`;
    }

    function cardHtml(t, dupSlider, flagDups) {
        const inWb = window.workbenchHas(t.item_id);
        const dupDist = (t.duplicate_of && typeof t.duplicate_of.distance === 'number') ? t.duplicate_of.distance : null;
        const isDup = flagDups && dupDist !== null && dupDist < dupSlider;
        const cardCls = inWb ? 'in-wb' : (isDup ? 'dup-warn' : '');
        const m = rowMeta(t);
        const inf = window.workbenchGetInfluence(t.item_id);
        const infInfo = getInfluenceInfo(inf);

        const action = inWb ? `
            <button type="button" class="curator-added-text" data-wb-remove="${m.id}">${ICONS.check} Added</button>` : `
            <button type="button" class="curator-btn" data-kind="success" data-size="sm" data-extend-add="${m.id}">+ Add</button>`;
        const inflBtn = inWb
            ? `<div style="margin-top:6px;"><button type="button" class="curator-influence-btn" data-level="${inf}" data-influence-id="${m.id}" title="${escHtml(infInfo.tip)}">${escHtml(infInfo.label)}</button></div>`
            : '';
        const dupTooltipCard = dupDist !== null ? buildDupTooltip(t.duplicate_of) : '';
        const dup = dupDist !== null
            ? `<div class="curator-dup-badge" style="margin-top:6px;${isDup ? '' : 'display:none;'}"><span class="curator-pill" data-tone="warn" title="${escHtml(dupTooltipCard)}">${ICONS.warn} Near-duplicate</span></div>`
            : '';
        const dupAttr = dupDist !== null ? ` data-dup-distance="${dupDist}"` : '';

        return `<div class="curator-track-card ${cardCls}" data-row-id="${m.id}"${dupAttr}>
            <div style="display:flex;align-items:center;gap:10px;width:100%;">
                <button type="button" class="curator-icon-btn" data-stream="${escHtml(m.stream)}" data-item-id="${m.id}" data-title="${m.title}" data-artist="${m.artist}">${ICONS.play}</button>
                <div class="curator-track-card-meta">
                    <div class="curator-track-card-title">${m.title}</div>
                    <div class="curator-track-card-sub">${m.artist} · dist ${m.distance}</div>
                </div>
                ${action}
            </div>
            ${inflBtn}${dup}
        </div>`;
    }

    // Read current dup-mode state straight from the DOM so per-row swaps stay
    // in sync with whatever the user has selected without re-running the full filter.
    function currentDupState() {
        const slider = document.getElementById('curator-dup-threshold');
        const mode = document.getElementById('curator-dup-mode');
        return {
            dupSlider: slider ? parseFloat(slider.value) : 0.05,
            flagDups: !!mode && mode.value !== 'off',
        };
    }

    function updateRowsForChanges(changedIds) {
        if (!Array.isArray(changedIds) || changedIds.length === 0) return false;
        if (lastResults.length === 0) return false;
        const idSet = new Set(changedIds.map(String));
        const tracks = lastResults.filter(r => idSet.has(String(r.item_id)));
        if (tracks.length === 0) return false;
        const { dupSlider, flagDups } = currentDupState();
        let updated = false;
        tracks.forEach(t => {
            const sel = `[data-row-id="${CSS.escape(String(t.item_id))}"]`;
            const tr = document.querySelector('tr' + sel);
            if (tr) {
                const tmp = document.createElement('tbody');
                tmp.innerHTML = rowHtml(t, dupSlider, flagDups);
                const next = tmp.firstElementChild;
                if (next) { tr.replaceWith(next); updated = true; }
            }
            const card = document.querySelector('.curator-track-card' + sel);
            if (card) {
                const tmp = document.createElement('div');
                tmp.innerHTML = cardHtml(t, dupSlider, flagDups);
                const next = tmp.firstElementChild;
                if (next) { card.replaceWith(next); updated = true; }
            }
        });
        return updated;
    }

    // ---------- Init ----------
    async function init() {
        await loadFilterOptions();
        refreshWorkbenchOption();
        await Promise.all([loadClusterPlaylists(), loadServerPlaylists()]);
        // Re-apply default selection if Workbench just became available after loading other playlists
        refreshWorkbenchOption();

        // Seed select change clears stale results AND auto-loads non-Workbench
        // playlists into the Workbench so the user can tune per-track influence.
        const seedSelect = document.getElementById('curator-seed-select');
        if (seedSelect) seedSelect.addEventListener('change', () => {
            lastResults = [];
            const section = document.getElementById('curator-results-section');
            if (section) section.classList.add('hidden');
            window.curatorSetStatus('curator-extender-status', '', '');
            const v = seedSelect.value;
            if (v && v !== SEED_WORKBENCH) loadSeedIntoWorkbench(v);
        });

        // Tune toggle
        const tuneToggle = document.getElementById('curator-tune-toggle');
        const tuneGrid = document.getElementById('curator-tune-grid');
        if (tuneToggle && tuneGrid) {
            // Default: hidden on mobile, open on desktop
            const isMobile = window.matchMedia('(max-width: 768px)').matches;
            if (isMobile) {
                tuneGrid.classList.add('hidden');
            } else {
                tuneToggle.classList.add('open');
            }
            tuneToggle.addEventListener('click', () => {
                tuneGrid.classList.toggle('hidden');
                tuneToggle.classList.toggle('open');
            });
        }

        // Sliders + live values
        const simSlider = document.getElementById('curator-similarity');
        const simVal = document.getElementById('curator-similarity-value');
        if (simSlider && simVal) {
            simSlider.addEventListener('input', () => { simVal.textContent = parseFloat(simSlider.value).toFixed(2); });
            simVal.textContent = parseFloat(simSlider.value).toFixed(2);
        }
        const ratingSlider = document.getElementById('curator-min-rating');
        const ratingVal = document.getElementById('curator-min-rating-value');
        function paintRating() {
            const v = parseFloat(ratingSlider.value);
            ratingVal.textContent = v === 0 ? 'Off' : v.toFixed(1) + '★';
        }
        if (ratingSlider && ratingVal) {
            ratingSlider.addEventListener('input', paintRating);
            paintRating();
        }
        const dupSlider = document.getElementById('curator-dup-threshold');
        const dupVal = document.getElementById('curator-dup-threshold-value');
        if (dupSlider && dupVal) {
            dupSlider.addEventListener('input', () => {
                // Slider label updates immediately for responsiveness; the row
                // visibility pass is debounced so a drag doesn't fire 20×/sec.
                dupVal.textContent = parseFloat(dupSlider.value).toFixed(3);
                if (lastResults.length > 0) applyDupVisibilityDebounced();
            });
            dupVal.textContent = parseFloat(dupSlider.value).toFixed(3);
        }
        const dupMode = document.getElementById('curator-dup-mode');
        const dupSensWrap = document.getElementById('curator-dup-sens');
        function paintDupMode() {
            if (dupSensWrap) dupSensWrap.style.display = dupMode.value === 'off' ? 'none' : '';
            if (lastResults.length > 0) applyDupVisibility();
        }
        if (dupMode) { dupMode.addEventListener('change', paintDupMode); paintDupMode(); }

        // Year range sliders
        const yMin = document.getElementById('curator-year-min');
        const yMax = document.getElementById('curator-year-max');
        if (yMin && yMax) {
            yMin.addEventListener('input', () => {
                if (parseInt(yMin.value) > parseInt(yMax.value)) yMin.value = yMax.value;
                updateYearDisplay();
            });
            yMax.addEventListener('input', () => {
                if (parseInt(yMax.value) < parseInt(yMin.value)) yMax.value = yMin.value;
                updateYearDisplay();
            });
        }

        // Run button
        const runBtn = document.getElementById('curator-extender-run');
        if (runBtn) runBtn.addEventListener('click', runExtend);

        // Result row delegation: Add only. The [data-influence-id] click
        // handler lives in curator-shared.js so both pages share it (and to
        // avoid double-cycling when the same listener fires twice).
        document.addEventListener('click', (e) => {
            const addBtn = e.target.closest('[data-extend-add]');
            if (addBtn) {
                e.preventDefault();
                const id = addBtn.dataset.extendAdd;
                const track = lastResults.find(r => r.item_id === id);
                if (track) window.workbenchAdd(track, 'extend');
                return;
            }
        });

        // Re-render on workbench changes — surgical when possible.
        document.addEventListener('curator:workbench:changed', (e) => {
            refreshWorkbenchOption();
            if (lastResults.length === 0) return;
            const changedIds = e && e.detail ? e.detail.changedIds : null;
            if (changedIds && updateRowsForChanges(changedIds)) return;
            renderResults();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
